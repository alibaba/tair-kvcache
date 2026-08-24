#include "kv_cache_manager/optimizer/service/event_subscriber/kvcm_event_subscriber.h"

#include <algorithm>
#include <chrono>
#include <grpcpp/grpcpp.h>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/service_discovery.h"
#include "kv_cache_manager/common/service_discovery_factory.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_reporter.h"
#include "kv_cache_manager/optimizer/service/optimizer_call_guard.h"
#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"
#include "kv_cache_manager/service/util/common.h"

namespace kv_cache_manager {

namespace {

constexpr std::chrono::milliseconds kReconnectBaseDelay(500);
constexpr std::chrono::milliseconds kReconnectMaxDelay(30000);
constexpr std::chrono::seconds kStableStreamDuration(30);
constexpr int kReconnectJitterPercent = 20;
constexpr std::chrono::minutes kKeepaliveTime(6);
constexpr std::chrono::seconds kKeepaliveTimeout(20);
constexpr std::chrono::milliseconds kUnaryRpcTimeout(1000);

} // namespace

KvcmEventSubscriber::KvcmEventSubscriber(const KvcmEventSubscriptionConfig &config,
                                         std::shared_ptr<OptimizerServiceImpl> optimizer_service,
                                         std::shared_ptr<MetricsRegistry> metrics_registry,
                                         std::shared_ptr<OptimizerMetricsReporter> metrics_reporter)
    : config_(config)
    , optimizer_service_(std::move(optimizer_service))
    , metrics_registry_(std::move(metrics_registry))
    , metrics_reporter_(std::move(metrics_reporter)) {}

KvcmEventSubscriber::~KvcmEventSubscriber() { Stop(); }

bool KvcmEventSubscriber::Init() {
    if (!config_.enable()) {
        return true;
    }
    if (!optimizer_service_) {
        KVCM_LOG_ERROR("KvcmEventSubscriber: optimizer service is null");
        return false;
    }
    service_discovery_ = ServiceDiscoveryFactory::CreateServiceDiscovery(config_.service_discovery_url());
    if (!service_discovery_) {
        KVCM_LOG_ERROR("KvcmEventSubscriber: invalid service discovery URL[%s]",
                       config_.service_discovery_url().c_str());
        return false;
    }
    return true;
}

bool KvcmEventSubscriber::Start() {
    if (!config_.enable()) {
        return true;
    }
    if (!service_discovery_ || !optimizer_service_) {
        return false;
    }
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return true;
    }
    supervisor_thread_ = std::thread(&KvcmEventSubscriber::SupervisorLoop, this);
    KVCM_LOG_INFO("KvcmEventSubscriber: started, discovery_type=%s consumer_id=%s",
                  service_discovery_->GetType().c_str(),
                  config_.consumer_id().c_str());
    return true;
}

void KvcmEventSubscriber::Stop() {
    if (!running_.exchange(false)) {
        return;
    }
    wait_cv_.notify_all();
    if (supervisor_thread_.joinable()) {
        supervisor_thread_.join();
    }

    std::unique_ptr<EndpointWorker> worker;
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        worker = std::move(worker_);
    }
    if (worker) {
        StopWorker(std::move(worker));
    }
    KVCM_LOG_INFO("KvcmEventSubscriber: stopped");
}

void KvcmEventSubscriber::SupervisorLoop() {
    while (running_) {
        RefreshLeader();
        if (!WaitForSupervisor(std::chrono::milliseconds(config_.discovery_refresh_interval_ms()))) {
            break;
        }
    }
}

void KvcmEventSubscriber::RefreshLeader() {
    if (!service_discovery_->Refresh()) {
        KVCM_LOG_WARN("KvcmEventSubscriber: service discovery refresh failed");
        return;
    }

    std::string leader_endpoint;
    if (!DiscoverLeader(leader_endpoint)) {
        return;
    }

    if (!SyncConfiguration(leader_endpoint)) {
        return;
    }
    UpdateWorker(leader_endpoint);
}

bool KvcmEventSubscriber::DiscoverLeader(std::string &leader_endpoint) {
    std::vector<ServiceEndpoint> endpoints;
    if (!service_discovery_->GetAllEndpoints(endpoints)) {
        KVCM_LOG_WARN("KvcmEventSubscriber: no KVCM endpoints discovered");
        return false;
    }

    for (const auto &endpoint : endpoints) {
        if (!endpoint.healthy || endpoint.host.empty()) {
            continue;
        }
        auto channel = grpc::CreateChannel(endpoint.host, grpc::InsecureChannelCredentials());
        auto stub = proto::meta::MetaService::NewStub(channel);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + kUnaryRpcTimeout);
        proto::meta::GetClusterInfoRequest request;
        request.set_trace_id(config_.consumer_id() + "-leader-discovery");
        proto::meta::GetClusterInfoResponse response;
        const grpc::Status grpc_status = stub->GetClusterInfo(&context, request, &response);
        if (!grpc_status.ok() || !response.has_header() || response.header().status().code() != proto::meta::OK ||
            !response.has_leader_endpoint()) {
            KVCM_LOG_WARN("KvcmEventSubscriber: GetClusterInfo failed via seed[%s], grpc_code=%d",
                          endpoint.host.c_str(),
                          static_cast<int>(grpc_status.error_code()));
            continue;
        }
        const auto &leader = response.leader_endpoint();
        if (leader.host().empty() || leader.meta_rpc_port() <= 0 || leader.meta_rpc_port() > 65535) {
            KVCM_LOG_WARN("KvcmEventSubscriber: invalid leader endpoint returned by seed[%s]", endpoint.host.c_str());
            continue;
        }
        leader_endpoint = leader.host() + ":" + std::to_string(leader.meta_rpc_port());
        return true;
    }

    KVCM_LOG_WARN("KvcmEventSubscriber: failed to discover KVCM leader from all healthy seeds");
    return false;
}

bool KvcmEventSubscriber::SyncConfiguration(const std::string &leader_endpoint) {
    auto channel = grpc::CreateChannel(leader_endpoint, grpc::InsecureChannelCredentials());
    auto stub = proto::optimizer::OptimizerEventStreamService::NewStub(channel);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + kUnaryRpcTimeout);
    proto::optimizer::KvcmConfigurationRequest request;
    request.set_trace_id(config_.consumer_id() + "-configuration");
    proto::optimizer::KvcmConfigurationResponse response;
    const grpc::Status grpc_status = stub->GetConfiguration(&context, request, &response);
    if (!grpc_status.ok() || !response.has_header() || response.header().status().code() != proto::optimizer::OK) {
        KVCM_LOG_WARN("KvcmEventSubscriber: GetConfiguration failed from leader[%s], grpc_code=%d",
                      leader_endpoint.c_str(),
                      static_cast<int>(grpc_status.error_code()));
        return false;
    }

    std::unordered_set<std::string> unsupported_instance_ids;
    const ErrorCode ec = optimizer_service_->ApplyKvcmConfiguration(response, unsupported_instance_ids);
    if (ec != EC_OK) {
        KVCM_LOG_WARN("KvcmEventSubscriber: apply configuration from leader[%s] failed, ec=%d",
                      leader_endpoint.c_str(),
                      static_cast<int>(ec));
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(unsupported_instances_mutex_);
        unsupported_instance_ids_ = std::move(unsupported_instance_ids);
    }
    KVCM_LOG_INFO("KvcmEventSubscriber: configuration synchronized from leader[%s], groups=%d instances=%d",
                  leader_endpoint.c_str(),
                  response.instance_groups_size(),
                  response.instances_size());
    return true;
}

void KvcmEventSubscriber::UpdateWorker(const std::string &leader_endpoint) {
    std::unique_ptr<EndpointWorker> old_worker;
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        if (worker_ && worker_->endpoint == leader_endpoint) {
            return;
        }
        old_worker = std::move(worker_);
    }
    if (old_worker) {
        KVCM_LOG_INFO("KvcmEventSubscriber: disconnect old leader[%s]", old_worker->endpoint.c_str());
        StopWorker(std::move(old_worker));
    }
    if (!running_) {
        return;
    }

    auto worker = std::make_unique<EndpointWorker>(leader_endpoint);
    EndpointWorker *worker_ptr = worker.get();
    worker->thread = std::thread(&KvcmEventSubscriber::EndpointLoop, this, worker_ptr);
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        worker_ = std::move(worker);
    }
    KVCM_LOG_INFO("KvcmEventSubscriber: connected leader[%s]", leader_endpoint.c_str());
}

void KvcmEventSubscriber::EndpointLoop(EndpointWorker *worker) {
    uint32_t failed_attempts = 0;
    while (running_ && worker->running) {
        grpc::ChannelArguments channel_args;
        channel_args.SetInt(
            GRPC_ARG_KEEPALIVE_TIME_MS,
            static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(kKeepaliveTime).count()));
        channel_args.SetInt(
            GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
            static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(kKeepaliveTimeout).count()));
        auto channel = grpc::CreateCustomChannel(worker->endpoint, grpc::InsecureChannelCredentials(), channel_args);
        auto stub = proto::optimizer::OptimizerEventStreamService::NewStub(channel);
        grpc::ClientContext context;
        {
            std::lock_guard<std::mutex> lock(worker->context_mutex);
            if (!running_ || !worker->running) {
                break;
            }
            worker->context = &context;
        }

        proto::optimizer::OptimizerEventSubscriptionRequest request;
        request.set_consumer_id(config_.consumer_id());
        const auto stream_started_at = std::chrono::steady_clock::now();
        bool received_event = false;
        auto reader = stub->SubscribeEvents(&context, request);
        proto::optimizer::TraceQueryRequest event;
        std::string kvcm_ip;
        while (running_ && worker->running && reader->Read(&event)) {
            received_event = true;
            if (kvcm_ip.empty()) {
                kvcm_ip = ExtractIpFromPeer(context.peer());
            }
            ProcessEvent(event, kvcm_ip);
        }
        const grpc::Status status = reader->Finish();
        {
            std::lock_guard<std::mutex> lock(worker->context_mutex);
            worker->context = nullptr;
        }

        if (running_ && worker->running) {
            if (received_event || std::chrono::steady_clock::now() - stream_started_at >= kStableStreamDuration) {
                failed_attempts = 0;
            }
            const auto reconnect_delay = ComputeReconnectDelay(failed_attempts);
            if (failed_attempts < std::numeric_limits<uint32_t>::max()) {
                ++failed_attempts;
            }
            KVCM_LOG_WARN("KvcmEventSubscriber: stream[%s] closed, code=%d message=%s retry_in_ms=%lld",
                          worker->endpoint.c_str(),
                          static_cast<int>(status.error_code()),
                          status.error_message().c_str(),
                          static_cast<long long>(reconnect_delay.count()));
            if (!WaitForReconnect(worker, reconnect_delay)) {
                break;
            }
        }
    }
}

void KvcmEventSubscriber::StopWorker(std::unique_ptr<EndpointWorker> worker) {
    worker->running = false;
    wait_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lock(worker->context_mutex);
        if (worker->context) {
            worker->context->TryCancel();
        }
    }
    if (worker->thread.joinable()) {
        worker->thread.join();
    }
}

void KvcmEventSubscriber::ProcessEvent(const proto::optimizer::TraceQueryRequest &event, const std::string &kvcm_ip) {
    std::shared_ptr<OptimizerServiceMetricsCollector> collector;
    if (metrics_registry_) {
        collector = std::make_shared<OptimizerServiceMetricsCollector>(metrics_registry_);
        if (!collector->Init()) {
            collector.reset();
        } else {
            collector->set_instance_id(event.instance_id());
            collector->set_service_error_code_metrics(static_cast<double>(EC_OK));
        }
    }

    RequestContext request_context(event.trace_id(), collector);
    request_context.set_api_name("TraceQuery");
    request_context.set_client_ip(kvcm_ip);

    proto::optimizer::TraceQueryResponse response;
    ErrorCode ec = EC_OK;
    {
        OptimizerCallGuard guard(&request_context, metrics_reporter_.get());
        ec = optimizer_service_->ExecuteTraceQuery(event, &response);
        request_context.set_status_code(static_cast<int>(ec));
        if (ec == EC_OK) {
            if (collector) {
                collector->set_total_blocks(response.total_blocks());
                std::vector<PerCapacityHitInfo> per_capacity_hits;
                per_capacity_hits.reserve(response.capacity_results_size());
                for (const auto &capacity_result : response.capacity_results()) {
                    per_capacity_hits.push_back(
                        {capacity_result.capacity_gb(), capacity_result.cache_hit_count(), capacity_result.hit_rate()});
                }
                collector->set_per_capacity_hits(std::move(per_capacity_hits));
                collector->set_max_hit_count(response.theoretical_result().max_hit_count());
                if (response.theoretical_result().max_hit_count() >= 0) {
                    collector->set_max_hit_rate(response.theoretical_result().hit_rate());
                }
            }
        } else if (collector) {
            collector->set_service_error_code_metrics(static_cast<double>(ec));
        }
    }
    if (ec == EC_INSTANCE_NOT_EXIST) {
        std::lock_guard<std::mutex> lock(unsupported_instances_mutex_);
        if (unsupported_instance_ids_.find(event.instance_id()) != unsupported_instance_ids_.end()) {
            KVCM_LOG_DEBUG("KvcmEventSubscriber: ignore unsupported instance event, trace_id=%s instance_id=%s",
                           event.trace_id().c_str(),
                           event.instance_id().c_str());
            return;
        }
    }
    if (ec != EC_OK) {
        KVCM_LOG_WARN("KvcmEventSubscriber: drop event, trace_id=%s instance_id=%s ec=%d",
                      event.trace_id().c_str(),
                      event.instance_id().c_str(),
                      static_cast<int>(ec));
        if (ec == EC_INSTANCE_NOT_EXIST) {
            RequestConfigurationRefresh();
        }
    }
}

void KvcmEventSubscriber::RequestConfigurationRefresh() {
    {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        configuration_refresh_requested_ = true;
    }
    wait_cv_.notify_all();
}

bool KvcmEventSubscriber::WaitForSupervisor(std::chrono::milliseconds duration) {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    wait_cv_.wait_for(lock, duration, [this] { return !running_ || configuration_refresh_requested_; });
    configuration_refresh_requested_ = false;
    return running_;
}

bool KvcmEventSubscriber::WaitForReconnect(EndpointWorker *worker, std::chrono::milliseconds duration) {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    return !wait_cv_.wait_for(lock, duration, [this, worker] { return !running_ || !worker->running; });
}

std::chrono::milliseconds KvcmEventSubscriber::ComputeReconnectDelay(uint32_t failed_attempts) {
    constexpr uint32_t kMaxShift = 16;
    const int64_t multiplier = int64_t{1} << std::min(failed_attempts, kMaxShift);
    const int64_t nominal_ms = std::min(kReconnectBaseDelay.count() * multiplier, kReconnectMaxDelay.count());
    const int64_t lower_ms = nominal_ms * (100 - kReconnectJitterPercent) / 100;
    const int64_t upper_ms = std::min(nominal_ms * (100 + kReconnectJitterPercent) / 100, kReconnectMaxDelay.count());
    thread_local std::mt19937 generator(std::random_device{}());
    return std::chrono::milliseconds(std::uniform_int_distribution<int64_t>(lower_ms, upper_ms)(generator));
}

} // namespace kv_cache_manager
