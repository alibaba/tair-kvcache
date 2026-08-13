#include "kv_cache_manager/optimizer/service/kvcm_event_subscriber.h"

#include <algorithm>
#include <chrono>
#include <grpcpp/grpcpp.h>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/service_discovery.h"
#include "kv_cache_manager/common/service_discovery_factory.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

namespace {

constexpr std::chrono::milliseconds kReconnectDelay(500);
constexpr std::chrono::milliseconds kUnaryRpcTimeout(1000);
constexpr long double kBytesPerGb = 1024.0L * 1024.0L * 1024.0L;

bool IsFullLocationSpecGroup(const std::string &name) {
    return name.rfind("full", 0) == 0 || name.rfind("FULL", 0) == 0;
}

} // namespace

KvcmEventSubscriber::KvcmEventSubscriber(const KvcmEventSubscriptionConfig &config,
                                         std::shared_ptr<OnlineOptimizerManager> manager)
    : config_(config), manager_(std::move(manager)) {}

KvcmEventSubscriber::~KvcmEventSubscriber() { Stop(); }

bool KvcmEventSubscriber::Init() {
    if (!config_.enable()) {
        return true;
    }
    if (!manager_) {
        KVCM_LOG_ERROR("KvcmEventSubscriber: optimizer manager is null");
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
    if (!service_discovery_ || !manager_) {
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

    SyncConfiguration(leader_endpoint);
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

    auto registry = manager_->registry_manager();
    if (!registry) {
        KVCM_LOG_ERROR("KvcmEventSubscriber: optimizer registry manager is null");
        return false;
    }

    std::unordered_set<std::string> available_groups;
    std::size_t created_groups = 0;
    std::size_t registered_instances = 0;
    for (const auto &source : response.instance_groups()) {
        if (source.name().empty() || source.capacity_bytes() <= 0) {
            KVCM_LOG_WARN("KvcmEventSubscriber: skip invalid KVCM instance group[%s], capacity_bytes=%ld",
                          source.name().c_str(),
                          source.capacity_bytes());
            continue;
        }
        if (!registry->GetInstanceGroup(source.name())) {
            OptimizerInstanceGroup group;
            group.set_name(source.name());
            group.set_capacity_gb(
                {static_cast<double>(static_cast<long double>(source.capacity_bytes()) / kBytesPerGb)});
            group.set_eviction_policy("lru");
            group.set_enable_prefix_hash(true);
            const ErrorCode ec = manager_->CreateInstanceGroup(group);
            if (ec != EC_OK && ec != EC_DUPLICATE_ENTITY) {
                KVCM_LOG_WARN("KvcmEventSubscriber: create instance group[%s] failed, ec=%d",
                              source.name().c_str(),
                              static_cast<int>(ec));
                continue;
            }
            ++created_groups;
        }
        available_groups.insert(source.name());
    }

    for (const auto &source : response.instances()) {
        if (available_groups.find(source.instance_group_name()) == available_groups.end()) {
            KVCM_LOG_WARN("KvcmEventSubscriber: skip instance[%s], group[%s] is unavailable",
                          source.instance_id().c_str(),
                          source.instance_group_name().c_str());
            continue;
        }
        if (manager_->GetInstanceState(source.instance_id(), [](const InstanceState &) {}) == EC_OK) {
            continue;
        }

        std::vector<LocationSpecInfo> spec_infos;
        spec_infos.reserve(source.location_spec_infos_size());
        for (const auto &spec : source.location_spec_infos()) {
            spec_infos.emplace_back(spec.name(), spec.size());
        }

        std::unordered_set<std::string> full_spec_names;
        for (const auto &group : source.location_spec_groups()) {
            if (!IsFullLocationSpecGroup(group.name())) {
                continue;
            }
            full_spec_names.insert(group.spec_names().begin(), group.spec_names().end());
        }
        if (full_spec_names.empty()) {
            for (const auto &spec : spec_infos) {
                full_spec_names.insert(spec.name());
            }
        }
        std::vector<std::string> full_specs(full_spec_names.begin(), full_spec_names.end());
        std::sort(full_specs.begin(), full_specs.end());

        OptimizerInstanceInfo instance(source.instance_group_name(),
                                       source.instance_id(),
                                       source.block_size(),
                                       spec_infos,
                                       {LocationSpecGroup("full", full_specs)},
                                       0,
                                       OptimizerStateInfo("full", ""));
        RegisterInstanceResult result;
        const ErrorCode ec = manager_->RegisterInstance(instance, result);
        if (ec != EC_OK) {
            KVCM_LOG_WARN("KvcmEventSubscriber: register instance[%s] failed, ec=%d",
                          source.instance_id().c_str(),
                          static_cast<int>(ec));
            continue;
        }
        ++registered_instances;
    }

    KVCM_LOG_INFO("KvcmEventSubscriber: configuration synchronized from leader[%s], groups=%d instances=%d "
                  "created_groups=%zu registered_instances=%zu",
                  leader_endpoint.c_str(),
                  response.instance_groups_size(),
                  response.instances_size(),
                  created_groups,
                  registered_instances);
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
    while (running_ && worker->running) {
        auto channel = grpc::CreateChannel(worker->endpoint, grpc::InsecureChannelCredentials());
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
        auto reader = stub->SubscribeEvents(&context, request);
        proto::optimizer::TraceQueryRequest event;
        while (running_ && worker->running && reader->Read(&event)) {
            ProcessEvent(event);
        }
        const grpc::Status status = reader->Finish();
        {
            std::lock_guard<std::mutex> lock(worker->context_mutex);
            worker->context = nullptr;
        }

        if (running_ && worker->running) {
            KVCM_LOG_WARN("KvcmEventSubscriber: stream[%s] closed, code=%d message=%s",
                          worker->endpoint.c_str(),
                          static_cast<int>(status.error_code()),
                          status.error_message().c_str());
            if (!WaitForStop(kReconnectDelay)) {
                break;
            }
        }
    }
}

void KvcmEventSubscriber::StopWorker(std::unique_ptr<EndpointWorker> worker) {
    worker->running = false;
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

void KvcmEventSubscriber::ProcessEvent(const proto::optimizer::TraceQueryRequest &event) {
    std::vector<int64_t> block_keys(event.block_keys().begin(), event.block_keys().end());
    int64_t input_token_len = event.input_token_len();
    if (input_token_len == 0 && event.token_ids_size() > 0) {
        input_token_len = event.token_ids_size();
    }

    TraceQueryResult result;
    const ErrorCode ec =
        manager_->TraceQuery(event.instance_id(), block_keys, input_token_len, event.timestamp_ns(), result);
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

bool KvcmEventSubscriber::WaitForStop(std::chrono::milliseconds duration) {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    return !wait_cv_.wait_for(lock, duration, [this] { return !running_; });
}

} // namespace kv_cache_manager
