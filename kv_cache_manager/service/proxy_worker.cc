#include "kv_cache_manager/service/proxy_worker.h"

#include <algorithm>
#include <chrono>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/common/net_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/metrics_reporter_factory.h"
#include "kv_cache_manager/protocol/protobuf/proxy_control_service.grpc.pb.h"
#include "kv_cache_manager/routing/routing_table.h"
#include "kv_cache_manager/service/grpc_service/grpc_server_options.h"
#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"
#include "kv_cache_manager/service/http_service/meta_service_http.h"
#include "kv_cache_manager/service/proxy/meta_forwarder.h"
#include "kv_cache_manager/service/server_config.h"
#include "kv_cache_manager/service/service_request_executor.h"
#include "kv_cache_manager/service/util/report_event_body_budget.h"

namespace kv_cache_manager {
namespace {
constexpr auto kReplicaLease = std::chrono::seconds(10);

class ProxyControl final : public proto::proxy::ProxyControlService::Service {
public:
    ProxyControl(RoutingTable &routes, std::string cluster, uint64_t version)
        : routes_(routes), cluster_(std::move(cluster)), version_(version) {}

    grpc::Status ReportReplica(grpc::ServerContext *context,
                               const proto::proxy::ReplicaReport *report,
                               proto::proxy::ReportAck *) override {
        const auto &target = report->target();
        if (context->IsCancelled() || context->deadline() <= std::chrono::system_clock::now()) {
            return {grpc::StatusCode::DEADLINE_EXCEEDED, "registration expired"};
        }
        if (target.cluster() != cluster_ || target.version() != version_) {
            return {grpc::StatusCode::FAILED_PRECONDITION, "unexpected cluster or active route version"};
        }
        const auto &owners = target.bucket_owners();
        if (report->process_id().empty() || report->grpc_endpoint().empty() || !report->sequence() ||
            std::find(owners.begin(), owners.end(), target.partition()) == owners.end()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "invalid replica registration"};
        }
        if (!routes_.ApplyAssignment({target.version(), {owners.begin(), owners.end()}})) {
            return {grpc::StatusCode::FAILED_PRECONDITION, "conflicting bucket assignment"};
        }
        const auto now = RoutingTable::Clock::now();
        routes_.Expire(now, kReplicaLease);
        if (!routes_.Report({target.partition(),
                             report->process_id(),
                             target.version(),
                             report->sequence(),
                             report->grpc_endpoint(),
                             report->serving_leader()},
                            now)) {
            return {grpc::StatusCode::FAILED_PRECONDITION, "obsolete replica report"};
        }
        return grpc::Status::OK;
    }

private:
    RoutingTable &routes_;
    const std::string cluster_;
    const uint64_t version_;
};
} // namespace

struct ProxyWorker::Runtime {
    ServerConfig config;
    RoutingTable routes{std::make_shared<InstanceBucketPolicy>()};
    std::shared_ptr<MetricsRegistry> metrics = std::make_shared<MetricsRegistry>();
    std::shared_ptr<MetricsReporter> reporter;
    std::shared_ptr<ServiceRequestExecutor> executor = std::make_shared<ServiceRequestExecutor>();
    std::shared_ptr<MetaForwarder> forwarder;
    std::unique_ptr<ProxyControl> control;
    std::unique_ptr<MetaServiceGRpc> grpc_meta;
    std::unique_ptr<MetaServiceHttp> http_meta;
    std::shared_ptr<LoopThread> body_budget_sampler;
    std::unique_ptr<grpc::Server> public_server;
    std::unique_ptr<grpc::Server> control_server;
    bool http_started = false;
};

ProxyWorker::ProxyWorker() : runtime_(std::make_unique<Runtime>()) {}
ProxyWorker::~ProxyWorker() = default;

bool ProxyWorker::Init(const ServerConfig &config) {
    auto &r = *runtime_;
    r.config = config;
    if (!r.config.Check() || config.ProxyCluster().empty() || !config.ProxyRouteVersion() ||
        config.ProxyControlAddress().empty() || config.GetServiceRpcPort() <= 0 || config.GetServiceRpcPort() > 65535 ||
        config.GetServiceHttpPort() <= 0 || config.GetServiceHttpPort() > 65535 ||
        config.GetServiceRpcPort() == config.GetServiceHttpPort()) {
        KVCM_LOG_ERROR("proxy requires cluster, route_version, control_address and distinct public RPC/HTTP ports");
        return false;
    }
    proto::meta::MetaNodeEndpoint self;
    MetricsReporterFactory factory;
    factory.Init(nullptr, r.metrics);
    r.reporter = factory.Create(
        r.config.metrics_reporter_type(), r.config.metrics_reporter_config(), r.config.metrics_common_tags());
    if (!r.reporter) {
        return false;
    }
    self.set_host(config.GetAdvertisedHost().empty() ? NetUtil::GetLocalIp() : config.GetAdvertisedHost());
    self.set_meta_rpc_port(config.GetServiceRpcPort());
    self.set_meta_http_port(config.GetServiceHttpPort());
    self.set_node_id(self.host() + ":" + std::to_string(self.meta_rpc_port()));
    r.forwarder =
        std::make_shared<MetaForwarder>(r.routes, self, kReplicaLease, config.GetServiceWorkerQueueCapacity());
    r.control = std::make_unique<ProxyControl>(r.routes, config.ProxyCluster(), config.ProxyRouteVersion());
    r.grpc_meta = std::make_unique<MetaServiceGRpc>(r.metrics, nullptr, nullptr, r.executor);
    r.grpc_meta->SetForwarder(r.forwarder);
    const auto configured_budget = config.GetHttpReportEventBodyBudgetBytes();
    auto body_budget = std::make_shared<ReportEventBodyBudget>(configured_budget);
    if (configured_budget == 0) {
        r.body_budget_sampler = LoopThread::CreateLoopThread(
            [body_budget] { body_budget->Refresh(); }, ReportEventBodyBudget::kSampleIntervalUs, "proxy-body-budget");
        if (!r.body_budget_sampler) {
            return false;
        }
    }
    r.http_meta =
        std::make_unique<MetaServiceHttp>(r.metrics, nullptr, nullptr, r.executor, nullptr, body_budget, nullptr);
    r.http_meta->SetForwarder(r.forwarder);
    r.http_meta->RegisterHandler();
    return true;
}

bool ProxyWorker::Start() {
    auto &r = *runtime_;
    if (!r.executor->Init(r.config.ResolveServiceWorkerThreadNum(std::thread::hardware_concurrency()),
                          r.config.GetServiceWorkerQueueCapacity())) {
        return false;
    }
    grpc::ServerBuilder control;
    control.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    control.AddListeningPort(r.config.ProxyControlAddress(), grpc::InsecureServerCredentials());
    control.RegisterService(r.control.get());
    r.control_server = control.BuildAndStart();
    if (!r.control_server) {
        return false;
    }
    grpc::ServerBuilder builder;
    builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    ConfigureGrpcServer(builder,
                        r.config.GetServiceGrpcMaxThreads(),
                        r.config.GetServiceGrpcMemoryQuotaBytes(),
                        "proxy",
                        r.metrics,
                        r.reporter);
    builder.AddListeningPort("0.0.0.0:" + std::to_string(r.config.GetServiceRpcPort()),
                             grpc::InsecureServerCredentials());
    builder.RegisterService(r.grpc_meta.get());
    if (!r.grpc_meta->PrepareCompletionQueue(builder, r.config.GetServiceGrpcCqThreadNum())) {
        return false;
    }
    r.public_server = builder.BuildAndStart();
    if (!r.public_server) {
        return false;
    }
    r.grpc_meta->StartAccepting();
    const auto threads = r.config.GetServiceIoThreadNum();
    r.http_started = r.http_meta->StartListening(r.config.GetServiceHttpPort(), threads > 0 ? threads : 1);
    return r.http_started;
}

void ProxyWorker::Stop() {
    auto &r = *runtime_;
    if (r.body_budget_sampler) {
        r.body_budget_sampler->Stop();
    }
    if (r.control_server) {
        r.control_server->Shutdown();
    }
    r.grpc_meta->StopAccepting();
    r.executor->CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDiscardPending);
    r.forwarder->Close();
    if (r.public_server) {
        r.public_server->Shutdown(std::chrono::system_clock::now());
    }
    r.executor->Drain();
    if (r.http_started) {
        r.http_meta->Stop();
    }
    r.grpc_meta->WaitForCalls();
    r.grpc_meta->ShutdownCompletionQueueAndJoin();
    r.executor->Join();
}

bool ProxyWorker::Wait() { return !runtime_->http_started || runtime_->http_meta->WaitForExit(); }
} // namespace kv_cache_manager
