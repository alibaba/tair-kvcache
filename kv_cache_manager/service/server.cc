#include "kv_cache_manager/service/server.h"

#include <cstdio>
#include <grpcpp/grpcpp.h>

#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/config/distributed_lock_backend.h"
#include "kv_cache_manager/config/distributed_lock_backend_factory.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/log_event_publisher.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_reporter_factory.h"
#include "kv_cache_manager/service/admin_service_impl.h"
#include "kv_cache_manager/service/command_line.h"
#include "kv_cache_manager/service/debug_service_impl.h"
#include "kv_cache_manager/service/grpc_service/admin_service_grpc.h"
#include "kv_cache_manager/service/grpc_service/debug_service_grpc.h"
#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"
#include "kv_cache_manager/service/http_service/admin_service_http.h"
#include "kv_cache_manager/service/http_service/debug_service_http.h"
#include "kv_cache_manager/service/http_service/meta_service_http.h"
#include "kv_cache_manager/service/meta_service_impl.h"

namespace kv_cache_manager {

bool Server::Init(const ServerConfig &config) {
    KVCM_LOG_INFO("begin server init...\n");

    metrics_registry_ = std::make_shared<MetricsRegistry>();

    config_ = config;

    if (!CreateLeaderElector()) {
        return false;
    }

    auto registry_storage_uri = config_.GetRegistryStorageUri();
    registry_manager_.reset(new RegistryManager(registry_storage_uri, metrics_registry_));
    registry_manager_->Init();

    cache_manager_.reset(new CacheManager(metrics_registry_, registry_manager_));
    cache_manager_->Init(config_.GetSchedulePlanExecutorThreadCount());
    cache_manager_->PauseReclaimer(); // Resume after DoRecover

    CreateMetricsReporter();
    CreateAndRegisterEventPublisher();

    meta_impl_ = std::make_shared<MetaServiceImpl>(cache_manager_, metrics_reporter_);
    admin_impl_ = std::make_shared<AdminServiceImpl>(
        cache_manager_, metrics_reporter_, metrics_registry_, registry_manager_, leader_elector_);
    debug_impl_ = std::make_shared<DebugServiceImpl>(cache_manager_);

    meta_impl_->DisableLeaderOnlyRequests();
    admin_impl_->DisableLeaderOnlyRequests();

    KVCM_LOG_INFO("server init success.");
    return true;
}

void Server::OnBecomeLeader() {
    KVCM_LOG_INFO("Server promoted to leader, starting recover...");
    ErrorCode ec = registry_manager_->DoRecover();
    if (ec != EC_OK) {
        // TODO: 添加异常情况下回滚和重试
        KVCM_LOG_ERROR("registry_manager recover failed");
        return;
    }

    if (!is_startup_loaded_) {
        is_startup_loaded_ = true;
        StartupConfigLoader loader;
        loader.Init(registry_manager_);
        if (!loader.Load(config_.startup_config())) {
            KVCM_LOG_ERROR("Startup loader failed");
        }
    }

    ec = cache_manager_->DoRecover();
    if (ec != EC_OK) {
        // TODO: 添加异常情况下回滚和重试
        KVCM_LOG_ERROR("cache_manager recover failed");
        return;
    }
    cache_manager_->ResumeReclaimer();

    meta_impl_->EnableLeaderOnlyRequests();
    admin_impl_->EnableLeaderOnlyRequests();
    KVCM_LOG_INFO("recover end");
}

void Server::OnNoLongerLeader() {
    KVCM_LOG_INFO("Server demoted to standby, starting cleanup...");
    cache_manager_->PauseReclaimer();

    meta_impl_->DisableLeaderOnlyRequests();
    admin_impl_->DisableLeaderOnlyRequests();

    meta_impl_->WaitForAllLeaderOnlyRequestsToComplete();
    admin_impl_->WaitForAllLeaderOnlyRequestsToComplete();

    ErrorCode ec = cache_manager_->DoCleanup();
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("cache_manager DoCleanup failed");
    }
    ec = registry_manager_->DoCleanup();
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("registry_manager DoCleanup failed");
    }
    KVCM_LOG_INFO("Server cleanup completed");
}

bool Server::Start() {
    KVCM_LOG_INFO("server starting...");
    if (!StartMetricsReportThread()) {
        KVCM_LOG_ERROR("init metrics reporter failed");
        return false;
    }
    if (!StartRpcServer()) {
        KVCM_LOG_ERROR("init rpc server failed");
        return false;
    }
    if (!StartHttpServer()) {
        KVCM_LOG_ERROR("init http server failed");
        return false;
    }
    if (!leader_elector_->Start()) {
        KVCM_LOG_ERROR("leader_elector start failed");
        return false;
    }
    KVCM_LOG_INFO("\n%s\nkvcm server start OK!\nversion: %s\ncommit: %s\nbuild time: %s",
                  KVCM_ART,
                  SYS_GLB_VERSION,
                  SYS_GLB_GIT_INFO,
                  SYS_GLB_BUILD_TIME);
    return true;
}

bool Server::Wait() {
    rpc_server_->Wait();
    if (meta_http_thread_.joinable()) {
        meta_http_thread_.join();
    }
    if (admin_http_thread_.joinable()) {
        admin_http_thread_.join();
    }
    if (debug_http_thread_.joinable()) {
        debug_http_thread_.join();
    }
    return true;
}

bool Server::StartRpcServer() {
    int32_t rpc_port = config_.GetServiceRpcPort();
    int32_t admin_rpc_port = config_.GetServiceAdminRpcPort();
    bool use_separate_admin_server = admin_rpc_port != 0 && rpc_port != admin_rpc_port;
    std::string server_address = "0.0.0.0:" + std::to_string(rpc_port);
    // grpc::EnableDefaultHealthCheckService(true);
    // grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    meta_service_.reset(new MetaServiceGRpc(metrics_registry_, meta_impl_, registry_manager_));
    admin_service_.reset(new AdminServiceGRpc(metrics_registry_, admin_impl_));
    debug_service_.reset(new DebugServiceGRpc(metrics_registry_, debug_impl_));

    meta_service_->Init();
    admin_service_->Init();
    debug_service_->Init();

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(meta_service_.get());
    if (!use_separate_admin_server) {
        builder.RegisterService(admin_service_.get());
    }
    if (!use_separate_admin_server && config_.IsEnableDebugService()) {
        builder.RegisterService(debug_service_.get());
    }
    auto server = builder.BuildAndStart();
    if (!server) {
        KVCM_LOG_ERROR("Failed to start rpc server");
        return false;
    }
    rpc_server_.reset(server.release());
    KVCM_LOG_INFO("Server listening on %s success", server_address.c_str());
    if (use_separate_admin_server) {
        return StartSeparateAdminRpcServer();
    }
    return true;
}

bool Server::StartSeparateAdminRpcServer() {
    int32_t rpc_port = config_.GetServiceAdminRpcPort();
    std::string server_address = "0.0.0.0:" + std::to_string(rpc_port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(admin_service_.get());
    if (config_.IsEnableDebugService()) {
        builder.RegisterService(debug_service_.get());
    }
    auto server = builder.BuildAndStart();
    if (!server) {
        KVCM_LOG_ERROR("Failed to start admin rpc server");
        return false;
    }
    admin_rpc_server_.reset(server.release());
    KVCM_LOG_INFO("Admin Server listening on %s success", server_address.c_str());
    return true;
}

bool Server::StartHttpServer() {
    int32_t http_port = config_.GetServiceHttpPort();
    int32_t admin_http_port = config_.GetServiceAdminHttpPort();

    meta_http_service_ = std::make_shared<MetaServiceHttp>(metrics_registry_, meta_impl_, registry_manager_);
    admin_http_service_ = std::make_shared<AdminServiceHttp>(metrics_registry_, admin_impl_);
    debug_http_service_ = std::make_shared<DebugServiceHttp>(metrics_registry_, debug_impl_);

    meta_http_service_->Init();
    admin_http_service_->Init();
    debug_http_service_->Init();

    // 注册HTTP处理器
    meta_http_service_->RegisterHandler();
    admin_http_service_->RegisterHandler();

    if (config_.IsEnableDebugService()) {
        debug_http_service_->RegisterHandler();
    }

    // 在单独的线程中启动HTTP服务
    meta_http_thread_ = std::thread([this, http_port]() {
        KVCM_LOG_INFO("Meta http server starting on port %d", http_port);
        bool meta_started = meta_http_service_->Start(http_port);

        if (!meta_started) {
            KVCM_LOG_ERROR("Failed to start meta http server on port %d", http_port);
        } else {
            KVCM_LOG_INFO("Meta HTTP server exited on port %d", http_port);
        }
    });
    admin_http_thread_ = std::thread([this, admin_http_port]() {
        KVCM_LOG_INFO("Admin http server starting on port %d", admin_http_port);
        bool admin_started = admin_http_service_->Start(admin_http_port); // 使用不同端口启动admin服务
        if (!admin_started) {
            KVCM_LOG_ERROR("Failed to start admin http server on port %d", admin_http_port);
        } else {
            KVCM_LOG_INFO("Admin HTTP server exited on port %d", admin_http_port);
        }
    });
    // TODO HTTP框架允许各个API共用一个端口
    // 可以考虑把三个HTTP服务合并到一个端口上，重复的API只注册一次
    // 如果有同名的不同API，可以通过特殊字段区分
    if (config_.IsEnableDebugService()) {
        debug_http_thread_ = std::thread([this, http_port]() {
            int32_t debug_http_port = http_port + 3000;
            KVCM_LOG_INFO("Debug http server starting on port %d", debug_http_port);
            bool debug_started = debug_http_service_->Start(debug_http_port);
            if (!debug_started) {
                KVCM_LOG_ERROR("Failed to start debug http server on port %d", debug_http_port);
            } else {
                KVCM_LOG_INFO("Debug HTTP server exited on port %d", debug_http_port);
            }
        });
    }
    return true;
}

void Server::CreateMetricsReporter() {
    metrics_reporter_factory_.reset(new MetricsReporterFactory);
    metrics_reporter_factory_->Init(cache_manager_, metrics_registry_);
    auto reporter_type = config_.metrics_reporter_type();
    auto reporter_config = config_.metrics_reporter_config();
    metrics_reporter_ = metrics_reporter_factory_->Create(reporter_type, reporter_config);
    KVCM_LOG_INFO("create metrics reporter OK");
}

void Server::CreateAndRegisterEventPublisher() {
    auto event_manager = cache_manager_->event_manager();
    if (!event_manager) {
        KVCM_LOG_WARN("do not have event manager, skip create and register event publisher.");
        return;
    }
    auto log_publisher = std::make_shared<LogEventPublisher>();
    // 这里的logpublisher初始化配置需要修改
    auto event_publishers_configs = config_.event_publishers_configs();
    if (!log_publisher->Init(event_publishers_configs)) {
        KVCM_LOG_ERROR("init log event publisher failed");
        return;
    }
    if (!event_manager->RegisterPublisher("log_event_publisher", log_publisher)) {
        KVCM_LOG_ERROR("add log event publisher failed");
        return;
    }
    KVCM_LOG_INFO("create and register event publisher OK");
}
bool Server::CreateLeaderElector() {
    auto distributed_lock_uri = config_.GetDistributedLockUri();
    std::string node_id = config_.GetLeaderElectorNodeId();
    if (node_id.empty()) {
        // TODO: replace local_ip_placeholder with real local ip
        std::string local_ip = "local_ip_placeholder";
        node_id = local_ip + ":" + std::to_string(config_.GetServiceAdminHttpPort()) + "_" +
                  StringUtil::GenerateRandomString(16);
    }
    distributed_lock_backend_ =
        DistributedLockBackendFactory::CreateAndInitDistributedLockBackend(distributed_lock_uri);
    if (!distributed_lock_backend_) {
        KVCM_LOG_ERROR("distributed_lock_backend[%s] init failed", distributed_lock_uri.c_str());
        return false;
    }

    leader_elector_ = std::make_shared<LeaderElector>(distributed_lock_backend_,
                                                      kLeaderLockKey,
                                                      node_id,
                                                      config_.GetLeaderElectorLeaseMs(),
                                                      config_.GetLeaderElectorLoopIntervalMs());
    leader_elector_->SetBecomeLeaderHandler([this]() { OnBecomeLeader(); });
    leader_elector_->SetNoLongerLeaderHandler([this]() { OnNoLongerLeader(); });
    return true;
}

bool Server::StartMetricsReportThread() {
    if (!metrics_reporter_) {
        KVCM_LOG_ERROR("do not have metrics reporter, start report thread failed.");
        return false;
    }
    metrics_report_thread_ = LoopThread::CreateLoopThread(
        std::bind(&MetricsReporter::ReportInterval, metrics_reporter_), config_.metrics_report_interval_ms() * 1000);
    KVCM_LOG_INFO("start metrics reporter success.");
    return true;
}

void Server::Stop() {
    if (stop_) {
        return;
    }
    stop_ = true;
    KVCM_LOG_INFO("server stopping...");
    rpc_server_->Shutdown();
    KVCM_LOG_INFO("rpc server stopped.");
    if (meta_http_service_) {
        meta_http_service_->Stop();
    }
    KVCM_LOG_INFO("meta http server stopped.");
    if (admin_http_service_) {
        admin_http_service_->Stop();
    }
    if (debug_http_service_) {
        debug_http_service_->Stop();
        KVCM_LOG_INFO("debug http server stopped.");
    }
    metrics_report_thread_->Stop();
    KVCM_LOG_INFO("metrics reporter stopped.");
    KVCM_LOG_INFO("admin http server stopped.");
    KVCM_LOG_INFO("kvcm server stopped, goodbye!");
}

} // namespace kv_cache_manager
