#include "kv_cache_manager/service/server.h"

#include <cstdio>
#include <exception>
#include <grpcpp/grpcpp.h>

#include "kv_cache_manager/common/build_version.h"
#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/common/net_util.h"
#include "kv_cache_manager/config/coordination_backend.h"
#include "kv_cache_manager/config/coordination_backend_factory.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/node_endpoint_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/log_event_publisher.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/kv_meta_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_reporter_factory.h"
#include "kv_cache_manager/service/admin_service_impl.h"
#include "kv_cache_manager/service/command_line.h"
#include "kv_cache_manager/service/debug_service_impl.h"
#include "kv_cache_manager/service/grpc_service/admin_service_grpc.h"
#include "kv_cache_manager/service/grpc_service/debug_service_grpc.h"
#include "kv_cache_manager/service/grpc_service/kv_meta_service_grpc.h"
#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"
#include "kv_cache_manager/service/http_service/admin_service_http.h"
#include "kv_cache_manager/service/http_service/debug_service_http.h"
#include "kv_cache_manager/service/http_service/meta_service_http.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/kv_meta_service_impl.h"

namespace kv_cache_manager {

bool Server::Init(const ServerConfig &config) {
    KVCM_LOG_INFO("begin server init...\n");

    metrics_registry_ = std::make_shared<MetricsRegistry>();
    // single shared lifecycle handle: producers hold it as a reader
    // around metric registration; AdminServiceImpl::RemoveInstance and
    // RemoveInstanceGroup hold it as a writer for the entire removal
    metrics_lifecycle_ = std::make_shared<MetricsLifecycle>();

    config_ = config;

    if (!config_.Check()) {
        KVCM_LOG_ERROR("server config check failed");
        return false;
    }

    if (!CreateLeaderElector()) {
        return false;
    }

    auto registry_storage_uri = config_.GetRegistryStorageUri();
    registry_manager_.reset(new RegistryManager(registry_storage_uri, metrics_registry_));
    if (!registry_manager_->Init()) {
        KVCM_LOG_ERROR("registry manager init failed");
        return false;
    }

    cache_manager_.reset(new CacheManager(metrics_registry_, registry_manager_, metrics_lifecycle_));
    CacheReclaimerAsyncDeleteConfig async_delete_config;
    async_delete_config.inflight_delete_timeout_ms = config_.GetCacheReclaimerInflightDeleteTimeoutMs();
    async_delete_config.pending_location_limit_per_group_type =
        config_.GetCacheReclaimerPendingLocationLimitPerGroupType();
    async_delete_config.pending_bytes_limit_per_group_type = config_.GetCacheReclaimerPendingBytesLimitPerGroupType();
    async_delete_config.pending_delete_handler_limit = config_.GetCacheReclaimerPendingDeleteHandlerLimit();
    async_delete_config.pending_bytes_limit = config_.GetCacheReclaimerPendingBytesLimit();
    CacheGarbageCollector::Config cache_gc_config;
    cache_gc_config.enabled = config_.IsCacheGcEnabled();
    cache_gc_config.scan_interval_ms = config_.GetCacheGcScanIntervalMs();
    cache_gc_config.round_pause_ms = config_.GetCacheGcRoundPauseMs();
    cache_gc_config.scan_batch_size = static_cast<size_t>(config_.GetCacheGcScanBatchSize());
    cache_gc_config.orphan_writing_grace_period_ms = config_.GetCacheGcOrphanWritingGracePeriodMs();
    cache_gc_config.max_inflight_delete_requests = static_cast<size_t>(config_.GetCacheGcMaxInflightDeleteRequests());
    cache_gc_config.event_report_cleanup_enabled = config_.IsCacheGcEventReportCleanupEnabled();
    cache_gc_config.event_report_action_batch_size =
        static_cast<size_t>(config_.GetCacheGcEventReportActionBatchSize());
    if (!cache_manager_->Init(config_.GetSchedulePlanExecutorThreadCount(),
                              config_.GetCacheReclaimerKeySamplingSizeTotal(),
                              config_.GetCacheReclaimerKeySamplingSizePerTask(),
                              config_.GetCacheReclaimerDelBatchSize(),
                              config_.GetCacheReclaimerIdleIntervalMs(),
                              config_.GetCacheReclaimerWorkerSize(),
                              async_delete_config,
                              config_.GetSchedulePlanMigrationWorkerBudget(),
                              config_.GetMetaQueryWorkerCount(),
                              config_.GetMetaQueryParallelThreshold(),
                              config_.GetMetaQueryChunkSize(),
                              cache_gc_config)) {
        KVCM_LOG_ERROR("cache manager init failed");
        return false;
    }
    cache_manager_->PauseReclaimer(); // Resume after DoRecover

    // Set revisit interval histogram configuration
    auto boundaries = ServerConfig::ParseRevisitIntervalBuckets(config_.GetRevisitIntervalBuckets());
    if (boundaries.empty()) {
        boundaries = ServerConfig::GetDefaultRevisitIntervalBuckets();
    }
    cache_manager_->SetRevisitHistogramConfig(boundaries);

    CreateMetricsReporter();
    CreateAndRegisterEventPublisher();

    meta_impl_ = std::make_shared<MetaServiceImpl>(cache_manager_, metrics_reporter_, leader_elector_);
    admin_impl_ = std::make_shared<AdminServiceImpl>(
        cache_manager_, metrics_reporter_, metrics_registry_, registry_manager_, leader_elector_);
    debug_impl_ = std::make_shared<DebugServiceImpl>(cache_manager_);

    if (config_.GetKvMetaRpcPort() != 0) {
        kv_meta_manager_ = std::make_shared<KvMetaManager>(cache_manager_, registry_manager_);
        if (!kv_meta_manager_->Init()) {
            KVCM_LOG_ERROR("KVMeta manager init failed");
            return false;
        }
        kv_meta_impl_ = std::make_shared<KvMetaServiceImpl>(cache_manager_, kv_meta_manager_, metrics_reporter_);
        kv_meta_impl_->DisableLeaderOnlyRequests();
        KVCM_LOG_INFO("KVMeta service enabled on its isolated RPC port %d", config_.GetKvMetaRpcPort());
    }

    meta_impl_->DisableLeaderOnlyRequests();
    admin_impl_->DisableLeaderOnlyRequests();

    KVCM_LOG_INFO("server init success.");
    return true;
}

void Server::OnBecomeLeader() {
    KVCM_LOG_INFO("Server promoted to leader, starting recover...");
    ErrorCode ec = registry_manager_->DoRecover();
    if (ec != EC_OK) {
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
        KVCM_LOG_ERROR("cache_manager recover failed");
        return;
    }
    ec = cache_manager_->StartCacheGarbageCollector();
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("cache garbage collector start failed, ec[%d]", static_cast<int>(ec));
        return;
    }
    cache_manager_->ResumeReclaimer();
    cache_manager_->StartMigrationManager();

    meta_impl_->EnableLeaderOnlyRequests();
    admin_impl_->EnableLeaderOnlyRequests();
    KVCM_LOG_INFO("recover end");

    // KVMeta recovery can scan a large object namespace. Run it only after
    // the existing service has been made available, and never on the leader
    // transition thread that gates the main lifecycle.
    if (kv_meta_manager_) {
        StartKvMetaRecovery();
    }
}

void Server::OnNoLongerLeader() {
    KVCM_LOG_INFO("Server demoted to standby, starting cleanup...");
    cache_manager_->RequestStopCacheGarbageCollector();
    cache_manager_->PauseReclaimer();

    meta_impl_->DisableLeaderOnlyRequests();
    admin_impl_->DisableLeaderOnlyRequests();
    if (kv_meta_manager_) {
        kv_meta_recovery_epoch_.fetch_add(1, std::memory_order_acq_rel);
        kv_meta_impl_->DisableLeaderOnlyRequests();
        kv_meta_manager_->CancelMaintenance();
        // Stop the independent expiry worker before CacheManager teardown can
        // race it. Pending sessions are cleared in memory without per-session
        // backend I/O; the next leader's recovery reclaims active records.
        kv_meta_manager_->DoCleanup();
        CancelAndJoinKvMetaRecovery();
    }

    meta_impl_->WaitForAllLeaderOnlyRequestsToComplete();
    admin_impl_->WaitForAllLeaderOnlyRequestsToComplete();
    if (kv_meta_manager_) {
        kv_meta_impl_->WaitForAllLeaderOnlyRequestsToComplete();
    }

    cache_manager_->JoinCacheGarbageCollector();
    // Stop migration after leader-only requests drain and after GC has stopped consulting
    // active Copy reservations.
    cache_manager_->StopMigrationManager();

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

void Server::StartKvMetaRecovery() {
    CancelAndJoinKvMetaRecovery();
    if (!kv_meta_manager_ || !kv_meta_impl_ || stop_.load(std::memory_order_acquire)) {
        return;
    }

    const std::uint64_t epoch = kv_meta_recovery_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    std::lock_guard<std::mutex> lock(kv_meta_recovery_mutex_);
    if (stop_.load(std::memory_order_acquire) ||
        kv_meta_recovery_epoch_.load(std::memory_order_acquire) != epoch) {
        return;
    }
    try {
        kv_meta_recovery_thread_ = std::thread([this, epoch]() {
            const auto should_abort = [this, epoch]() {
                return stop_.load(std::memory_order_acquire) ||
                       kv_meta_recovery_epoch_.load(std::memory_order_acquire) != epoch;
            };
            const ErrorCode ec = kv_meta_manager_->DoRecover(should_abort);
            bool enabled = false;
            if (ec == EC_OK) {
                // Serialize the final epoch check and gate opening with
                // CancelAndJoinKvMetaRecovery. A demotion either invalidates
                // the epoch before this lock is acquired, or disables the
                // gate after this block; a standby can therefore never be
                // re-enabled by a finishing recovery thread.
                std::lock_guard<std::mutex> lock(kv_meta_recovery_mutex_);
                if (!should_abort()) {
                    if (kv_meta_manager_->ResumeMaintenance()) {
                        kv_meta_impl_->EnableLeaderOnlyRequests();
                        enabled = true;
                    } else {
                        KVCM_LOG_ERROR("KVMeta session worker restart failed; service remains disabled");
                    }
                }
            }
            if (enabled) {
                KVCM_LOG_INFO("KVMeta recovery completed; generic object service is ready");
            } else if (ec != EC_SERVICE_NOT_LEADER && ec != EC_OK) {
                KVCM_LOG_ERROR("KVMeta recover failed, generic object service remains disabled, ec[%d]",
                               static_cast<int>(ec));
            }
        });
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("failed to start KVMeta recovery thread: %s", e.what());
    }
}

void Server::CancelAndJoinKvMetaRecovery() {
    kv_meta_recovery_epoch_.fetch_add(1, std::memory_order_acq_rel);
    std::thread recovery_thread;
    {
        std::lock_guard<std::mutex> lock(kv_meta_recovery_mutex_);
        if (kv_meta_recovery_thread_.joinable()) {
            recovery_thread = std::move(kv_meta_recovery_thread_);
        }
    }
    if (recovery_thread.joinable()) {
        recovery_thread.join();
    }
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

    // wire up instance-removal callback: invalidate per-instance
    // collector caches on both gRPC and HTTP meta services
    // registered before leader election so no RemoveInstance can arrive
    // before the callback is in place (API_CALL_GUARD rejects requests
    // when not leader)
    std::weak_ptr<MetaServiceGRpc> grpc_weak = meta_service_;
    std::weak_ptr<MetaServiceHttp> http_weak = meta_http_service_;
    cache_manager_->SetOnInstanceRemoved(
        [grpc_weak = std::move(grpc_weak), http_weak = std::move(http_weak)](const std::string &instance_id) {
            if (const auto grpc = grpc_weak.lock()) {
                grpc->InvalidateCollectorCache(instance_id);
            }
            if (const auto http = http_weak.lock()) {
                http->InvalidateCollectorCache(instance_id);
            }
        });

    if (!leader_elector_->Start()) {
        KVCM_LOG_ERROR("leader_elector start failed");
        return false;
    }
    KVCM_LOG_INFO("\n%s\nkvcm server start OK!\nversion: %s\ncommit: %s\nbuild time: %s",
                  kKvcmArt,
                  kKvcmFullVersion,
                  kKvcmGitCommit,
                  kKvcmBuildTime);
    return true;
}

bool Server::Wait() {
    if (rpc_server_) {
        rpc_server_->Wait();
    }
    if (kv_meta_rpc_server_) {
        kv_meta_rpc_server_->Wait();
    }
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

    meta_service_.reset(new MetaServiceGRpc(metrics_registry_, meta_impl_, registry_manager_, metrics_lifecycle_));
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
    if (use_separate_admin_server && !StartSeparateAdminRpcServer()) {
        return false;
    }
    return StartKvMetaRpcServer();
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

bool Server::StartKvMetaRpcServer() {
    const int32_t rpc_port = config_.GetKvMetaRpcPort();
    if (rpc_port == 0) {
        return true;
    }
    if (!kv_meta_impl_ || !kv_meta_manager_) {
        KVCM_LOG_ERROR("KVMeta RPC port is configured but KVMeta manager is unavailable");
        return false;
    }

    kv_meta_service_ = std::make_shared<KvMetaServiceGRpc>(metrics_registry_, kv_meta_impl_);
    kv_meta_service_->Init();

    const std::string server_address = "0.0.0.0:" + std::to_string(rpc_port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(kv_meta_service_.get());
    auto server = builder.BuildAndStart();
    if (!server) {
        KVCM_LOG_ERROR("Failed to start isolated KVMeta RPC server on %s", server_address.c_str());
        return false;
    }
    kv_meta_rpc_server_.reset(server.release());
    KVCM_LOG_INFO("KVMeta Server listening on %s success", server_address.c_str());
    return true;
}

bool Server::StartHttpServer() {
    int32_t http_port = config_.GetServiceHttpPort();
    int32_t admin_http_port = config_.GetServiceAdminHttpPort();

    const int32_t configured_io_thread_num = config_.GetServiceIoThreadNum();
    const size_t io_thread_num = configured_io_thread_num > 0 ? static_cast<size_t>(configured_io_thread_num)
                                                              : std::thread::hardware_concurrency();
    KVCM_LOG_INFO("HTTP server io_thread_num=%zu (configured=%d)", io_thread_num, configured_io_thread_num);

    meta_http_service_ =
        std::make_shared<MetaServiceHttp>(metrics_registry_, meta_impl_, registry_manager_, metrics_lifecycle_);
    admin_http_service_ = std::make_shared<AdminServiceHttp>(
        metrics_registry_, admin_impl_, config_.enable_prometheus(), config_.prometheus_prefix());
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
    meta_http_thread_ = std::thread([this, http_port, io_thread_num]() {
        KVCM_LOG_INFO("Meta http server starting on port %d", http_port);
        bool meta_started = meta_http_service_->Start(http_port, io_thread_num);

        if (!meta_started) {
            KVCM_LOG_ERROR("Failed to start meta http server on port %d", http_port);
        } else {
            KVCM_LOG_INFO("Meta HTTP server exited on port %d", http_port);
        }
    });
    admin_http_thread_ = std::thread([this, admin_http_port, io_thread_num]() {
        KVCM_LOG_INFO("Admin http server starting on port %d", admin_http_port);
        bool admin_started = admin_http_service_->Start(admin_http_port, io_thread_num); // 使用不同端口启动admin服务
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
        debug_http_thread_ = std::thread([this, http_port, io_thread_num]() {
            int32_t debug_http_port = http_port + 3000;
            KVCM_LOG_INFO("Debug http server starting on port %d", debug_http_port);
            bool debug_started = debug_http_service_->Start(debug_http_port, io_thread_num);
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
    auto coordination_uri = config_.GetCoordinationUri();
    std::string node_id = config_.GetLeaderElectorNodeId();
    std::string host = config_.GetAdvertisedHost();
    if (host.empty()) {
        host = NetUtil::GetLocalIp();
    }
    if (node_id.empty()) {
        node_id =
            host + ":" + std::to_string(config_.GetServiceAdminHttpPort()) + "_" + StringUtil::GenerateRandomString(16);
    }
    coordination_backend_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(coordination_uri);
    if (!coordination_backend_) {
        KVCM_LOG_ERROR("coordination_backend[%s] init failed", coordination_uri.c_str());
        return false;
    }

    leader_elector_ = std::make_shared<LeaderElector>(coordination_backend_,
                                                      kLeaderLockKey,
                                                      node_id,
                                                      config_.GetLeaderElectorLeaseMs(),
                                                      config_.GetLeaderElectorLoopIntervalMs());
    leader_elector_->SetBecomeLeaderHandler([this]() { OnBecomeLeader(); });
    leader_elector_->SetNoLongerLeaderHandler([this]() { OnNoLongerLeader(); });

    // 写入本节点的连接信息到协调后端
    {
        NodeEndpointInfo node_info(node_id,
                                   host,
                                   config_.GetServiceRpcPort(),
                                   config_.GetServiceHttpPort(),
                                   config_.GetServiceAdminRpcPort(),
                                   config_.GetServiceAdminHttpPort(),
                                   config_.GetCustomInfo());

        ErrorCode ec = leader_elector_->SetSelfNodeInfo(node_info);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("failed to write node info for node_id[%s], ec=%d", node_id.c_str(), ec);
            return false;
        }
        KVCM_LOG_INFO("node info written for node_id[%s]", node_id.c_str());
    }

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
    if (kv_meta_manager_) {
        kv_meta_impl_->DisableLeaderOnlyRequests();
        kv_meta_manager_->CancelMaintenance();
        kv_meta_manager_->DoCleanup();
        CancelAndJoinKvMetaRecovery();
    }
    if (kv_meta_rpc_server_) {
        kv_meta_rpc_server_->Shutdown();
    }
    if (kv_meta_manager_) {
        kv_meta_impl_->WaitForAllLeaderOnlyRequestsToComplete();
    }
    if (rpc_server_) {
        rpc_server_->Shutdown();
    }
    if (admin_rpc_server_) {
        admin_rpc_server_->Shutdown();
    }
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
    if (metrics_report_thread_) {
        metrics_report_thread_->Stop();
        KVCM_LOG_INFO("metrics reporter stopped.");
    }
    KVCM_LOG_INFO("admin http server stopped.");
    KVCM_LOG_INFO("kvcm server stopped, goodbye!");
}

} // namespace kv_cache_manager
