#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/service/server_config.h"

namespace grpc {
class Server;
}

namespace kv_cache_manager {
class DebugServiceImpl;
class KvMetaServiceImpl;
class AdminServiceImpl;
class MetaServiceImpl;
class CoordinationBackend;
class LeaderElector;
class RegistryManager;
class CacheManager;
class KvMetaManager;

class MetaServiceGRpc;
class AdminServiceGRpc;
class DebugServiceGRpc;
class KvMetaServiceGRpc;
class MetaServiceHttp;
class AdminServiceHttp;
class DebugServiceHttp;
class MetricsRegistry;
class MetricsReporter;
class MetricsReporterFactory;
struct MetricsLifecycle;
class LoopThread;

class Server {
public:
    bool Init(const ServerConfig &config);
    bool Start();
    bool Wait();
    void Stop();

private:
    bool StartRpcServer();
    bool StartSeparateAdminRpcServer();
    bool StartKvMetaRpcServer();
    bool StartHttpServer();
    void CreateMetricsReporter();
    bool StartMetricsReportThread();
    void CreateAndRegisterEventPublisher();
    bool CreateLeaderElector();
    void StartKvMetaRecovery();
    void CancelAndJoinKvMetaRecovery();

    void OnBecomeLeader();
    void OnNoLongerLeader();

private:
    const std::string kLeaderLockKey = "_TAIR_KVCM_LEADER_KEY";

    std::atomic<bool> stop_{false};
    bool is_startup_loaded_ = false;
    ServerConfig config_;
    std::shared_ptr<MetaServiceImpl> meta_impl_;
    std::shared_ptr<AdminServiceImpl> admin_impl_;
    std::shared_ptr<DebugServiceImpl> debug_impl_;
    std::shared_ptr<KvMetaServiceImpl> kv_meta_impl_;
    std::shared_ptr<MetaServiceGRpc> meta_service_;
    std::shared_ptr<AdminServiceGRpc> admin_service_;
    std::shared_ptr<DebugServiceGRpc> debug_service_;
    std::shared_ptr<KvMetaServiceGRpc> kv_meta_service_;
    std::shared_ptr<grpc::Server> rpc_server_;
    std::shared_ptr<grpc::Server> admin_rpc_server_;
    std::shared_ptr<grpc::Server> kv_meta_rpc_server_;
    std::shared_ptr<MetaServiceHttp> meta_http_service_;
    std::shared_ptr<AdminServiceHttp> admin_http_service_;
    std::shared_ptr<DebugServiceHttp> debug_http_service_;

    std::thread meta_http_thread_;
    std::thread admin_http_thread_;
    std::thread debug_http_thread_;
    std::thread kv_meta_recovery_thread_;
    std::mutex kv_meta_recovery_mutex_;
    std::atomic<std::uint64_t> kv_meta_recovery_epoch_{0};

    std::shared_ptr<CoordinationBackend> coordination_backend_;
    std::shared_ptr<LeaderElector> leader_elector_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<KvMetaManager> kv_meta_manager_;

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<MetricsLifecycle> metrics_lifecycle_;
    std::shared_ptr<MetricsReporterFactory> metrics_reporter_factory_;
    std::shared_ptr<MetricsReporter> metrics_reporter_;
    std::shared_ptr<LoopThread> metrics_report_thread_;
};
} // namespace kv_cache_manager
