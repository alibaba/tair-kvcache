#pragma once

#include <atomic>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "kv_cache_manager/optimizer/service/online_optimizer_server_config.h"

namespace kv_cache_manager {

class OnlineOptimizerManager;
class OptimizerServiceImpl;
class OptimizerServiceGRpc;
class OptimizerServiceHttp;
class OptimizerRegistryManager;
class OptimizerMetricsReporter;
class MetricsRegistry;

class OnlineOptimizerServer {
public:
    using EnvironMap = std::unordered_map<std::string, std::string>;

    OnlineOptimizerServer() = default;
    ~OnlineOptimizerServer();

    OnlineOptimizerServer(const OnlineOptimizerServer &) = delete;
    OnlineOptimizerServer &operator=(const OnlineOptimizerServer &) = delete;

    bool Init(const std::string &config_file, const EnvironMap &environ = {});
    bool Start();
    void Stop();
    void RequestShutdown();
    void WaitForShutdown();

private:
    bool InitRpcServer();
    bool InitHttpServer();
    void DoStop();
    void MetricsReportLoop();
    void RecoveryRetryLoop();

    OnlineOptimizerServerConfig config_;
    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<OptimizerServiceImpl> service_impl_;
    std::shared_ptr<OptimizerServiceGRpc> grpc_service_;
    std::shared_ptr<OptimizerServiceHttp> http_service_;
    std::shared_ptr<OptimizerRegistryManager> registry_manager_;
    std::shared_ptr<OptimizerMetricsReporter> metrics_reporter_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;

    std::unique_ptr<grpc::Server> grpc_server_;
    std::thread http_thread_;
    std::thread metrics_thread_;
    std::thread recovery_thread_;
    std::atomic<bool> running_{false};
    std::once_flag stop_flag_;
    std::atomic<bool> recovery_needed_{false};
};

} // namespace kv_cache_manager
