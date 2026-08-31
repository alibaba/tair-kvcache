#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/optimizer/service/online_optimizer_server_config.h"

namespace grpc {
class ClientContext;
}

namespace kv_cache_manager {

class OptimizerServiceImpl;
class ServiceDiscovery;

namespace proto::optimizer {
class TraceQueryRequest;
}

class KvcmEventSubscriber {
public:
    KvcmEventSubscriber(const KvcmEventSubscriptionConfig &config,
                        std::shared_ptr<OptimizerServiceImpl> optimizer_service);
    ~KvcmEventSubscriber();

    KvcmEventSubscriber(const KvcmEventSubscriber &) = delete;
    KvcmEventSubscriber &operator=(const KvcmEventSubscriber &) = delete;

    bool Init();
    bool Start();
    void Stop();

private:
    struct EndpointWorker {
        explicit EndpointWorker(std::string endpoint) : endpoint(std::move(endpoint)) {}

        std::string endpoint;
        std::atomic<bool> running{true};
        std::thread thread;
        std::mutex context_mutex;
        grpc::ClientContext *context = nullptr;
    };

    void SupervisorLoop();
    void RefreshLeader();
    bool DiscoverLeader(std::string &leader_endpoint);
    bool SyncConfiguration(const std::string &leader_endpoint);
    void UpdateWorker(const std::string &leader_endpoint);
    void EndpointLoop(EndpointWorker *worker);
    void StopWorker(std::unique_ptr<EndpointWorker> worker);
    void ProcessEvent(const proto::optimizer::TraceQueryRequest &event);
    void RequestConfigurationRefresh();
    bool WaitForSupervisor(std::chrono::milliseconds duration);
    bool WaitForReconnect(EndpointWorker *worker, std::chrono::milliseconds duration);
    static std::chrono::milliseconds ComputeReconnectDelay(uint32_t failed_attempts);

    KvcmEventSubscriptionConfig config_;
    std::shared_ptr<OptimizerServiceImpl> optimizer_service_;
    std::unique_ptr<ServiceDiscovery> service_discovery_;

    std::atomic<bool> running_{false};
    std::thread supervisor_thread_;
    std::mutex worker_mutex_;
    std::unique_ptr<EndpointWorker> worker_;
    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
    bool configuration_refresh_requested_ = false;
    std::mutex unsupported_instances_mutex_;
    std::unordered_set<std::string> unsupported_instance_ids_;
};

} // namespace kv_cache_manager
