#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/event_reporting_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {
class VineyardBackend : public DataStorageBackend, public EventReportingBackend {
public:
    VineyardBackend() = delete;
    explicit VineyardBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~VineyardBackend() override;

    DataStorageType GetType() override;

    bool Available() override;

    double GetStorageUsageRatio(const std::string &trace_id) const override;

    ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) override;
    ErrorCode Close() override;

    // EventReportingBackend interface
    void SetCleanupCallback(CleanupCallback cb) override;
    bool IsCleanupCallbackSet() const override { return cleanup_cb_set_.load(std::memory_order_acquire); }

    ErrorCode RegisterNode(const std::string &instance_id,
                           const std::string &host_ip_port,
                           const std::vector<std::string> &mediums) override;

    ErrorCode UnregisterNode(const std::string &instance_id, const std::string &host_ip_port) override;

    ErrorCode OnHeartbeat(const std::string &instance_id,
                          const std::string &host_ip_port,
                          const std::map<std::string, std::string> &system_status) override;
    void SetNodeUnavailable(const std::string &instance_id, const std::string &host_ip_port) override;
    bool IsNodeAvailable(const std::string &instance_id, const std::string &host_ip_port) const;

    uint64_t GetNodeGeneration(const std::string &instance_id, const std::string &host_ip_port) const override;

    std::string BuildLocationId(const std::string &medium, const std::string &host_ip_port) const override;
    std::string HostSuffix(const std::string &host_ip_port) const override;
    DataStorageType GetStorageType() const override;
    std::string GetProtocol() const override;

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override;

    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override;

    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<bool> MightExist(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override;

private:
    void LivenessCheckerLoop();

    static int64_t NowMillis() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }

    struct NodeInfo {
        std::string instance_id;
        std::atomic<int64_t> last_heartbeat_ms{0};
        std::atomic<bool> available{true};
        std::atomic<int64_t> unavailable_since_ms{0};

        std::vector<std::string> mediums;
        mutable std::mutex status_mutex;
        std::map<std::string, std::string> last_system_status;
        MetricsTags metrics_tags;
    };

    VineyardStorageSpec spec_;

    mutable std::shared_mutex nodes_mutex_;
    // instance_id -> (host_ip_port -> NodeInfo)
    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<NodeInfo>>> instance_nodes_;
    // Persists across unregister/register to fence stale cleanup.
    // instance_id -> (host_ip_port -> generation)
    std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>> node_generation_;

    std::thread liveness_checker_thread_;
    std::atomic<bool> liveness_checker_running_{false};

    int64_t heartbeat_timeout_ms_ = VineyardStorageSpec::kDefaultHeartbeatTimeoutMs;
    int64_t cleanup_grace_ms_ = VineyardStorageSpec::kDefaultCleanupGraceMs;
    int64_t liveness_check_interval_ms_ = VineyardStorageSpec::kDefaultLivenessCheckIntervalMs;

    void ClearNodeGauges(const NodeInfo &info);

    mutable std::mutex cleanup_cb_mutex_;
    CleanupCallback cleanup_callback_;
    std::atomic<bool> cleanup_cb_set_{false};
};

} // namespace kv_cache_manager
