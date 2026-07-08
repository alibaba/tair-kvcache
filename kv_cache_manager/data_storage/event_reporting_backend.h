#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

// Unified backend for event-reporting storage types (V6D / RTP-LLM / VLLM).
// Engine-specific parameters (location_id prefix, metrics prefix, protocol)
// are derived from config_.type() at DoOpen time.
class EventReportingBackend : public DataStorageBackend {
public:
    using CleanupCallback =
        std::function<void(const std::string &instance_id, const std::string &host_ip_port, uint64_t generation)>;

    EventReportingBackend() = delete;
    explicit EventReportingBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~EventReportingBackend() override;

    // --- DataStorageBackend interface ---
    DataStorageType GetType() override;
    bool Available() override;
    double GetStorageUsageRatio(const std::string &trace_id) const override;
    ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) override;
    ErrorCode Close() override;

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

    // --- Event reporting methods ---
    void SetCleanupCallback(CleanupCallback cb);
    bool IsCleanupCallbackSet() const { return cleanup_cb_set_.load(std::memory_order_acquire); }

    ErrorCode RegisterNode(const std::string &instance_id,
                           const std::string &host_ip_port,
                           const std::vector<std::string> &mediums);
    ErrorCode UnregisterNode(const std::string &instance_id, const std::string &host_ip_port);
    ErrorCode OnHeartbeat(const std::string &instance_id,
                          const std::string &host_ip_port,
                          const std::map<std::string, std::string> &system_status);
    void SetNodeUnavailable(const std::string &instance_id, const std::string &host_ip_port);
    bool IsNodeAvailable(const std::string &instance_id, const std::string &host_ip_port) const;
    uint64_t GetNodeGeneration(const std::string &instance_id, const std::string &host_ip_port) const;

    std::string BuildLocationId(const std::string &medium, const std::string &host_ip_port) const;
    std::string HostSuffix(const std::string &host_ip_port) const;
    DataStorageType GetStorageType() const;
    std::string GetProtocol() const;

private:
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

    void LivenessCheckerLoop();
    void ClearNodeGauges(const NodeInfo &info);
    static int64_t NowMillis() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }

    // Engine-specific parameter derivation from config_.type()
    static std::string GetBackendIdentifier(DataStorageType type);
    static std::string GetMetricsPrefix(DataStorageType type);
    static std::string GetProtocolStr(DataStorageType type);

    EventReportingStorageSpec spec_;

    mutable std::shared_mutex nodes_mutex_;
    // instance_id -> (host_ip_port -> NodeInfo)
    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<NodeInfo>>> instance_nodes_;
    // Persists across unregister/register to fence stale cleanup.
    // instance_id -> (host_ip_port -> generation)
    std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>> node_generation_;

    std::thread liveness_checker_thread_;
    std::atomic<bool> liveness_checker_running_{false};

    int64_t heartbeat_timeout_ms_ = EventReportingStorageSpec::kDefaultHeartbeatTimeoutMs;
    int64_t cleanup_grace_ms_ = EventReportingStorageSpec::kDefaultCleanupGraceMs;
    int64_t liveness_check_interval_ms_ = EventReportingStorageSpec::kDefaultLivenessCheckIntervalMs;

    mutable std::mutex cleanup_cb_mutex_;
    CleanupCallback cleanup_callback_;
    std::atomic<bool> cleanup_cb_set_{false};
};

} // namespace kv_cache_manager
