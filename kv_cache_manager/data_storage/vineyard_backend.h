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
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// V6D cluster storage backend. V6D nodes drive liveness via EVENT_HEARTBEAT.
// LivenessCheckerLoop runs a three-stage decision per node:
//   healthy -> unavailable (heartbeat timeout) -> dead (cleanup grace exceeded).
// Does NOT manage allocation (Create/Delete return EC_UNIMPLEMENTED).
class VineyardBackend : public DataStorageBackend {
public:
    // Invoked when a node has been unavailable for >= cleanup_grace_ms.
    // generation_at_trigger fences stale cleanup against re-registration.
    using CleanupCallback = std::function<void(const std::string &host_ip_port, uint64_t generation_at_trigger)>;

    VineyardBackend() = delete;
    explicit VineyardBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~VineyardBackend() override;

    DataStorageType GetType() override;

    bool Available() override;

    // Returns 1.0 to prevent DataStorageSelector from picking this backend.
    double GetStorageUsageRatio(const std::string &trace_id) const override;

    ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) override;
    ErrorCode Close() override;

    void SetCleanupCallback(CleanupCallback cb);
    bool IsCleanupCallbackSet() const { return cleanup_cb_set_.load(std::memory_order_acquire); }

    // Idempotent. Re-registration merges mediums and bumps generation counter.
    ErrorCode RegisterNode(const std::string &host_ip_port, const std::vector<std::string> &mediums);

    // Does NOT reset generation counter so in-flight cleanup can still be fenced.
    ErrorCode UnregisterNode(const std::string &host_ip_port);

    void OnHeartbeat(const std::string &host_ip_port, const std::map<std::string, std::string> &system_status);
    void SetNodeUnavailable(const std::string &host_ip_port);
    bool IsNodeAvailable(const std::string &host_ip_port) const;
    bool IsNodeRegistered(const std::string &host_ip_port) const;

    // Extracts host_ip_port from location_id suffix and queries IsNodeAvailable.
    bool IsLocationAvailable(const std::string &location_id) const;

    // Returns 0 if the host was never registered.
    uint64_t GetNodeGeneration(const std::string &host_ip_port) const;

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
        std::atomic<int64_t> last_heartbeat_ms{0};
        std::atomic<bool> available{true};
        std::atomic<int64_t> unavailable_since_ms{0};

        std::vector<std::string> mediums;
        std::map<std::string, std::string> last_system_status;
    };

    VineyardStorageSpec spec_;

    mutable std::shared_mutex nodes_mutex_;
    std::unordered_map<std::string, std::unique_ptr<NodeInfo>> nodes_;
    // Bumped on every RegisterNode; persists across unregister/register cycles
    // to fence stale cleanup. Protected by nodes_mutex_.
    std::unordered_map<std::string, uint64_t> node_generation_;

    std::thread liveness_checker_thread_;
    std::atomic<bool> liveness_checker_running_{false};

    int64_t heartbeat_timeout_ms_ = VineyardStorageSpec::kDefaultHeartbeatTimeoutMs;
    int64_t cleanup_grace_ms_ = VineyardStorageSpec::kDefaultCleanupGraceMs;
    int64_t liveness_check_interval_ms_ = VineyardStorageSpec::kDefaultLivenessCheckIntervalMs;

    mutable std::mutex cleanup_cb_mutex_;
    CleanupCallback cleanup_callback_;
    std::atomic<bool> cleanup_cb_set_{false};
};

} // namespace kv_cache_manager
