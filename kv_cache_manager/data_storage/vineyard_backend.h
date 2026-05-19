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

// VineyardBackend represents an entire V6D cluster as a single storage backend.
// V8 model: KVCM does NOT actively probe V6D nodes. V6D nodes drive the
// liveness via EVENT_HEARTBEAT (default 10s cadence). VineyardBackend uses a
// single LivenessCheckerLoop that walks the node table on
// liveness_check_interval_ms cadence and runs the three-stage decision:
//
//   healthy:                now - last_heartbeat <= heartbeat_timeout_ms
//   unavailable (lazy):     timeout exceeded -> mark available=false,
//                           record unavailable_since_ms
//   dead (active cleanup):  unavailable for >= cleanup_grace_ms
//                           -> invoke cleanup_callback to drop all
//                              locations whose location_id ends with the host
//
// Unlike data-handling backends, VineyardBackend does NOT manage allocation:
// Create() and Delete() return EC_UNIMPLEMENTED. Per-node availability is
// surfaced through IsLocationAvailable; CacheManager wires it into
// check_loc_data_exist for the matching path.
class VineyardBackend : public DataStorageBackend {
public:
    // Cleanup hook signature: (host_ip_port). Invoked from LivenessCheckerLoop
    // when a node has been unavailable for >= cleanup_grace_ms. Implementation
    // is provided by CacheManager (see CleanupHostLocations).
    using CleanupCallback = std::function<void(const std::string &host_ip_port)>;

    VineyardBackend() = delete;
    explicit VineyardBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~VineyardBackend() override;

    DataStorageType GetType() override;

    // Cluster-level Available: true while the backend is Open. Per-node health
    // is independent and surfaced via IsLocationAvailable.
    bool Available() override;

    // Returns 1.0 to keep DataStorageSelector from picking this backend for
    // write allocation -- V6D manages its own capacity.
    double GetStorageUsageRatio(const std::string &trace_id) const override;

    ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) override;
    ErrorCode Close() override;

    // CacheManager registers a cleanup hook before opening; LivenessCheckerLoop
    // calls it for hosts that pass cleanup_grace_ms. Idempotent: setting null
    // disables active cleanup but leaves liveness flags in place.
    void SetCleanupCallback(CleanupCallback cb);

    // Register a V6D node and the mediums it advertises (e.g. {"mem","disk"}).
    // Idempotent: re-registration merges new mediums into the existing entry,
    // refreshes last_heartbeat_ms, and clears the unavailable flag.
    ErrorCode RegisterNode(const std::string &host_ip_port, const std::vector<std::string> &mediums);

    // Remove the node from the table. Subsequent IsNodeAvailable returns false.
    ErrorCode UnregisterNode(const std::string &host_ip_port);

    // EVENT_HEARTBEAT entry point. Refreshes last_heartbeat_ms, restores
    // available=true if the node was previously flagged unavailable, and
    // stashes system_status for observability.
    void OnHeartbeat(const std::string &host_ip_port, const std::map<std::string, std::string> &system_status);

    // Mark the node unavailable (called by EVENT_HOST_DOWN handler or
    // LivenessCheckerLoop). Idempotent.
    void SetNodeUnavailable(const std::string &host_ip_port);

    // Whether the host is registered AND currently flagged available.
    bool IsNodeAvailable(const std::string &host_ip_port) const;

    // Whether the host is registered in nodes_ (regardless of available flag).
    // Used by MightExist: registered = data physically still on the V6D node.
    bool IsNodeRegistered(const std::string &host_ip_port) const;

    // location_id format: "kvs#v6d#{medium}#{ip:port}" (V8 §2.1.4).
    // Slices the host_ip_port off the suffix and queries IsNodeAvailable.
    bool IsLocationAvailable(const std::string &location_id) const;

    // V6D manages its own data; KVCM callers should never reach these methods.
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

        // mediums registered by V6D side (NODE_REGISTER); informational.
        std::vector<std::string> mediums;
        // Most recent heartbeat payload (CPU / capacity / version / ...).
        // Read by observability paths only; mutated under nodes_mutex_.
        std::map<std::string, std::string> last_system_status;
    };

    VineyardStorageSpec spec_;

    mutable std::shared_mutex nodes_mutex_;
    std::unordered_map<std::string, std::unique_ptr<NodeInfo>> nodes_;

    std::thread liveness_checker_thread_;
    std::atomic<bool> liveness_checker_running_{false};

    // cached from spec_ for hot paths.
    int64_t heartbeat_timeout_ms_ = VineyardStorageSpec::kDefaultHeartbeatTimeoutMs;
    int64_t cleanup_grace_ms_ = VineyardStorageSpec::kDefaultCleanupGraceMs;
    int64_t liveness_check_interval_ms_ = VineyardStorageSpec::kDefaultLivenessCheckIntervalMs;

    // Set by CacheManager. Guarded by cleanup_cb_mutex_ so SetCleanupCallback
    // can replace it safely while LivenessCheckerLoop is running.
    mutable std::mutex cleanup_cb_mutex_;
    CleanupCallback cleanup_callback_;
};

} // namespace kv_cache_manager
