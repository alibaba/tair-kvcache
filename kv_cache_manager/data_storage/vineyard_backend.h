#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

// VineyardBackend represents an entire V6D cluster as a single storage backend.
// Individual V6D nodes register/unregister themselves at runtime via
// RegisterNode / UnregisterNode, each of which starts/stops a dedicated
// per-node probe thread.
//
// Unlike other backends, VineyardBackend does NOT manage data allocation:
// Create() and Delete() return EC_UNIMPLEMENTED.
//
// Node-level availability is tracked independently.  Callers use
// IsLocationAvailable("kvs_vineyard_{ip:port}") to check whether the node
// backing a particular CacheLocation is currently healthy.
class VineyardBackend : public DataStorageBackend {
public:
    VineyardBackend() = delete;
    explicit VineyardBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~VineyardBackend() override;

    DataStorageType GetType() override;

    // Cluster-level availability: true as long as the backend is Open.
    // Per-node health is tracked separately via IsLocationAvailable().
    bool Available() override;

    // Returns 1.0 to prevent DataStorageSelector from selecting this backend
    // for write allocation (V6D manages its own capacity).
    double GetStorageUsageRatio(const std::string &trace_id) const override;

    ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) override;
    ErrorCode Close() override;

    // Register a V6D node: adds it to the internal node table and starts a
    // probe thread for it.  Idempotent if the node is already registered.
    ErrorCode RegisterNode(const std::string &host_ip_port);

    // Unregister a V6D node: stops its probe thread and removes the entry.
    ErrorCode UnregisterNode(const std::string &host_ip_port);

    // Set the availability of a specific node (called by ProbeNodeLoop).
    void SetNodeAvailable(const std::string &host_ip_port, bool available);

    // Return true when the node identified by host_ip_port is registered
    // and currently healthy.
    bool IsNodeAvailable(const std::string &host_ip_port) const;

    // Resolve a location ID of the form "kvs_vineyard_{ip:port}" and return
    // whether the corresponding node is available.  Used as the
    // check_loc_data_exist callback in GetCacheLocation queries.
    bool IsLocationAvailable(const std::string &location_id) const;

    // These interfaces should never be called on VineyardBackend.
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override;

    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override;

    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override;

private:
    // Per-node probe loop: runs in a dedicated thread, probes the node every
    // 10 s with up to 10 retries.  When availability changes from true to
    // false, triggers an async HOST_DOWN cleanup.
    void ProbeNodeLoop(const std::string &host_ip_port);

    // Attempt a lightweight health check against host_ip_port.
    bool DetectNodeHealthy(const std::string &host_ip_port) const;

    bool IsNodeRegistered(const std::string &host_ip_port) const;

    struct NodeInfo {
        std::atomic<bool> available{true};
        std::thread probe_thread;
    };

    VineyardStorageSpec spec_;
    mutable std::shared_mutex nodes_mutex_;
    std::unordered_map<std::string, std::unique_ptr<NodeInfo>> nodes_;
};

} // namespace kv_cache_manager
