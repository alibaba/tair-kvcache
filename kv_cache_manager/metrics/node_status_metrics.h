#pragma once

#include <memory>

namespace kv_cache_manager {

class DataStorageManager;
class MetricsRegistry;

// Node-level read/write capability published through the access_mode gauge.
// Every data plane API is leader-only, so a follower always reports NONE.
enum class NodeAccessMode : int {
    NONE = 0,       // serves neither read nor write
    READ_ONLY = 1,  // serves read, rejects write
    READ_WRITE = 2, // serves both read and write
};

// Publishes the two node-scoped gauges, which carry no type / unique_name
// label because they describe the process rather than one storage backend:
//
//   is_leader   -> kvcm_is_leader    1 while this node holds the leader lease
//   access_mode -> kvcm_access_mode  the NodeAccessMode value above
class NodeStatusMetrics {
public:
    NodeStatusMetrics() = delete;

    static constexpr const char *kIsLeaderName = "is_leader";
    static constexpr const char *kAccessModeName = "access_mode";

    // Reduces every registered storage backend to one node-level mode: a node
    // can serve a request as soon as one backend can, which matches how
    // DataStorageSelector picks a backend per request. A null manager, or one
    // holding no usable backend, yields NONE.
    static NodeAccessMode ComputeAccessMode(const std::shared_ptr<DataStorageManager> &data_storage_manager);

    // Writes both gauges. A follower is published as NONE without consulting
    // the storage manager, since it answers no data plane request.
    static void Publish(const std::shared_ptr<MetricsRegistry> &metrics_registry,
                        bool is_leader,
                        const std::shared_ptr<DataStorageManager> &data_storage_manager);
};

} // namespace kv_cache_manager
