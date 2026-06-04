#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/affinity/affinity_strategy.h"
#include "kv_cache_manager/affinity/frequency_sketch.h"
#include "kv_cache_manager/affinity/hint_suppressor.h"
#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/write_hints.h"

namespace kv_cache_manager {

class DataStorageManager;

struct AffinityResolveContext {
    std::string instance_strategy_json;
    std::string group_strategy_json;
    std::string caller_node_id;
    std::string caller_supernode_id;
    std::string instance_id;
    std::string instance_group_name;
    std::string trace_id;
};

// CacheAffinityManager owns the affinity strategy instances and the runtime
// snapshot of node metrics. It is the single entry point for write / read /
// eviction affinity decisions.
//
// Strategies are organized into three priority tiers, from highest to lowest:
//   1. instance level       -- per-instance JSON, persisted on InstanceInfo
//   2. instance_group level -- per-group JSON, persisted on InstanceGroup
//   3. process level        -- loaded once via LoadProcessStrategyFromJson*
//
// At resolve time the highest-priority non-empty strategy whose JSON parses
// is used. The other tiers are silently ignored for that call. Strategies
// parsed from per-instance / per-group JSON are memoized by raw JSON text so
// identical configs share a parsed AffinityStrategy.
//
// Threading: thread-safe. All mutating and reading APIs take an internal
// mutex; parsed AffinityStrategy objects are treated as immutable once cached.
class CacheAffinityManager {
public:
    CacheAffinityManager() = default;
    ~CacheAffinityManager();

    CacheAffinityManager(const CacheAffinityManager &) = delete;
    CacheAffinityManager &operator=(const CacheAffinityManager &) = delete;

    // ---- Process-level strategy configuration ------------------------------
    // Both loaders accept the file-level envelope { "strategy": { ... } } as
    // well as a bare strategy object. error_msg is populated on failure when
    // non-null. Re-loading replaces the previous process-level strategy
    // atomically.
    bool LoadProcessStrategyFromJsonFile(const std::string &path, std::string *error_msg = nullptr);
    bool LoadProcessStrategyFromJsonString(const std::string &json, std::string *error_msg = nullptr);

    // ---- Node metrics ingestion (v1: stub; future: registry/heartbeat) ----
    void UpsertNodeMetrics(const NodeMetrics &metrics);
    void RemoveNode(const std::string &node_id);
    std::vector<NodeMetrics> SnapshotNodes() const;

    // ---- Resolve decisions (write / read / eviction) -----------------------

    WriteDecision ResolveWrite(const AffinityResolveContext &ctx);

    ReadDecision ResolveRead(const ReadRequest &req, const AffinityResolveContext &ctx);

    // Returns the set of node IDs whose load exceeds the high-water threshold
    // (with hysteresis applied). Empty set means no node-level eviction needed.
    std::unordered_set<std::string> ResolveEviction(const AffinityResolveContext &ctx);

    // Accumulate evicted bytes for hysteresis estimation.
    void ReportEvictedBytes(const std::string &node_id, int64_t bytes);

    // Start a background thread that periodically pulls per-node metrics from
    // each backend via DataStorageManager. Idempotent; stopped on destruction.
    void StartMetricsPullLoop(std::shared_ptr<DataStorageManager> dsm, uint32_t interval_seconds = 5);
    void StopMetricsPullLoop();

    // Global kill-switch: when disabled, GetStrategy always returns Noop.
    void SetGloballyDisabled(bool disabled) { globally_disabled_.store(disabled, std::memory_order_relaxed); }
    bool GloballyDisabled() const { return globally_disabled_.load(std::memory_order_relaxed); }

private:
    std::shared_ptr<AffinityStrategy> GetStrategy(const std::string &instance_strategy_json,
                                                  const std::string &instance_group_strategy_json) const;
    std::function<const NodeMetrics *(const std::string &)> MakeNodeMetricsAccessor() const;
    StrategyContext BuildStrategyContext(const AffinityResolveContext &ctx) const;

    std::shared_ptr<AffinityStrategy> ParseOrCacheAffinityLocked(const std::string &json) const;

    mutable std::mutex mux_;
    std::shared_ptr<AffinityStrategy> process_affinity_strategy_;
    std::unordered_map<std::string, NodeMetrics> nodes_;
    // Memoized parsed strategies keyed by raw JSON text.
    mutable std::unordered_map<std::string, std::shared_ptr<AffinityStrategy>> affinity_strategy_cache_;
    // Per-(caller, key) frequency sketch; mutable because read path updates it
    // even through const methods (GetStrategy etc. remain logically const).
    mutable FrequencySketch sketch_;
    mutable HintSuppressor suppressor_;

    // NodeMetrics pull background thread
    std::thread metrics_thread_;
    std::atomic<bool> metrics_stop_{false};
    std::mutex metrics_cv_mu_;
    std::condition_variable metrics_cv_;
    std::shared_ptr<DataStorageManager> metrics_dsm_;
    uint32_t metrics_interval_seconds_ = 5;

    std::atomic<bool> globally_disabled_{false};

    // Node-level eviction hysteresis state (protected by mux_)
    std::unordered_map<std::string, int64_t> node_evicted_bytes_;
    int64_t node_metrics_last_reset_us_ = 0;
};

} // namespace kv_cache_manager
