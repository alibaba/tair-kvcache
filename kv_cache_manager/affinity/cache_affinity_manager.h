#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/affinity/strategy.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/write_hints.h"

namespace kv_cache_manager {

// CacheAffinityManager owns the affinity Strategy(es) and the runtime
// snapshot of node metrics. It is consulted on the write path to translate
// caller information + candidate storage nodes into a WriteHints struct that
// the storage backend can act on.
//
// Strategies are organized into three priority tiers, from highest to lowest:
//   1. instance level       -- per-instance JSON, persisted on InstanceInfo
//   2. instance_group level -- per-group JSON, persisted on InstanceGroup
//   3. process level        -- loaded once via LoadProcessStrategyFromJson*
//
// At resolve time the highest-priority non-empty strategy whose JSON parses
// is used. The other tiers are silently ignored for that call. Strategies
// parsed from per-instance / per-group JSON are memoized by raw JSON text so
// identical configs share a parsed Strategy.
//
// Threading: thread-safe. All mutating and reading APIs take an internal
// mutex; parsed Strategy objects are treated as immutable once cached.
class CacheAffinityManager {
public:
    CacheAffinityManager() = default;
    ~CacheAffinityManager() = default;

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

    // ---- Write-path decision ----------------------------------------------
    struct ResolveContext {
        std::string caller_node_ip;
        std::size_t block_count = 0;
        std::size_t bytes_per_block = 0;
        std::string trace_id;

        // Identity of the write target. instance_id is informational; the
        // priority chain is driven solely by which strategy JSON is non-empty.
        std::string instance_id;
        std::string instance_group_name;

        // Per-instance strategy JSON; empty = no override at instance level.
        std::string instance_strategy_json;
        // Per-instance-group strategy JSON; empty = no override at group level.
        std::string instance_group_strategy_json;
    };

    // Compute write hints for `candidates` (storage node ids that the chosen
    // backend currently exposes).
    //
    // Priority chain at the call:
    //   instance_strategy_json > instance_group_strategy_json > process strategy
    // The first non-empty JSON whose parse succeeds is used. Parse failures
    // for an override fall through to the next tier silently.
    //
    // Return codes:
    //   EC_OK    - strategy ran successfully. out_hints.preferred_node_ids
    //              may be empty, meaning "no preference; let the backend
    //              decide".
    //   EC_ERROR - strategy aborted (e.g. prefer_local with on_miss=abort
    //              found no local candidate).
    //
    // When no strategy is configured at any tier, returns EC_OK + empty hints.
    ErrorCode
    Resolve(const ResolveContext &ctx, const std::vector<std::string> &candidates, WriteHints &out_hints) const;

private:
    // Parse `json` as a Strategy, memoized by raw JSON text. Caller must hold
    // mux_. Returns nullptr on parse failure.
    std::shared_ptr<Strategy> ParseOrCacheLocked(const std::string &json) const;

    mutable std::mutex mux_;
    std::shared_ptr<Strategy> process_strategy_;
    std::unordered_map<std::string, NodeMetrics> nodes_;
    // Memoized parsed strategies keyed by raw JSON text.
    mutable std::unordered_map<std::string, std::shared_ptr<Strategy>> parsed_strategy_cache_;
};

} // namespace kv_cache_manager
