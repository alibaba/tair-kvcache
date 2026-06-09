// Probe tests: targeted edge-case and bug-hunting tests for
// CacheAffinityManager new logic. Each test probes a specific concern.

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/affinity/cache_affinity_manager.h"
#include "kv_cache_manager/affinity/local_replica_strategy.h"
#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

namespace {
std::vector<std::unique_ptr<CacheAffinityManager>> g_mgr_pool;
CacheAffinityManager &NewManager() {
    g_mgr_pool.push_back(std::make_unique<CacheAffinityManager>());
    return *g_mgr_pool.back();
}
} // namespace

class AffinityProbeTest : public TESTBASE {};

static const char *kWriteFilterJson = R"({
    "type": "local_replica",
    "write": {
        "ops": {
            "filter": { "metric": "load_ratio", "max": 0.90 },
            "prefer_local": { "on_miss": "passthrough" },
            "sort": [{ "metric": "free_bytes", "weight": 1.0 }]
        }
    }
})";

// ===================================================================
// Probe 1: MakeNodeMetricsAccessor TLS staleness
//
// MakeNodeMetricsAccessor uses thread_local keyed by `this`. On the
// same thread + same manager, calling UpsertNodeMetrics between two
// ResolveWrite calls does NOT refresh the TLS snapshot. The second
// Resolve sees stale metrics.
//
// Scenario: node_a starts at load 0.50 (passes filter max 0.90).
// After first ResolveWrite, update node_a to load 0.95 (should fail filter).
// Second ResolveWrite should filter out node_a, but TLS returns stale 0.50.
// ===================================================================

TEST_F(AffinityProbeTest, TlsStaleMetricsInWriteFilter) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kWriteFilterJson));

    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 300000, 0.50, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", DataStorageType{}, 800000, 0.30, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "tls-probe-1";

    // First resolve: node_a at 0.50, passes filter → preferred
    {
        WriteDecision dec = mgr.ResolveWrite(ctx);
        ASSERT_FALSE(dec.hints.preferred_node_ids.empty());
        EXPECT_EQ("node_a", dec.hints.preferred_node_ids[0]);
    }

    // Update node_a to 0.95: should be filtered out on next resolve
    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 50000, 0.95, 0, 0, 2});

    // Second resolve: node_a should be filtered out (load 0.95 > max 0.90)
    // Expected: only node_b survives → preferred = [node_b]
    // Actual (if TLS is stale): TLS still has node_a at 0.50 → node_a passes
    {
        WriteDecision dec = mgr.ResolveWrite(ctx);
        ASSERT_FALSE(dec.hints.preferred_node_ids.empty());

        // If this passes, TLS correctly refreshed. If it gets "node_a", TLS is stale.
        EXPECT_EQ("node_b", dec.hints.preferred_node_ids[0])
            << "BUG: TLS returned stale metrics (node_a load=0.50 instead of 0.95)."
               " MakeNodeMetricsAccessor never refreshes for the same `this` pointer.";
    }
}

// ===================================================================
// Probe 2: TLS staleness affects read capacity gate
//
// Same TLS issue but in the read path: caller_capacity_threshold check
// uses get_node_metrics which goes through the TLS accessor.
// ===================================================================

TEST_F(AffinityProbeTest, TlsStaleMetricsInReadCapacityGate) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "read": {
            "on_miss": {
                "enabled": true,
                "replication_hot_threshold": 1,
                "caller_capacity_threshold": 0.85,
                "caller_capacity_buffer": 0.05
            }
        }
    })"));

    // caller node_a starts at low load → capacity gate should pass
    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 500000, 0.30, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", DataStorageType{}, 800000, 0.20, 0, 0, 1});

    LocationSpec remote("tp0", "tair://node_b/x", "node_b");
    CacheLocation winner;
    winner.push_location_spec(LocationSpec("tp0", "tair://node_b/x", "node_b"));

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "tls-read-probe";

    // First read: threshold=1, load=0.30 < gate=0.80 → hint emitted
    {
        ReadRequest req;
        req.block_key = 1;
        req.spec_candidates["tp0"] = {&remote};
        req.winner_tier = &winner;
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        ASSERT_EQ(1u, dec.side_effects.size()) << "first read should emit hint";
    }

    // Now overload node_a: load 0.90 > gate 0.80
    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 100000, 0.90, 0, 0, 2});

    // Second read (different block_key to avoid dedup): caller overloaded,
    // hint should be suppressed.
    // With TLS stale: sees old load 0.30 → gate passes → hint emitted (wrong)
    {
        ReadRequest req;
        req.block_key = 2;
        req.spec_candidates["tp0"] = {&remote};
        req.winner_tier = &winner;
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        EXPECT_TRUE(dec.side_effects.empty()) << "BUG: hint emitted despite caller load 0.90 > gate 0.80."
                                                 " TLS returned stale load=0.30.";
    }
}

// ===================================================================
// Probe 3: Hysteresis reset when NodeMetrics refresh
//
// After eviction bytes accumulate and reduce estimated load below low,
// new metrics (with higher updated_at_us) should clear evicted_bytes.
// The node becomes exceeded again if raw load_ratio still exceeds threshold.
// ===================================================================

TEST_F(AffinityProbeTest, HysteresisResetsOnMetricsRefresh) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "eviction": {
            "ops": [{"op": "node_water_level", "threshold": 0.85, "low": 0.70}]
        }
    })"));

    // node_a: load 0.90, free 100K → total ~1M
    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 100000, 0.90, 0, 0, 100});

    AffinityResolveContext ctx;

    // Step 1: exceeded
    {
        auto ex = mgr.ResolveEviction(ctx);
        ASSERT_EQ(1u, ex.size());
    }

    // Step 2: evict 300K → estimated 0.60 < low 0.70 → not exceeded
    mgr.ReportEvictedBytes("node_a", 300000);
    {
        auto ex = mgr.ResolveEviction(ctx);
        EXPECT_TRUE(ex.empty()) << "hysteresis: estimated load 0.60 < low 0.70";
    }

    // Step 3: metrics refresh with same high load but new updated_at_us
    // This should clear evicted_bytes → estimated goes back to raw 0.90
    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 100000, 0.90, 0, 0, 200});
    {
        auto ex = mgr.ResolveEviction(ctx);
        EXPECT_EQ(1u, ex.size()) << "After metrics refresh, evicted_bytes should be cleared."
                                    " Node raw load 0.90 > threshold 0.85 → exceeded again.";
    }
}

// ===================================================================
// Probe 4: Hysteresis with updated_at_us = 0 (edge case)
//
// If all nodes have updated_at_us = 0, the hysteresis reset condition
// `max_updated_at > node_metrics_last_reset_us_` is 0 > 0 = false.
// evicted_bytes never reset. Is this intentional?
// ===================================================================

TEST_F(AffinityProbeTest, HysteresisWithZeroTimestamp) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "eviction": {
            "ops": [{"op": "node_water_level", "threshold": 0.85, "low": 0.70}]
        }
    })"));

    // All timestamps = 0
    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 100000, 0.90, 0, 0, 0});

    AffinityResolveContext ctx;

    // Exceeded
    {
        auto ex = mgr.ResolveEviction(ctx);
        ASSERT_EQ(1u, ex.size());
    }

    // Evict enough to drop below low
    mgr.ReportEvictedBytes("node_a", 300000);
    {
        auto ex = mgr.ResolveEviction(ctx);
        EXPECT_TRUE(ex.empty());
    }

    // Re-insert same metrics with timestamp still 0
    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 100000, 0.90, 0, 0, 0});

    // With timestamp=0: max_updated_at=0, last_reset=0 → 0 > 0 false
    // evicted_bytes NOT cleared → still not exceeded
    // This means metrics at timestamp 0 can never trigger hysteresis reset.
    {
        auto ex = mgr.ResolveEviction(ctx);
        // If this is empty, it means evicted_bytes were NOT cleared
        // (the hysteresis "sticks" forever with timestamp=0).
        // Whether this is a bug depends on design intent.
        EXPECT_TRUE(ex.empty()) << "With updated_at_us=0, evicted_bytes should NOT be cleared"
                                   " (0 > 0 is false). Document this known edge case.";
    }
}

// ===================================================================
// Probe 5: FrequencySketch survives strategy reload
//
// The sketch is owned by CacheAffinityManager, not the strategy.
// Reloading process strategy should preserve accumulated frequency.
// ===================================================================

TEST_F(AffinityProbeTest, SketchSurvivesStrategyReload) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "read": { "on_miss": { "enabled": true, "replication_hot_threshold": 3 } }
    })"));

    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 500000, 0.30, 0, 0, 1});

    LocationSpec remote("tp0", "uri_b", "node_b");
    CacheLocation winner;
    winner.push_location_spec(LocationSpec("tp0", "uri_b", "node_b"));

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "sketch-reload";

    // Accumulate 2 reads (count=2, below threshold=3)
    for (int i = 0; i < 2; ++i) {
        ReadRequest req;
        req.block_key = 77;
        req.spec_candidates["tp0"] = {&remote};
        req.winner_tier = &winner;
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        EXPECT_TRUE(dec.side_effects.empty());
    }

    // Reload strategy with LOWER threshold (2)
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "read": { "on_miss": { "enabled": true, "replication_hot_threshold": 2 } }
    })"));

    // Next read: sketch count was 2, new threshold is 2.
    // Observe increments to 3, check 3 >= 2 → hint.
    // This proves sketch survived reload.
    {
        ReadRequest req;
        req.block_key = 77;
        req.spec_candidates["tp0"] = {&remote};
        req.winner_tier = &winner;
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        EXPECT_EQ(1u, dec.side_effects.size()) << "Sketch count should survive strategy reload."
                                                  " count was 2, new threshold=2, 3rd read → hint.";
    }
}

// ===================================================================
// Probe 6: ResolveEviction StrategyContext missing get_node_metrics
//
// ResolveEviction builds StrategyContext manually, NOT via
// BuildStrategyContext. It does NOT set get_node_metrics. If the
// eviction strategy tried to call get_node_metrics, it would be null.
// Currently LocalReplicaAffinityStrategy::ResolveEviction only uses
// ctx.all_nodes, so this is safe. But verify it doesn't crash.
// ===================================================================

TEST_F(AffinityProbeTest, EvictionContextHasNoGetNodeMetrics) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "eviction": {
            "ops": [{"op": "node_water_level", "threshold": 0.85}]
        }
    })"));

    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 100000, 0.90, 0, 0, 1});

    AffinityResolveContext ctx;
    // No crash even though get_node_metrics is null in the StrategyContext
    auto ex = mgr.ResolveEviction(ctx);
    EXPECT_EQ(1u, ex.size());
}

// ===================================================================
// Probe 7: ReadDecision with multiple spec names
//
// When read has multiple spec names (e.g. "tp0" on remote, "tp1" on
// local), PickLocalSpec should independently select the best for each.
// ===================================================================

TEST_F(AffinityProbeTest, ReadMultipleSpecNamesPickIndependently) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica"
    })"));

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "multi-spec";

    LocationSpec tp0_remote("tp0", "uri_b", "node_b");
    LocationSpec tp1_local("tp1", "uri_a", "node_a");
    LocationSpec tp1_remote("tp1", "uri_c", "node_c");

    ReadRequest req;
    req.block_key = 1;
    req.spec_candidates["tp0"] = {&tp0_remote};             // only remote
    req.spec_candidates["tp1"] = {&tp1_remote, &tp1_local}; // local available
    CacheLocation winner;
    req.winner_tier = &winner;

    ReadDecision dec = mgr.ResolveRead(req, ctx);

    // tp0: no local candidate → falls back to first (remote)
    ASSERT_NE(nullptr, dec.picked_specs["tp0"]);
    EXPECT_EQ("node_b", dec.picked_specs["tp0"]->node_id());

    // tp1: local candidate exists → picks node_a
    ASSERT_NE(nullptr, dec.picked_specs["tp1"]);
    EXPECT_EQ("node_a", dec.picked_specs["tp1"]->node_id());
}

// ===================================================================
// Probe 8: ResolveWrite candidates come from SnapshotNodes, not from
// caller. If a node is removed between two resolves, it disappears.
// ===================================================================

TEST_F(AffinityProbeTest, WriteReflectsNodeAddRemove) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "write": { "ops": { "prefer_local": {"on_miss": "passthrough"} } }
    })"));

    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 500000, 0.30, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", DataStorageType{}, 800000, 0.20, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "node-remove";

    // Both nodes present: node_a preferred
    {
        WriteDecision dec = mgr.ResolveWrite(ctx);
        ASSERT_FALSE(dec.hints.preferred_node_ids.empty());
        EXPECT_EQ("node_a", dec.hints.preferred_node_ids[0]);
    }

    // Remove node_a: only node_b remains
    mgr.RemoveNode("node_a");

    {
        WriteDecision dec = mgr.ResolveWrite(ctx);
        // prefer_local: node_a not found → passthrough [node_b]
        ASSERT_FALSE(dec.hints.preferred_node_ids.empty());
        EXPECT_EQ("node_b", dec.hints.preferred_node_ids[0]);
    }

    // Remove node_b: no nodes → empty hints
    mgr.RemoveNode("node_b");

    {
        WriteDecision dec = mgr.ResolveWrite(ctx);
        EXPECT_TRUE(dec.hints.preferred_node_ids.empty());
    }
}

// ===================================================================
// Probe 9: Eviction total capacity calculation edge case
//
// When load_ratio is exactly 1.0, free_bytes should be 0. The formula:
//   total = free_bytes / max(1.0 - load_ratio, 0.01)
// With load_ratio=1.0, 1-1.0=0 → clamped to 0.01.
// free_bytes=0 → total=0/0.01=0.
// estimated -= evicted_bytes/0 → inf → always below low? Or NaN?
//
// Actually total=0, so evicted_bytes/total would be division by zero.
// But wait: 0/0.01 = 0. So total=0. Then evicted_bytes/0 = inf.
// estimated_load = 1.0 - inf = -inf → definitely <= low → not exceeded.
// But the node IS at 100% load. This seems like a bug.
// ===================================================================

TEST_F(AffinityProbeTest, EvictionAtFullCapacity) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "eviction": {
            "ops": [{"op": "node_water_level", "threshold": 0.85, "low": 0.70}]
        }
    })"));

    // load_ratio = 1.0, free_bytes = 0 → total capacity = 0/0.01 = 0
    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 0, 1.0, 0, 0, 1});

    AffinityResolveContext ctx;

    // Without evicted bytes: raw load 1.0 > threshold 0.85 → exceeded
    {
        auto ex = mgr.ResolveEviction(ctx);
        ASSERT_EQ(1u, ex.size());
    }

    // Report even 1 byte evicted: total=0, evicted_bytes/total = 1/0 = +inf
    // estimated = 1.0 - inf = -inf → -inf <= low(0.70) → skip
    // But the node still has load_ratio > high AND has eviction state...
    // Line 96: if (node.load_ratio > high || it != ctx.evicted_bytes.end())
    // Both conditions are true, but we skipped at line 93-95 (estimated <= low).
    // So the node is NOT reported exceeded. Is this correct?
    // The node has 100% load and we only evicted 1 byte.
    mgr.ReportEvictedBytes("node_a", 1);
    {
        auto ex = mgr.ResolveEviction(ctx);
        // With total=0 and any eviction, estimated_load becomes -inf.
        // The skip at `estimated_load <= low` kicks in.
        // This is arguably a bug: 1 byte evicted from a 100%-full node
        // shouldn't stop eviction.
        if (ex.empty()) {
            // Expected bug: node at 100% capacity with 1 byte evicted
            // incorrectly considered "below low watermark" due to
            // total_capacity=0 causing infinite eviction ratio.
            KVCM_LOG_WARN("KNOWN ISSUE: eviction at load_ratio=1.0 with "
                          "free_bytes=0 causes total_capacity=0, making any "
                          "eviction report cause -inf estimated load");
        }
        // Document: this test probes the behavior, not asserts correctness
    }
}

// ===================================================================
// Probe 10: Strategy memoization — same JSON shares parsed strategy
// ===================================================================

TEST_F(AffinityProbeTest, StrategyMemoizationSharesParsedInstance) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({"type":"noop"})"));

    mgr.UpsertNodeMetrics({"node_a", "node_a", DataStorageType{}, 500000, 0.30, 0, 0, 1});

    const char *json = R"({"type":"local_replica","write":{"ops":{"prefer_local":{"on_miss":"passthrough"}}}})";

    // Two resolves with identical instance_strategy_json
    AffinityResolveContext ctx1;
    ctx1.instance_strategy_json = json;
    ctx1.caller_node.node_id = "node_a";

    AffinityResolveContext ctx2;
    ctx2.instance_strategy_json = json;
    ctx2.caller_node.node_id = "node_a";

    WriteDecision d1 = mgr.ResolveWrite(ctx1);
    WriteDecision d2 = mgr.ResolveWrite(ctx2);

    // Both should produce same result (memoized strategy)
    ASSERT_FALSE(d1.hints.preferred_node_ids.empty());
    ASSERT_FALSE(d2.hints.preferred_node_ids.empty());
    EXPECT_EQ(d1.hints.preferred_node_ids[0], d2.hints.preferred_node_ids[0]);
}

} // namespace kv_cache_manager
