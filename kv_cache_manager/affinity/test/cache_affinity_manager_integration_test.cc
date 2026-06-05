// End-to-end integration test for CacheAffinityManager facade.
//
// These tests exercise the full adaptive hot-key retention loop described in
// the design doc (CLAUDE.md §四):
//
//   write(remote) → read(remote miss) → frequency accumulates → ReplicationHint
//   → local replica created → read(local hit, no hint) → node overloaded
//   → eviction identifies node → evicted bytes reduce estimated load
//
// Each test loads a realistic JSON strategy config, injects NodeMetrics, and
// drives the manager through ResolveWrite / ResolveRead / ResolveEviction to
// verify the decisions match expected behavior.
//
// NOTE: CacheAffinityManager is heap-allocated via NewManager() to ensure each
// test gets a unique address. MakeNodeMetricsAccessor uses a thread-local cache
// keyed by `this`; stack-local managers across TEST_F share the same address,
// causing stale metrics from the previous test to leak through.

#include <fstream>
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
// Keep managers alive across tests to prevent heap address reuse.
std::vector<std::unique_ptr<CacheAffinityManager>> g_mgr_pool;
CacheAffinityManager &NewManager() {
    g_mgr_pool.push_back(std::make_unique<CacheAffinityManager>());
    return *g_mgr_pool.back();
}
} // namespace

class CacheAffinityManagerIntegrationTest : public TESTBASE {};

// Full local_replica JSON config used across most tests. Enables all three
// aspects (write/read/eviction) with the 5-stage pipeline, on_miss replication,
// and node water level eviction.
static const char *kFullStrategyJson = R"({
    "type": "local_replica",
    "write": {
        "ops": {
            "filter": {
                "metric": "load_ratio", "max": 0.90
            },
            "prefer_local": { "on_miss": "passthrough" },
            "sort": [{ "metric": "free_bytes", "weight": 1.0 }],
            "limit": 2
        }
    },
    "read": {
        "on_miss": {
            "enabled": true,
            "replication_hot_threshold": 3,
            "caller_capacity_threshold": 0.85,
            "caller_capacity_buffer": 0.05
        }
    },
    "eviction": {
        "ops": [
            { "op": "node_water_level", "threshold": 0.85, "critical": 0.95, "low": 0.70 }
        ]
    }
})";

// Helper: build a ReadRequest with a single spec on a given node.
static ReadRequest MakeRemoteReadRequest(int64_t block_key, LocationSpec *spec, CacheLocation *winner) {
    ReadRequest req;
    req.block_key = block_key;
    req.spec_candidates["tp0"] = {spec};
    req.winner_tier = winner;
    return req;
}

// ---------------------------------------------------------------------------
// Test 1: Full adaptive loop — the core design scenario
//
// Simulates: key written to remote → repeated reads → replication hint →
// local replica available → reads converge to local → no more hints
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, AdaptiveHotKeyRetentionLoop) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    // Cluster: 3 nodes, caller is on node_a
    mgr.UpsertNodeMetrics({"node_a", "node_a", 400000, 0.60, 10, 10, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 800000, 0.20, 5, 5, 1});
    mgr.UpsertNodeMetrics({"node_c", "node_c", 600000, 0.30, 5, 5, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.instance_id = "inst-001";
    ctx.trace_id = "adaptive-loop";

    // ---- Phase 1: Write path prefers local node ----
    {
        WriteDecision dec = mgr.ResolveWrite(ctx);
        ASSERT_EQ(AffinityStatus::kOk, dec.status);
        ASSERT_FALSE(dec.hints.preferred_node_ids.empty());
        EXPECT_EQ("node_a", dec.hints.preferred_node_ids[0]);
    }

    // ---- Phase 2: Data ended up on remote node_b (older key / local-full) ----
    // Reader on node_a repeatedly reads key 42 — only remote spec available.

    LocationSpec remote_spec("tp0", "tair://node_b/block/42", "node_b");
    CacheLocation winner;
    winner.set_type(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
    winner.push_location_spec(LocationSpec("tp0", "tair://node_b/block/42", "node_b"));

    // Reads 1-2: sketch accumulates, below threshold 3 → no hint
    for (int i = 0; i < 2; ++i) {
        auto req = MakeRemoteReadRequest(42, &remote_spec, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        ASSERT_EQ(1u, dec.picked_specs.count("tp0"));
        EXPECT_EQ("node_b", dec.picked_specs["tp0"]->node_id());
        EXPECT_TRUE(dec.side_effects.empty()) << "read #" << (i + 1) << ": count < threshold, no hint expected";
    }

    // Read 3: sketch count reaches threshold → ReplicationHint emitted
    {
        auto req = MakeRemoteReadRequest(42, &remote_spec, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        ASSERT_EQ(1u, dec.side_effects.size());
        auto *hint = dynamic_cast<ReplicationHintSideEffect *>(dec.side_effects[0].get());
        ASSERT_NE(nullptr, hint);
        EXPECT_EQ(42, hint->block_key);
        EXPECT_EQ("node_a", hint->target_node_id);
        EXPECT_EQ("tair://node_b/block/42", hint->source_uri);
    }

    // ---- Phase 3: Client acts on hint — local replica now exists ----
    LocationSpec local_spec("tp0", "tair://node_a/block/42", "node_a");

    {
        ReadRequest req;
        req.block_key = 42;
        req.spec_candidates["tp0"] = {&remote_spec, &local_spec};
        req.winner_tier = &winner;

        ReadDecision dec = mgr.ResolveRead(req, ctx);
        ASSERT_EQ(1u, dec.picked_specs.count("tp0"));
        EXPECT_EQ("node_a", dec.picked_specs["tp0"]->node_id());
        EXPECT_TRUE(dec.side_effects.empty());
    }

    // ---- Phase 4: Subsequent reads all go to local, no hints ----
    for (int i = 0; i < 5; ++i) {
        ReadRequest req;
        req.block_key = 42;
        req.spec_candidates["tp0"] = {&remote_spec, &local_spec};
        req.winner_tier = &winner;

        ReadDecision dec = mgr.ResolveRead(req, ctx);
        EXPECT_EQ("node_a", dec.picked_specs["tp0"]->node_id());
        EXPECT_TRUE(dec.side_effects.empty()) << "stable local read #" << (i + 1) << " should never emit hint";
    }
}

// ---------------------------------------------------------------------------
// Test 2: Write pipeline — filter + sort + limit with real metrics
//
// Validates that the 5-stage pipeline filters out overloaded nodes,
// sorts survivors by free_bytes, and respects the limit.
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, WritePipelineFilterSortLimit) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    // node_a: load 0.95 → filtered out by "load_ratio max 0.90"
    // node_b: load 0.40, free 200000
    // node_c: load 0.30, free 800000
    // node_d: load 0.50, free 500000
    mgr.UpsertNodeMetrics({"node_a", "node_a", 50000, 0.95, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 200000, 0.40, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_c", "node_c", 800000, 0.30, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_d", "node_d", 500000, 0.50, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a"; // caller is on the overloaded node
    ctx.trace_id = "pipeline-test";

    WriteDecision dec = mgr.ResolveWrite(ctx);
    ASSERT_EQ(AffinityStatus::kOk, dec.status);

    // node_a filtered out (load > 0.90)
    // prefer_local: node_a not in survivors → passthrough
    // sort by free_bytes desc: node_c(800k) > node_d(500k) > node_b(200k)
    // limit 2: [node_c, node_d]
    ASSERT_EQ(2u, dec.hints.preferred_node_ids.size());
    EXPECT_EQ("node_c", dec.hints.preferred_node_ids[0]);
    EXPECT_EQ("node_d", dec.hints.preferred_node_ids[1]);
}

// ---------------------------------------------------------------------------
// Test 3: Write pipeline — prefer_local interacts with filter
//
// Caller's node passes filter → prefer_local keeps only the local match.
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, WritePipelinePreferLocalWithFilter) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    // node_a (caller): load 0.50, passes filter (max 0.90)
    // node_b: load 0.30, highest free_bytes
    mgr.UpsertNodeMetrics({"node_a", "node_a", 300000, 0.50, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 800000, 0.30, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "prefer-local-filter";

    WriteDecision dec = mgr.ResolveWrite(ctx);
    ASSERT_EQ(AffinityStatus::kOk, dec.status);
    ASSERT_FALSE(dec.hints.preferred_node_ids.empty());
    // prefer_local finds node_a in survivors → result narrows to [node_a]
    EXPECT_EQ("node_a", dec.hints.preferred_node_ids[0]);
}

// ---------------------------------------------------------------------------
// Test 4: Read on_miss — caller capacity gate blocks hint
//
// Caller node is too loaded → replication hint suppressed even though
// frequency exceeds threshold. This prevents piling replicas onto an
// already-overloaded node.
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, ReadOnMissCallerCapacityGateBlocksHint) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    // caller node_a: load 0.82 > threshold(0.85) - buffer(0.05) = 0.80
    mgr.UpsertNodeMetrics({"node_a", "node_a", 180000, 0.82, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 800000, 0.20, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "capacity-gate";

    LocationSpec remote("tp0", "tair://node_b/block/99", "node_b");
    CacheLocation winner;
    winner.push_location_spec(LocationSpec("tp0", "tair://node_b/block/99", "node_b"));

    // Read 10 times — frequency well above threshold (3), but caller too loaded.
    for (int i = 0; i < 10; ++i) {
        auto req = MakeRemoteReadRequest(99, &remote, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        EXPECT_TRUE(dec.side_effects.empty())
            << "read #" << (i + 1) << ": caller load 0.82 > gate 0.80, hint suppressed";
    }
}

// ---------------------------------------------------------------------------
// Test 5: Eviction loop with hysteresis
//
// Node exceeds threshold → eviction reports node → evicted bytes accumulate
// → estimated load drops below low watermark → eviction stops.
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, EvictionLoopWithHysteresis) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    // node_a: load 0.90 (> threshold 0.85), free 100000 → total ~1000000
    // node_b: load 0.50 (healthy)
    mgr.UpsertNodeMetrics({"node_a", "node_a", 100000, 0.90, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 500000, 0.50, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.trace_id = "eviction-loop";

    // Round 1: node_a exceeded
    {
        auto exceeded = mgr.ResolveEviction(ctx);
        ASSERT_EQ(1u, exceeded.size());
        EXPECT_TRUE(exceeded.count("node_a"));
        EXPECT_FALSE(exceeded.count("node_b"));
    }

    // Simulate reclaimer evicting 100KB
    mgr.ReportEvictedBytes("node_a", 100000);

    // Round 2: estimated = 0.90 - 100000/1000000 = 0.80, still > low(0.70)
    {
        auto exceeded = mgr.ResolveEviction(ctx);
        EXPECT_EQ(1u, exceeded.size());
        EXPECT_TRUE(exceeded.count("node_a"));
    }

    // Simulate more eviction: total 300KB evicted
    mgr.ReportEvictedBytes("node_a", 200000);

    // Round 3: estimated = 0.90 - 300000/1000000 = 0.60, below low(0.70)
    {
        auto exceeded = mgr.ResolveEviction(ctx);
        EXPECT_TRUE(exceeded.empty());
    }
}

// ---------------------------------------------------------------------------
// Test 6: Multi-caller convergence (design doc Case A)
//
// Two callers on different nodes read the same hot key from remote.
// Both should independently trigger replication hints to their own nodes.
// After convergence each caller reads from its own local replica.
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, MultiCallerBothGetReplicationHints) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    mgr.UpsertNodeMetrics({"node_a", "node_a", 500000, 0.30, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 500000, 0.30, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_c", "node_c", 800000, 0.20, 0, 0, 1});

    LocationSpec remote("tp0", "tair://node_c/block/7", "node_c");
    CacheLocation winner;
    winner.push_location_spec(LocationSpec("tp0", "tair://node_c/block/7", "node_c"));

    // Caller A reads 3 times → hint targets node_a
    {
        AffinityResolveContext ctx_a;
        ctx_a.caller_node.node_id = "node_a";
        ctx_a.trace_id = "multi-caller-a";

        for (int i = 0; i < 2; ++i) {
            auto req = MakeRemoteReadRequest(7, &remote, &winner);
            ReadDecision dec = mgr.ResolveRead(req, ctx_a);
            EXPECT_TRUE(dec.side_effects.empty());
        }
        auto req = MakeRemoteReadRequest(7, &remote, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx_a);
        ASSERT_EQ(1u, dec.side_effects.size());
        auto *hint = dynamic_cast<ReplicationHintSideEffect *>(dec.side_effects[0].get());
        ASSERT_NE(nullptr, hint);
        EXPECT_EQ("node_a", hint->target_node_id);
    }

    // Caller B reads 3 times → independent sketch → hint targets node_b
    {
        AffinityResolveContext ctx_b;
        ctx_b.caller_node.node_id = "node_b";
        ctx_b.trace_id = "multi-caller-b";

        for (int i = 0; i < 2; ++i) {
            auto req = MakeRemoteReadRequest(7, &remote, &winner);
            ReadDecision dec = mgr.ResolveRead(req, ctx_b);
            EXPECT_TRUE(dec.side_effects.empty());
        }
        auto req = MakeRemoteReadRequest(7, &remote, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx_b);
        ASSERT_EQ(1u, dec.side_effects.size());
        auto *hint = dynamic_cast<ReplicationHintSideEffect *>(dec.side_effects[0].get());
        ASSERT_NE(nullptr, hint);
        EXPECT_EQ("node_b", hint->target_node_id);
    }
}

TEST_F(CacheAffinityManagerIntegrationTest, HintDedupSuppressionWindow) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "read": {
            "on_miss": {
                "enabled": true,
                "replication_hot_threshold": 3,
                "suppression_window_ms": 60000
            }
        }
    })"));

    mgr.UpsertNodeMetrics({"node_a", "node_a", 500000, 0.30, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 800000, 0.20, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "dedup-window";

    LocationSpec remote("tp0", "tair://node_b/block/55", "node_b");
    CacheLocation winner;
    winner.push_location_spec(LocationSpec("tp0", "tair://node_b/block/55", "node_b"));

    for (int i = 0; i < 2; ++i) {
        auto req = MakeRemoteReadRequest(55, &remote, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        EXPECT_TRUE(dec.side_effects.empty());
    }
    {
        auto req = MakeRemoteReadRequest(55, &remote, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        ASSERT_EQ(1u, dec.side_effects.size()) << "3rd read should trigger first hint";
    }

    for (int i = 0; i < 10; ++i) {
        auto req = MakeRemoteReadRequest(55, &remote, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        EXPECT_TRUE(dec.side_effects.empty()) << "read #" << (i + 4) << " within suppression window must not emit hint";
    }

    LocationSpec remote56("tp0", "tair://node_b/block/56", "node_b");
    CacheLocation winner56;
    winner56.push_location_spec(LocationSpec("tp0", "tair://node_b/block/56", "node_b"));
    for (int i = 0; i < 2; ++i) {
        auto req = MakeRemoteReadRequest(56, &remote56, &winner56);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        EXPECT_TRUE(dec.side_effects.empty());
    }
    {
        auto req = MakeRemoteReadRequest(56, &remote56, &winner56);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        ASSERT_EQ(1u, dec.side_effects.size()) << "different block key: independent suppressor entry";
    }
}

// ---------------------------------------------------------------------------
// Test 8: 3-tier priority with realistic configs
//
// Process: full local_replica (threshold=3)
// Group: overrides only read.on_miss threshold to 100
// Instance: overrides to noop (disables all affinity)
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, ThreeTierPriorityRealisticOverrides) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    mgr.UpsertNodeMetrics({"node_a", "node_a", 500000, 0.30, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 800000, 0.20, 0, 0, 1});

    LocationSpec remote("tp0", "tair://node_b/block/10", "node_b");
    CacheLocation winner;
    winner.push_location_spec(LocationSpec("tp0", "tair://node_b/block/10", "node_b"));

    // ---- Tier: process (threshold=3) — 3 reads → hint ----
    {
        AffinityResolveContext ctx;
        ctx.caller_node.node_id = "node_a";
        ctx.trace_id = "tier-process";

        for (int i = 0; i < 2; ++i) {
            auto req = MakeRemoteReadRequest(10, &remote, &winner);
            ReadDecision dec = mgr.ResolveRead(req, ctx);
            EXPECT_TRUE(dec.side_effects.empty());
        }
        auto req = MakeRemoteReadRequest(10, &remote, &winner);
        ReadDecision dec = mgr.ResolveRead(req, ctx);
        ASSERT_EQ(1u, dec.side_effects.size()) << "process threshold=3, 3rd read → hint";
    }

    // ---- Tier: group overrides threshold to 100 ----
    {
        AffinityResolveContext ctx;
        ctx.caller_node.node_id = "node_a";
        ctx.trace_id = "tier-group";
        ctx.group_strategy_json = R"({
            "type": "local_replica",
            "read": { "on_miss": { "enabled": true, "replication_hot_threshold": 100 } }
        })";

        // 10 reads: count reaches 10, well below threshold 100 → no hint
        for (int i = 0; i < 10; ++i) {
            auto req = MakeRemoteReadRequest(200, &remote, &winner);
            ReadDecision dec = mgr.ResolveRead(req, ctx);
            EXPECT_TRUE(dec.side_effects.empty()) << "group threshold=100, read #" << (i + 1) << " should not hint";
        }
    }

    // ---- Tier: instance overrides to noop ----
    {
        AffinityResolveContext ctx;
        ctx.caller_node.node_id = "node_a";
        ctx.trace_id = "tier-instance";
        ctx.instance_strategy_json = R"({"type":"noop"})";
        ctx.group_strategy_json = kFullStrategyJson;

        WriteDecision wdec = mgr.ResolveWrite(ctx);
        EXPECT_TRUE(wdec.hints.preferred_node_ids.empty()) << "noop: no affinity";

        auto req = MakeRemoteReadRequest(300, &remote, &winner);
        ReadDecision rdec = mgr.ResolveRead(req, ctx);
        EXPECT_TRUE(rdec.picked_specs.empty()) << "noop: no spec picking";
        EXPECT_TRUE(rdec.side_effects.empty()) << "noop: no side effects";
    }
}

// ---------------------------------------------------------------------------
// Test 9: Globally disabled — kill switch mid-flight
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, GlobalKillSwitchMidFlight) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    mgr.UpsertNodeMetrics({"node_a", "node_a", 100000, 0.90, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 500000, 0.50, 0, 0, 1});

    // Before: eviction identifies node_a
    {
        AffinityResolveContext ctx;
        auto exceeded = mgr.ResolveEviction(ctx);
        EXPECT_EQ(1u, exceeded.size());
    }

    mgr.SetGloballyDisabled(true);

    // After: everything noop
    {
        AffinityResolveContext ctx;
        ctx.caller_node.node_id = "node_a";

        WriteDecision wdec = mgr.ResolveWrite(ctx);
        EXPECT_TRUE(wdec.hints.preferred_node_ids.empty());

        auto exceeded = mgr.ResolveEviction(ctx);
        EXPECT_TRUE(exceeded.empty());
    }

    mgr.SetGloballyDisabled(false);

    // Recovery: decisions resume
    {
        AffinityResolveContext ctx;
        ctx.caller_node.node_id = "node_a";

        WriteDecision wdec = mgr.ResolveWrite(ctx);
        ASSERT_FALSE(wdec.hints.preferred_node_ids.empty());
    }
}

// ---------------------------------------------------------------------------
// Test 10: JSON file loading — full strategy from file with envelope
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, LoadFullStrategyFromFile) {
    auto &mgr = NewManager();
    std::string path = GetPrivateTestRuntimeDataPath() + "strategy.json";
    {
        std::ofstream f(path);
        ASSERT_TRUE(f.is_open());
        f << R"({"strategy": )" << kFullStrategyJson << "}";
    }

    std::string err;
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonFile(path, &err)) << err;

    mgr.UpsertNodeMetrics({"node_a", "node_a", 600000, 0.40, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 200000, 0.30, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "file-load";

    WriteDecision dec = mgr.ResolveWrite(ctx);
    ASSERT_EQ(AffinityStatus::kOk, dec.status);
    ASSERT_FALSE(dec.hints.preferred_node_ids.empty());
    EXPECT_EQ("node_a", dec.hints.preferred_node_ids[0]);
}

// ---------------------------------------------------------------------------
// Test 11: No nodes registered — graceful degradation
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, NoNodesGracefulDegradation) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(kFullStrategyJson));

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "no-nodes";

    WriteDecision wdec = mgr.ResolveWrite(ctx);
    EXPECT_EQ(AffinityStatus::kOk, wdec.status);
    EXPECT_TRUE(wdec.hints.preferred_node_ids.empty());

    auto exceeded = mgr.ResolveEviction(ctx);
    EXPECT_TRUE(exceeded.empty());
}

// ---------------------------------------------------------------------------
// Test 12: Per-aspect toggles — write disabled, read/eviction active
//
// Verifies that enabled_aspects.write=false suppresses write hints while
// read on_miss and eviction continue to function.
// ---------------------------------------------------------------------------

TEST_F(CacheAffinityManagerIntegrationTest, WriteAspectDisabledReadEvictionOn) {
    auto &mgr = NewManager();
    ASSERT_TRUE(mgr.LoadProcessStrategyFromJsonString(R"({
        "type": "local_replica",
        "enabled_aspects": { "write": false, "read": true, "eviction": true },
        "read": { "on_miss": { "enabled": true, "replication_hot_threshold": 1 } },
        "eviction": { "ops": [{"op": "node_water_level", "threshold": 0.85}] }
    })"));

    // node_a (caller): low load so capacity gate passes
    // node_b: high load, eviction target
    mgr.UpsertNodeMetrics({"node_a", "node_a", 500000, 0.30, 0, 0, 1});
    mgr.UpsertNodeMetrics({"node_b", "node_b", 100000, 0.90, 0, 0, 1});

    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "node_a";
    ctx.trace_id = "toggle-test";

    // Write disabled: empty hints even though nodes are present
    WriteDecision wdec = mgr.ResolveWrite(ctx);
    EXPECT_TRUE(wdec.hints.preferred_node_ids.empty());

    // Read still works: threshold=1, first remote read → hint
    LocationSpec remote("tp0", "tair://node_b/x", "node_b");
    CacheLocation winner;
    winner.push_location_spec(LocationSpec("tp0", "tair://node_b/x", "node_b"));
    auto req = MakeRemoteReadRequest(1, &remote, &winner);
    ReadDecision rdec = mgr.ResolveRead(req, ctx);
    ASSERT_EQ(1u, rdec.side_effects.size());

    // Eviction still works: node_b exceeded (load 0.90 > threshold 0.85)
    auto exceeded = mgr.ResolveEviction(ctx);
    EXPECT_EQ(1u, exceeded.size());
    EXPECT_TRUE(exceeded.count("node_b"));
}

} // namespace kv_cache_manager
