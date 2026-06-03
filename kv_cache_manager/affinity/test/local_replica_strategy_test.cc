// affinity v1: LocalReplicaAffinityStrategy 3 个一级 method 行为单测
//
// ResolveWrite     —— 委托 v0 5 段流水线
// ResolveRead      —— PickLocalSpec + on_miss 复制触发（私有 helper 间接验证）
// ResolveEviction  —— 节点水位匹配

#include <memory>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/affinity/frequency_sketch.h"
#include "kv_cache_manager/affinity/local_replica_strategy.h"
#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/affinity/pipeline/candidate_pipeline.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class LocalReplicaStrategyTest : public TESTBASE {};

// === ResolveWrite ===

TEST_F(LocalReplicaStrategyTest, ResolveWriteNoPipelineReturnsEmpty) {
    LocalReplicaAffinityStrategy s;
    StrategyContext ctx;
    WriteDecision dec = s.ResolveWrite({"node_a", "node_b"}, ctx);
    EXPECT_EQ(AffinityStatus::kOk, dec.status);
    EXPECT_TRUE(dec.hints.preferred_node_ids.empty());
}

TEST_F(LocalReplicaStrategyTest, ResolveWriteDelegatesToV0Pipeline) {
    // 配 v0 prefer_local on_miss=passthrough：caller 在 candidates 内时排前。
    // 这是 v0 写时亲和（caller_node_id → preferred_node_ids）的完整继承，但
    // 通过 v1 新接口 strategy.ResolveWrite。
    auto v0 = CandidatePipeline::ParseJsonString(R"({"prefer_local":{"on_miss":"passthrough"}})", nullptr);
    ASSERT_NE(nullptr, v0);

    LocalReplicaAffinityStrategy::Params p;
    p.write_pipeline = std::move(v0);
    LocalReplicaAffinityStrategy s(std::move(p));

    StrategyContext ctx;
    ctx.caller_node_id = "node_b";
    WriteDecision dec = s.ResolveWrite({"node_a", "node_b", "node_c"}, ctx);
    ASSERT_EQ(AffinityStatus::kOk, dec.status);
    ASSERT_FALSE(dec.hints.preferred_node_ids.empty());
    EXPECT_EQ("node_b", dec.hints.preferred_node_ids.front());
}

TEST_F(LocalReplicaStrategyTest, ResolveWriteAbortPropagates) {
    auto v0 = CandidatePipeline::ParseJsonString(R"({"prefer_local":{"on_miss":"abort"}})", nullptr);
    ASSERT_NE(nullptr, v0);

    LocalReplicaAffinityStrategy::Params p;
    p.write_pipeline = std::move(v0);
    LocalReplicaAffinityStrategy s(std::move(p));

    StrategyContext ctx;
    ctx.caller_node_id = "node_X"; // 不在候选列表
    WriteDecision dec = s.ResolveWrite({"node_a", "node_b"}, ctx);
    EXPECT_EQ(AffinityStatus::kAbort, dec.status);
}

// === ResolveRead: pick spec ===

TEST_F(LocalReplicaStrategyTest, ResolveReadPicksLocalSpec) {
    LocalReplicaAffinityStrategy s;
    StrategyContext ctx;
    ctx.caller_node_id = "node_a";

    LocationSpec remote("tp0", "uri_b", "node_b");
    LocationSpec local("tp0", "uri_a", "node_a");

    ReadRequest req;
    req.block_key = 1;
    req.spec_candidates["tp0"] = {&remote, &local};
    CacheLocation winner; // 任意 winner_tier，仅 ShouldEmitReplicationHint 需要
    winner.set_type(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
    req.winner_tier = &winner;

    ReadDecision dec = s.ResolveRead(req, ctx);
    ASSERT_EQ(1u, dec.picked_specs.count("tp0"));
    ASSERT_NE(nullptr, dec.picked_specs["tp0"]);
    EXPECT_EQ("node_a", dec.picked_specs["tp0"]->node_id());
    // 已有本地 ⇒ 不发复制提示
    EXPECT_TRUE(dec.side_effects.empty());
}

TEST_F(LocalReplicaStrategyTest, ResolveReadFallsBackWhenNoLocal) {
    LocalReplicaAffinityStrategy s;
    StrategyContext ctx;
    ctx.caller_node_id = "node_x"; // 没有 caller 本地候选

    LocationSpec a("tp0", "u1", "node_a");
    LocationSpec b("tp0", "u2", "node_b");
    ReadRequest req;
    req.block_key = 1;
    req.spec_candidates["tp0"] = {&a, &b};
    CacheLocation winner;
    req.winner_tier = &winner;

    ReadDecision dec = s.ResolveRead(req, ctx);
    EXPECT_EQ(&a, dec.picked_specs["tp0"]); // 退化为首个
}

// === ResolveRead: on_miss 复制触发（4 gate 集成验证）===

TEST_F(LocalReplicaStrategyTest, ResolveReadEmitsReplicationHintWhenHot) {
    // 预先把 sketch 喂到阈值之上（sketch 现在构造期注入 Params）
    FrequencySketch sketch;
    // 模拟前几次远端读已经被累积
    sketch.Observe("node_a", 42);
    sketch.Observe("node_a", 42);
    sketch.Observe("node_a", 42);
    // 此时 RemoteCount = 3，ResolveRead 调一次会变 4，>= 阈值

    LocalReplicaAffinityStrategy::Params p;
    p.replication_hot_threshold = 3;
    p.sketch = &sketch;
    LocalReplicaAffinityStrategy s(std::move(p));

    StrategyContext ctx;
    ctx.caller_node_id = "node_a";

    LocationSpec remote("tp0", "uri_remote", "node_b");
    ReadRequest req;
    req.block_key = 42;
    req.spec_candidates["tp0"] = {&remote};
    CacheLocation winner;
    winner.push_location_spec(LocationSpec("tp0", "uri_remote", "node_b"));
    req.winner_tier = &winner;

    ReadDecision dec = s.ResolveRead(req, ctx);
    ASSERT_EQ(1u, dec.side_effects.size());
    auto *hint = dynamic_cast<ReplicationHint *>(dec.side_effects[0].get());
    ASSERT_NE(nullptr, hint);
    EXPECT_EQ(42, hint->block_key);
    EXPECT_EQ("node_a", hint->target_node_id);
    EXPECT_EQ("uri_remote", hint->source_uri);
    // 发完后 sketch 应被 Reset
    EXPECT_EQ(0u, sketch.RemoteCount("node_a", 42));
}

TEST_F(LocalReplicaStrategyTest, ResolveReadNoHintWhenOnMissDisabled) {
    FrequencySketch sketch;
    LocalReplicaAffinityStrategy::Params p;
    p.enable_on_miss = false; // 关闭 read on_miss 子项
    p.replication_hot_threshold = 0;
    p.sketch = &sketch;
    LocalReplicaAffinityStrategy s(std::move(p));

    StrategyContext ctx;
    ctx.caller_node_id = "node_a";

    LocationSpec remote("tp0", "uri_remote", "node_b");
    ReadRequest req;
    req.block_key = 99;
    req.spec_candidates["tp0"] = {&remote};
    CacheLocation winner;
    req.winner_tier = &winner;

    ReadDecision dec = s.ResolveRead(req, ctx);
    EXPECT_TRUE(dec.side_effects.empty());
    EXPECT_EQ(0u, sketch.RemoteCount("node_a", 99)); // 也不喂 sketch
}

TEST_F(LocalReplicaStrategyTest, ResolveReadNoHintWhenCallerLoadHigh) {
    FrequencySketch sketch;
    LocalReplicaAffinityStrategy::Params p;
    p.replication_hot_threshold = 0;
    p.caller_capacity_threshold = 0.85;
    p.caller_capacity_buffer = 0.05; // gate = 0.80
    p.sketch = &sketch;
    LocalReplicaAffinityStrategy s(std::move(p));

    NodeMetrics m;
    m.node_id = "node_a";
    m.load_ratio = 0.90; // 超 gate

    StrategyContext ctx;
    ctx.caller_node_id = "node_a";
    ctx.get_node_metrics = [&](const std::string &id) -> const NodeMetrics * { return id == "node_a" ? &m : nullptr; };

    LocationSpec remote("tp0", "uri_r", "node_b");
    ReadRequest req;
    req.block_key = 7;
    req.spec_candidates["tp0"] = {&remote};
    CacheLocation winner;
    req.winner_tier = &winner;

    ReadDecision dec = s.ResolveRead(req, ctx);
    EXPECT_TRUE(dec.side_effects.empty());
}

// === ResolveEviction ===

TEST_F(LocalReplicaStrategyTest, ResolveEvictionReturnsExceededNodeIds) {
    LocalReplicaAffinityStrategy::Params p;
    p.node_water_threshold = 0.85;
    p.node_water_low = 0.70;
    LocalReplicaAffinityStrategy s(std::move(p));

    NodeMetrics a, b;
    a.node_id = "node_a";
    a.load_ratio = 0.90;
    a.free_bytes = 100000;
    b.node_id = "node_b";
    b.load_ratio = 0.50;
    b.free_bytes = 500000;

    StrategyContext ctx;
    ctx.all_nodes = {a, b};

    auto result = s.ResolveEviction(ctx);
    ASSERT_EQ(1u, result.size());
    EXPECT_TRUE(result.count("node_a"));
}

TEST_F(LocalReplicaStrategyTest, ResolveEvictionReturnsEmptyWhenAllBelowThreshold) {
    LocalReplicaAffinityStrategy::Params p;
    p.node_water_threshold = 0.85;
    p.node_water_low = 0.70;
    LocalReplicaAffinityStrategy s(std::move(p));

    NodeMetrics b;
    b.node_id = "node_b";
    b.load_ratio = 0.50;
    b.free_bytes = 500000;

    StrategyContext ctx;
    ctx.all_nodes = {b};

    EXPECT_TRUE(s.ResolveEviction(ctx).empty());
}

TEST_F(LocalReplicaStrategyTest, ResolveEvictionReturnsEmptyWhenNoNodes) {
    LocalReplicaAffinityStrategy::Params p;
    p.node_water_threshold = 0.85;
    LocalReplicaAffinityStrategy s(std::move(p));

    StrategyContext ctx;
    EXPECT_TRUE(s.ResolveEviction(ctx).empty());
}

TEST_F(LocalReplicaStrategyTest, ResolveEvictionHysteresisReducesEstimatedLoad) {
    LocalReplicaAffinityStrategy::Params p;
    p.node_water_threshold = 0.85;
    p.node_water_low = 0.70;
    LocalReplicaAffinityStrategy s(std::move(p));

    NodeMetrics a;
    a.node_id = "node_a";
    a.load_ratio = 0.90;
    a.free_bytes = 100000; // total = 100000 / (1-0.9) = 1000000

    StrategyContext ctx;
    ctx.all_nodes = {a};
    // evicted 250000 bytes → estimated = 0.90 - 250000/1000000 = 0.65 < low=0.70
    ctx.evicted_bytes = {{"node_a", 250000}};

    EXPECT_TRUE(s.ResolveEviction(ctx).empty());
}

TEST_F(LocalReplicaStrategyTest, ResolveEvictionDisabledReturnsEmpty) {
    LocalReplicaAffinityStrategy::Params p;
    p.enable_eviction = false;
    p.node_water_threshold = 0.85;
    LocalReplicaAffinityStrategy s(std::move(p));

    NodeMetrics a;
    a.node_id = "node_a";
    a.load_ratio = 0.95;
    a.free_bytes = 50000;

    StrategyContext ctx;
    ctx.all_nodes = {a};

    EXPECT_TRUE(s.ResolveEviction(ctx).empty());
}

} // namespace kv_cache_manager
