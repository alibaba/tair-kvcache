// Release acceptance contracts for known gaps. Explicit manual target: these
// assertions intentionally expose current missing behavior, never XFAIL it.
#include "kv_cache_manager/affinity/cache_affinity_manager.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
class AffinityPendingContractTest : public TESTBASE {};

TEST_F(AffinityPendingContractTest, PartialLocalBlockStillRequestsMissingComponents) {
    CacheAffinityManager manager;
    ASSERT_TRUE(manager.LoadProcessStrategyFromJsonString(R"({
        "type":"local_replica","read":{"on_miss":{"replication_hot_threshold":1}}
    })"));
    AffinityResolveContext ctx;
    ctx.instance_id = "multi_spec";
    ctx.caller_node.node_id = "reader";
    LocationSpec local_kv("kv", "tair://reader/kv", "reader");
    LocationSpec remote_state("state", "tair://writer/state", "writer");
    CacheLocation winner;
    winner.push_location_spec(local_kv);
    winner.push_location_spec(remote_state);
    ReadRequest req;
    req.block_key = 100;
    req.winner_tier = &winner;
    req.spec_candidates["kv"] = {&local_kv};
    req.spec_candidates["state"] = {&remote_state};
    auto decision = manager.ResolveRead(req, ctx);
    EXPECT_EQ(1u, decision.side_effects.size()) << "one local component is not a complete local block";
}

TEST_F(AffinityPendingContractTest, RemoteFrequencyDoesNotCrossInstances) {
    CacheAffinityManager manager;
    ASSERT_TRUE(manager.LoadProcessStrategyFromJsonString(R"({
        "type":"local_replica","read":{"on_miss":{
            "replication_hot_threshold":2,"suppression_window_ms":0}}
    })"));
    LocationSpec remote("tp0", "tair://writer/key", "writer");
    CacheLocation winner;
    winner.push_location_spec(remote);
    ReadRequest req;
    req.block_key = 101;
    req.winner_tier = &winner;
    req.spec_candidates["tp0"] = {&remote};
    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "reader";
    ctx.instance_id = "instance_a";
    EXPECT_TRUE(manager.ResolveRead(req, ctx).side_effects.empty());
    ctx.instance_id = "instance_b";
    EXPECT_TRUE(manager.ResolveRead(req, ctx).side_effects.empty())
        << "instance B has only one remote query, not two";
}

TEST_F(AffinityPendingContractTest, HintSuppressionDoesNotCrossInstances) {
    CacheAffinityManager manager;
    ASSERT_TRUE(manager.LoadProcessStrategyFromJsonString(R"({
        "type":"local_replica","read":{"on_miss":{
            "replication_hot_threshold":1,"suppression_window_ms":60000}}
    })"));
    LocationSpec remote("tp0", "tair://writer/key", "writer");
    CacheLocation winner;
    winner.push_location_spec(remote);
    ReadRequest req;
    req.block_key = 102;
    req.winner_tier = &winner;
    req.spec_candidates["tp0"] = {&remote};
    AffinityResolveContext ctx;
    ctx.caller_node.node_id = "reader";
    ctx.instance_id = "instance_a";
    EXPECT_EQ(1u, manager.ResolveRead(req, ctx).side_effects.size());
    ctx.instance_id = "instance_b";
    EXPECT_EQ(1u, manager.ResolveRead(req, ctx).side_effects.size())
        << "a hint for A must not suppress an independent replica for B";
}
} // namespace kv_cache_manager
