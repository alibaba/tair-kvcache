// Regression contracts for instance isolation and complete local replicas.
#include "kv_cache_manager/affinity/cache_affinity_manager.h"
#include "kv_cache_manager/affinity/local_replica_strategy.h"
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
    winner.push_location_spec(LocationSpec(local_kv));
    winner.push_location_spec(LocationSpec(remote_state));
    ReadRequest req;
    req.block_key = 100;
    req.winner_tier = &winner;
    req.spec_candidates["kv"] = {&local_kv};
    req.spec_candidates["state"] = {&remote_state};
    auto decision = manager.ResolveRead(req, ctx);
    ASSERT_EQ(1u, decision.side_effects.size()) << "one local component is not a complete local block";
    auto *hint = dynamic_cast<ReplicationHintSideEffect *>(decision.side_effects.front().get());
    ASSERT_NE(nullptr, hint);
    ASSERT_EQ(2u, hint->source_specs.size());
    EXPECT_EQ("kv", hint->source_specs[0].spec_name);
    EXPECT_EQ(local_kv.uri(), hint->source_specs[0].uri);
    EXPECT_EQ("state", hint->source_specs[1].spec_name);
    EXPECT_EQ(remote_state.uri(), hint->source_specs[1].uri);
    EXPECT_TRUE(hint->source_uri.empty());
    LocationSpec local_state("state", "tair://reader/state", "reader");
    req.spec_candidates["state"] = {&remote_state, &local_state};
    EXPECT_TRUE(manager.ResolveRead(req, ctx).side_effects.empty());
}

TEST_F(AffinityPendingContractTest, RemoteFrequencyDoesNotCrossInstances) {
    CacheAffinityManager manager;
    ASSERT_TRUE(manager.LoadProcessStrategyFromJsonString(R"({
        "type":"local_replica","read":{"on_miss":{
            "replication_hot_threshold":2,"suppression_window_ms":0}}
    })"));
    LocationSpec remote("tp0", "tair://writer/key", "writer");
    CacheLocation winner;
    winner.push_location_spec(LocationSpec(remote));
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
    winner.push_location_spec(LocationSpec(remote));
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
