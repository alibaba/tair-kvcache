#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/reclaim_indexer/node_lru_reclaim_indexer.h"

namespace kv_cache_manager {

class NodeLruReclaimIndexerTest : public TESTBASE {};

// Helper: build a CacheLocationMap with one location containing one spec.
namespace {
CacheLocationMap MakeLocMap(const std::string &loc_id, const std::string &spec_name, const std::string &node_id) {
    auto loc = std::make_shared<CacheLocation>();
    loc->set_id(loc_id);
    loc->push_location_spec(LocationSpec(spec_name, "uri://" + spec_name, node_id));
    return {{loc_id, loc}};
}
} // namespace

// --- Add ---

TEST_F(NodeLruReclaimIndexerTest, AddAndSampleBasic) {
    NodeLruReclaimIndexer indexer;
    KeyTypeVec keys = {100, 200};
    CacheLocationMapVector locs = {
        MakeLocMap("loc1", "spec0", "nodeA"),
        MakeLocMap("loc2", "spec0", "nodeA"),
    };
    auto results = indexer.Add(keys, locs, {EC_OK, EC_OK});
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), results);

    KeyTypeVec sampled;
    EXPECT_EQ(EC_OK, indexer.Sample(10, {"nodeA"}, sampled));
    EXPECT_EQ(2u, sampled.size());
}

TEST_F(NodeLruReclaimIndexerTest, AddSizeMismatchReturnsError) {
    NodeLruReclaimIndexer indexer;
    KeyTypeVec keys = {1, 2};
    CacheLocationMapVector locs = {MakeLocMap("loc1", "spec0", "nodeA")};
    auto results = indexer.Add(keys, locs, {EC_OK, EC_OK});
    EXPECT_EQ(EC_BADARGS, results[0]);
}

TEST_F(NodeLruReclaimIndexerTest, AddSkipsPriorErrors) {
    NodeLruReclaimIndexer indexer;
    KeyTypeVec keys = {10, 20};
    CacheLocationMapVector locs = {
        MakeLocMap("loc1", "spec0", "nodeA"),
        MakeLocMap("loc2", "spec0", "nodeA"),
    };
    // key 10 had prior error — should be skipped, key 20 should be added.
    auto results = indexer.Add(keys, locs, {EC_ERROR, EC_OK});
    EXPECT_EQ(EC_ERROR, results[0]);
    EXPECT_EQ(EC_OK, results[1]);

    KeyTypeVec sampled;
    indexer.Sample(10, {"nodeA"}, sampled);
    EXPECT_EQ(1u, sampled.size());
    EXPECT_EQ(20, sampled[0]);
}

TEST_F(NodeLruReclaimIndexerTest, AddMultipleNodes) {
    NodeLruReclaimIndexer indexer;
    KeyTypeVec keys = {10, 20};
    CacheLocationMapVector locs = {
        MakeLocMap("loc1", "spec0", "nodeA"),
        MakeLocMap("loc2", "spec0", "nodeB"),
    };
    auto results = indexer.Add(keys, locs, {EC_OK, EC_OK});
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), results);

    KeyTypeVec sampled_a, sampled_b;
    indexer.Sample(10, {"nodeA"}, sampled_a);
    indexer.Sample(10, {"nodeB"}, sampled_b);
    EXPECT_EQ(1u, sampled_a.size());
    EXPECT_EQ(10, sampled_a[0]);
    EXPECT_EQ(1u, sampled_b.size());
    EXPECT_EQ(20, sampled_b[0]);
}

// --- Touch ---

TEST_F(NodeLruReclaimIndexerTest, TouchPromotesKey) {
    NodeLruReclaimIndexer indexer;
    // Add 3 keys to same node. Oldest first: 1, 2, 3
    KeyTypeVec keys = {1, 2, 3};
    CacheLocationMapVector locs = {
        MakeLocMap("l1", "spec0", "nodeA"),
        MakeLocMap("l2", "spec0", "nodeA"),
        MakeLocMap("l3", "spec0", "nodeA"),
    };
    indexer.Add(keys, locs, {EC_OK, EC_OK, EC_OK});

    // Touch key 1 — it should move to hot end.
    auto results = indexer.Touch({1}, {EC_OK});
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), results);

    // Sample 1 — should get key 2 (now oldest).
    KeyTypeVec sampled;
    indexer.Sample(1, {"nodeA"}, sampled);
    EXPECT_EQ(1u, sampled.size());
    EXPECT_EQ(2, sampled[0]);
}

TEST_F(NodeLruReclaimIndexerTest, TouchSkipsPriorErrors) {
    NodeLruReclaimIndexer indexer;
    indexer.Add({1, 2}, {MakeLocMap("l1", "spec0", "nodeA"), MakeLocMap("l2", "spec0", "nodeA")}, {EC_OK, EC_OK});

    // Touch with key 1 having prior error — should propagate EC_ERROR, key 2 OK.
    auto results = indexer.Touch({1, 2}, {EC_ERROR, EC_OK});
    EXPECT_EQ(EC_ERROR, results[0]);
    EXPECT_EQ(EC_OK, results[1]);
}

// --- Remove (partial) ---

TEST_F(NodeLruReclaimIndexerTest, RemovePartialByLocation) {
    NodeLruReclaimIndexer indexer;
    // Key 1 has two locations on the same node.
    auto loc = std::make_shared<CacheLocation>();
    loc->set_id("locA");
    loc->push_location_spec(LocationSpec("sp0", "uri://sp0", "nodeA"));
    CacheLocationMap map1a = {{"locA", loc}};

    auto loc2 = std::make_shared<CacheLocation>();
    loc2->set_id("locB");
    loc2->push_location_spec(LocationSpec("sp0", "uri://sp0", "nodeA"));
    CacheLocationMap map1b = {{"locB", loc2}};

    // Add key=1 with locA, then again with locB.
    indexer.Add({1}, {map1a}, {EC_OK});
    indexer.Add({1}, {map1b}, {EC_OK});

    // Remove only locA — key should still be in nodeA's cache via locB.
    LocationIdsPerKey lids = {{"locA"}};
    auto results = indexer.Remove({1}, lids, {EC_OK});
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), results);

    KeyTypeVec sampled;
    indexer.Sample(10, {"nodeA"}, sampled);
    EXPECT_EQ(1u, sampled.size());
    EXPECT_EQ(1, sampled[0]);

    // Remove locB — now key should be gone.
    lids = {{"locB"}};
    indexer.Remove({1}, lids, {EC_OK});
    sampled.clear();
    indexer.Sample(10, {"nodeA"}, sampled);
    EXPECT_TRUE(sampled.empty());
}

TEST_F(NodeLruReclaimIndexerTest, RemovePartialSizeMismatch) {
    NodeLruReclaimIndexer indexer;
    auto results = indexer.Remove({1, 2}, {{"loc1"}}, {EC_OK, EC_OK});
    EXPECT_EQ(EC_BADARGS, results[0]);
}

// --- Remove (full) ---

TEST_F(NodeLruReclaimIndexerTest, RemoveFullKey) {
    NodeLruReclaimIndexer indexer;
    indexer.Add({42}, {MakeLocMap("loc1", "sp0", "nodeA")}, {EC_OK});

    KeyTypeVec sampled;
    indexer.Sample(10, {"nodeA"}, sampled);
    EXPECT_FALSE(sampled.empty());

    auto results = indexer.Remove({42}, {EC_OK});
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), results);

    indexer.Sample(10, {"nodeA"}, sampled);
    EXPECT_TRUE(sampled.empty());
}

// --- Sample ---

TEST_F(NodeLruReclaimIndexerTest, SampleUnknownNodeReturnsEmpty) {
    NodeLruReclaimIndexer indexer;
    indexer.Add({1}, {MakeLocMap("loc1", "sp0", "nodeA")}, {EC_OK});

    KeyTypeVec sampled;
    indexer.Sample(10, {"nodeX"}, sampled);
    EXPECT_TRUE(sampled.empty());
}

TEST_F(NodeLruReclaimIndexerTest, SampleDistributesAcrossNodes) {
    NodeLruReclaimIndexer indexer;
    // 3 keys on nodeA, 3 on nodeB.
    for (int i = 0; i < 3; ++i) {
        indexer.Add({i + 1}, {MakeLocMap("lA" + std::to_string(i), "sp0", "nodeA")}, {EC_OK});
        indexer.Add({i + 100}, {MakeLocMap("lB" + std::to_string(i), "sp0", "nodeB")}, {EC_OK});
    }

    // Sample 4 across both nodes — should get 2 per node.
    KeyTypeVec sampled;
    indexer.Sample(4, {"nodeA", "nodeB"}, sampled);
    EXPECT_EQ(4u, sampled.size());
    size_t sum_a = 0, sum_b = 0;
    for (const KeyType &key : sampled) {
        key < 100 ? sum_a++ : sum_b++;
    }
    EXPECT_EQ(2, sum_a);
    EXPECT_EQ(2, sum_b);
}

TEST_F(NodeLruReclaimIndexerTest, SampleZeroCount) {
    NodeLruReclaimIndexer indexer;
    indexer.Add({1}, {MakeLocMap("loc1", "sp0", "nodeA")}, {EC_OK});
    KeyTypeVec sampled;
    indexer.Sample(0, {"nodeA"}, sampled);
    EXPECT_TRUE(sampled.empty());
}

TEST_F(NodeLruReclaimIndexerTest, SampleEmptyNodes) {
    NodeLruReclaimIndexer indexer;
    indexer.Add({1}, {MakeLocMap("loc1", "sp0", "nodeA")}, {EC_OK});
    KeyTypeVec sampled;
    indexer.Sample(10, {}, sampled);
    EXPECT_TRUE(sampled.empty());
}

} // namespace kv_cache_manager
