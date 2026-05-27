#include <memory>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/lru.h"
#include "kv_cache_manager/optimizer/index/radix_tree_index.h"

using namespace kv_cache_manager;

class RadixTreeIndexTest : public TESTBASE {
public:
    void SetUp() override {
        LruParams params;
        params.sample_rate = 1.0;
        auto policy = std::make_shared<LruEvictionPolicy>("test_lru", params);
        index_ = std::make_shared<RadixTreeIndex>("test_instance", policy);
    }

protected:
    std::shared_ptr<RadixTreeIndex> index_;
};

TEST_F(RadixTreeIndexTest, BasicInitialization) { EXPECT_NE(index_, nullptr); }

TEST_F(RadixTreeIndexTest, InsertOnly) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    auto result = index_->InsertOnly(block_keys, 1000);
    EXPECT_EQ(result.inserted_keys.size(), 5);
}

TEST_F(RadixTreeIndexTest, InsertOnlyDuplicate) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 再次插入相同的块
    auto result = index_->InsertOnly(block_keys, 2000);
    EXPECT_EQ(result.inserted_keys.size(), 0);
}

TEST_F(RadixTreeIndexTest, PrefixQueryNoHit) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 查询不同的序列
    std::vector<int64_t> query_keys = {10, 11, 12};
    BlockMask block_mask = std::vector<bool>(query_keys.size(), true);

    index_->PrefixQuery(query_keys, block_mask, 2000);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, PrefixQueryWithHit) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 查询相同的前缀
    std::vector<int64_t> query_keys = {1, 2, 3};
    BlockMask block_mask = std::vector<bool>(query_keys.size(), true);

    index_->PrefixQuery(query_keys, block_mask, 2000);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, PrefixQueryPartialMask) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 查询时mask部分块
    std::vector<int64_t> query_keys = {1, 2, 3};
    BlockMask block_mask = std::vector<bool>{true, false, true}; // 只查询第1和第3个块

    index_->PrefixQuery(query_keys, block_mask, 2000);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, PrefixQueryCountsMixedLocalRemoteNodeOnce) {
    std::vector<int64_t> block_keys = {1, 2, 3};
    index_->InsertOnly(block_keys, 1000);

    BlockMask block_mask = std::vector<bool>{true, false, true};
    QueryHit query_hit;
    index_->PrefixQuery(block_keys, block_mask, 2000, &query_hit);

    const auto *root = index_->GetRoot();
    ASSERT_NE(root, nullptr);
    auto child_it = root->children.find(1);
    ASSERT_NE(child_it, root->children.end());

    EXPECT_EQ(child_it->second->stat.access_count, 1);
    EXPECT_EQ(child_it->second->stat.last_access_time, 2000);
    EXPECT_EQ(query_hit.local_hit_block_num, 2);
    EXPECT_EQ(query_hit.remote_hit_block_num, 1);
    ASSERT_EQ(query_hit.per_tier_hit_block_num.size(), 1);
    EXPECT_EQ(query_hit.per_tier_hit_block_num[0], 3);
}

TEST_F(RadixTreeIndexTest, PromoteCopiesThroughIntermediateHigherTiers) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::CASCADING);
    index->set_enable_promote(true);
    index->InsertOnly({10}, 1000);

    auto *block = index->GetRoot()->children.at(10)->blocks[0].get();
    block->location_map.clear();
    AppendBlockLocation(block, "l3", 1000);

    QueryHit query_hit;
    BlockMask block_mask = std::vector<bool>{false};
    const bool read_triggered_tier_write = index->PrefixQuery({10}, block_mask, 2000, &query_hit);

    EXPECT_EQ(block->location_map.count("l1"), 1);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.count("l3"), 1);
    EXPECT_TRUE(read_triggered_tier_write);
}

TEST_F(RadixTreeIndexTest, PromoteDoesNotCopyToLowerTiers) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::CASCADING);
    index->set_enable_promote(true);
    index->InsertOnly({20}, 1000);

    auto *block = index->GetRoot()->children.at(20)->blocks[0].get();
    block->location_map.clear();
    AppendBlockLocation(block, "l2", 1000);

    QueryHit query_hit;
    BlockMask block_mask = std::vector<bool>{false};
    const bool read_triggered_tier_write = index->PrefixQuery({20}, block_mask, 2000, &query_hit);

    EXPECT_EQ(block->location_map.count("l1"), 1);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.count("l3"), 0);
    EXPECT_TRUE(read_triggered_tier_write);
}

TEST_F(RadixTreeIndexTest, WriteThroughPropagatesAccessToLowerTierByDefault) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
    };
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH);
    index->InsertOnly({30}, 1000);

    auto *block = index->GetRoot()->children.at(30)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    ASSERT_EQ(block->location_map.count("l2"), 1);

    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({30}, block_mask, 2000);

    EXPECT_EQ(block->location_map.at("l1").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 2000);
}

TEST_F(RadixTreeIndexTest, WriteThroughCanDisableAccessPropagationToLowerTier) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
    };
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH, 0, 2, false);
    index->InsertOnly({40}, 1000);

    auto *block = index->GetRoot()->children.at(40)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    ASSERT_EQ(block->location_map.count("l2"), 1);

    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({40}, block_mask, 2000);

    EXPECT_EQ(block->location_map.at("l1").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 1000);
}

TEST_F(RadixTreeIndexTest, CascadingCanDisableAccessPropagationToLowerTier) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
    };
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::CASCADING, 0, 2, false);
    index->InsertOnly({45}, 1000);

    auto *block = index->GetRoot()->children.at(45)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    AppendBlockLocation(block, "l2", 1000);
    ASSERT_EQ(block->location_map.count("l2"), 1);

    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({45}, block_mask, 2000);

    EXPECT_EQ(block->location_map.at("l1").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 1000);
}

TEST_F(RadixTreeIndexTest, SelectiveWriteToNextTierAfterThreshold) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
    };
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH_SELECTIVE, 0, 2, true);
    index->InsertOnly({46}, 1000);

    auto *block = index->GetRoot()->children.at(46)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    ASSERT_EQ(block->location_map.count("l2"), 0);

    BlockMask block_mask = std::vector<bool>{false};
    EXPECT_FALSE(index->PrefixQuery({46}, block_mask, 2000));
    EXPECT_EQ(block->location_map.count("l2"), 0);

    const bool read_triggered_tier_write = index->PrefixQuery({46}, block_mask, 3000);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_TRUE(read_triggered_tier_write);
}

TEST_F(RadixTreeIndexTest, TierFlowsControlInitialWritePerEdge) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    std::vector<TierFlowStrategy> flows(2);
    flows[0].write_mode = TierWriteMode::WRITE_THROUGH;
    flows[1].write_mode = TierWriteMode::CASCADING;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH, 0, 2, true, flows);
    index->InsertOnly({50}, 1000);

    auto *block = index->GetRoot()->children.at(50)->blocks[0].get();
    EXPECT_EQ(block->location_map.count("l1"), 1);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.count("l3"), 0);
}

TEST_F(RadixTreeIndexTest, TierFlowsStopAccessPropagationAtDisabledEdge) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    std::vector<TierFlowStrategy> flows(2);
    flows[0].write_mode = TierWriteMode::WRITE_THROUGH;
    flows[1].write_mode = TierWriteMode::WRITE_THROUGH;
    flows[1].access_propagation_enabled = false;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH, 0, 2, true, flows);
    index->InsertOnly({60}, 1000);

    auto *block = index->GetRoot()->children.at(60)->blocks[0].get();
    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({60}, block_mask, 2000);

    EXPECT_EQ(block->location_map.at("l1").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l3").last_access_time, 1000);
}

TEST_F(RadixTreeIndexTest, TierFlowsPromoteOnlyAcrossEnabledEdges) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    std::vector<TierFlowStrategy> flows(2);
    flows[0].write_mode = TierWriteMode::CASCADING;
    flows[0].promote_enabled = false;
    flows[1].write_mode = TierWriteMode::CASCADING;
    flows[1].promote_enabled = true;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::CASCADING, 0, 2, true, flows);
    index->InsertOnly({70}, 1000);

    auto *block = index->GetRoot()->children.at(70)->blocks[0].get();
    block->location_map.clear();
    AppendBlockLocation(block, "l3", 1000);

    QueryHit query_hit;
    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({70}, block_mask, 2000, &query_hit);

    EXPECT_EQ(block->location_map.count("l1"), 0);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.count("l3"), 1);
}

TEST_F(RadixTreeIndexTest, MultipleInsertions) {
    std::vector<int64_t> block_keys1 = {1, 2, 3};
    index_->InsertOnly(block_keys1, 1000);

    std::vector<int64_t> block_keys2 = {4, 5, 6};
    index_->InsertOnly(block_keys2, 2000);

    std::vector<int64_t> block_keys3 = {7, 8, 9};
    index_->InsertOnly(block_keys3, 3000);

    // 查询第一个序列
    std::vector<int64_t> query_keys = {1, 2, 3};
    BlockMask block_mask = std::vector<bool>(query_keys.size(), true);

    index_->PrefixQuery(query_keys, block_mask, 4000);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, CleanEmptyBlocks) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 创建空的BlockEntry指针列表
    std::vector<BlockEntry *> empty_blocks;
    index_->CleanEmptyBlocks(empty_blocks, 2000);

    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, LargeBlockSequence) {
    // 插入一个较长的序列
    std::vector<int64_t> block_keys;
    for (int i = 0; i < 100; i++) {
        block_keys.push_back(i);
    }

    auto result = index_->InsertOnly(block_keys, 1000);
    EXPECT_EQ(result.inserted_keys.size(), 100);

    // 查询前50个
    std::vector<int64_t> query_keys;
    for (int i = 0; i < 50; i++) {
        query_keys.push_back(i);
    }

    BlockMask block_mask = std::vector<bool>(query_keys.size(), true);

    index_->PrefixQuery(query_keys, block_mask, 2000);
    // 不应该崩溃
    SUCCEED();
}
