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

TEST_F(RadixTreeIndexTest, InsertOnlyTouchesExistingBlocksOnWrite) {
    std::vector<int64_t> block_keys = {1, 2, 3};
    index_->InsertOnly(block_keys, 1000);

    const auto *root = index_->GetRoot();
    ASSERT_NE(root, nullptr);
    auto child_it = root->children.find(1);
    ASSERT_NE(child_it, root->children.end());
    ASSERT_EQ(child_it->second->blocks.size(), 3);

    auto *block1 = child_it->second->blocks[0].get();
    auto *block2 = child_it->second->blocks[1].get();
    auto *block3 = child_it->second->blocks[2].get();

    auto result = index_->InsertOnly({1, 2}, 2000);
    EXPECT_EQ(result.inserted_keys.size(), 0);

    EXPECT_EQ(block1->writing_time, 1000);
    EXPECT_EQ(block1->last_access_time, 2000);
    EXPECT_EQ(block1->location_map.at("test_lru").writing_time, 1000);
    EXPECT_EQ(block1->location_map.at("test_lru").last_access_time, 2000);
    EXPECT_EQ(block2->writing_time, 1000);
    EXPECT_EQ(block2->last_access_time, 2000);
    EXPECT_EQ(block2->location_map.at("test_lru").writing_time, 1000);
    EXPECT_EQ(block2->location_map.at("test_lru").last_access_time, 2000);

    EXPECT_EQ(block3->writing_time, 1000);
    EXPECT_EQ(block3->last_access_time, 1000);
    EXPECT_EQ(block3->location_map.at("test_lru").writing_time, 1000);
    EXPECT_EQ(block3->location_map.at("test_lru").last_access_time, 1000);
}

TEST_F(RadixTreeIndexTest, FillPathOnlyDoesNotTouchExistingPrefixOrCountWriteTouch) {
    index_->InsertOnly({1}, 1000);

    auto *prefix_block = index_->GetRoot()->children.at(1)->blocks[0].get();
    ASSERT_EQ(prefix_block->location_map.count("test_lru"), 1);

    auto result = index_->FillPathOnly({1, 2}, {1}, 2000);
    ASSERT_EQ(result.inserted_keys, (std::vector<int64_t>{2}));

    EXPECT_EQ(prefix_block->last_access_time, 1000);
    EXPECT_EQ(prefix_block->location_map.at("test_lru").last_access_time, 1000);
    EXPECT_EQ(prefix_block->location_map.at("test_lru").write_touch_count, 1);

    auto *filled_block = index_->GetRoot()->children.at(1)->blocks[1].get();
    ASSERT_EQ(filled_block->key, 2);
    ASSERT_EQ(filled_block->location_map.count("test_lru"), 1);
    EXPECT_EQ(filled_block->location_map.at("test_lru").write_touch_count, 0);
}

TEST_F(RadixTreeIndexTest, FillPathOnlyKeepsSparseBatchHitsOnPrefixPath) {
    auto result = index_->FillPathOnly({1, 2, 3}, {2}, 1000);
    ASSERT_EQ(result.inserted_keys, (std::vector<int64_t>{3}));

    const auto *root = index_->GetRoot();
    ASSERT_NE(root, nullptr);
    ASSERT_EQ(root->children.count(1), 1);
    EXPECT_EQ(root->children.count(3), 0);

    const auto *path_node = root->children.at(1).get();
    ASSERT_EQ(path_node->blocks.size(), 3);
    EXPECT_TRUE(path_node->blocks[0]->location_map.empty());
    EXPECT_TRUE(path_node->blocks[1]->location_map.empty());
    ASSERT_EQ(path_node->blocks[2]->location_map.count("test_lru"), 1);
    EXPECT_EQ(path_node->blocks[2]->location_map.at("test_lru").write_touch_count, 0);

    QueryHit prefix_hit;
    index_->PrefixQuery({1, 2, 3}, BlockMaskVector{}, 1100, &prefix_hit);
    EXPECT_EQ(prefix_hit.remote_hit_block_num, 0);

    QueryHit batch_hit;
    index_->BatchQuery({3}, BlockMaskVector{}, 1200, &batch_hit);
    EXPECT_EQ(batch_hit.remote_hit_block_num, 1);
}

TEST_F(RadixTreeIndexTest, FillPathOnlyRecordsPromoteReasonAcrossWriteThroughTiers) {
    LruParams params;
    auto l1 = std::make_shared<LruEvictionPolicy>("l1", params);
    auto l2 = std::make_shared<LruEvictionPolicy>("l2", params);
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {l1, l2};
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH);

    auto result = index->FillPathOnly({10}, {0}, 1000);
    const auto &events = result.tier_flow.events();
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events[0].kind, TierFlowEventKind::ENTER_TIER);
    EXPECT_EQ(events[0].reason, TierFlowEventReason::PROMOTE);
    EXPECT_EQ(events[0].to_tier, "l1");
    EXPECT_EQ(events[1].kind, TierFlowEventKind::ENTER_TIER);
    EXPECT_EQ(events[1].reason, TierFlowEventReason::PROMOTE);
    EXPECT_EQ(events[1].to_tier, "l2");
}

TEST_F(RadixTreeIndexTest, InsertOnlyRefreshAffectsLruEvictionOrder) {
    LruParams params;
    params.sample_rate = 1.0;
    auto policy = std::make_shared<LruEvictionPolicy>("test_lru", params);
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policy);

    index->InsertOnly({1, 2, 3}, 1000);

    const auto *root = index->GetRoot();
    ASSERT_NE(root, nullptr);
    auto child_it = root->children.find(1);
    ASSERT_NE(child_it, root->children.end());
    ASSERT_EQ(child_it->second->blocks.size(), 3);

    auto *block1 = child_it->second->blocks[0].get();
    auto *block2 = child_it->second->blocks[1].get();
    auto *block3 = child_it->second->blocks[2].get();

    index->InsertOnly({1}, 2000);
    auto evicted_blocks = policy->EvictBlocks(1);

    ASSERT_EQ(evicted_blocks.size(), 1);
    EXPECT_EQ(evicted_blocks[0]->key, 2);
    EXPECT_FALSE(block1->location_map.empty());
    EXPECT_TRUE(block2->location_map.empty());
    EXPECT_FALSE(block3->location_map.empty());
}

TEST_F(RadixTreeIndexTest, InsertOnlyWriteThroughRefreshesOnlyEntryTier) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
    };
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH);
    index->InsertOnly({1, 2}, 1000);

    auto *block = index->GetRoot()->children.at(1)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    ASSERT_EQ(block->location_map.count("l2"), 1);

    index->InsertOnly({1}, 2000);

    EXPECT_EQ(block->location_map.at("l1").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 1000);
}

TEST_F(RadixTreeIndexTest, InsertOnlyTouchesHighestExistingTierWhenEntryTierMissing) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
    };
    auto index = std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH);
    index->InsertOnly({1}, 1000);

    auto *block = index->GetRoot()->children.at(1)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    ASSERT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.at("l2").write_touch_count, 0);

    block->location_map.erase("l1");
    auto result = index->InsertOnly({1}, 2000);

    EXPECT_TRUE(result.inserted_keys.empty());
    EXPECT_EQ(block->location_map.count("l1"), 0);
    ASSERT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").write_touch_count, 1);
}

TEST_F(RadixTreeIndexTest, InsertOnlyWriteCanPropagateTouchToLowerTier) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
    };
    std::vector<TierFlowStrategy> flows(1);
    flows[0].write_mode = TierWriteMode::WRITE_THROUGH;
    flows[0].write_propagation_enabled = true;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH, 0, 2, true, flows);
    index->InsertOnly({1, 2}, 1000);

    auto *block = index->GetRoot()->children.at(1)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    ASSERT_EQ(block->location_map.count("l2"), 1);

    index->InsertOnly({1}, 2000);

    EXPECT_EQ(block->location_map.at("l1").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l1").write_touch_count, 2);
    EXPECT_EQ(block->location_map.at("l2").write_touch_count, 0);
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
    index->InsertOnly({10}, 1000);

    auto *block = index->GetRoot()->children.at(10)->blocks[0].get();
    block->location_map.clear();
    AppendBlockLocation(block, "l3", 1000);

    QueryHit query_hit;
    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({10}, block_mask, 2000, &query_hit);

    EXPECT_EQ(block->location_map.count("l1"), 1);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.count("l3"), 1);
    EXPECT_EQ(block->location_map.at("l1").write_touch_count, 0);
    EXPECT_EQ(block->location_map.at("l2").write_touch_count, 0);
    EXPECT_TRUE(index->ConsumeReadTriggeredTierWrite());
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
    index->InsertOnly({20}, 1000);

    auto *block = index->GetRoot()->children.at(20)->blocks[0].get();
    block->location_map.clear();
    AppendBlockLocation(block, "l2", 1000);

    QueryHit query_hit;
    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({20}, block_mask, 2000, &query_hit);

    EXPECT_EQ(block->location_map.count("l1"), 1);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.count("l3"), 0);
    EXPECT_TRUE(index->ConsumeReadTriggeredTierWrite());
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

TEST_F(RadixTreeIndexTest, SelectiveWriteToNextTierAfterWriteTouchThreshold) {
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
    EXPECT_EQ(block->location_map.at("l1").write_touch_count, 1);

    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({46}, block_mask, 2000);
    EXPECT_FALSE(index->ConsumeReadTriggeredTierWrite());
    EXPECT_EQ(block->location_map.count("l2"), 0);

    index->PrefixQuery({46}, block_mask, 3000);
    EXPECT_FALSE(index->ConsumeReadTriggeredTierWrite());
    EXPECT_EQ(block->location_map.count("l2"), 0);
    EXPECT_EQ(block->access_count, 2);
    EXPECT_EQ(block->location_map.at("l1").access_count, 2);

    index->InsertOnly({46}, 4000);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.at("l2").access_count, 0);
    EXPECT_EQ(block->location_map.at("l1").write_touch_count, 2);
    EXPECT_EQ(block->location_map.at("l2").write_touch_count, 0);
    EXPECT_FALSE(index->ConsumeReadTriggeredTierWrite());
}

TEST_F(RadixTreeIndexTest, WritePropagationDoesNotTriggerLowerSelectiveWrite) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    std::vector<TierFlowStrategy> flows(2);
    flows[0].write_mode = TierWriteMode::WRITE_THROUGH;
    flows[0].write_propagation_enabled = true;
    flows[1].write_mode = TierWriteMode::WRITE_THROUGH_SELECTIVE;
    flows[1].selective_write_threshold = 2;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH, 0, 2, true, flows);
    index->InsertOnly({47}, 1000);

    auto *block = index->GetRoot()->children.at(47)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    ASSERT_EQ(block->location_map.count("l2"), 1);
    ASSERT_EQ(block->location_map.count("l3"), 0);
    EXPECT_EQ(block->location_map.at("l2").write_touch_count, 0);

    index->InsertOnly({47}, 2000);

    EXPECT_EQ(block->location_map.at("l1").write_touch_count, 2);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").write_touch_count, 0);
    EXPECT_EQ(block->location_map.count("l3"), 0);
}

TEST_F(RadixTreeIndexTest, SelectiveWriteThroughContinuesWhenTargetTierExists) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    std::vector<TierFlowStrategy> flows(2);
    flows[0].write_mode = TierWriteMode::WRITE_THROUGH_SELECTIVE;
    flows[0].selective_write_threshold = 2;
    flows[1].write_mode = TierWriteMode::WRITE_THROUGH;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH, 0, 2, true, flows);
    index->InsertOnly({48}, 1000);

    auto *block = index->GetRoot()->children.at(48)->blocks[0].get();
    ASSERT_EQ(block->location_map.count("l1"), 1);
    AppendBlockLocation(block, "l2", 1000, 0);
    ASSERT_EQ(block->location_map.count("l2"), 1);
    ASSERT_EQ(block->location_map.count("l3"), 0);

    index->InsertOnly({48}, 2000);

    EXPECT_EQ(block->location_map.at("l1").write_touch_count, 2);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.count("l3"), 1);
    EXPECT_EQ(block->location_map.at("l3").write_touch_count, 0);
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

TEST_F(RadixTreeIndexTest, AccessPropagationTouchesAllReachableLowerTiers) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    std::vector<TierFlowStrategy> flows(2);
    flows[0].write_mode = TierWriteMode::WRITE_THROUGH;
    flows[0].access_propagation_enabled = true;
    flows[1].write_mode = TierWriteMode::WRITE_THROUGH;
    flows[1].access_propagation_enabled = true;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::WRITE_THROUGH, 0, 2, true, flows);
    index->InsertOnly({61}, 1000);

    auto *block = index->GetRoot()->children.at(61)->blocks[0].get();
    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({61}, block_mask, 2000);

    EXPECT_EQ(block->location_map.at("l1").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l3").last_access_time, 2000);
    EXPECT_EQ(block->location_map.at("l1").access_count, 1);
    EXPECT_EQ(block->location_map.at("l2").access_count, 0);
    EXPECT_EQ(block->location_map.at("l3").access_count, 0);
}

TEST_F(RadixTreeIndexTest, TierFlowsPromoteAcrossAllHigherTiers) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
        std::make_shared<LruEvictionPolicy>("l3", params),
    };
    std::vector<TierFlowStrategy> flows(2);
    flows[0].write_mode = TierWriteMode::CASCADING;
    flows[1].write_mode = TierWriteMode::CASCADING;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::CASCADING, 0, 2, true, flows);
    index->InsertOnly({70}, 1000);

    auto *block = index->GetRoot()->children.at(70)->blocks[0].get();
    block->location_map.clear();
    AppendBlockLocation(block, "l3", 1000);

    QueryHit query_hit;
    BlockMask block_mask = std::vector<bool>{false};
    index->PrefixQuery({70}, block_mask, 2000, &query_hit);

    EXPECT_EQ(block->location_map.count("l1"), 1);
    EXPECT_EQ(block->location_map.count("l2"), 1);
    EXPECT_EQ(block->location_map.count("l3"), 1);
}

TEST_F(RadixTreeIndexTest, LocalMaskTouchDoesNotCountAsReadOrPromote) {
    LruParams params;
    params.sample_rate = 1.0;
    std::vector<std::shared_ptr<EvictionPolicy>> policies = {
        std::make_shared<LruEvictionPolicy>("l1", params),
        std::make_shared<LruEvictionPolicy>("l2", params),
    };
    std::vector<TierFlowStrategy> flows(1);
    flows[0].write_mode = TierWriteMode::CASCADING;
    auto index =
        std::make_shared<RadixTreeIndex>("test_instance", policies, TierWriteMode::CASCADING, 0, 2, true, flows);
    index->InsertOnly({72}, 1000);

    auto *block = index->GetRoot()->children.at(72)->blocks[0].get();
    block->location_map.clear();
    AppendBlockLocation(block, "l2", 1000);

    QueryHit query_hit;
    BlockMask block_mask = std::vector<bool>{true};
    index->PrefixQuery({72}, block_mask, 2000, &query_hit, true, true, false);

    EXPECT_EQ(query_hit.local_hit_block_num, 0);
    EXPECT_EQ(query_hit.remote_hit_block_num, 0);
    EXPECT_TRUE(query_hit.per_tier_hit_block_num.empty());
    EXPECT_EQ(block->access_count, 0);
    EXPECT_EQ(block->last_access_time, 2000);
    EXPECT_EQ(block->owner_node->stat.access_count, 0);
    EXPECT_EQ(block->owner_node->stat.last_access_time, 2000);
    EXPECT_EQ(block->location_map.count("l1"), 0);
    EXPECT_EQ(block->location_map.at("l2").access_count, 0);
    EXPECT_EQ(block->location_map.at("l2").last_access_time, 2000);
    EXPECT_FALSE(index->ConsumeReadTriggeredTierWrite());
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
