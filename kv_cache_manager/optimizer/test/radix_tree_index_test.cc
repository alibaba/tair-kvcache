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
