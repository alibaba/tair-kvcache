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
    auto inserted = index_->InsertOnly(block_keys, 1000);
    EXPECT_EQ(inserted.size(), 5);
}

TEST_F(RadixTreeIndexTest, InsertOnlyDuplicate) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 再次插入相同的块
    auto inserted = index_->InsertOnly(block_keys, 2000);
    EXPECT_EQ(inserted.size(), 0); // 应该没有新插入的块
}

TEST_F(RadixTreeIndexTest, InsertWithQueryNoHit) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    std::vector<std::vector<int64_t>> hits;

    auto inserted = index_->InsertWithQuery(block_keys, 1000, hits);
    EXPECT_EQ(inserted.size(), 5);
    EXPECT_EQ(hits.size(), 0); // 第一次插入不应该有命中
}

TEST_F(RadixTreeIndexTest, InsertWithQueryWithHit) {
    // 第一次插入
    std::vector<int64_t> block_keys1 = {1, 2, 3, 4, 5};
    std::vector<std::vector<int64_t>> hits1;
    index_->InsertWithQuery(block_keys1, 1000, hits1);

    // 第二次插入,应该命中
    std::vector<int64_t> block_keys2 = {1, 2, 3, 4, 5};
    std::vector<std::vector<int64_t>> hits2;
    auto inserted = index_->InsertWithQuery(block_keys2, 2000, hits2);

    EXPECT_EQ(inserted.size(), 0); // 所有块都已存在
    EXPECT_EQ(hits2.size(), 1);    // 应该有一个命中
    EXPECT_EQ(hits2[0].size(), 5); // 命中了5个块
}

TEST_F(RadixTreeIndexTest, PrefixQueryNoHit) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 查询不同的序列
    std::vector<int64_t> query_keys = {10, 11, 12};
    BlockMask block_mask = std::vector<bool>(query_keys.size(), true);
    std::vector<std::vector<int64_t>> external_hits;
    std::vector<std::vector<int64_t>> internal_hits;

    index_->PrefixQuery(query_keys, block_mask, 2000, external_hits, internal_hits);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, PrefixQueryWithHit) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 查询相同的前缀
    std::vector<int64_t> query_keys = {1, 2, 3};
    BlockMask block_mask = std::vector<bool>(query_keys.size(), true);
    std::vector<std::vector<int64_t>> external_hits;
    std::vector<std::vector<int64_t>> internal_hits;

    index_->PrefixQuery(query_keys, block_mask, 2000, external_hits, internal_hits);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, PrefixQueryPartialMask) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 查询时mask部分块
    std::vector<int64_t> query_keys = {1, 2, 3};
    BlockMask block_mask = std::vector<bool>{true, false, true}; // 只查询第1和第3个块
    std::vector<std::vector<int64_t>> external_hits;
    std::vector<std::vector<int64_t>> internal_hits;

    index_->PrefixQuery(query_keys, block_mask, 2000, external_hits, internal_hits);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, InsertWithQueryPartialHit) {
    // 第一次插入
    std::vector<int64_t> block_keys1 = {1, 2, 3, 4, 5};
    std::vector<std::vector<int64_t>> hits1;
    index_->InsertWithQuery(block_keys1, 1000, hits1);

    // 第二次插入,部分命中
    std::vector<int64_t> block_keys2 = {1, 2, 6, 7, 8};
    std::vector<std::vector<int64_t>> hits2;
    auto inserted = index_->InsertWithQuery(block_keys2, 2000, hits2);

    EXPECT_EQ(inserted.size(), 3); // 6, 7, 8是新的
    EXPECT_GT(hits2.size(), 0);    // 1, 2应该命中
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
    std::vector<std::vector<int64_t>> external_hits;
    std::vector<std::vector<int64_t>> internal_hits;

    index_->PrefixQuery(query_keys, block_mask, 4000, external_hits, internal_hits);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, CleanEmptyBlocks) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    index_->InsertOnly(block_keys, 1000);

    // 创建空的BlockEntry指针列表
    std::vector<BlockEntry *> empty_blocks;
    index_->CleanEmptyBlocks(empty_blocks);

    // 不应该崩溃
    SUCCEED();
}

TEST_F(RadixTreeIndexTest, InsertWithQueryAndUpdateAccessTime) {
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    std::vector<std::vector<int64_t>> hits;

    // 第一次插入
    index_->InsertWithQuery(block_keys, 1000, hits);

    // 第二次查询相同的数据,应该更新访问时间
    std::vector<std::vector<int64_t>> hits2;
    auto inserted = index_->InsertWithQuery(block_keys, 2000, hits2);

    EXPECT_EQ(inserted.size(), 0);
    EXPECT_EQ(hits2.size(), 1);
}

TEST_F(RadixTreeIndexTest, LargeBlockSequence) {
    // 插入一个较长的序列
    std::vector<int64_t> block_keys;
    for (int i = 0; i < 100; i++) {
        block_keys.push_back(i);
    }

    auto inserted = index_->InsertOnly(block_keys, 1000);
    EXPECT_EQ(inserted.size(), 100);

    // 查询前50个
    std::vector<int64_t> query_keys;
    for (int i = 0; i < 50; i++) {
        query_keys.push_back(i);
    }

    BlockMask block_mask = std::vector<bool>(query_keys.size(), true);
    std::vector<std::vector<int64_t>> external_hits;
    std::vector<std::vector<int64_t>> internal_hits;

    index_->PrefixQuery(query_keys, block_mask, 2000, external_hits, internal_hits);
    // 不应该崩溃
    SUCCEED();
}