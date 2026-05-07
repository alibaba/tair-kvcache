#include <memory>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/lru.h"

using namespace kv_cache_manager;

class LruEvictionPolicyTest : public TESTBASE {
public:
    void SetUp() override {
        LruParams params;
        params.sample_rate = 1.0;
        policy_ = std::make_shared<LruEvictionPolicy>("test_lru", params);
    }

protected:
    BlockEntry CreateBlock(int64_t key, int64_t last_access_time) {
        BlockEntry block;
        block.key = key;
        block.last_access_time = last_access_time;
        block.location_map[policy_->name()] = TierStat{0, last_access_time, -1};
        return block;
    }

    void AccessBlock(BlockEntry *block, int64_t timestamp) {
        block->last_access_time = timestamp;
        block->location_map[policy_->name()].last_access_time = timestamp;
        policy_->OnBlockAccessedWithOptions(block, timestamp, true);
    }

    std::shared_ptr<LruEvictionPolicy> policy_;
};

TEST_F(LruEvictionPolicyTest, BasicInitialization) {
    EXPECT_EQ(policy_->name(), "test_lru");
    EXPECT_EQ(policy_->size(), 0);
}

TEST_F(LruEvictionPolicyTest, OnBlockWritten) {
    auto block1 = CreateBlock(1, 1000);
    auto block2 = CreateBlock(2, 2000);

    policy_->OnBlockWritten(&block1);
    EXPECT_EQ(policy_->size(), 1);

    policy_->OnBlockWritten(&block2);
    EXPECT_EQ(policy_->size(), 2);
}

TEST_F(LruEvictionPolicyTest, OnBlockAccessed) {
    auto block1 = CreateBlock(1, 1000);
    auto block2 = CreateBlock(2, 2000);

    policy_->OnBlockWritten(&block1);
    policy_->OnBlockWritten(&block2);

    // 访问block1,将其移到LRU链表头部
    AccessBlock(&block1, 3000);
    EXPECT_EQ(block1.last_access_time, 3000);

    // 驱逐应该先驱逐block2(最久未使用)
    auto evicted = policy_->EvictBlocks(1);
    EXPECT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 2);
}

TEST_F(LruEvictionPolicyTest, EvictBlocks) {
    auto block1 = CreateBlock(1, 1000);
    auto block2 = CreateBlock(2, 2000);
    auto block3 = CreateBlock(3, 3000);

    policy_->OnBlockWritten(&block1);
    policy_->OnBlockWritten(&block2);
    policy_->OnBlockWritten(&block3);

    EXPECT_EQ(policy_->size(), 3);

    // 驱逐2个块
    auto evicted = policy_->EvictBlocks(2);
    EXPECT_EQ(evicted.size(), 2);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_EQ(evicted[1]->key, 2);

    // 剩余1个块
    EXPECT_EQ(policy_->size(), 1);
}

TEST_F(LruEvictionPolicyTest, EvictAllBlocks) {
    auto block1 = CreateBlock(1, 1000);
    auto block2 = CreateBlock(2, 2000);

    policy_->OnBlockWritten(&block1);
    policy_->OnBlockWritten(&block2);

    EXPECT_EQ(policy_->size(), 2);

    // 驱逐所有块
    auto evicted = policy_->EvictBlocks(10);
    EXPECT_EQ(evicted.size(), 2);
    EXPECT_EQ(policy_->size(), 0);
}

TEST_F(LruEvictionPolicyTest, OnNodeWritten) {
    auto block1 = CreateBlock(1, 1000);
    auto block2 = CreateBlock(2, 2000);

    std::vector<BlockEntry *> blocks = {&block1, &block2};
    policy_->OnNodeWritten(blocks);

    EXPECT_EQ(policy_->size(), 2);
}

TEST_F(LruEvictionPolicyTest, EvictMoreThanAvailable) {
    auto block1 = CreateBlock(1, 1000);

    policy_->OnBlockWritten(&block1);
    EXPECT_EQ(policy_->size(), 1);

    // 尝试驱逐比可用数量更多的块
    auto evicted = policy_->EvictBlocks(10);
    EXPECT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_EQ(policy_->size(), 0);
}

TEST_F(LruEvictionPolicyTest, LruOrderAfterMultipleAccesses) {
    auto block1 = CreateBlock(1, 1000);
    auto block2 = CreateBlock(2, 2000);
    auto block3 = CreateBlock(3, 3000);

    policy_->OnBlockWritten(&block1);
    policy_->OnBlockWritten(&block2);
    policy_->OnBlockWritten(&block3);

    // 多次访问不同的块
    AccessBlock(&block1, 4000);
    AccessBlock(&block3, 5000);
    AccessBlock(&block2, 6000);

    // block3应该是最久未使用的(最后访问时间是5000,而block1是4000,block2是6000)
    // LRU驱逐最久未访问的,即最后访问时间最小的
    auto evicted = policy_->EvictBlocks(1);
    EXPECT_EQ(evicted[0]->key, 1); // block1的最后访问时间是4000,是最小的
    EXPECT_EQ(policy_->size(), 2);
}
