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
    std::shared_ptr<LruEvictionPolicy> policy_;
};

TEST_F(LruEvictionPolicyTest, BasicInitialization) {
    EXPECT_EQ(policy_->name(), "test_lru");
    EXPECT_EQ(policy_->size(), 0);
}

TEST_F(LruEvictionPolicyTest, OnBlockWritten) {
    BlockEntry block1;
    block1.key = 1;
    block1.last_access_time = 1000;

    BlockEntry block2;
    block2.key = 2;
    block2.last_access_time = 2000;

    policy_->OnBlockWritten(&block1);
    EXPECT_EQ(policy_->size(), 1);

    policy_->OnBlockWritten(&block2);
    EXPECT_EQ(policy_->size(), 2);
}

TEST_F(LruEvictionPolicyTest, OnBlockAccessed) {
    BlockEntry block1;
    block1.key = 1;
    block1.last_access_time = 1000;

    BlockEntry block2;
    block2.key = 2;
    block2.last_access_time = 2000;

    policy_->OnBlockWritten(&block1);
    policy_->OnBlockWritten(&block2);

    // 访问block1,将其移到LRU链表头部
    policy_->OnBlockAccessed(&block1, 3000);
    EXPECT_EQ(block1.last_access_time, 3000);

    // 驱逐应该先驱逐block2(最久未使用)
    auto evicted = policy_->EvictBlocks(1);
    EXPECT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 2);
}

TEST_F(LruEvictionPolicyTest, EvictBlocks) {
    BlockEntry block1;
    block1.key = 1;
    block1.last_access_time = 1000;

    BlockEntry block2;
    block2.key = 2;
    block2.last_access_time = 2000;

    BlockEntry block3;
    block3.key = 3;
    block3.last_access_time = 3000;

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
    BlockEntry block1;
    block1.key = 1;
    block1.last_access_time = 1000;

    BlockEntry block2;
    block2.key = 2;
    block2.last_access_time = 2000;

    policy_->OnBlockWritten(&block1);
    policy_->OnBlockWritten(&block2);

    EXPECT_EQ(policy_->size(), 2);

    // 驱逐所有块
    auto evicted = policy_->EvictBlocks(10);
    EXPECT_EQ(evicted.size(), 2);
    EXPECT_EQ(policy_->size(), 0);
}

TEST_F(LruEvictionPolicyTest, OnNodeWritten) {
    BlockEntry block1;
    block1.key = 1;
    block1.last_access_time = 1000;

    BlockEntry block2;
    block2.key = 2;
    block2.last_access_time = 2000;

    std::vector<BlockEntry *> blocks = {&block1, &block2};
    policy_->OnNodeWritten(blocks);

    EXPECT_EQ(policy_->size(), 2);
}

TEST_F(LruEvictionPolicyTest, EvictMoreThanAvailable) {
    BlockEntry block1;
    block1.key = 1;
    block1.last_access_time = 1000;

    policy_->OnBlockWritten(&block1);
    EXPECT_EQ(policy_->size(), 1);

    // 尝试驱逐比可用数量更多的块
    auto evicted = policy_->EvictBlocks(10);
    EXPECT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_EQ(policy_->size(), 0);
}

TEST_F(LruEvictionPolicyTest, LruOrderAfterMultipleAccesses) {
    BlockEntry block1;
    block1.key = 1;
    block1.last_access_time = 1000;

    BlockEntry block2;
    block2.key = 2;
    block2.last_access_time = 2000;

    BlockEntry block3;
    block3.key = 3;
    block3.last_access_time = 3000;

    policy_->OnBlockWritten(&block1);
    policy_->OnBlockWritten(&block2);
    policy_->OnBlockWritten(&block3);

    // 多次访问不同的块
    policy_->OnBlockAccessed(&block1, 4000);
    policy_->OnBlockAccessed(&block3, 5000);
    policy_->OnBlockAccessed(&block2, 6000);

    // block3应该是最久未使用的(最后访问时间是5000,而block1是4000,block2是6000)
    // LRU驱逐最久未访问的,即最后访问时间最小的
    auto evicted = policy_->EvictBlocks(1);
    EXPECT_EQ(evicted[0]->key, 1); // block1的最后访问时间是4000,是最小的
    EXPECT_EQ(policy_->size(), 2);
}