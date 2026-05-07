#include <memory>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/ttl.h"

using namespace kv_cache_manager;

class TtlEvictionPolicyTest : public TESTBASE {
protected:
    void SetUp() override { policy_ = std::make_shared<TtlEvictionPolicy>("shared", false); }

    BlockEntry MakeBlock(int64_t key, int64_t last_access = 0, int64_t ttl_ns = 0, bool with_location = false) {
        BlockEntry b;
        b.key = key;
        b.last_access_time = last_access;
        b.ttl_anchor_time = last_access;
        b.ttl_ns = ttl_ns;
        if (with_location) {
            b.location_map["shared"] = TierStat{};
        }
        return b;
    }

    std::shared_ptr<TtlEvictionPolicy> policy_;
};

// ============================================================
//  基础行为
// ============================================================

TEST_F(TtlEvictionPolicyTest, BasicInit) {
    EXPECT_EQ(policy_->name(), "shared");
    EXPECT_EQ(policy_->size(), 0);
}

TEST_F(TtlEvictionPolicyTest, WriteAndSize) {
    auto b1 = MakeBlock(1);
    auto b2 = MakeBlock(2);
    policy_->OnBlockWritten(&b1);
    policy_->OnBlockWritten(&b2);
    EXPECT_EQ(policy_->size(), 2);
}

// ============================================================
//  纯 TTL 驱逐：只回收过期的
// ============================================================

TEST_F(TtlEvictionPolicyTest, OnlyExpiredBlocksEvicted) {
    auto b1 = MakeBlock(1, 1000, 100, true); // TTL=100us
    auto b2 = MakeBlock(2, 900, 0, true);    // 无 TTL

    policy_->OnBlockWritten(&b1);
    policy_->OnBlockWritten(&b2);

    // t=1200: b1 过期 (1000+100<1200), b2 永不过期
    policy_->OnBlockAccessedWithOptions(&b2, 1200, true);

    auto evicted = policy_->EvictBlocks(10);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_EQ(policy_->size(), 1);
}

TEST_F(TtlEvictionPolicyTest, NoExpiredNothingEvicted) {
    auto b1 = MakeBlock(1, 1000, 0, true);     // 永不过期
    auto b2 = MakeBlock(2, 2000, 99999, true); // TTL 足够大

    policy_->OnBlockWritten(&b1);
    policy_->OnBlockWritten(&b2);

    policy_->OnBlockAccessedWithOptions(&b1, 3000, true);

    auto evicted = policy_->EvictBlocks(100);
    EXPECT_EQ(evicted.size(), 0);
    EXPECT_EQ(policy_->size(), 2);
}

TEST_F(TtlEvictionPolicyTest, AllExpiredAllEvicted) {
    auto b1 = MakeBlock(1, 100, 50, true);
    auto b2 = MakeBlock(2, 200, 50, true);
    auto b3 = MakeBlock(3, 300, 50, true);

    policy_->OnBlockWritten(&b1);
    policy_->OnBlockWritten(&b2);
    policy_->OnBlockWritten(&b3);

    // 用一个无 TTL 的 block 推进时间到 t=1000，避免 OnBlockAccessed 续命已有 block
    auto anchor = MakeBlock(99, 1000, 0, true);
    policy_->OnBlockWritten(&anchor);

    // t=1000: b1(150) b2(250) b3(350) 全部过期，anchor 永不过期
    auto evicted = policy_->EvictBlocks(10);
    EXPECT_EQ(evicted.size(), 3);
    EXPECT_EQ(policy_->size(), 1); // anchor 留存
}

TEST_F(TtlEvictionPolicyTest, EvictAllExpiredIgnoresCount) {
    auto b1 = MakeBlock(1, 100, 10, true);
    auto b2 = MakeBlock(2, 100, 10, true);
    auto b3 = MakeBlock(3, 100, 10, true);

    policy_->OnBlockWritten(&b1);
    policy_->OnBlockWritten(&b2);
    policy_->OnBlockWritten(&b3);

    auto anchor = MakeBlock(99, 1000, 0, true);
    policy_->OnBlockWritten(&anchor);

    // 3 个全过期，count=2 但 TTL 无视 count，全部清走
    auto evicted = policy_->EvictBlocks(2);
    EXPECT_EQ(evicted.size(), 3);
    EXPECT_EQ(policy_->size(), 1); // anchor 留存
}

// ============================================================
//  IsExpired 边界测试
// ============================================================

TEST_F(TtlEvictionPolicyTest, BlockEntryIsExpired) {
    auto b = MakeBlock(1, 1000, 500);
    EXPECT_FALSE(b.IsExpired(1000));
    EXPECT_FALSE(b.IsExpired(1499));
    EXPECT_FALSE(b.IsExpired(1500));
    EXPECT_TRUE(b.IsExpired(1501));
}

TEST_F(TtlEvictionPolicyTest, ZeroTtlNeverExpires) {
    auto b = MakeBlock(1, 1000, 0);
    EXPECT_FALSE(b.IsExpired(999999999));
}

// ============================================================
//  Sliding Window TTL（访问刷新 last_access_time 续命）
// ============================================================

TEST_F(TtlEvictionPolicyTest, AccessRefreshesTtl) {
    auto b = MakeBlock(1, 1000, 500, true);
    policy_->OnBlockWritten(&b);

    EXPECT_FALSE(b.IsExpired(1400));

    // 模拟 RadixTreeIndex::OnBlockAccessed 的 block 级字段更新
    // （生产路径中 block 级统计由 RadixTreeIndex 统一负责，策略层不写）
    b.last_access_time = 1400;
    policy_->OnBlockAccessedWithOptions(&b, 1400, true);
    EXPECT_EQ(b.last_access_time, 1400);

    // t=1800: 距新 ttl_anchor 400us，仍未过期
    EXPECT_FALSE(b.IsExpired(1800));

    // t=1901: 超过 1400+500=1900，过期
    EXPECT_TRUE(b.IsExpired(1901));
}

// ============================================================
//  Clear
// ============================================================

TEST_F(TtlEvictionPolicyTest, ClearRemovesAll) {
    auto b1 = MakeBlock(1, 0, 0, true);
    auto b2 = MakeBlock(2, 0, 0, true);

    policy_->OnBlockWritten(&b1);
    policy_->OnBlockWritten(&b2);
    EXPECT_EQ(policy_->size(), 2);

    policy_->Clear();
    EXPECT_EQ(policy_->size(), 0);
    EXPECT_TRUE(b1.location_map.empty());
    EXPECT_TRUE(b2.location_map.empty());
}

// ============================================================
//  混合场景：部分过期 + 部分存活（含永不过期的 block）
// ============================================================

TEST_F(TtlEvictionPolicyTest, MixedExpiredAndAlive) {
    auto b1 = MakeBlock(1, 100, 50, true);  // 过期
    auto b2 = MakeBlock(2, 200, 0, true);   // 永不过期
    auto b3 = MakeBlock(3, 300, 100, true); // 未过期

    policy_->OnBlockWritten(&b1);
    policy_->OnBlockWritten(&b2);
    policy_->OnBlockWritten(&b3);

    // t=350: b1 过期 (100+50<350), b3 未过期 (300+100=400>350), b2 永不过期
    policy_->OnBlockAccessedWithOptions(&b3, 350, true);

    auto evicted = policy_->EvictBlocks(10);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_EQ(policy_->size(), 2);
}

// ============================================================
//  写入路径推进时间也能触发过期判定
// ============================================================

TEST_F(TtlEvictionPolicyTest, WriteAdvancesTimestamp) {
    auto b1 = MakeBlock(1, 1000, 200, true); // TTL=200, 将在 t>1200 过期

    policy_->OnBlockWritten(&b1);

    // 不通过 OnBlockAccessed，而是通过后续写入推进时间
    auto b2 = MakeBlock(2, 1500, 0, true); // 写入时间 1500 > 1200
    policy_->OnBlockWritten(&b2);

    auto evicted = policy_->EvictBlocks(10);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 1);
    EXPECT_EQ(policy_->size(), 1);
}

// ============================================================
//  Fallback on pressure（兜底 LRU）
// ============================================================

TEST_F(TtlEvictionPolicyTest, FallbackEvictsOldestWhenNotEnoughExpired) {
    auto fallback_policy = std::make_shared<TtlEvictionPolicy>("shared", true);

    // b1: 最旧，不过期；b2: 过期；b3: 最新，不过期
    auto b1 = MakeBlock(1, 100, 0, true);
    auto b2 = MakeBlock(2, 200, 50, true); // 过期于 t>250
    auto b3 = MakeBlock(3, 300, 0, true);

    fallback_policy->OnBlockWritten(&b1);
    fallback_policy->OnBlockWritten(&b2);
    fallback_policy->OnBlockWritten(&b3);

    // 推进时间到 500
    auto anchor = MakeBlock(99, 500, 0, true);
    fallback_policy->OnBlockWritten(&anchor);

    // count=3: Phase1 清 b2（过期），Phase2 兜底补 2 个 → 从尾部取 b1（最旧）再取 b3
    auto evicted = fallback_policy->EvictBlocks(3);
    EXPECT_EQ(evicted.size(), 3);
    EXPECT_EQ(fallback_policy->size(), 1); // anchor 留存
}

TEST_F(TtlEvictionPolicyTest, FallbackStopsWhenListEmpty) {
    auto fallback_policy = std::make_shared<TtlEvictionPolicy>("shared", true);

    auto b1 = MakeBlock(1, 100, 50, true);
    fallback_policy->OnBlockWritten(&b1);

    auto anchor = MakeBlock(99, 500, 0, true);
    fallback_policy->OnBlockWritten(&anchor);

    // count=100: Phase1 清 b1，Phase2 兜底取 anchor，链表空 → 停止
    auto evicted = fallback_policy->EvictBlocks(100);
    EXPECT_EQ(evicted.size(), 2);
    EXPECT_EQ(fallback_policy->size(), 0);
}

TEST_F(TtlEvictionPolicyTest, NoFallbackLeavesExcess) {
    // fallback=false 时，过期不够不会强制驱逐
    auto no_fallback = std::make_shared<TtlEvictionPolicy>("shared", false);

    auto b1 = MakeBlock(1, 100, 0, true);  // 永不过期
    auto b2 = MakeBlock(2, 200, 50, true); // 过期于 t>250
    auto b3 = MakeBlock(3, 300, 0, true);  // 永不过期

    no_fallback->OnBlockWritten(&b1);
    no_fallback->OnBlockWritten(&b2);
    no_fallback->OnBlockWritten(&b3);

    auto anchor = MakeBlock(99, 500, 0, true);
    no_fallback->OnBlockWritten(&anchor);

    // count=10: 只有 b2 过期，无兜底 → 只驱逐 1 个
    auto evicted = no_fallback->EvictBlocks(10);
    EXPECT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0]->key, 2);
    EXPECT_EQ(no_fallback->size(), 3);
}
