#include <atomic>
#include <thread>
#include <vector>

#include "kv_cache_manager/affinity/hint_suppressor.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

namespace {

struct MockClock {
    int64_t now_us = 0;
    std::function<int64_t()> Fn() {
        return [this]() { return this->now_us; };
    }
};

constexpr uint32_t kWindowMs = 60000;

} // namespace

class HintSuppressorTest : public TESTBASE {};

TEST_F(HintSuppressorTest, EmptyTargetNodeReturnsTrueAndDoesNotRecord) {
    HintSuppressor s;
    EXPECT_TRUE(s.TryEmit(1, "", kWindowMs));
    EXPECT_TRUE(s.TryEmit(2, "", 0));
    EXPECT_EQ(0u, s.Size());
}

TEST_F(HintSuppressorTest, FirstEmitAlwaysPasses) {
    MockClock clk;
    HintSuppressor s(100, clk.Fn());
    EXPECT_TRUE(s.TryEmit(42, "node_a", kWindowMs));
    EXPECT_EQ(1u, s.Size());
}

TEST_F(HintSuppressorTest, SecondEmitWithinWindowSuppressed) {
    MockClock clk;
    clk.now_us = 1'000'000;
    HintSuppressor s(100, clk.Fn());

    EXPECT_TRUE(s.TryEmit(42, "node_a", kWindowMs));
    clk.now_us += 1'000;
    EXPECT_FALSE(s.TryEmit(42, "node_a", kWindowMs));
    clk.now_us = 1'000'000 + 59'999'000;
    EXPECT_FALSE(s.TryEmit(42, "node_a", kWindowMs));
}

TEST_F(HintSuppressorTest, EmitAfterWindowExpirationPasses) {
    MockClock clk;
    clk.now_us = 1'000'000;
    HintSuppressor s(100, clk.Fn());

    EXPECT_TRUE(s.TryEmit(42, "node_a", kWindowMs));
    // 边界：差 == window_us ⇒ 放行（严格 <）
    clk.now_us = 1'000'000 + 60'000'000;
    EXPECT_TRUE(s.TryEmit(42, "node_a", kWindowMs));
    // 放行同时刷新 last_emit_us，下次立刻再 emit 又被抑制
    clk.now_us += 1'000;
    EXPECT_FALSE(s.TryEmit(42, "node_a", kWindowMs));
}

// window_ms == 0 仍占用容量
TEST_F(HintSuppressorTest, ZeroWindowAlwaysPasses) {
    MockClock clk;
    HintSuppressor s(100, clk.Fn());
    EXPECT_TRUE(s.TryEmit(42, "node_a", 0));
    EXPECT_TRUE(s.TryEmit(42, "node_a", 0));
    EXPECT_TRUE(s.TryEmit(42, "node_a", 0));
    EXPECT_EQ(1u, s.Size());
}

TEST_F(HintSuppressorTest, DifferentKeysIsolated) {
    MockClock clk;
    HintSuppressor s(100, clk.Fn());
    EXPECT_TRUE(s.TryEmit(1, "node_a", kWindowMs));
    EXPECT_TRUE(s.TryEmit(1, "node_b", kWindowMs));
    EXPECT_TRUE(s.TryEmit(2, "node_a", kWindowMs));
    EXPECT_EQ(3u, s.Size());
    EXPECT_FALSE(s.TryEmit(1, "node_a", kWindowMs));
    EXPECT_FALSE(s.TryEmit(1, "node_b", kWindowMs));
    EXPECT_FALSE(s.TryEmit(2, "node_a", kWindowMs));
}

TEST_F(HintSuppressorTest, LRUEvictsOldestEntryWhenFull) {
    MockClock clk;
    HintSuppressor s(3, clk.Fn());
    EXPECT_TRUE(s.TryEmit(1, "n", kWindowMs));
    EXPECT_TRUE(s.TryEmit(2, "n", kWindowMs));
    EXPECT_TRUE(s.TryEmit(3, "n", kWindowMs));
    EXPECT_EQ(3u, s.Size());

    // 写第 4 个 ⇒ 淘汰最老的 (1, n)
    EXPECT_TRUE(s.TryEmit(4, "n", kWindowMs));
    EXPECT_EQ(3u, s.Size());

    // (1,n) 已被淘汰 ⇒ 重新 emit 当作首次
    EXPECT_TRUE(s.TryEmit(1, "n", kWindowMs));
    EXPECT_EQ(3u, s.Size());

    EXPECT_FALSE(s.TryEmit(3, "n", kWindowMs));
    EXPECT_FALSE(s.TryEmit(4, "n", kWindowMs));
}

TEST_F(HintSuppressorTest, SuccessfulEmitPromotesToMRU) {
    MockClock clk;
    HintSuppressor s(3, clk.Fn());
    EXPECT_TRUE(s.TryEmit(1, "n", kWindowMs));
    EXPECT_TRUE(s.TryEmit(2, "n", kWindowMs));
    EXPECT_TRUE(s.TryEmit(3, "n", kWindowMs));

    // 过窗口后 (1,n) 再次放行，被提到 MRU 前端
    clk.now_us = 60'000'000;
    EXPECT_TRUE(s.TryEmit(1, "n", kWindowMs));

    // 写 (4,n) ⇒ 应淘汰 (2,n)（LRU 末尾），(1,n) 保留
    EXPECT_TRUE(s.TryEmit(4, "n", kWindowMs));
    EXPECT_EQ(3u, s.Size());

    EXPECT_TRUE(s.TryEmit(2, "n", kWindowMs));
    EXPECT_FALSE(s.TryEmit(1, "n", kWindowMs));
}

TEST_F(HintSuppressorTest, DefaultClockWorks) {
    HintSuppressor s;
    EXPECT_TRUE(s.TryEmit(42, "node_a", kWindowMs));
    EXPECT_FALSE(s.TryEmit(42, "node_a", kWindowMs));
}

TEST_F(HintSuppressorTest, ConcurrentTryEmitIsThreadSafe) {
    constexpr size_t kCap = 1024;
    HintSuppressor s(kCap);
    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 5000;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> ts;
    ts.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&, t]() {
            ++ready;
            while (!go.load()) {}
            for (int i = 0; i < kOpsPerThread; ++i) {
                int64_t k = (t * 1000 + i) % 2000;
                s.TryEmit(k, "node_a", 60000);
            }
        });
    }
    while (ready.load() < kThreads) {}
    go.store(true);
    for (auto &th : ts) {
        th.join();
    }
    EXPECT_LE(s.Size(), kCap);
}

} // namespace kv_cache_manager
