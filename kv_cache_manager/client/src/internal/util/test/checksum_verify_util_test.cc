#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/util/checksum_verify_util.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class ChecksumVerifyUtilTest : public TESTBASE {};

// Fast path：所有 block 一致 -> mismatch=false, faulty_indices 空。
TEST_F(ChecksumVerifyUtilTest, FastPathAllMatch) {
    std::vector<int64_t> expected = {0x1111, 0x2222, 0x3333};
    std::vector<int64_t> actual = expected;
    auto r = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    EXPECT_FALSE(r.mismatch);
    EXPECT_TRUE(r.faulty_indices.empty());
}

// Fast path：检测到不匹配后应该回填 faulty_indices 让上层逐块打日志。
TEST_F(ChecksumVerifyUtilTest, FastPathDetectsMismatchAndLocatesIndex) {
    std::vector<int64_t> expected = {0x1111, 0x2222, 0x3333};
    std::vector<int64_t> actual = {0x1111, 0xFFFF, 0x3333}; // block #1 错
    auto r = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 1u);
    EXPECT_EQ(r.faulty_indices[0], 1u);
}

// 多个错位 block：fallback 阶段把所有错的都列出来。
TEST_F(ChecksumVerifyUtilTest, FastPathListsAllFaultyBlocks) {
    std::vector<int64_t> expected = {0x1111, 0x2222, 0x3333, 0x4444};
    std::vector<int64_t> actual = {0xAAAA, 0x2222, 0xBBBB, 0x4444}; // #0, #2 错
    auto r = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 2u);
    EXPECT_EQ(r.faulty_indices[0], 0u);
    EXPECT_EQ(r.faulty_indices[1], 2u);
}

// expected[i] == 0 是 sentinel (legacy data / legacy client)，跳过比对。
TEST_F(ChecksumVerifyUtilTest, FastPathSentinelZeroIsSkipped) {
    std::vector<int64_t> expected = {0x1111, 0, 0x3333};        // block #1 没有 checksum
    std::vector<int64_t> actual = {0x1111, 0xDEADBEEF, 0x3333}; // #1 的 actual 不会被比较
    auto r = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    EXPECT_FALSE(r.mismatch);
}

// 全 sentinel：fast 路径没有可比较的项，等同于全 match。
TEST_F(ChecksumVerifyUtilTest, FastPathAllSentinelsTreatedAsMatch) {
    std::vector<int64_t> expected = {0, 0, 0};
    std::vector<int64_t> actual = {0xAA, 0xBB, 0xCC};
    auto r = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    EXPECT_FALSE(r.mismatch);
}

// Size mismatch (上层 bug): mismatch=true 且 faulty_indices 空 -> 上层走 size 错误日志。
TEST_F(ChecksumVerifyUtilTest, SizeMismatchReturnsMismatchWithoutIndices) {
    std::vector<int64_t> expected = {0x1111, 0x2222};
    std::vector<int64_t> actual = {0x1111};
    auto r = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    EXPECT_TRUE(r.mismatch);
    EXPECT_TRUE(r.faulty_indices.empty());
}

// Strict mode：行为跟 fast fallback 一致 (per-block 比对)，但跳过 fast 聚合阶段。
TEST_F(ChecksumVerifyUtilTest, StrictModeMatchesFastFallback) {
    std::vector<int64_t> expected = {0x1111, 0, 0x3333, 0x4444};
    std::vector<int64_t> actual = {0xAAAA, 0x2222, 0x3333, 0xBBBB}; // #0, #3 错；#1 是 sentinel
    auto r = VerifyBatchChecksums(expected, actual, /*strict_mode=*/true);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 2u);
    EXPECT_EQ(r.faulty_indices[0], 0u);
    EXPECT_EQ(r.faulty_indices[1], 3u);
}

// Strict mode + all match: 仍然 mismatch=false。
TEST_F(ChecksumVerifyUtilTest, StrictModeAllMatch) {
    std::vector<int64_t> expected = {0x1111, 0x2222, 0x3333};
    std::vector<int64_t> actual = expected;
    auto r = VerifyBatchChecksums(expected, actual, /*strict_mode=*/true);
    EXPECT_FALSE(r.mismatch);
}

// Block swap (读串): expected=[A,B], actual=[B,A]. 老 XOR 聚合无序会漏，
// 加了 position-dependent 奇数乘子后 fast path 也能识别并回填两个 faulty index。
TEST_F(ChecksumVerifyUtilTest, FastPathCatchesBlockSwap) {
    std::vector<int64_t> expected = {0xAAAA, 0xBBBB};
    std::vector<int64_t> actual = {0xBBBB, 0xAAAA};
    auto r_fast = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    ASSERT_TRUE(r_fast.mismatch);
    ASSERT_EQ(r_fast.faulty_indices.size(), 2u);
    EXPECT_EQ(r_fast.faulty_indices[0], 0u);
    EXPECT_EQ(r_fast.faulty_indices[1], 1u);
    // Strict 一致。
    auto r_strict = VerifyBatchChecksums(expected, actual, /*strict_mode=*/true);
    ASSERT_TRUE(r_strict.mismatch);
    EXPECT_EQ(r_strict.faulty_indices.size(), 2u);
}

// Same-delta 成对突变：每个 block 都被同一 delta 改写 (expected=[A,B],
// actual=[A^X, B^X])。老 XOR fast 会 delta 对消而漏；新的乘法聚合不再有 GF(2)-
// 线性，所以两条路径都能识别。这里同时断言，防止将来实现回退成纯 XOR 时漏检回归。
TEST_F(ChecksumVerifyUtilTest, DetectsSameDeltaPairedMutation) {
    constexpr int64_t kDelta = 0x0F0F0F0F0F0F0F0FLL;
    std::vector<int64_t> expected = {0xAAAA, 0xBBBB};
    std::vector<int64_t> actual = {0xAAAA ^ kDelta, 0xBBBB ^ kDelta};
    auto r_fast = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    ASSERT_TRUE(r_fast.mismatch);
    EXPECT_EQ(r_fast.faulty_indices.size(), 2u);
    auto r_strict = VerifyBatchChecksums(expected, actual, /*strict_mode=*/true);
    ASSERT_TRUE(r_strict.mismatch);
    EXPECT_EQ(r_strict.faulty_indices.size(), 2u);
}
