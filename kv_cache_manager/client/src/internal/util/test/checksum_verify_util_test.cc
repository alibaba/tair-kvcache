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

// 极端：fast 路径 XOR 偶然抵消的构造场景。Strict mode 必须仍能发现。
// 构造：block #0 expected=0xAAAA actual=0xBBBB，block #1 expected=0xBBBB actual=0xAAAA
// XOR 都是 0xAAAA ^ 0xBBBB，fast aggregate 相等 -> fast 漏；strict 不漏。
TEST_F(ChecksumVerifyUtilTest, StrictModeCatchesPairedXorCancellation) {
    std::vector<int64_t> expected = {0xAAAA, 0xBBBB};
    std::vector<int64_t> actual = {0xBBBB, 0xAAAA};
    // Fast: 漏 (XOR 都是 0xAAAA^0xBBBB)
    auto r_fast = VerifyBatchChecksums(expected, actual, /*strict_mode=*/false);
    EXPECT_FALSE(r_fast.mismatch);
    // Strict: 抓到两个
    auto r_strict = VerifyBatchChecksums(expected, actual, /*strict_mode=*/true);
    ASSERT_TRUE(r_strict.mismatch);
    EXPECT_EQ(r_strict.faulty_indices.size(), 2u);
}
