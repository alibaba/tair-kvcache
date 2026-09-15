#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/util/checksum_verify_util.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class ChecksumVerifyUtilTest : public TESTBASE {};

// 所有 block 一致 -> mismatch=false, faulty_indices 空。
TEST_F(ChecksumVerifyUtilTest, AllMatch) {
    std::vector<int64_t> expected = {0x1111, 0x2222, 0x3333};
    std::vector<int64_t> actual = expected;
    auto r = VerifyBatchChecksums(expected, actual);
    EXPECT_FALSE(r.mismatch);
    EXPECT_TRUE(r.faulty_indices.empty());
    EXPECT_EQ(r.stage, ChecksumValidationStage::CVS_UNSPECIFIED);
}

// 检测到不匹配后应该回填 faulty_indices 让上层逐块打日志。
TEST_F(ChecksumVerifyUtilTest, DetectsMismatchAndLocatesIndex) {
    std::vector<int64_t> expected = {0x1111, 0x2222, 0x3333};
    std::vector<int64_t> actual = {0x1111, 0xFFFF, 0x3333}; // block #1 错
    auto r = VerifyBatchChecksums(expected, actual);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 1u);
    EXPECT_EQ(r.faulty_indices[0], 1u);
}

// 多个错位 block：把所有错的都列出来。
TEST_F(ChecksumVerifyUtilTest, ListsAllFaultyBlocks) {
    std::vector<int64_t> expected = {0x1111, 0x2222, 0x3333, 0x4444};
    std::vector<int64_t> actual = {0xAAAA, 0x2222, 0xBBBB, 0x4444}; // #0, #2 错
    auto r = VerifyBatchChecksums(expected, actual);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 2u);
    EXPECT_EQ(r.faulty_indices[0], 0u);
    EXPECT_EQ(r.faulty_indices[1], 2u);
}

// zero 是合法 checksum，presence=true 时必须参与比较。
TEST_F(ChecksumVerifyUtilTest, PresentZeroIsCompared) {
    std::vector<int64_t> expected = {0x1111, 0, 0x3333};
    std::vector<int64_t> actual = {0x1111, 0xDEADBEEF, 0x3333};
    auto r = VerifyBatchChecksums(expected, actual);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 1u);
    EXPECT_EQ(r.faulty_indices[0], 1u);
}

// expected presence=false 表示 legacy/no-checksum，对应位置跳过。
TEST_F(ChecksumVerifyUtilTest, MissingExpectedValuesAreSkipped) {
    std::vector<int64_t> expected = {0x11, 0, 0x33};
    std::vector<int64_t> actual = {0x11, 0x22, 0x33};
    std::vector<bool> expected_present = {true, false, true};
    auto r = VerifyBatchChecksums(expected, actual, ChecksumValidationStage::CVS_READ_OUTPUT, expected_present);
    EXPECT_FALSE(r.mismatch);
    EXPECT_EQ(r.stage, ChecksumValidationStage::CVS_READ_OUTPUT);
}

// 用户有可信值但 meta 返回缺失，属于 META_ROUND_TRIP mismatch。
TEST_F(ChecksumVerifyUtilTest, MissingActualValueIsMismatch) {
    std::vector<int64_t> expected = {0x11, 0, 0x33};
    std::vector<int64_t> actual = expected;
    std::vector<bool> actual_present = {true, false, true};
    auto r = VerifyBatchChecksums(expected, actual, ChecksumValidationStage::CVS_META_ROUND_TRIP, {}, actual_present);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 1u);
    EXPECT_EQ(r.faulty_indices[0], 1u);
    EXPECT_EQ(r.stage, ChecksumValidationStage::CVS_META_ROUND_TRIP);
}

// Size mismatch (上层 bug): mismatch=true 且 faulty_indices 空 -> 上层走 size 错误日志。
TEST_F(ChecksumVerifyUtilTest, SizeMismatchReturnsMismatchWithoutIndices) {
    std::vector<int64_t> expected = {0x1111, 0x2222};
    std::vector<int64_t> actual = {0x1111};
    auto r = VerifyBatchChecksums(expected, actual);
    EXPECT_TRUE(r.mismatch);
    EXPECT_TRUE(r.faulty_indices.empty());
}

TEST_F(ChecksumVerifyUtilTest, PresenceSizeMismatchReturnsMismatchWithoutIndices) {
    std::vector<int64_t> expected = {0x1111, 0x2222};
    std::vector<int64_t> actual = expected;
    auto r = VerifyBatchChecksums(expected, actual, ChecksumValidationStage::CVS_READ_OUTPUT, std::vector<bool>{true});
    EXPECT_TRUE(r.mismatch);
    EXPECT_TRUE(r.faulty_indices.empty());
}

// Block swap (读串): expected=[A,B], actual=[B,A]. 必须识别并回填两个 faulty index。
TEST_F(ChecksumVerifyUtilTest, CatchesBlockSwap) {
    std::vector<int64_t> expected = {0xAAAA, 0xBBBB};
    std::vector<int64_t> actual = {0xBBBB, 0xAAAA};
    auto r = VerifyBatchChecksums(expected, actual);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 2u);
    EXPECT_EQ(r.faulty_indices[0], 0u);
    EXPECT_EQ(r.faulty_indices[1], 1u);
}

// Same-delta 成对突变：每个 block 都被同一 delta 改写。
// 逐块比对不能让这种 batch 通过。
TEST_F(ChecksumVerifyUtilTest, DetectsSameDeltaPairedMutation) {
    constexpr int64_t kDelta = 0x0F0F0F0F0F0F0F0FLL;
    std::vector<int64_t> expected = {0xAAAA, 0xBBBB};
    std::vector<int64_t> actual = {0xAAAA ^ kDelta, 0xBBBB ^ kDelta};
    auto r = VerifyBatchChecksums(expected, actual);
    ASSERT_TRUE(r.mismatch);
    EXPECT_EQ(r.faulty_indices.size(), 2u);
}

// High-bit 成对突变：逐块比对不能接受这种 batch。
TEST_F(ChecksumVerifyUtilTest, DetectsHighBitPairedMutation) {
    constexpr int64_t kHighBit = static_cast<int64_t>(0x8000000000000000ULL);
    std::vector<int64_t> expected = {0x1111, 0x2222, 0x3333};
    std::vector<int64_t> actual = {0x1111 ^ kHighBit, 0x2222 ^ kHighBit, 0x3333};

    auto r = VerifyBatchChecksums(expected, actual);
    ASSERT_TRUE(r.mismatch);
    ASSERT_EQ(r.faulty_indices.size(), 2u);
    EXPECT_EQ(r.faulty_indices[0], 0u);
    EXPECT_EQ(r.faulty_indices[1], 1u);
}
