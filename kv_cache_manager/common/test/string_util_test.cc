#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class StringUtilTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// --- ParseBucketBoundaries ---

TEST_F(StringUtilTest, ParseEmptyStringReturnsEmpty) {
    auto result = StringUtil::ParseBucketBoundaries("");
    EXPECT_TRUE(result.empty());
}

TEST_F(StringUtilTest, ParseValidBoundaries) {
    auto result = StringUtil::ParseBucketBoundaries("1,5,30,60,300,3600");
    ASSERT_EQ(result.size(), 6);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 30.0);
    EXPECT_DOUBLE_EQ(result[3], 60.0);
    EXPECT_DOUBLE_EQ(result[4], 300.0);
    EXPECT_DOUBLE_EQ(result[5], 3600.0);
}

TEST_F(StringUtilTest, ParseWithSpaces) {
    auto result = StringUtil::ParseBucketBoundaries(" 1 , 5 , 30 ");
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 30.0);
}

TEST_F(StringUtilTest, ParseNonAscendingRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("5,1,30").empty());
}

TEST_F(StringUtilTest, ParseDuplicateRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("1,5,5,30").empty());
}

TEST_F(StringUtilTest, ParseNegativeRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("-1,5,30").empty());
}

TEST_F(StringUtilTest, ParseZeroRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("0,5,30").empty());
}

TEST_F(StringUtilTest, ParseTrailingCharsRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("1s,5,30").empty());
}

TEST_F(StringUtilTest, ParseEmptyTokenRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("1,,5").empty());
}

TEST_F(StringUtilTest, ParseLeadingCommaRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries(",1,5").empty());
}

TEST_F(StringUtilTest, ParseTrailingCommaRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("1,5,").empty());
}

TEST_F(StringUtilTest, ParseNonNumericRejected) {
    EXPECT_TRUE(StringUtil::ParseBucketBoundaries("abc,5,30").empty());
}

TEST_F(StringUtilTest, ParseFractionalBoundaries) {
    auto result = StringUtil::ParseBucketBoundaries("0.5,1.5,5.0");
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 0.5);
    EXPECT_DOUBLE_EQ(result[1], 1.5);
    EXPECT_DOUBLE_EQ(result[2], 5.0);
}

} // namespace kv_cache_manager
