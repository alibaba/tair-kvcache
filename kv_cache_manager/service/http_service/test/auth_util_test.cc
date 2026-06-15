#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/http_service/auth/auth_util.h"

using namespace kv_cache_manager;

class AuthUtilTest : public TESTBASE {};

TEST_F(AuthUtilTest, ConstantTimeEqualsBasic) {
    ASSERT_TRUE(AuthUtil::ConstantTimeEquals("", ""));
    ASSERT_TRUE(AuthUtil::ConstantTimeEquals("abc", "abc"));
    ASSERT_FALSE(AuthUtil::ConstantTimeEquals("abc", "abd"));
}

TEST_F(AuthUtilTest, ConstantTimeEqualsLengthMismatch) {
    ASSERT_FALSE(AuthUtil::ConstantTimeEquals("abc", "abcd"));
    ASSERT_FALSE(AuthUtil::ConstantTimeEquals("abcd", "abc"));
    ASSERT_FALSE(AuthUtil::ConstantTimeEquals("", "x"));
}

TEST_F(AuthUtilTest, ConstantTimeEqualsBinarySafe) {
    std::string a("ab\0cd", 5);
    std::string b("ab\0cd", 5);
    std::string c("ab\0ce", 5);
    ASSERT_TRUE(AuthUtil::ConstantTimeEquals(a, b));
    ASSERT_FALSE(AuthUtil::ConstantTimeEquals(a, c));
}

TEST_F(AuthUtilTest, ICaseEqualsAscii) {
    ASSERT_TRUE(AuthUtil::ICaseEqualsAscii("Bearer", "bearer"));
    ASSERT_TRUE(AuthUtil::ICaseEqualsAscii("BEARER", "bearer"));
    ASSERT_TRUE(AuthUtil::ICaseEqualsAscii("BeArEr", "bEaReR"));
    ASSERT_FALSE(AuthUtil::ICaseEqualsAscii("Bearer", "Basic"));
    ASSERT_FALSE(AuthUtil::ICaseEqualsAscii("Bearer", "Bearers"));
    ASSERT_FALSE(AuthUtil::ICaseEqualsAscii("", "x"));
    ASSERT_TRUE(AuthUtil::ICaseEqualsAscii("", ""));
}
