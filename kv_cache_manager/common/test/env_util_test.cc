#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class EnvUtilTest : public TESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(EnvUtilTest, TestSimple) {
    ScopedEnv env1("TEST_ENV_VAR", "1");
    ASSERT_EQ(1, EnvUtil::GetEnv("TEST_ENV_VAR", -1));
}
} // namespace kv_cache_manager
