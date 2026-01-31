#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"

using namespace kv_cache_manager;

class RegistryManagerTest : public TESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(RegistryManagerTest, TestSimple) {
    RegistryManager registry_manager("fake_uri", nullptr);
    auto cache_config = registry_manager.GetCacheConfig("fake");
    ASSERT_FALSE(cache_config);
}
