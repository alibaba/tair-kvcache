#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"

using namespace kv_cache_manager;

class RegistryManagerTest : public TESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(RegistryManagerTest, TestSimple) {
    // RegistryManager requires a valid storage backend; basic construction test only.
    RegistryManager registry_manager("fake_uri", nullptr);
}
