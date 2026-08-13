#include <gtest/gtest.h>
#include <typeinfo>

#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

TEST(SdkFactoryTest, TairMempoolDramAndSsdCreateSameSdkType) {
#ifdef ENABLE_TAIR_MEMPOOL
    const auto dram_sdk = SdkFactory::CreateSdkInstance(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
    const auto ssd_sdk = SdkFactory::CreateSdkInstance(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD);

    ASSERT_NE(nullptr, dram_sdk);
    ASSERT_NE(nullptr, ssd_sdk);
    EXPECT_EQ(SdkType::TAIR_MEMPOOL, dram_sdk->Type());
    EXPECT_EQ(SdkType::TAIR_MEMPOOL, ssd_sdk->Type());
    EXPECT_EQ(typeid(*dram_sdk), typeid(*ssd_sdk));
#else
    GTEST_SKIP() << "TairMempool SDK is disabled";
#endif
}
