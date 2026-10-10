#include <array>
#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/sdk/mooncake_sdk.h"

namespace {
int identity = 0;
int destroy_count = 0;
int fail_registration = -1;
std::vector<std::pair<void *, size_t>> registrations;
}

extern "C" client_t __wrap_mooncake_client_create(const char *, const char *, const char *, const char *, const char *) {
    return &identity;
}
extern "C" void __wrap_mooncake_client_destroy(client_t) { ++destroy_count; }
extern "C" ErrorCode_t __wrap_mooncake_client_register_local_memory(client_t, void *base, size_t size,
                                                                  const char *, bool, bool) {
    registrations.emplace_back(base, size);
    return static_cast<int>(registrations.size()) == fail_registration ? MOONCAKE_ERROR_RPC_FAIL : MOONCAKE_ERROR_OK;
}

class MooncakeRegistrationTest : public testing::Test {
protected:
    void SetUp() override {
        destroy_count = 0;
        fail_registration = -1;
        registrations.clear();
        config = std::make_shared<kv_cache_manager::MooncakeSdkConfig>();
        config->set_local_mem_ptr(primary.data());
        config->set_local_buffer_size(primary.size());
        config->set_self_location_spec_name("tp0_F0");
        config->set_spec_byte_sizes_per_block({{"tp0_F0", 32}});
        auto spec = std::make_shared<kv_cache_manager::MooncakeStorageSpec>();
        spec->set_protocol("rdma");
        storage = std::make_shared<kv_cache_manager::StorageConfig>(
            kv_cache_manager::DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, "test", spec);
    }
    std::map<std::string, uint64_t> Span(std::array<char, 32> &pool) {
        return {{"base", reinterpret_cast<uintptr_t>(pool.data())}, {"size", pool.size()}};
    }
    std::array<char, 32> primary{}, hbm{}, indexer{};
    std::shared_ptr<kv_cache_manager::MooncakeSdkConfig> config;
    std::shared_ptr<kv_cache_manager::StorageConfig> storage;
};

TEST_F(MooncakeRegistrationTest, RegistersPrimaryAndBothAdditionalPools) {
    config->set_additional_local_memory_spans({Span(hbm), Span(indexer)});
    {
        kv_cache_manager::MooncakeSdk sdk;
        ASSERT_EQ(kv_cache_manager::ER_OK, sdk.Init(config, storage));
        ASSERT_EQ(3, registrations.size());
        EXPECT_EQ(primary.data(), registrations[0].first);
        EXPECT_EQ(hbm.data(), registrations[1].first);
        EXPECT_EQ(indexer.data(), registrations[2].first);
        EXPECT_EQ(kv_cache_manager::ER_OK, sdk.Close());
        EXPECT_EQ(kv_cache_manager::ER_OK, sdk.Close());
    }
    EXPECT_EQ(1, destroy_count);
}

TEST_F(MooncakeRegistrationTest, PartialRegistrationFailureDestroysClientOnce) {
    config->set_additional_local_memory_spans({Span(hbm), Span(indexer)});
    fail_registration = 2;
    {
        kv_cache_manager::MooncakeSdk sdk;
        EXPECT_EQ(kv_cache_manager::ER_SDKINIT_ERROR, sdk.Init(config, storage));
        EXPECT_EQ(2, registrations.size());
        EXPECT_EQ(1, destroy_count);
    }
    EXPECT_EQ(1, destroy_count);
}

TEST_F(MooncakeRegistrationTest, RejectsOverlapWithPrimaryBeforeCreatingClient) {
    config->set_additional_local_memory_spans({Span(primary)});
    kv_cache_manager::MooncakeSdk sdk;
    EXPECT_EQ(kv_cache_manager::ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(config, storage));
    EXPECT_TRUE(registrations.empty());
    EXPECT_EQ(0, destroy_count);
}

TEST_F(MooncakeRegistrationTest, LegacySinglePoolRegistersOnce) {
    kv_cache_manager::MooncakeSdk sdk;
    EXPECT_EQ(kv_cache_manager::ER_OK, sdk.Init(config, storage));
    EXPECT_EQ(1, registrations.size());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
