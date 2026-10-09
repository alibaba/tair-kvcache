#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/mooncake_backend.h"

namespace {
int destroy_count = 0;
client_t destroyed_client = nullptr;
} // namespace

extern "C" void __wrap_mooncake_client_destroy(client_t client) {
    ++destroy_count;
    destroyed_client = client;
}

TEST(MooncakeBackendCloseTest, ExplicitCloseThenDestructionReleasesClientOnce) {
    int client_identity = 0;
    destroy_count = 0;
    destroyed_client = nullptr;
    {
        kv_cache_manager::MooncakeBackend backend(nullptr);
        backend.client_ = &client_identity;
        backend.SetOpen(true);
        backend.SetAvailable(true);
        ASSERT_EQ(kv_cache_manager::EC_OK, backend.Close());
        EXPECT_FALSE(backend.IsOpen());
        EXPECT_FALSE(backend.Available());
        EXPECT_EQ(1, destroy_count);
        EXPECT_EQ(&client_identity, destroyed_client);
        EXPECT_EQ(kv_cache_manager::EC_OK, backend.Close());
        EXPECT_EQ(1, destroy_count);
    }
    EXPECT_EQ(1, destroy_count);
}

TEST(MooncakeBackendCloseTest, UnopenedBackendDoesNotDestroyClient) {
    destroy_count = 0;
    {
        kv_cache_manager::MooncakeBackend backend(nullptr);
        EXPECT_EQ(kv_cache_manager::EC_OK, backend.Close());
    }
    EXPECT_EQ(0, destroy_count);
}
