#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/mooncake_backend.h"

namespace {
int destroy_count = 0;
client_t destroyed_client = nullptr;
std::string created_hostname;
std::string created_device;
std::string created_protocol;
} // namespace

extern "C" client_t __wrap_mooncake_client_create(const char *hostname,
                                                   const char *,
                                                   const char *protocol,
                                                   const char *device,
                                                   const char *) {
    created_hostname = hostname;
    created_device = device;
    created_protocol = protocol;
    return nullptr;
}

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

TEST(MooncakeBackendCloseTest, LocalDeviceOverridePreservesWorkerStorageSpec) {
    kv_cache_manager::ScopedEnv hostname("KVCM_MOONCAKE_LOCAL_HOSTNAME", "controller:50770");
    kv_cache_manager::ScopedEnv device("KVCM_MOONCAKE_RDMA_DEVICE", "erdma_0");
    auto spec = std::make_shared<kv_cache_manager::MooncakeStorageSpec>();
    spec->set_local_hostname("worker:50770");
    spec->set_rdma_device("mlx5_bond_0");
    spec->set_protocol("rdma");
    kv_cache_manager::StorageConfig config(kv_cache_manager::DataStorageType::DATA_STORAGE_TYPE_MOONCAKE,
                                          "storage",
                                          spec);
    kv_cache_manager::MooncakeBackend backend(nullptr);
    EXPECT_EQ(kv_cache_manager::EC_ERROR, backend.DoOpen(config, "test"));
    EXPECT_EQ(0, created_hostname.find("controller:50770_kvcm_"));
    EXPECT_EQ("erdma_0", created_device);
    EXPECT_EQ("rdma", created_protocol);
    EXPECT_EQ("worker:50770", backend.spec_.local_hostname());
    EXPECT_EQ("mlx5_bond_0", backend.spec_.rdma_device());
    EXPECT_EQ("rdma", backend.spec_.protocol());
}

TEST(MooncakeBackendCloseTest, EmptyDeviceOverrideAllowsLocalDiscovery) {
    kv_cache_manager::ScopedEnv device("KVCM_MOONCAKE_RDMA_DEVICE", "");
    auto spec = std::make_shared<kv_cache_manager::MooncakeStorageSpec>();
    spec->set_local_hostname("worker:50770");
    spec->set_rdma_device("mlx5_bond_0");
    spec->set_protocol("rdma");
    kv_cache_manager::StorageConfig config(kv_cache_manager::DataStorageType::DATA_STORAGE_TYPE_MOONCAKE,
                                          "storage",
                                          spec);
    kv_cache_manager::MooncakeBackend backend(nullptr);
    EXPECT_EQ(kv_cache_manager::EC_ERROR, backend.DoOpen(config, "test"));
    EXPECT_EQ(0, created_hostname.find("worker:50770_kvcm_"));
    EXPECT_TRUE(created_device.empty());
    EXPECT_EQ("rdma", created_protocol);
    EXPECT_EQ("mlx5_bond_0", backend.spec_.rdma_device());
}
