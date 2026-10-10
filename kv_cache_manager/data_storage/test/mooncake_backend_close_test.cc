#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/mooncake_backend.h"

namespace {
int destroy_count = 0;
client_t destroyed_client = nullptr;
std::string created_hostname;
std::string created_device;
std::string created_protocol;
std::mutex status_mutex;
std::condition_variable status_cv;
bool release_status = false;
int status_calls = 0;
int active_status_calls = 0;
int max_active_status_calls = 0;

void ResetStatus() {
    std::lock_guard<std::mutex> guard(status_mutex);
    release_status = false;
    status_calls = active_status_calls = max_active_status_calls = 0;
}

bool WaitForStatus() {
    std::unique_lock<std::mutex> guard(status_mutex);
    return status_cv.wait_for(guard, std::chrono::seconds(5), [] { return status_calls > 0; });
}

void ReleaseStatus() {
    std::lock_guard<std::mutex> guard(status_mutex);
    release_status = true;
    status_cv.notify_all();
}
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

extern "C" ErrorCode_t __wrap_mooncake_client_get_store_status(client_t, MooncakeStoreStatus_t *status) {
    std::unique_lock<std::mutex> guard(status_mutex);
    ++status_calls;
    ++active_status_calls;
    max_active_status_calls = std::max(max_active_status_calls, active_status_calls);
    status_cv.notify_all();
    status_cv.wait_for(guard, std::chrono::seconds(5), [] { return release_status; });
    --active_status_calls;
    *status = {};
    status->healthy = true;
    status->used_ratio = 0.25;
    return MOONCAKE_ERROR_OK;
}

TEST(MooncakeBackendCloseTest, MetricsQueriesSerializeSharedHttpClient) {
    ResetStatus();
    int identity = 0;
    kv_cache_manager::MooncakeBackend backend(nullptr);
    backend.client_ = &identity;
    backend.SetOpen(true);
    auto first = std::async(std::launch::async, [&] { return backend.GetStorageUsageRatio("first"); });
    EXPECT_TRUE(WaitForStatus());
    auto second = std::async(std::launch::async, [&] { return backend.GetStorageUsageRatio("second"); });
    {
        std::unique_lock<std::mutex> guard(status_mutex);
        EXPECT_FALSE(status_cv.wait_for(guard, std::chrono::milliseconds(100), [] { return status_calls > 1; }));
    }
    ReleaseStatus();
    EXPECT_DOUBLE_EQ(0.25, first.get());
    EXPECT_DOUBLE_EQ(0.25, second.get());
    EXPECT_EQ(2, status_calls);
    EXPECT_EQ(1, max_active_status_calls);
}

TEST(MooncakeBackendCloseTest, CloseWaitsForMetricsBeforeDestroyingClient) {
    ResetStatus();
    destroy_count = 0;
    int identity = 0;
    kv_cache_manager::MooncakeBackend backend(nullptr);
    backend.client_ = &identity;
    backend.SetOpen(true);
    auto metrics = std::async(std::launch::async, [&] { return backend.GetStorageUsageRatio("metrics"); });
    EXPECT_TRUE(WaitForStatus());
    auto close = std::async(std::launch::async, [&] { return backend.Close(); });
    EXPECT_EQ(std::future_status::timeout, close.wait_for(std::chrono::milliseconds(100)));
    ReleaseStatus();
    EXPECT_DOUBLE_EQ(0.25, metrics.get());
    EXPECT_EQ(kv_cache_manager::EC_OK, close.get());
    EXPECT_EQ(1, destroy_count);
    EXPECT_DOUBLE_EQ(0.0, backend.GetStorageUsageRatio("closed"));
    EXPECT_EQ(1, status_calls);
    EXPECT_EQ(kv_cache_manager::EC_OK, backend.Close());
    EXPECT_EQ(1, destroy_count);
}

TEST(MooncakeBackendCloseTest, ConcurrentCloseReleasesClientOnce) {
    destroy_count = 0;
    int identity = 0;
    kv_cache_manager::MooncakeBackend backend(nullptr);
    backend.client_ = &identity;
    backend.SetOpen(true);
    auto first = std::async(std::launch::async, [&] { return backend.Close(); });
    auto second = std::async(std::launch::async, [&] { return backend.Close(); });
    EXPECT_EQ(kv_cache_manager::EC_OK, first.get());
    EXPECT_EQ(kv_cache_manager::EC_OK, second.get());
    EXPECT_EQ(1, destroy_count);
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
