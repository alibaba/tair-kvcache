#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/sdk/deadline_util.h"
#include "kv_cache_manager/client/src/internal/sdk/mooncake_sdk.h"
#include "kv_cache_manager/common/unittest.h"
#ifdef USING_CUDA
#include <cuda_runtime.h>
#endif

using namespace kv_cache_manager;

class MooncakeSdkTest : public TESTBASE {
public:
    /*
     * 测试说明：
     * 1. 编译指令：bazel test ... --config=client_with_cuda,目前只测了tcp的情况
     * 2.
     * mooncake单机部署：https://alidocs.dingtalk.com/i/nodes/20eMKjyp810mMdK4HQ06BE9qJxAZB1Gv?cid=319308789%3A5750508195&utm_source=im&utm_scene=team_space&iframeQuery=utm_medium%3Dim_card%26utm_source%3Dim&utm_medium=im_card&corpId=dingd8e1123006514592
     */
    void SetUp() override {
        sdk_backend_config_ = std::make_shared<MooncakeSdkConfig>();
        sdk_backend_config_->set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
        sdk_backend_config_->set_local_buffer_size(128 * 1024 * 1024);
        sdk_backend_config_->set_location("*");
        sdk_backend_config_->set_spec_byte_sizes_per_block({{"default", 1024}});
        sdk_backend_config_->set_self_location_spec_name("tp0_F0");

        storage_config_ = std::make_shared<StorageConfig>();
        auto mooncake_spec = std::make_shared<MooncakeStorageSpec>();
        mooncake_spec->set_local_hostname("localhost");
        mooncake_spec->set_metadata_connstring("http://0.0.0.0:8090/metadata"); // Metadata 地址
        mooncake_spec->set_protocol("tcp");
        mooncake_spec->set_master_server_entry("localhost:50051"); // Master Server 地址
        storage_config_->set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
        storage_config_->set_storage_spec(mooncake_spec);
    }

    void TearDown() override {}

private:
    std::shared_ptr<MooncakeSdkConfig> sdk_backend_config_;
    std::shared_ptr<StorageConfig> storage_config_;
};

TEST_F(MooncakeSdkTest, TestInit) {
    MooncakeSdk sdk;
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(nullptr, nullptr));
    ASSERT_EQ(ER_INVALID_STORAGE_CONFIG, sdk.Init(sdk_backend_config_, nullptr));
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(nullptr, storage_config_));
    // invalid backend config
    auto nfs_sdk_backend_config = std::make_shared<NfsSdkConfig>();
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(nfs_sdk_backend_config, storage_config_));
    // invalid storage config
    auto empty_storage_config = std::make_shared<StorageConfig>();
    ASSERT_EQ(ER_INVALID_STORAGE_CONFIG, sdk.Init(sdk_backend_config_, empty_storage_config));
    // invalid spec_byte_sizes_per_block
    sdk_backend_config_->set_spec_byte_sizes_per_block({});
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(sdk_backend_config_, storage_config_));
    sdk_backend_config_->set_spec_byte_sizes_per_block({{"default", 1024}});
    // empty local buffer
    ASSERT_EQ(nullptr, sdk_backend_config_->local_mem_ptr());
    sdk_backend_config_->set_self_location_spec_name("tp1_F0");
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, storage_config_));
    ASSERT_EQ(ER_OK, sdk.Close());
    // success
    auto client_buffer_allocator = std::shared_ptr<char[]>(new char[128 * 1024 * 1024]);
    sdk_backend_config_->set_local_mem_ptr(client_buffer_allocator.get());
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, storage_config_));
    sdk_backend_config_->set_local_mem_ptr(nullptr);
}

TEST_F(MooncakeSdkTest, TestPutGetWithCpu) {
    MooncakeSdk sdk;
    auto client_buffer_allocator = std::shared_ptr<char[]>(new char[128 * 1024 * 1024]);
    sdk_backend_config_->set_local_mem_ptr(client_buffer_allocator.get());
    sdk_backend_config_->set_self_location_spec_name("tp2_F0");
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, storage_config_));
    std::string key = "testkey";
    DataStorageUri uri("mooncake://na61_mc_bucket_01?key=" + key);
    BlockBuffer buf;
    BlockBuffer invalid_buf;
    const char *test_data = "this is mooncake test";
    size_t len1 = strlen(test_data);
    const char *test_data_2 = "and this is another mooncake test";
    size_t len2 = strlen(test_data_2);

    auto put_buffer = malloc(1024 * 1024); // 向putbuffer中写入数据
    std::memcpy(put_buffer, test_data, len1);
    std::memcpy(static_cast<char *>(put_buffer) + len1, test_data_2, len2);

    Iov iov1; // "this is mooncake test"
    iov1.base = put_buffer;
    iov1.size = len1;
    iov1.type = MemoryType::CPU;
    iov1.ignore = false;
    buf.iovs.push_back(iov1);
    invalid_buf.iovs.push_back(iov1);

    Iov iov2;                                                // "and this is another mooncake test";
    iov2.base = static_cast<char *>(put_buffer) + iov1.size; // pointer arithmetic ok
    iov2.size = len2;
    iov2.type = MemoryType::CPU;
    iov2.ignore = false;
    buf.iovs.push_back(iov2);
    iov2.size += 1024;
    invalid_buf.iovs.push_back(iov2);

    const std::vector<DataStorageUri> &remote_uris = {uri};
    BlockBuffers local_buffers = {buf};
    BlockBuffers invalid_local_buffers = {invalid_buf};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_INVALID_PARAMS, sdk.Put(remote_uris, invalid_local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    free(put_buffer);

    // get
    auto get_buffer = malloc(1024 * 1024);
    size_t offset = 0;
    for (auto &iov : local_buffers[0].iovs) {
        iov.base = static_cast<char *>(get_buffer) + offset;
        offset += iov.size;
    }
    ASSERT_EQ(ER_INVALID_PARAMS, sdk.Get(remote_uris, invalid_local_buffers, /*deadline_ms=*/0));
    ASSERT_EQ(ER_OK, sdk.Get(remote_uris, local_buffers, /*deadline_ms=*/0));
    auto &iov1_res = local_buffers[0].iovs[0];
    ASSERT_EQ(std::memcmp(iov1_res.base, test_data, iov1_res.size), 0);
    auto &iov2_res = local_buffers[0].iovs[1];
    ASSERT_EQ(std::memcmp(iov2_res.base, test_data_2, iov2_res.size), 0);
    free(get_buffer);
}

TEST_F(MooncakeSdkTest, TestMultipleUriWithCpu) {
    MooncakeSdk sdk;
    auto client_buffer_allocator = std::shared_ptr<char[]>(new char[128 * 1024 * 1024]);
    sdk_backend_config_->set_local_mem_ptr(client_buffer_allocator.get());
    sdk_backend_config_->set_self_location_spec_name("tp3_F0");
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, storage_config_));
    // put
    DataStorageUri uri1("mooncake://na61_mc_bucket_01?key=test1");
    DataStorageUri uri2("mooncake://na61_mc_bucket_01?key=test2");

    BlockBuffers local_buffers(2);

    const char *test_data_1 = "this is mooncake test";
    size_t len1 = strlen(test_data_1);
    const char *test_data_2 = "and this is another mooncake test";
    size_t len2 = strlen(test_data_2);

    auto put_buffer_1 = malloc(1024 * 1024);
    std::memcpy(put_buffer_1, test_data_1, len1);
    auto put_buffer_2 = malloc(1024 * 1024);
    std::memcpy(put_buffer_2, test_data_2, len2);

    Iov iov1;
    iov1.base = put_buffer_1;
    iov1.size = len1;
    iov1.type = MemoryType::CPU;
    iov1.ignore = false;
    local_buffers[0].iovs.push_back(iov1);

    Iov iov2;
    iov2.base = put_buffer_2;
    iov2.size = len2;
    iov2.type = MemoryType::CPU;
    iov2.ignore = false;
    local_buffers[1].iovs.push_back(iov2);

    const std::vector<DataStorageUri> &remote_uris = {uri1, uri2};
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();

    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(actual_remote_uris->size(), 2);

    free(put_buffer_1);
    free(put_buffer_2);

    // get
    auto get_buffer_1 = malloc(1024 * 1024);
    auto get_buffer_2 = malloc(1024 * 1024);

    local_buffers[0].iovs[0].base = get_buffer_1;
    local_buffers[1].iovs[0].base = get_buffer_2;

    ASSERT_EQ(ER_OK, sdk.Get(*actual_remote_uris, local_buffers, /*deadline_ms=*/0));
    auto &iov1_res = local_buffers[0].iovs[0];
    ASSERT_EQ(std::memcmp(iov1_res.base, test_data_1, iov1_res.size), 0);
    auto &iov2_res = local_buffers[1].iovs[0];
    ASSERT_EQ(std::memcmp(iov2_res.base, test_data_2, iov2_res.size), 0);
    free(get_buffer_1);
    free(get_buffer_2);
}

TEST_F(MooncakeSdkTest, TestPutGetWithGpu) {
#ifdef USING_CUDA
    MooncakeSdk sdk;
    sdk_backend_config_->set_local_buffer_size(128 * 1024 * 1024);
    void *device_ptr = nullptr;
    cudaMalloc(&device_ptr, 128 * 1024 * 1024);
    sdk_backend_config_->set_local_mem_ptr(device_ptr);
    sdk_backend_config_->set_self_location_spec_name("tp4_F0");
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, storage_config_));
    // put
    std::string key = "testkey";
    DataStorageUri uri("mooncake://na61_mc_bucket_01?key=" + key);
    BlockBuffer buf;
    BlockBuffer invalid_buf;
    const char *test_data = "this is mooncake test";
    size_t len1 = strlen(test_data);
    const char *test_data_2 = "and this is another mooncake test";
    size_t len2 = strlen(test_data_2);

    void *host_put_buffer = malloc(len1 + len2);
    std::memcpy(static_cast<char *>(host_put_buffer), test_data, len1);
    std::memcpy(static_cast<char *>(host_put_buffer) + len1, test_data_2, len2);

    void *gpu_put_buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_put_buffer, len1 + len2), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gpu_put_buffer, host_put_buffer, len1 + len2, cudaMemcpyHostToDevice), cudaSuccess);

    Iov iov1;
    iov1.base = gpu_put_buffer;
    iov1.size = len1;
    iov1.type = MemoryType::GPU;
    iov1.ignore = false;
    buf.iovs.push_back(iov1);
    invalid_buf.iovs.push_back(iov1);

    Iov iov2;
    iov2.base = static_cast<char *>(gpu_put_buffer) + len1;
    iov2.size = len2;
    iov2.type = MemoryType::GPU;
    iov2.ignore = false;
    buf.iovs.push_back(iov2);
    iov2.size += 1024;
    invalid_buf.iovs.push_back(iov2);

    const std::vector<DataStorageUri> &remote_uris = {uri};
    BlockBuffers local_buffers = {buf};
    BlockBuffers invalid_local_buffers = {invalid_buf};

    // put
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_INVALID_PARAMS, sdk.Put(remote_uris, invalid_local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris, /*deadline_ms=*/0));

    free(host_put_buffer);
    cudaFree(gpu_put_buffer);

    // get
    void *gpu_get_buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_get_buffer, len1 + len2), cudaSuccess);

    size_t offset = 0;
    for (auto &iov : local_buffers[0].iovs) {
        iov.base = static_cast<char *>(gpu_get_buffer) + offset;
        offset += iov.size;
    }

    ASSERT_EQ(ER_INVALID_PARAMS, sdk.Get(remote_uris, invalid_local_buffers, /*deadline_ms=*/0));
    ASSERT_EQ(ER_OK, sdk.Get(remote_uris, local_buffers, /*deadline_ms=*/0));
    void *host_get_buffer = malloc(len1 + len2);
    ASSERT_EQ(cudaMemcpy(host_get_buffer, gpu_get_buffer, len1 + len2, cudaMemcpyDeviceToHost), cudaSuccess);

    auto &iov1_res = local_buffers[0].iovs[0];
    ASSERT_EQ(std::memcmp(host_get_buffer, test_data, iov1_res.size), 0);
    auto &iov2_res = local_buffers[0].iovs[1];
    ASSERT_EQ(std::memcmp(static_cast<char *>(host_get_buffer) + len1, test_data_2, iov2_res.size), 0);

    free(host_get_buffer);
    cudaFree(gpu_get_buffer);
    cudaFree(device_ptr);
#else
    GTEST_SKIP() << "CUDA not enabled, skipping GPU buffer test";
#endif
}

TEST_F(MooncakeSdkTest, TestMultipleUriWithGpu) {
#ifdef USING_CUDA
    MooncakeSdk sdk;
    sdk_backend_config_->set_local_buffer_size(128 * 1024 * 1024);
    void *device_ptr = nullptr;
    cudaMalloc(&device_ptr, 128 * 1024 * 1024);
    sdk_backend_config_->set_local_mem_ptr(device_ptr);
    sdk_backend_config_->set_self_location_spec_name("tp5_F0");
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, storage_config_));
    // put
    DataStorageUri uri1("mooncake://na61_mc_bucket_01?key=test1");
    DataStorageUri uri2("mooncake://na61_mc_bucket_01?key=test2");

    BlockBuffers local_buffers(2);

    const char *test_data_1 = "this is mooncake test";
    size_t len1 = strlen(test_data_1);
    const char *test_data_2 = "and this is another mooncake test";
    size_t len2 = strlen(test_data_2);

    void *host_put_buffer_1 = malloc(len1);
    void *host_put_buffer_2 = malloc(len2);
    std::memcpy(static_cast<char *>(host_put_buffer_1), test_data_1, len1);
    std::memcpy(static_cast<char *>(host_put_buffer_2), test_data_2, len2);

    void *gpu_put_buffer_1 = nullptr;
    void *gpu_put_buffer_2 = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_put_buffer_1, len1), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gpu_put_buffer_1, host_put_buffer_1, len1, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gpu_put_buffer_2, len2), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gpu_put_buffer_2, host_put_buffer_2, len2, cudaMemcpyHostToDevice), cudaSuccess);

    Iov iov1;
    iov1.base = gpu_put_buffer_1;
    iov1.size = len1;
    iov1.type = MemoryType::GPU;
    iov1.ignore = false;
    local_buffers[0].iovs.push_back(iov1);

    Iov iov2;
    iov2.base = gpu_put_buffer_2;
    iov2.size = len2;
    iov2.type = MemoryType::GPU;
    iov2.ignore = false;
    local_buffers[1].iovs.push_back(iov2);

    const std::vector<DataStorageUri> &remote_uris = {uri1, uri2};

    // put
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(actual_remote_uris->size(), 2);
    free(host_put_buffer_1);
    free(host_put_buffer_2);
    cudaFree(gpu_put_buffer_1);
    cudaFree(gpu_put_buffer_2);

    // get
    void *gpu_get_buffer_1 = nullptr;
    void *gpu_get_buffer_2 = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_get_buffer_1, len1), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gpu_get_buffer_2, len2), cudaSuccess);
    local_buffers[0].iovs[0].base = static_cast<char *>(gpu_get_buffer_1);
    local_buffers[1].iovs[0].base = static_cast<char *>(gpu_get_buffer_2);

    ASSERT_EQ(ER_OK, sdk.Get(*actual_remote_uris, local_buffers, /*deadline_ms=*/0));

    void *host_get_buffer_1 = malloc(len1);
    void *host_get_buffer_2 = malloc(len2);
    ASSERT_EQ(cudaMemcpy(host_get_buffer_1, gpu_get_buffer_1, len1, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_get_buffer_2, gpu_get_buffer_2, len2, cudaMemcpyDeviceToHost), cudaSuccess);

    auto &iov1_res = local_buffers[0].iovs[0];
    ASSERT_EQ(std::memcmp(host_get_buffer_1, test_data_1, iov1_res.size), 0);
    auto &iov2_res = local_buffers[1].iovs[0];
    ASSERT_EQ(std::memcmp(host_get_buffer_2, test_data_2, iov2_res.size), 0);

    free(host_get_buffer_1);
    free(host_get_buffer_2);
    cudaFree(gpu_get_buffer_1);
    cudaFree(gpu_get_buffer_2);
#else
    GTEST_SKIP() << "CUDA not enabled, skipping GPU buffer test";
#endif
}

// ===========================================================================
// 无 mooncake 服务也能跑的用例：逐 key 准入检查发生在 mooncake_client_get/put
// 之前，因此不需要真实服务。通过 -fno-access-control 直接注入 sdk_backend_config_
// 绕过 Init（client_ 保持 nullptr —— 若检查失效走到网络调用，nullptr 解引用会
// crash 而不是返回 ER_SDK_TIMEOUT，所以"返回超时码"本身就是"检查先于 I/O"的证明）。
// ===========================================================================

TEST_F(MooncakeSdkTest, TestGetSkipsIoWhenDeadlineExpired) {
    MooncakeSdk sdk;
    // 注入配置绕过 Init：不需要真实 mooncake 服务。
    sdk.sdk_backend_config_ = sdk_backend_config_;
    sdk.storage_config_ = storage_config_;

    const std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("mooncake://na61_mc_bucket_01?key=expired_get_key")};
    std::vector<char> storage(1024);
    BlockBuffers local_buffers(1);
    Iov iov;
    iov.base = storage.data();
    iov.size = storage.size();
    iov.type = MemoryType::CPU;
    iov.ignore = false;
    local_buffers[0].iovs.push_back(iov);

    // 已过期的 deadline：任何一次 I/O 都不允许发起。
    // 检查先于 mooncake_client_get，返回超时而不是 crash / ER_SDKREAD_ERROR。
    ASSERT_EQ(ER_SDK_TIMEOUT, sdk.Get(remote_uris, local_buffers, SteadyClockMs() - 1'000));
}

TEST_F(MooncakeSdkTest, TestPutSkipsIoWhenDeadlineExpired) {
    MooncakeSdk sdk;
    // 注入配置绕过 Init：不需要真实 mooncake 服务。
    sdk.sdk_backend_config_ = sdk_backend_config_;
    sdk.storage_config_ = storage_config_;

    const std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("mooncake://na61_mc_bucket_01?key=expired_put_key")};
    std::vector<char> storage(1024);
    BlockBuffers local_buffers(1);
    Iov iov;
    iov.base = storage.data();
    iov.size = storage.size();
    iov.type = MemoryType::CPU;
    iov.ignore = false;
    local_buffers[0].iovs.push_back(iov);
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();

    // 已过期的 deadline：任何一次 I/O 都不允许发起。
    // 检查先于 mooncake_client_put，返回超时而不是 crash / ER_SDKWRITE_ERROR。
    ASSERT_EQ(ER_SDK_TIMEOUT, sdk.Put(remote_uris, local_buffers, actual_remote_uris, SteadyClockMs() - 1'000));
}

TEST_F(MooncakeSdkTest, TestPutActualUrisSameOrder) {
    // 保序契约：Put 的 actual_remote_uris 由
    // 末尾的 Alloc 整体赋值（alloc_uris = remote_uris）填充，顺序与 remote_uris 天然
    // 一致。直接测 Alloc（protected，经 -fno-access-control 访问），不需要 mooncake 服务。
    MooncakeSdk sdk;
    const std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("mooncake://na61_mc_bucket_01?key=k0"),
        DataStorageUri("mooncake://na61_mc_bucket_01?key=k1"),
        DataStorageUri("mooncake://na61_mc_bucket_01?key=k2"),
    };
    std::vector<DataStorageUri> actual_uris;
    ASSERT_EQ(ER_OK, sdk.Alloc(remote_uris, actual_uris));
    // 逐一同序断言：actual_uris[i] 必须对应 remote_uris[i]（下标即 block 身份）。
    ASSERT_EQ(remote_uris.size(), actual_uris.size());
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        ASSERT_EQ(remote_uris[i].ToUriString(), actual_uris[i].ToUriString());
    }
}