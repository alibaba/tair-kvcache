#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/sdk/local_file_sdk.h"
#include "kv_cache_manager/common/unittest.h"
#ifdef USING_CUDA
#include <cuda_runtime.h>
#endif

using namespace kv_cache_manager;

class LocalFileSdkTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        sdk_backend_config_ = std::make_shared<NfsSdkConfig>();
        ;
        sdk_backend_config_->set_byte_size_per_block(1024);
    }
    void TearDown() override {}

private:
    std::string root_path_;
    std::shared_ptr<NfsSdkConfig> sdk_backend_config_;
};

TEST_F(LocalFileSdkTest, TestInit) {
    LocalFileSdk sdk;
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(nullptr, nullptr));
    sdk_backend_config_->set_byte_size_per_block(-1);
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(sdk_backend_config_, nullptr));
    sdk_backend_config_->set_byte_size_per_block(1024);
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
}

TEST_F(LocalFileSdkTest, TestPutGetWithCpu) {
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
    // put
    DataStorageUri uri("file://" + root_path_ + "/local_file/test.txt");
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "1024");
    DataStorageUri invalid_uri = uri;
    invalid_uri.SetParam("size", "1023");
    BlockBuffer buf;

    const char *test_data = "this is local file test";
    size_t len1 = strlen(test_data);
    const char *test_data_2 = "and this is another local file test";
    size_t len2 = strlen(test_data_2);

    auto put_buffer = malloc(1024 * 1024);
    std::memcpy(put_buffer, test_data, len1);
    std::memcpy(static_cast<char *>(put_buffer) + len1, test_data_2, len2);

    Iov iov1;
    iov1.base = put_buffer;
    iov1.size = len1;
    iov1.type = MemoryType::CPU;
    iov1.ignore = false;
    buf.iovs.push_back(iov1);

    Iov iov2;
    iov2.base = static_cast<char *>(put_buffer) + iov1.size; // pointer arithmetic ok
    iov2.size = len2;
    iov2.type = MemoryType::CPU;
    iov2.ignore = false;
    buf.iovs.push_back(iov2);

    const std::vector<DataStorageUri> &remote_uris = {uri};
    const std::vector<DataStorageUri> &invalid_remote_uris = {invalid_uri};
    BlockBuffers local_buffers = {buf};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_SDKWRITE_ERROR, sdk.Put(invalid_remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(actual_remote_uris->size(), 1);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), uri.ToUriString());
    free(put_buffer);

    // get
    auto get_buffer = malloc(1024 * 1024);
    size_t offset = 0;
    for (auto &iov : local_buffers[0].iovs) {
        iov.base = static_cast<char *>(get_buffer) + offset;
        offset += iov.size;
    }
    ASSERT_EQ(ER_SDKREAD_ERROR, sdk.Get(invalid_remote_uris, local_buffers));
    ASSERT_EQ(ER_OK, sdk.Get(remote_uris, local_buffers));
    auto &iov1_res = local_buffers[0].iovs[0];
    ASSERT_EQ(std::memcmp(iov1_res.base, test_data, iov1_res.size), 0);
    auto &iov2_res = local_buffers[0].iovs[1];
    ASSERT_EQ(std::memcmp(iov2_res.base, test_data_2, iov2_res.size), 0);
    free(get_buffer);
}

TEST_F(LocalFileSdkTest, TestPutGetWithGpu) {
#ifdef USING_CUDA
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
    // put
    DataStorageUri uri("file://" + root_path_ + "/local_file/test_gpu.txt");
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "1024");
    DataStorageUri invalid_uri = uri;
    invalid_uri.SetParam("size", "1023");

    BlockBuffer buf;
    const char *test_data = "this is local file test";
    size_t len1 = strlen(test_data);
    const char *test_data_2 = "and this is another local file test";
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

    Iov iov2;
    iov2.base = static_cast<char *>(gpu_put_buffer) + len1;
    iov2.size = len2;
    iov2.type = MemoryType::GPU;
    iov2.ignore = false;
    buf.iovs.push_back(iov2);

    const std::vector<DataStorageUri> &remote_uris = {uri};
    const std::vector<DataStorageUri> &invalid_remote_uris = {invalid_uri};
    BlockBuffers local_buffers = {buf};

    // put
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_SDKWRITE_ERROR, sdk.Put(invalid_remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(actual_remote_uris->size(), 1);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), uri.ToUriString());

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

    ASSERT_EQ(ER_SDKREAD_ERROR, sdk.Get(invalid_remote_uris, local_buffers));
    ASSERT_EQ(ER_OK, sdk.Get(remote_uris, local_buffers));

    void *host_get_buffer = malloc(len1 + len2);
    ASSERT_EQ(cudaMemcpy(host_get_buffer, gpu_get_buffer, len1 + len2, cudaMemcpyDeviceToHost), cudaSuccess);

    auto &iov1_res = local_buffers[0].iovs[0];
    ASSERT_EQ(std::memcmp(host_get_buffer, test_data, iov1_res.size), 0);
    auto &iov2_res = local_buffers[0].iovs[1];
    ASSERT_EQ(std::memcmp(static_cast<char *>(host_get_buffer) + len1, test_data_2, iov2_res.size), 0);

    free(host_get_buffer);
    cudaFree(gpu_get_buffer);
#else
    GTEST_SKIP() << "CUDA not enabled, skipping GPU buffer test";
#endif
}
