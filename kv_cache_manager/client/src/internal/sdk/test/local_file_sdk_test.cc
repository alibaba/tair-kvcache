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
        sdk_backend_config_->set_spec_byte_sizes_per_block({{"default", 1024}});
    }
    void TearDown() override {}

private:
    std::string root_path_;
    std::shared_ptr<NfsSdkConfig> sdk_backend_config_;
};

TEST_F(LocalFileSdkTest, TestInit) {
    LocalFileSdk sdk;
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(nullptr, nullptr));
    sdk_backend_config_->set_spec_byte_sizes_per_block({});
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(sdk_backend_config_, nullptr));
    sdk_backend_config_->set_spec_byte_sizes_per_block({{"default", 1024}});
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

// 同 backend（同一次 Put 调用）内多个不同 path 交错出现、且 payload 各不相同。
// 修复前 SplitByPath 按 unordered_map 迭代顺序回填 actual_remote_uris，导致
// 返回顺序与输入顺序不一致；本测试固定"同序契约"：actual_remote_uris[i] 必须
// 对应 remote_uris[i]，且数据真正写入各自 path 对应的文件。
TEST_F(LocalFileSdkTest, TestPutMultiPathInterleavedOrdering) {
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));

    // 交错输入：fileA(blk0), fileB(blk0), fileA(blk1)
    DataStorageUri uri_a0("file://" + root_path_ + "/multi_path/fileA");
    uri_a0.SetParam("blkid", "0");
    uri_a0.SetParam("size", "1024");
    DataStorageUri uri_b0("file://" + root_path_ + "/multi_path/fileB");
    uri_b0.SetParam("blkid", "0");
    uri_b0.SetParam("size", "1024");
    DataStorageUri uri_a1("file://" + root_path_ + "/multi_path/fileA");
    uri_a1.SetParam("blkid", "1");
    uri_a1.SetParam("size", "1024");

    const char *payload_a0 = "payload for fileA block0";
    const char *payload_b0 = "payload for fileB block0";
    const char *payload_a1 = "payload for fileA block1";
    size_t len_a0 = strlen(payload_a0);
    size_t len_b0 = strlen(payload_b0);
    size_t len_a1 = strlen(payload_a1);

    // 三个 block 使用相互独立、内容各异的 buffer
    void *buf_a0 = malloc(1024);
    void *buf_b0 = malloc(1024);
    void *buf_a1 = malloc(1024);
    std::memcpy(buf_a0, payload_a0, len_a0);
    std::memcpy(buf_b0, payload_b0, len_b0);
    std::memcpy(buf_a1, payload_a1, len_a1);

    auto make_buffer = [](void *base, size_t size) {
        BlockBuffer bb;
        Iov iov;
        iov.base = base;
        iov.size = size;
        iov.type = MemoryType::CPU;
        iov.ignore = false;
        bb.iovs.push_back(iov);
        return bb;
    };

    std::vector<DataStorageUri> remote_uris = {uri_a0, uri_b0, uri_a1};
    BlockBuffers local_buffers = {make_buffer(buf_a0, len_a0), make_buffer(buf_b0, len_b0), make_buffer(buf_a1, len_a1)};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris));

    // 同序契约：返回顺序必须与输入一致
    ASSERT_EQ(actual_remote_uris->size(), remote_uris.size());
    EXPECT_EQ(actual_remote_uris->at(0).ToUriString(), uri_a0.ToUriString());
    EXPECT_EQ(actual_remote_uris->at(1).ToUriString(), uri_b0.ToUriString());
    EXPECT_EQ(actual_remote_uris->at(2).ToUriString(), uri_a1.ToUriString());

    // 用全新的 buffer 读回，验证数据确实写入了各自 path 的文件
    void *get_a0 = malloc(1024);
    void *get_b0 = malloc(1024);
    void *get_a1 = malloc(1024);
    BlockBuffers get_buffers = {make_buffer(get_a0, len_a0), make_buffer(get_b0, len_b0), make_buffer(get_a1, len_a1)};
    ASSERT_EQ(ER_OK, sdk.Get(*actual_remote_uris, get_buffers));
    EXPECT_EQ(std::memcmp(get_a0, payload_a0, len_a0), 0);
    EXPECT_EQ(std::memcmp(get_b0, payload_b0, len_b0), 0);
    EXPECT_EQ(std::memcmp(get_a1, payload_a1, len_a1), 0);

    free(buf_a0);
    free(buf_b0);
    free(buf_a1);
    free(get_a0);
    free(get_b0);
    free(get_a1);
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
