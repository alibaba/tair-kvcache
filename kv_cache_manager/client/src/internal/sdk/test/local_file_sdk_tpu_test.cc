#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/sdk/local_file_sdk.h"
#include "kv_cache_manager/common/unittest.h"
#ifdef USING_TPU
#include "kv_cache_manager/client/src/internal/sdk/tpu_client.h"
#endif

#include <cstring>
#include <vector>

using namespace kv_cache_manager;

class LocalFileSdkTpuTest : public TESTBASE {
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

// Test TPU client initialization
TEST_F(LocalFileSdkTpuTest, TestInit) {
#ifdef USING_TPU
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));

    // Verify TpuClient is properly initialized via -fno-access-control
    ASSERT_NE(sdk.tpu_client_.api_, nullptr) << "PJRT_Api should be initialized";
    ASSERT_NE(sdk.tpu_client_.client_, nullptr) << "PJRT_Client should be initialized";
    ASSERT_NE(sdk.tpu_client_.device_, nullptr) << "PJRT_Device should be initialized";
#else
    GTEST_SKIP() << "TPU not enabled, skipping TPU init test";
#endif
}

// Test TpuClient BufferFromHost and BufferToHost methods
TEST_F(LocalFileSdkTpuTest, TestTpuClientBufferMethods) {
#ifdef USING_TPU
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));

    // Test data
    const char* test_data = "TpuClient buffer test data content!";
    size_t data_size = strlen(test_data);

    // Test BufferFromHost: create a buffer from host data
    std::vector<char> host_src(data_size);
    std::memcpy(host_src.data(), test_data, data_size);

    PJRT_Buffer* tpu_buffer = nullptr;
    auto ec = sdk.tpu_client_.BufferFromHost(host_src.data(), data_size, tpu_buffer);
    ASSERT_EQ(ec, ER_OK) << "BufferFromHost failed";
    ASSERT_NE(tpu_buffer, nullptr);

    // Test BufferToHost: read buffer data back to host
    std::vector<char> host_dst(data_size, 0);
    ec = sdk.tpu_client_.BufferToHost(tpu_buffer, host_dst.data(), data_size);
    ASSERT_EQ(ec, ER_OK) << "BufferToHost failed";
    ASSERT_EQ(std::memcmp(host_dst.data(), test_data, data_size), 0)
        << "BufferToHost data mismatch";

    // Cleanup
    sdk.tpu_client_.DestroyBuffer(tpu_buffer);
#else
    GTEST_SKIP() << "TPU not enabled, skipping TpuClient buffer methods test";
#endif
}

// Test TPU DMA map/unmap for host memory registration
TEST_F(LocalFileSdkTpuTest, TestTpuDmaMapUnmap) {
#ifdef USING_TPU
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));

    // Allocate a page-aligned host buffer for DMA mapping
    size_t buf_size = 4096;
    void* host_buf = std::aligned_alloc(4096, buf_size);
    ASSERT_NE(host_buf, nullptr);
    std::memset(host_buf, 0xAB, buf_size);

    // Test DmaMap
    auto ec = sdk.tpu_client_.DmaMap(host_buf, buf_size);
    EXPECT_EQ(ec, ER_OK) << "DmaMap should succeed";

    // Test DmaUnmap
    ec = sdk.tpu_client_.DmaUnmap(host_buf);
    EXPECT_EQ(ec, ER_OK) << "DmaUnmap should succeed";

    free(host_buf);
#else
    GTEST_SKIP() << "TPU not enabled, skipping TPU DMA map test";
#endif
}

// Test SDK Put/Get with TPU memory (full round-trip)
TEST_F(LocalFileSdkTpuTest, TestSdkPutGetWithTpuMemory) {
#ifdef USING_TPU
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));

    // Prepare test data
    const char* test_data = "this is tpu local file test data!!";
    size_t len1 = strlen(test_data);
    const char* test_data_2 = "second tpu iov test data content!!";
    size_t len2 = strlen(test_data_2);
    size_t total_size = len1 + len2;

    // Allocate host buffer with test data
    std::vector<char> host_put_data(total_size);
    std::memcpy(host_put_data.data(), test_data, len1);
    std::memcpy(host_put_data.data() + len1, test_data_2, len2);

    // Create TPU buffers for Put: upload host data to TPU
    PJRT_Buffer* tpu_buf1 = nullptr;
    auto ec = sdk.tpu_client_.BufferFromHost(host_put_data.data(), len1, tpu_buf1);
    ASSERT_EQ(ec, ER_OK) << "BufferFromHost for iov1 failed";
    ASSERT_NE(tpu_buf1, nullptr);

    PJRT_Buffer* tpu_buf2 = nullptr;
    ec = sdk.tpu_client_.BufferFromHost(host_put_data.data() + len1, len2, tpu_buf2);
    ASSERT_EQ(ec, ER_OK) << "BufferFromHost for iov2 failed";
    ASSERT_NE(tpu_buf2, nullptr);

    // Build IOVs with TPU memory type for Put
    BlockBuffer put_buf;
    Iov put_iov1;
    put_iov1.base = reinterpret_cast<void*>(tpu_buf1);
    put_iov1.size = len1;
    put_iov1.type = MemoryType::TPU;
    put_iov1.ignore = false;
    put_buf.iovs.push_back(put_iov1);

    Iov put_iov2;
    put_iov2.base = reinterpret_cast<void*>(tpu_buf2);
    put_iov2.size = len2;
    put_iov2.type = MemoryType::TPU;
    put_iov2.ignore = false;
    put_buf.iovs.push_back(put_iov2);

    DataStorageUri uri("file://" + root_path_ + "/local_file/test_tpu.txt");
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "1024");

    const std::vector<DataStorageUri>& remote_uris = {uri};
    BlockBuffers local_buffers = {put_buf};

    // Put: TPU memory -> local file
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    auto put_ec = sdk.Put(remote_uris, local_buffers, actual_remote_uris);
    ASSERT_EQ(put_ec, ER_OK) << "Put failed";
    ASSERT_EQ(actual_remote_uris->size(), 1);

    // Cleanup put buffers
    sdk.tpu_client_.DestroyBuffer(tpu_buf1);
    sdk.tpu_client_.DestroyBuffer(tpu_buf2);

    // Get: local file -> TPU memory
    // For Get, iov.base will be filled by SDK with newly created PJRT_Buffer*
    BlockBuffer get_buf;
    Iov get_iov1;
    get_iov1.base = nullptr;  // Will be filled by SDK
    get_iov1.size = len1;
    get_iov1.type = MemoryType::TPU;
    get_iov1.ignore = false;
    get_buf.iovs.push_back(get_iov1);

    Iov get_iov2;
    get_iov2.base = nullptr;  // Will be filled by SDK
    get_iov2.size = len2;
    get_iov2.type = MemoryType::TPU;
    get_iov2.ignore = false;
    get_buf.iovs.push_back(get_iov2);

    local_buffers[0] = get_buf;

    auto get_ec = sdk.Get(remote_uris, local_buffers);
    ASSERT_EQ(get_ec, ER_OK) << "Get failed";

    // SDK should have filled iov.base with new PJRT_Buffer* handles
    PJRT_Buffer* get_tpu_buf1 = reinterpret_cast<PJRT_Buffer*>(local_buffers[0].iovs[0].base);
    PJRT_Buffer* get_tpu_buf2 = reinterpret_cast<PJRT_Buffer*>(local_buffers[0].iovs[1].base);
    ASSERT_NE(get_tpu_buf1, nullptr) << "Get should create TPU buffer for iov1";
    ASSERT_NE(get_tpu_buf2, nullptr) << "Get should create TPU buffer for iov2";

    // Read back from TPU buffers to verify
    std::vector<char> host_get_data1(len1, 0);
    ec = sdk.tpu_client_.BufferToHost(get_tpu_buf1, host_get_data1.data(), len1);
    ASSERT_EQ(ec, ER_OK) << "BufferToHost for iov1 failed";

    std::vector<char> host_get_data2(len2, 0);
    ec = sdk.tpu_client_.BufferToHost(get_tpu_buf2, host_get_data2.data(), len2);
    ASSERT_EQ(ec, ER_OK) << "BufferToHost for iov2 failed";

    // Verify data integrity
    ASSERT_EQ(std::memcmp(host_get_data1.data(), test_data, len1), 0)
        << "iov1 data mismatch";
    ASSERT_EQ(std::memcmp(host_get_data2.data(), test_data_2, len2), 0)
        << "iov2 data mismatch";

    // Cleanup get buffers
    sdk.tpu_client_.DestroyBuffer(get_tpu_buf1);
    sdk.tpu_client_.DestroyBuffer(get_tpu_buf2);
#else
    GTEST_SKIP() << "TPU not enabled, skipping TPU SDK Put/Get test";
#endif
}

// Test SDK Put/Get with CPU memory (verifies SDK works in TPU build)
TEST_F(LocalFileSdkTpuTest, TestSdkPutGetWithCpuMemory) {
#ifdef USING_TPU
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));

    // Prepare test data
    const char* test_data = "this is cpu memory test data!!!!!!!";
    size_t len1 = strlen(test_data);
    const char* test_data_2 = "second cpu iov test data content!!";
    size_t len2 = strlen(test_data_2);
    size_t total_size = len1 + len2;

    // Allocate host buffers
    std::vector<char> host_put_data(total_size);
    std::memcpy(host_put_data.data(), test_data, len1);
    std::memcpy(host_put_data.data() + len1, test_data_2, len2);

    // Build IOVs with CPU memory type
    BlockBuffer buf;
    Iov iov1;
    iov1.base = host_put_data.data();
    iov1.size = len1;
    iov1.type = MemoryType::CPU;
    iov1.ignore = false;
    buf.iovs.push_back(iov1);

    Iov iov2;
    iov2.base = host_put_data.data() + len1;
    iov2.size = len2;
    iov2.type = MemoryType::CPU;
    iov2.ignore = false;
    buf.iovs.push_back(iov2);

    DataStorageUri uri("file://" + root_path_ + "/local_file/test_cpu.txt");
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "1024");

    const std::vector<DataStorageUri>& remote_uris = {uri};
    BlockBuffers local_buffers = {buf};

    // Put: CPU memory -> local file
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    auto put_ec = sdk.Put(remote_uris, local_buffers, actual_remote_uris);
    ASSERT_EQ(put_ec, ER_OK) << "Put failed";
    ASSERT_EQ(actual_remote_uris->size(), 1);

    // Get: local file -> CPU memory
    std::vector<char> host_get_data(total_size, 0);
    local_buffers[0].iovs[0].base = host_get_data.data();
    local_buffers[0].iovs[1].base = host_get_data.data() + len1;

    auto get_ec = sdk.Get(remote_uris, local_buffers);
    ASSERT_EQ(get_ec, ER_OK) << "Get failed";

    // Verify data
    ASSERT_EQ(std::memcmp(host_get_data.data(), test_data, len1), 0);
    ASSERT_EQ(std::memcmp(host_get_data.data() + len1, test_data_2, len2), 0);
#else
    GTEST_SKIP() << "TPU not enabled, skipping CPU memory test";
#endif
}

// Test invalid size rejection
TEST_F(LocalFileSdkTpuTest, TestInvalidSize) {
#ifdef USING_TPU
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));

    DataStorageUri uri("file://" + root_path_ + "/local_file/test_invalid.txt");
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "1023"); // invalid: not in spec_byte_sizes_per_block

    BlockBuffer buf;
    Iov iov;
    std::vector<char> dummy_data(23, 0);
    iov.base = dummy_data.data();
    iov.size = 23;
    iov.type = MemoryType::CPU;
    iov.ignore = false;
    buf.iovs.push_back(iov);

    const std::vector<DataStorageUri>& remote_uris = {uri};
    BlockBuffers local_buffers = {buf};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    auto ec = sdk.Put(remote_uris, local_buffers, actual_remote_uris);
    EXPECT_NE(ec, ER_OK) << "Put should fail with invalid size";
#else
    GTEST_SKIP() << "TPU not enabled, skipping invalid size test";
#endif
}
