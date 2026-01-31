#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/mooncake_backend.h"

using namespace kv_cache_manager;

/*
 * 测试说明：
 * 1. 编译选项带上 --config=mooncake
 * 2. 本地需要启动相同版本的mooncake master和metadata服务，默认端口 50051, 8080
 * 2. segment_provider_buffer_ 通过 mountSegment 挂载内存，提供存储
 * 3. 采用tcp协议，如果测试rdma，要register_local_memory
 */
class MooncakeBackendTest : public TESTBASE {
public:
    void SetUp() override {
        InitTestClient();
        InitSegmentProviderClient();
    }
    void TearDown() override {
        auto ec = mooncake_backend_.Close();
        ASSERT_EQ(EC_OK, ec);
        mooncake_client_destroy(segment_provider_buffer_);
    }

protected:
    void InitTestClient();
    void InitSegmentProviderClient();

private:
    MooncakeBackend mooncake_backend_;
    client_t segment_provider_buffer_;
    std::unique_ptr<char[]> client_buffer_allocator_ = std::make_unique<char[]>(128 * 1024 * 1024);
};

void MooncakeBackendTest::InitTestClient() {
    MooncakeBackendConfig mooncake_backend_config;
    mooncake_backend_config.local_hostname = "localhost:17812";
    mooncake_backend_config.metadata_connstring = "http://localhost:8080/metadata";
    mooncake_backend_config.protocol = "tcp";
    mooncake_backend_config.master_server_entry = "localhost:50051";

    DataStorageConfig data_storage_config;
    data_storage_config.type = DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
    data_storage_config.backend_config = mooncake_backend_config;

    ASSERT_EQ(EC_OK, mooncake_backend_.Open(data_storage_config));
}

void MooncakeBackendTest::InitSegmentProviderClient() {
    segment_provider_buffer_ =
        mooncake_client_create("localhost:17813", "http://localhost:8080/metadata", "tcp", "", "localhost:50051");
    ASSERT_NE(nullptr, segment_provider_buffer_);
    ErrorCode_t err = mooncake_client_mount_segment(segment_provider_buffer_, 512 * 1024 * 1024);
    ASSERT_EQ(MOONCAKE_ERROR_OK, err);
}

TEST_F(MooncakeBackendTest, TestSimple) {
    DataStorageItem item;
    item.key = "test_key";

    // clear
    ASSERT_EQ(MOONCAKE_ERROR_OK, mooncake_client_remove_all(mooncake_backend_.client_));

    // put
    std::vector<Slice_t> slices;
    const std::string test_data = "test_data";
    Slice_t slice = {(void *)test_data.data(), test_data.size()};
    slices.push_back(slice);
    ReplicateConfig_t config;
    ASSERT_EQ(MOONCAKE_ERROR_OK,
              mooncake_client_put(mooncake_backend_.client_, item.key.data(), slices.data(), slices.size(), config));

    // Exist
    ASSERT_TRUE(mooncake_backend_.Exist(item));

    // Delete
    ASSERT_EQ(EC_ERROR, mooncake_backend_.Delete(item, []() { /* do nothing */ }));
    sleep(5); // wait ttl
    ASSERT_EQ(EC_OK, mooncake_backend_.Delete(item, []() { /* do nothing */ }));

    // Not Exist
    ASSERT_FALSE(mooncake_backend_.Exist(item));
}