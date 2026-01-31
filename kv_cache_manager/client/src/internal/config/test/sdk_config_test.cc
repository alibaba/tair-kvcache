#include <fstream>
#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/config/test/config_test_base.h"
#include "kv_cache_manager/common/logger.h"

using namespace kv_cache_manager;

class SdkBackendConfigTest : public ConfigTestBase {};

TEST_F(SdkBackendConfigTest, TestHf3fsSdkConfigSuccess) {
    Hf3fsSdkConfig sdk_backend_config;
    std::string file_content = getFileContent("sdk_config_hf3fs_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_backend_config.FromJsonString(file_content));
    ASSERT_TRUE(sdk_backend_config.Validate());
}

TEST_F(SdkBackendConfigTest, TestHf3fsSdkConfigInvalidType) {
    Hf3fsSdkConfig sdk_backend_config;
    std::string file_content = getFileContent("sdk_config_hf3fs_invalid_type.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_backend_config.FromJsonString(file_content));
    ASSERT_FALSE(sdk_backend_config.Validate());
}

TEST_F(SdkBackendConfigTest, TestMooncakeSdkConfigSuccess) {
    MooncakeSdkConfig sdk_backend_config;
    std::string file_content = getFileContent("sdk_config_mooncake_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_backend_config.FromJsonString(file_content));
    ASSERT_TRUE(sdk_backend_config.Validate());
}

TEST_F(SdkBackendConfigTest, TestMooncakeSdkConfigEmptyLocation) {
    MooncakeSdkConfig sdk_backend_config;
    std::string file_content = getFileContent("sdk_config_mooncake_empty_location.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_backend_config.FromJsonString(file_content));
    ASSERT_FALSE(sdk_backend_config.Validate());
}

TEST_F(SdkBackendConfigTest, TestMooncakeSdkConfigInvalidReplicaNum) {
    MooncakeSdkConfig sdk_backend_config;
    std::string file_content = getFileContent("sdk_config_mooncake_invalid_replica_num.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_backend_config.FromJsonString(file_content));
    ASSERT_FALSE(sdk_backend_config.Validate());
}

TEST_F(SdkBackendConfigTest, TestTairMempoolSdkConfigSuccess) {
    TairMempoolSdkConfig sdk_backend_config;
    std::string file_content = getFileContent("sdk_config_tair_mempool_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_backend_config.FromJsonString(file_content));
    ASSERT_TRUE(sdk_backend_config.Validate());
    ASSERT_EQ("logs/pace_client.log", sdk_backend_config.sdk_log_file_path());
    ASSERT_EQ("DEBUG", sdk_backend_config.sdk_log_level());
}

TEST_F(SdkBackendConfigTest, TestNfsSdkConfigSuccess) {
    NfsSdkConfig sdk_backend_config;
    std::string file_content = getFileContent("sdk_config_nfs_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_backend_config.FromJsonString(file_content));
    ASSERT_TRUE(sdk_backend_config.Validate());
}

// SdkWrapperConfig tests
TEST_F(SdkBackendConfigTest, TestSdkWrapperConfigSuccess) {
    SdkWrapperConfig sdk_wrapper_config;
    std::string file_content = getFileContent("sdk_wrapper_config_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_wrapper_config.FromJsonString(file_content));
    ASSERT_TRUE(sdk_wrapper_config.Validate());
}

TEST_F(SdkBackendConfigTest, TestSdkWrapperConfigInvalidTimeout) {
    SdkWrapperConfig sdk_wrapper_config;
    std::string file_content = getFileContent("sdk_wrapper_config_invalid_timeout.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_wrapper_config.FromJsonString(file_content));
    ASSERT_FALSE(sdk_wrapper_config.Validate());
}

TEST_F(SdkBackendConfigTest, TestSdkWrapperConfigEmptySdkConfigs) {
    SdkWrapperConfig sdk_wrapper_config;
    std::string file_content = getFileContent("sdk_wrapper_config_empty_sdk_configs.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_wrapper_config.FromJsonString(file_content));
    ASSERT_TRUE(sdk_wrapper_config.Validate());
}

TEST_F(SdkBackendConfigTest, TestSdkWrapperConfigGetSdkConfig) {
    SdkWrapperConfig sdk_wrapper_config;
    std::string file_content = getFileContent("sdk_wrapper_config_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_wrapper_config.FromJsonString(file_content));

    auto hf3fs_config = sdk_wrapper_config.GetSdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    ASSERT_NE(hf3fs_config, nullptr);

    auto mooncake_config = sdk_wrapper_config.GetSdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
    ASSERT_NE(mooncake_config, nullptr);

    auto tair_config = sdk_wrapper_config.GetSdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
    ASSERT_NE(tair_config, nullptr);

    auto nfs_config = sdk_wrapper_config.GetSdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_NE(nfs_config, nullptr);

    auto unknown_config = sdk_wrapper_config.GetSdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN);
    ASSERT_EQ(unknown_config, nullptr);
}

TEST_F(SdkBackendConfigTest, TestDuplicatedSdkConfig) {
    SdkWrapperConfig sdk_wrapper_config;
    std::string file_content = getFileContent("sdk_wrapper_config_duplicated_sdk_config.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(sdk_wrapper_config.FromJsonString(file_content));
    ASSERT_TRUE(sdk_wrapper_config.Validate());

    auto mooncake_config = sdk_wrapper_config.GetSdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
    ASSERT_NE(mooncake_config, nullptr);
    ASSERT_EQ(8, std::dynamic_pointer_cast<MooncakeSdkConfig>(mooncake_config)->put_replica_num());
}

// Test cases for operator== functions
TEST_F(SdkBackendConfigTest, TestSdkTimeoutConfigEquality) {
    SdkTimeoutConfig config1;
    config1.set_put_timeout_ms(1000);
    config1.set_get_timeout_ms(2000);

    SdkTimeoutConfig config2;
    config2.set_put_timeout_ms(1000);
    config2.set_get_timeout_ms(2000);

    SdkTimeoutConfig config3;
    config3.set_put_timeout_ms(1500);
    config3.set_get_timeout_ms(2000);

    // Test equality
    ASSERT_TRUE(config1 == config2);
    ASSERT_FALSE(config1 != config2);

    // Test inequality
    ASSERT_FALSE(config1 == config3);
    ASSERT_TRUE(config1 != config3);

    // Test self-equality
    ASSERT_TRUE(config1 == config1);
    ASSERT_FALSE(config1 != config1);
}

TEST_F(SdkBackendConfigTest, TestSdkBackendConfigEquality) {
    SdkBackendConfig config1;
    config1.set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    config1.set_sdk_log_file_path("log1.txt");
    config1.set_sdk_log_level("DEBUG");
    config1.set_byte_size_per_block(1024);

    SdkBackendConfig config2;
    config2.set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    config2.set_sdk_log_file_path("log1.txt");
    config2.set_sdk_log_level("DEBUG");
    config2.set_byte_size_per_block(1024);

    SdkBackendConfig config3;
    config3.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    config3.set_sdk_log_file_path("log1.txt");
    config3.set_sdk_log_level("DEBUG");
    config3.set_byte_size_per_block(1024);

    // Test equality
    ASSERT_TRUE(config1 == config2);
    ASSERT_FALSE(config1 != config2);

    // Test inequality
    ASSERT_FALSE(config1 == config3);
    ASSERT_TRUE(config1 != config3);

    // Test self-equality
    ASSERT_TRUE(config1 == config1);
    ASSERT_FALSE(config1 != config1);
}

TEST_F(SdkBackendConfigTest, TestHf3fsSdkConfigEquality) {
    Hf3fsSdkConfig config1;
    config1.set_read_iov_block_size(1024);
    config1.set_read_iov_size(1ULL << 32);
    config1.set_write_iov_block_size(1ULL << 20);
    config1.set_write_iov_size(1ULL << 32);

    Hf3fsSdkConfig config2;
    config2.set_read_iov_block_size(1024);
    config2.set_read_iov_size(1ULL << 32);
    config2.set_write_iov_block_size(1ULL << 20);
    config2.set_write_iov_size(1ULL << 32);

    Hf3fsSdkConfig config3;
    config3.set_read_iov_block_size(2048); // Different value
    config3.set_read_iov_size(1ULL << 32);
    config3.set_write_iov_block_size(1ULL << 20);
    config3.set_write_iov_size(1ULL << 32);

    // Test equality
    ASSERT_TRUE(config1 == config2);
    ASSERT_FALSE(config1 != config2);

    // Test inequality
    ASSERT_FALSE(config1 == config3);
    ASSERT_TRUE(config1 != config3);

    // Test self-equality
    ASSERT_TRUE(config1 == config1);
    ASSERT_FALSE(config1 != config1);
}

TEST_F(SdkBackendConfigTest, TestMooncakeSdkConfigEquality) {
    MooncakeSdkConfig config1;
    config1.set_local_buffer_size(1024);
    config1.set_location("localhost");
    config1.set_put_replica_num(2);

    MooncakeSdkConfig config2;
    config2.set_local_buffer_size(1024);
    config2.set_location("localhost");
    config2.set_put_replica_num(2);

    MooncakeSdkConfig config3;
    config3.set_local_buffer_size(2048); // Different value
    config3.set_location("localhost");
    config3.set_put_replica_num(2);

    // Test equality (note: local_mem_ptr_ is not compared)
    ASSERT_TRUE(config1 == config2);
    ASSERT_FALSE(config1 != config2);

    // Test inequality
    ASSERT_FALSE(config1 == config3);
    ASSERT_TRUE(config1 != config3);

    // Test self-equality
    ASSERT_TRUE(config1 == config1);
    ASSERT_FALSE(config1 != config1);
}

TEST_F(SdkBackendConfigTest, TestTairMempoolSdkConfigEquality) {
    TairMempoolSdkConfig config1;
    config1.set_type(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
    config1.set_sdk_log_file_path("tair.log");
    config1.set_sdk_log_level("INFO");
    config1.set_byte_size_per_block(2048);

    TairMempoolSdkConfig config2;
    config2.set_type(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
    config2.set_sdk_log_file_path("tair.log");
    config2.set_sdk_log_level("INFO");
    config2.set_byte_size_per_block(2048);

    TairMempoolSdkConfig config3;
    config3.set_type(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
    config3.set_sdk_log_file_path("tair.log");
    config3.set_sdk_log_level("DEBUG"); // Different value
    config3.set_byte_size_per_block(2048);

    // Test equality
    ASSERT_TRUE(config1 == config2);
    ASSERT_FALSE(config1 != config2);

    // Test inequality
    ASSERT_FALSE(config1 == config3);
    ASSERT_TRUE(config1 != config3);

    // Test self-equality
    ASSERT_TRUE(config1 == config1);
    ASSERT_FALSE(config1 != config1);
}

TEST_F(SdkBackendConfigTest, TestNfsSdkConfigEquality) {
    NfsSdkConfig config1;
    config1.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    config1.set_sdk_log_file_path("nfs.log");
    config1.set_sdk_log_level("DEBUG");
    config1.set_byte_size_per_block(4096);

    NfsSdkConfig config2;
    config2.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    config2.set_sdk_log_file_path("nfs.log");
    config2.set_sdk_log_level("DEBUG");
    config2.set_byte_size_per_block(4096);

    NfsSdkConfig config3;
    config3.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    config3.set_sdk_log_file_path("nfs.log");
    config3.set_sdk_log_level("INFO"); // Different value
    config3.set_byte_size_per_block(4096);

    // Test equality
    ASSERT_TRUE(config1 == config2);
    ASSERT_FALSE(config1 != config2);

    // Test inequality
    ASSERT_FALSE(config1 == config3);
    ASSERT_TRUE(config1 != config3);

    // Test self-equality
    ASSERT_TRUE(config1 == config1);
    ASSERT_FALSE(config1 != config1);
}

TEST_F(SdkBackendConfigTest, TestSdkWrapperConfigEquality) {
    // Create first config
    SdkWrapperConfig config1;
    config1.set_thread_num(8);
    config1.set_queue_size(2000);

    SdkTimeoutConfig timeout1;
    timeout1.set_put_timeout_ms(1000);
    timeout1.set_get_timeout_ms(2000);
    config1.set_timeout_config(timeout1);

    // Create second identical config
    SdkWrapperConfig config2;
    config2.set_thread_num(8);
    config2.set_queue_size(2000);

    SdkTimeoutConfig timeout2;
    timeout2.set_put_timeout_ms(1000);
    timeout2.set_get_timeout_ms(2000);
    config2.set_timeout_config(timeout2);

    // Create third different config
    SdkWrapperConfig config3;
    config3.set_thread_num(16); // Different value
    config3.set_queue_size(2000);

    SdkTimeoutConfig timeout3;
    timeout3.set_put_timeout_ms(1000);
    timeout3.set_get_timeout_ms(2000);
    config3.set_timeout_config(timeout3);

    // Test equality
    ASSERT_TRUE(config1 == config2);
    ASSERT_FALSE(config1 != config2);

    // Test inequality
    ASSERT_FALSE(config1 == config3);
    ASSERT_TRUE(config1 != config3);

    // Test self-equality
    ASSERT_TRUE(config1 == config1);
    ASSERT_FALSE(config1 != config1);
}
