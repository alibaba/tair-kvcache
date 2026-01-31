#include <fstream>
#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/config/test/config_test_base.h"
#include "kv_cache_manager/common/logger.h"

using namespace kv_cache_manager;

class ClientConfigTest : public ConfigTestBase {};

TEST_F(ClientConfigTest, TestClientConfigSuccess) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(client_config.FromJsonString(file_content));
}

TEST_F(ClientConfigTest, TestClientConfigTairMemPoolSuccess) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_tair_mempool_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(client_config.FromJsonString(file_content));
    auto sdk_wrapper_config = client_config.sdk_wrapper_config();
    ASSERT_TRUE(sdk_wrapper_config);
    auto sdk_backend_configs_map = sdk_wrapper_config->sdk_backend_configs_map();
    // 默认就加了3个
    ASSERT_EQ(4, sdk_backend_configs_map.size());
    auto tair_mempool_config = sdk_backend_configs_map[DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL];
    ASSERT_TRUE(tair_mempool_config);
    ASSERT_EQ("logs/pace_client.log", tair_mempool_config->sdk_log_file_path());
    ASSERT_EQ("DEBUG", tair_mempool_config->sdk_log_level());
    // 需要在sdk wrapper的init阶段填充
    ASSERT_EQ(0, tair_mempool_config->byte_size_per_block());

    // 再次序列化
    std::string json_string2 = client_config.ToJsonString();
    ASSERT_TRUE(json_string2.find("sdk_log_file_path") != std::string::npos);
    ClientConfig client_config2;
    ASSERT_TRUE(client_config2.FromJsonString(json_string2));
    ASSERT_EQ(client_config, client_config2);
    ASSERT_EQ(json_string2, client_config2.ToJsonString());
}

TEST_F(ClientConfigTest, TestClientConfigEmptyInstanceGroup) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_empty_instance_group.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_FALSE(client_config.FromJsonString(file_content));
}

TEST_F(ClientConfigTest, TestClientConfigEmptyInstanceId) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_empty_instance_id.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_FALSE(client_config.FromJsonString(file_content));
}

TEST_F(ClientConfigTest, TestClientConfigEmptyLocationSpecInfos) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_location_spec_infos_empty.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_FALSE(client_config.FromJsonString(file_content));
}

TEST_F(ClientConfigTest, TestClientConfigInvalidLocationSpecInfo) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_invalid_location_spec_info.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_FALSE(client_config.FromJsonString(file_content));
}

TEST_F(ClientConfigTest, TestClientConfigZeroLocationSpecInfo) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_zero_location_spec_info.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_FALSE(client_config.FromJsonString(file_content));
}

TEST_F(ClientConfigTest, TestClientConfigEquality) {
    // Test with two identical configs loaded from the same file
    ClientConfig client_config1;
    std::string file_content = getFileContent("client_config_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(client_config1.FromJsonString(file_content));

    ClientConfig client_config2;
    ASSERT_TRUE(client_config2.FromJsonString(file_content));

    // They should be equal
    ASSERT_TRUE(client_config1 == client_config2);
    ASSERT_FALSE(client_config1 != client_config2);

    // Test with different configs
    ClientConfig client_config3;
    std::string file_content2 = getFileContent("client_config_tair_mempool_success.json");
    ASSERT_FALSE(file_content2.empty());
    ASSERT_TRUE(client_config3.FromJsonString(file_content2));

    // They should not be equal
    ASSERT_FALSE(client_config1 == client_config3);
    ASSERT_TRUE(client_config1 != client_config3);
}

TEST_F(ClientConfigTest, TestClientConfigSelfEquality) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_success.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_TRUE(client_config.FromJsonString(file_content));

    // A config should be equal to itself
    ASSERT_TRUE(client_config == client_config);
    ASSERT_FALSE(client_config != client_config);
}

TEST_F(ClientConfigTest, TestClientConfigInvalidLocationSpecGroup) {
    ClientConfig client_config;
    std::string file_content = getFileContent("client_config_success_invalid_location_spec_groups.json");
    ASSERT_FALSE(file_content.empty());
    ASSERT_FALSE(client_config.FromJsonString(file_content));
}