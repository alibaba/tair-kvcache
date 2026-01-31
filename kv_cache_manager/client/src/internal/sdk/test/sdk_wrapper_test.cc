#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

using namespace kv_cache_manager;

class SdkWrapperTest : public TESTBASE {
public:
    void SetUp() override {
        client_config_ = CreateTestClientConfig();
        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = CreateTestStorageConfigs();
        root_path_ = GetPrivateTestRuntimeDataPath();
    }

    void TearDown() override {
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

private:
    std::unique_ptr<ClientConfig> CreateTestClientConfig() {
        auto client_config = std::make_unique<ClientConfig>();
        std::string client_config_str = R"({
            "instance_group": "group",
            "instance_id": "instance",
            "address": [
                "127.0.0.1:8080"
            ],
            "block_size": 128,
            "sdk_config": {
                "thread_num": 8,
                "queue_size": 2000,
                "sdk_backend_configs": [
                    {
                        "type": "file"
                    }
                ],
                "timeout_config": {
                    "put_timeout_ms": 2000,
                    "get_timeout_ms": 2000
                }
            },
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": false,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1
            },
            "location_spec_infos": {
                "tp0": 1024
            }
        })";
        client_config->FromJsonString(client_config_str);
        return client_config;
    }

    std::string CreateTestStorageConfigs() {
        return "["
               // #ifdef ENABLE_HF3FS
               //                R"({
               //             "type": "hf3fs",
               //             "global_unique_name": "3fs_test",
               //             "storage_spec": {
               //                 "cluster_name": "3fs_cluster",
               //                 "mountpoint": "/3fs/stage/3fs",
               //                 "root_dir": "3fs_test/",
               //                 "key_count_per_file": 2
               //             }
               //         },)"
               // #endif
               R"({
            "type": "file",
            "global_unique_name": "nfs_test",
            "storage_spec": {
                "root_path": "/nfs/",
                "key_count_per_file": 2
            }
        }
    ])";
    }

private:
    std::unique_ptr<ClientConfig> client_config_;
    InitParams init_params_;
    std::string root_path_;
};

TEST_F(SdkWrapperTest, TestInit) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));
}

TEST_F(SdkWrapperTest, TestInitWithEmptyWrapperConfig) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_INVALID_CLIENT_CONFIG, sdk_wrapper.Init(nullptr, init_params_));
}

TEST_F(SdkWrapperTest, TestInitWithEmptyStorageConfigs) {
    SdkWrapper sdk_wrapper;
    InitParams init_params = init_params_;
    init_params.storage_configs = "[]";
    ASSERT_EQ(ER_INVALID_STORAGE_CONFIG, sdk_wrapper.Init(client_config_, init_params));
}

TEST_F(SdkWrapperTest, TestInitWithInvalidStorageConfigs) {
    SdkWrapper sdk_wrapper;
    InitParams init_params = init_params_;
    init_params.storage_configs = "[invalid json]";
    ASSERT_EQ(ER_INVALID_STORAGE_CONFIG, sdk_wrapper.Init(client_config_, init_params));
}

// TODO: mock mooncake
//  TEST_F(SdkWrapperTest, TestInitWithMooncake) {
//  #ifdef ENABLE_MOONCAKE
//      auto wrapper_config = CreateTestWrapperConfig();
//      auto mooncake_config = std::make_shared<MooncakeSdkConfig>();
//      mooncake_config->set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
//      mooncake_config->set_location("*");
//      mooncake_config->set_put_replica_num(2);
//      wrapper_config->sdk_config_map_[DataStorageType::DATA_STORAGE_TYPE_MOONCAKE] = mooncake_config;
//      InitParams init_params;
//      {
//          SdkWrapper sdk_wrapper;
//          ASSERT_FALSE(sdk_wrapper.Init(wrapper_config, init_params));
//      }
//      {
//          SdkWrapper sdk_wrapper;
//          ASSERT_TRUE(sdk_wrapper.Init(wrapper_config, init_params_));
//      }
//      ASSERT_TRUE(false);
//  #else
//      GTEST_SKIP() << "mooncake not enabled, skipping init sdk wrapper with mooncake config";
//  #endif
//  }

TEST_F(SdkWrapperTest, TestPutAndGet) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));
    std::vector<DataStorageUri> remote_uris;
    remote_uris.push_back(DataStorageUri("file://nfs_test/" + root_path_ + "/nfs/0/0/1?blkid=0&size=1024"));
    BlockBuffers local_buffers;
    BlockBuffer buffer;
    local_buffers.push_back(buffer);
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk_wrapper.Put(remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(actual_remote_uris->size(), 1);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), remote_uris[0].ToUriString());

    ASSERT_EQ(ER_OK, sdk_wrapper.Get(*actual_remote_uris, local_buffers));
}

TEST_F(SdkWrapperTest, TestValid) {
    SdkWrapper sdk_wrapper;
    std::vector<DataStorageUri> remote_uris;
    remote_uris.push_back(DataStorageUri("file://nfs_test/nfs/0/0/1?blkid=0"));
    BlockBuffers local_buffers;
    ASSERT_EQ(ER_INVALID_PARAMS, sdk_wrapper.Valid(remote_uris, local_buffers));
    BlockBuffer buffer;
    local_buffers.push_back(buffer);
    ASSERT_EQ(ER_OK, sdk_wrapper.Valid(remote_uris, local_buffers));
}

TEST_F(SdkWrapperTest, TestGetSdk) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));
    DataStorageUri remote_uri("file://nfs_test/nfs/0/0/1?blkid=0");
    ASSERT_TRUE(sdk_wrapper.GetSdk(remote_uri));
    remote_uri = DataStorageUri("file://invalid/nfs/1/0/1?blkid=0");
    ASSERT_FALSE(sdk_wrapper.GetSdk(remote_uri));
#if ENABLE_HF3FS && USING_CUDA
    remote_uri = DataStorageUri("3fs://3fs_test/3fs_test/0/1?blkid=0");
    // ASSERT_TRUE(sdk_wrapper.GetSdk(remote_uri));
#endif
    remote_uri = DataStorageUri("invalid:///mnt/nfs/0/0/1?blkid=0");
    ASSERT_FALSE(sdk_wrapper.GetSdk(remote_uri));
}

TEST_F(SdkWrapperTest, TestUpdateMooncakeSdkConfig) {
    SdkWrapper sdk_wrapper;
    RegistSpan span;
    auto sdk_backend_config =
        client_config_->sdk_wrapper_config()->GetSdkBackendConfig(DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_TRUE(sdk_backend_config);
    ASSERT_EQ(ER_OK, sdk_wrapper.UpdateMooncakeSdkConfig(sdk_backend_config, nullptr, ""));
    ASSERT_EQ(ER_OK, sdk_wrapper.UpdateMooncakeSdkConfig(sdk_backend_config, &span, ""));
#ifdef ENABLE_MOONCAKE
    auto mooncake_config = std::make_shared<MooncakeSdkConfig>();
    mooncake_config->set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
    mooncake_config->set_location("*");
    mooncake_config->set_put_replica_num(2);
    ASSERT_EQ(ER_INVALID_PARAMS, sdk_wrapper.UpdateMooncakeSdkConfig(mooncake_config, nullptr, ""));
    ASSERT_EQ(ER_OK, sdk_wrapper.UpdateMooncakeSdkConfig(mooncake_config, &span, ""));
#endif
}
