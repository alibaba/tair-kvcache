#include <atomic>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/lock_free_thread_pool.h"
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

TEST_F(SdkWrapperTest, TestPrepareSharedMemoryRegistrationOwnsFd) {
    FILE *file = tmpfile();
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(ftruncate(fileno(file), static_cast<off_t>(init_params_.regist_span->size)), 0);

    SharedMemoryRegistration registration;
    registration.base = init_params_.regist_span->base;
    registration.size = init_params_.regist_span->size;
    registration.fd = fileno(file);

    SdkWrapper sdk_wrapper;
    SharedMemoryRegistration prepared_registration;
    ASSERT_EQ(ER_OK, sdk_wrapper.PrepareSharedMemoryRegistration(registration, prepared_registration));
    EXPECT_NE(prepared_registration.fd, registration.fd);
    const int fd_flags = fcntl(prepared_registration.fd, F_GETFD);
    ASSERT_GE(fd_flags, 0);
    EXPECT_NE(fd_flags & FD_CLOEXEC, 0);

    ASSERT_EQ(fclose(file), 0);
    struct stat file_stat{};
    EXPECT_EQ(fstat(prepared_registration.fd, &file_stat), 0);
}

TEST_F(SdkWrapperTest, TestDestructorKeepsSharedMemoryFdAliveForRunningTasks) {
    FILE *file = tmpfile();
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(ftruncate(fileno(file), static_cast<off_t>(init_params_.regist_span->size)), 0);

    SharedMemoryRegistration registration;
    registration.base = init_params_.regist_span->base;
    registration.size = init_params_.regist_span->size;
    registration.fd = fileno(file);

    auto sdk_wrapper = std::make_unique<SdkWrapper>();
    SharedMemoryRegistration prepared_registration;
    ASSERT_EQ(ER_OK, sdk_wrapper->PrepareSharedMemoryRegistration(registration, prepared_registration));
    const int owned_fd = prepared_registration.fd;

    sdk_wrapper->wait_task_thread_pool_ = std::make_unique<LockFreeThreadPool>(1, 1, "SdkWrapperTestPool");
    ASSERT_TRUE(sdk_wrapper->wait_task_thread_pool_->start());

    std::promise<void> task_started;
    auto task_started_future = task_started.get_future();
    std::promise<void> allow_task_finish;
    auto allow_task_finish_future = allow_task_finish.get_future().share();
    std::atomic<bool> fd_valid_in_task{false};
    auto task_result = sdk_wrapper->wait_task_thread_pool_->async([&]() {
        task_started.set_value();
        allow_task_finish_future.wait();
        struct stat file_stat{};
        fd_valid_in_task.store(fstat(owned_fd, &file_stat) == 0);
        return ER_OK;
    });
    ASSERT_EQ(task_started_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);

    std::promise<void> destructor_started;
    auto destructor_started_future = destructor_started.get_future();
    std::thread destroyer([&]() {
        destructor_started.set_value();
        sdk_wrapper.reset();
    });
    EXPECT_EQ(destructor_started_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    allow_task_finish.set_value();
    destroyer.join();

    EXPECT_EQ(task_result.get(), ER_OK);
    EXPECT_TRUE(fd_valid_in_task.load());
    EXPECT_EQ(fcntl(owned_fd, F_GETFD), -1);
    ASSERT_EQ(fclose(file), 0);
}

TEST_F(SdkWrapperTest, TestPrepareSharedMemoryRegistrationRejectsInvalidValues) {
    SdkWrapper sdk_wrapper;
    SharedMemoryRegistration prepared_registration;
    SharedMemoryRegistration registration;

    registration.fd = 0;
    EXPECT_EQ(ER_INVALID_PARAMS, sdk_wrapper.PrepareSharedMemoryRegistration(registration, prepared_registration));

    registration = SharedMemoryRegistration();
    registration.base = init_params_.regist_span->base;
    EXPECT_EQ(ER_INVALID_PARAMS, sdk_wrapper.PrepareSharedMemoryRegistration(registration, prepared_registration));

    FILE *file = tmpfile();
    ASSERT_NE(file, nullptr);
    registration = SharedMemoryRegistration();
    registration.base = init_params_.regist_span->base;
    registration.size = init_params_.regist_span->size;
    registration.fd = fileno(file);
    EXPECT_EQ(ER_INVALID_PARAMS, sdk_wrapper.PrepareSharedMemoryRegistration(registration, prepared_registration));
    ASSERT_EQ(fclose(file), 0);
}

TEST_F(SdkWrapperTest, TestUpdateTairMempoolSdkConfigWithSharedMemory) {
    FILE *file = tmpfile();
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(ftruncate(fileno(file), static_cast<off_t>(init_params_.regist_span->size)), 0);

    SharedMemoryRegistration registration;
    registration.base = init_params_.regist_span->base;
    registration.size = init_params_.regist_span->size;
    registration.fd = fileno(file);

    SdkWrapper sdk_wrapper;
    SharedMemoryRegistration prepared_registration;
    ASSERT_EQ(ER_OK, sdk_wrapper.PrepareSharedMemoryRegistration(registration, prepared_registration));

    for (const auto type : {DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL,
                            DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD}) {
        SCOPED_TRACE(static_cast<int>(type));
        auto config = std::make_shared<TairMempoolSdkConfig>(type);
        ASSERT_EQ(ER_OK, sdk_wrapper.UpdateTairMempoolSdkConfig(config, &prepared_registration));
        EXPECT_EQ(config->shm_fd(), prepared_registration.fd);
        EXPECT_EQ(config->shm_size(), registration.size);
        EXPECT_EQ(config->client_base(), registration.base);
    }

    ASSERT_EQ(fclose(file), 0);
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

// ============================================================================
// Multi-storage test fixture
// ============================================================================
class SdkWrapperMultiStorageTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        client_config_ = CreateTestClientConfig();
        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = CreateMultiStorageConfigs();
    }

    void TearDown() override {
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

protected:
    std::unique_ptr<ClientConfig> client_config_;
    InitParams init_params_;
    std::string root_path_;

private:
    std::unique_ptr<ClientConfig> CreateTestClientConfig() {
        auto client_config = std::make_unique<ClientConfig>();
        std::string client_config_str = R"({
            "instance_group": "group",
            "instance_id": "instance",
            "address": ["127.0.0.1:8080"],
            "block_size": 128,
            "sdk_config": {
                "thread_num": 8,
                "queue_size": 2000,
                "sdk_backend_configs": [{"type": "file"}],
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
            "location_spec_infos": {"tp0": 1024}
        })";
        client_config->FromJsonString(client_config_str);
        return client_config;
    }

    std::string CreateMultiStorageConfigs() {
        return "["
               R"({"type":"file","global_unique_name":"nfs_a","storage_spec":{"root_path":")" +
               root_path_ + R"(/nfs_a/","key_count_per_file":2}},)"
               R"({"type":"file","global_unique_name":"nfs_b","storage_spec":{"root_path":")" +
               root_path_ + R"(/nfs_b/","key_count_per_file":2}})"
               "]";
    }
};

TEST_F(SdkWrapperMultiStorageTest, TestMixedStoragePutAndGet) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));

    // 同 backend 使用同 path（不同 blkid），避免 SDK 内部 SplitByPath 导致重排
    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=0&size=1024"),
        DataStorageUri("file://nfs_b/" + root_path_ + "/nfs_b/0/0/1?blkid=1&size=1024"),
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=2&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer(), BlockBuffer()};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk_wrapper.Put(remote_uris, local_buffers, actual_remote_uris));

    ASSERT_EQ(actual_remote_uris->size(), 3);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), remote_uris[0].ToUriString());
    ASSERT_EQ(actual_remote_uris->at(1).ToUriString(), remote_uris[1].ToUriString());
    ASSERT_EQ(actual_remote_uris->at(2).ToUriString(), remote_uris[2].ToUriString());

    ASSERT_EQ(ER_OK, sdk_wrapper.Get(*actual_remote_uris, local_buffers));
}

TEST_F(SdkWrapperMultiStorageTest, TestSingleStorageBackwardCompat) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));

    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=0&size=1024"),
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=1&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer()};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk_wrapper.Put(remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(actual_remote_uris->size(), 2);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), remote_uris[0].ToUriString());
    ASSERT_EQ(actual_remote_uris->at(1).ToUriString(), remote_uris[1].ToUriString());

    ASSERT_EQ(ER_OK, sdk_wrapper.Get(*actual_remote_uris, local_buffers));
}

TEST_F(SdkWrapperMultiStorageTest, TestMixedStorageWithInvalidSdk) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));

    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=0&size=1024"),
        DataStorageUri("file://unknown_backend/" + root_path_ + "/unknown/0/0/1?blkid=1&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer()};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_GETSDK_ERROR, sdk_wrapper.Put(remote_uris, local_buffers, actual_remote_uris));
    ASSERT_EQ(ER_GETSDK_ERROR, sdk_wrapper.Get(remote_uris, local_buffers));
}

TEST_F(SdkWrapperMultiStorageTest, TestGroupBySdk) {
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, sdk_wrapper.Init(client_config_, init_params_));

    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/1?blkid=0&size=1024"),
        DataStorageUri("file://nfs_b/" + root_path_ + "/nfs_b/0/0/2?blkid=1&size=1024"),
        DataStorageUri("file://nfs_a/" + root_path_ + "/nfs_a/0/0/3?blkid=2&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer(), BlockBuffer()};

    std::vector<SdkWrapper::SdkGroup> groups;
    ASSERT_EQ(ER_OK, sdk_wrapper.GroupBySdk(remote_uris, local_buffers, groups));
    ASSERT_EQ(groups.size(), 2);

    // 第一组 nfs_a: indices 0, 2
    ASSERT_EQ(groups[0].indices.size(), 2);
    ASSERT_EQ(groups[0].indices[0], 0);
    ASSERT_EQ(groups[0].indices[1], 2);
    ASSERT_EQ(groups[0].uris.size(), 2);
    ASSERT_EQ(groups[0].buffers.size(), 2);

    // 第二组 nfs_b: index 1
    ASSERT_EQ(groups[1].indices.size(), 1);
    ASSERT_EQ(groups[1].indices[0], 1);
    ASSERT_EQ(groups[1].uris.size(), 1);
    ASSERT_EQ(groups[1].buffers.size(), 1);
}
