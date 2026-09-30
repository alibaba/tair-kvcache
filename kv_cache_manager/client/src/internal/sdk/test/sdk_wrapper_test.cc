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
#include "kv_cache_manager/client/src/internal/sdk/deadline_util.h"
#include "kv_cache_manager/client/src/internal/sdk/lock_free_thread_pool.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
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
               root_path_ +
               R"(/nfs_a/","key_count_per_file":2}},)"
               R"({"type":"file","global_unique_name":"nfs_b","storage_spec":{"root_path":")" +
               root_path_ +
               R"(/nfs_b/","key_count_per_file":2}})"
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

// ============================================================================
// ============================================================================

// 可控 fake SDK：记录 Get/Put 调用，可注入延迟；Init 观测 wrapper 注入的静态预算。
struct FakeSdkControl {
    std::atomic<int> get_call_count{0};
    std::atomic<int> put_call_count{0};
    std::atomic<int> get_delay_ms{0}; // Get 内的睡眠时长（模拟慢 I/O）
    std::atomic<int> get_result{static_cast<int>(ER_OK)};
    // Init 时观测到的注入预算（TestStaticBudgetInjection 用）：wrapper 应把自身
    // timeout_config 注入 SdkBackendConfig，后端据此自律。
    std::atomic<int> observed_get_budget_ms{0};
    std::atomic<int> observed_put_budget_ms{0};
};

class FakeSdk : public SdkInterface {
public:
    explicit FakeSdk(std::shared_ptr<FakeSdkControl> ctrl) : ctrl_(std::move(ctrl)) {}

    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                         const std::shared_ptr<StorageConfig> &) override {
        if (sdk_backend_config) {
            ctrl_->observed_get_budget_ms.store(sdk_backend_config->timeout_config().get_timeout_ms());
            ctrl_->observed_put_budget_ms.store(sdk_backend_config->timeout_config().put_timeout_ms());
        }
        return ER_OK;
    }
    SdkType Type() override { return SdkType::LOCAL_FILE; }
    ClientErrorCode Get(const std::vector<DataStorageUri> &, const BlockBuffers &) override {
        ctrl_->get_call_count.fetch_add(1);
        int delay_ms = ctrl_->get_delay_ms.load();
        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        return static_cast<ClientErrorCode>(ctrl_->get_result.load());
    }
    ClientErrorCode Put(const std::vector<DataStorageUri> &,
                        const BlockBuffers &,
                        std::shared_ptr<std::vector<DataStorageUri>>) override {
        ctrl_->put_call_count.fetch_add(1);
        return ER_OK;
    }

protected:
    ClientErrorCode Alloc(const std::vector<DataStorageUri> &, std::vector<DataStorageUri> &) override { return ER_OK; }

private:
    std::shared_ptr<FakeSdkControl> ctrl_;
};

// Test-only factory returning FakeSdk for NFS; production code has no test hooks.
class FakeSdkFactory : public SdkFactory {
public:
    explicit FakeSdkFactory(std::shared_ptr<FakeSdkControl> ctrl) : ctrl_(std::move(ctrl)) {}

    std::shared_ptr<SdkInterface> CreateSdk(const DataStorageType &type,
                                            const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                            const std::shared_ptr<StorageConfig> &storage_config) override {
        if (type == DataStorageType::DATA_STORAGE_TYPE_NFS) {
            auto sdk = std::make_shared<FakeSdk>(ctrl_);
            return sdk->Init(sdk_backend_config, storage_config) == ER_OK ? sdk : nullptr;
        }
        return SdkFactory::CreateSdk(type, sdk_backend_config, storage_config);
    }

private:
    std::shared_ptr<FakeSdkControl> ctrl_;
};

// Init wrapper with a fake factory injected (-fno-access-control).
// Caller keeps the returned factory alive for the wrapper's lifetime.
std::unique_ptr<FakeSdkFactory> InitWrapperWithFake(SdkWrapper &wrapper,
                                                    const std::shared_ptr<FakeSdkControl> &ctrl,
                                                    const std::unique_ptr<ClientConfig> &client_config,
                                                    InitParams &init_params,
                                                    ClientErrorCode &ec) {
    auto factory = std::make_unique<FakeSdkFactory>(ctrl);
    wrapper.sdk_factory_ = factory.get();
    ec = wrapper.Init(client_config, init_params);
    return factory;
}

// 覆盖 timeout_config 的 ClientConfig（其余字段与 fixture 一致）。
std::unique_ptr<ClientConfig> MakeClientConfigWithTimeouts(int put_timeout_ms, int get_timeout_ms) {
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
                "put_timeout_ms": )" + std::to_string(put_timeout_ms) + R"(,
                "get_timeout_ms": )" + std::to_string(get_timeout_ms) + R"(
            }
        },
        "model_deployment": {
            "model_name": "test_model", "dtype": "FP8", "use_mla": false,
            "tp_size": 1, "dp_size": 1, "pp_size": 1
        },
        "location_spec_infos": {"tp0": 1024}
    })";
    client_config->FromJsonString(client_config_str);
    return client_config;
}

// Wait for a condition to hold (deadline/admission events happen asynchronously).
bool WaitForTrue(const std::function<bool()> &cond, int max_retry = 300) {
    for (int i = 0; i < max_retry; ++i) {
        if (cond()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

// 验证准入修复：排队超过静态预算的任务不得再发起 I/O。
// 方法：get_timeout_ms 调小到 50ms + 占满线程池（8 线程各睡 500ms）→
// 分组任务必然在队列里等过 wrapper deadline → 启动时被准入检查拦下。
TEST_F(SdkWrapperTest, TestAdmissionRejectOnExpiredDeadline) {
    auto ctrl = std::make_shared<FakeSdkControl>();
    auto client_config = MakeClientConfigWithTimeouts(/*put_timeout_ms=*/2000, /*get_timeout_ms=*/50);
    SdkWrapper sdk_wrapper;
    ClientErrorCode init_ec = ER_OK;
    auto fake_factory = InitWrapperWithFake(sdk_wrapper, ctrl, client_config, init_params_, init_ec);
    ASSERT_EQ(ER_OK, init_ec);

    // 占满线程池的全部 8 个线程，每个睡眠 500ms。
    std::vector<std::future<ClientErrorCode>> blockers;
    for (size_t i = 0; i < 8; ++i) {
        blockers.push_back(sdk_wrapper.wait_task_thread_pool_->async([]() -> ClientErrorCode {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return ER_OK;
        }));
    }

    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_test/" + root_path_ + "/nfs/0/0/1?blkid=0&size=1024")};
    BlockBuffers local_buffers = {BlockBuffer()};

    auto start = std::chrono::steady_clock::now();
    ClientErrorCode ec = sdk_wrapper.Get(remote_uris, local_buffers);
    int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // 核心断言：fake SDK 的 Get 从未被调用（I/O 未发起），返回超时，且不等待在飞任务。
    ASSERT_EQ(ER_SDK_TIMEOUT, ec);
    ASSERT_EQ(0, ctrl->get_call_count.load());
    ASSERT_LT(elapsed_ms, 400); // 占位任务要 500ms 才结束；若等待它们则此处必超

    // 回收占位任务，避免线程池析构时等待未完成任务。
    for (auto &f : blockers) {
        f.get();
    }
}

// 验证超时有界返回：fake SDK 睡 1500ms、get_timeout_ms=200 → wrapper 必须在远小于
// fake 睡眠时长内返回（证明没有 drain / 等待 in-flight I/O 的阻塞）。
TEST_F(SdkWrapperTest, TestNoUnboundedWaitOnTimeout) {
    auto ctrl = std::make_shared<FakeSdkControl>();
    ctrl->get_delay_ms.store(1500);
    auto client_config = MakeClientConfigWithTimeouts(/*put_timeout_ms=*/2000, /*get_timeout_ms=*/200);
    SdkWrapper sdk_wrapper;
    ClientErrorCode init_ec = ER_OK;
    auto fake_factory = InitWrapperWithFake(sdk_wrapper, ctrl, client_config, init_params_, init_ec);
    ASSERT_EQ(ER_OK, init_ec);

    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_test/" + root_path_ + "/nfs/0/0/1?blkid=0&size=1024")};
    BlockBuffers local_buffers = {BlockBuffer()};

    auto start = std::chrono::steady_clock::now();
    ClientErrorCode ec = sdk_wrapper.Get(remote_uris, local_buffers);
    int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    ASSERT_EQ(ER_SDK_TIMEOUT, ec);
    ASSERT_LT(elapsed_ms, 1000); // 若有人改回"等 in-flight 完成"，此断言会立刻变红

    // 等 fake 的睡眠结束，避免线程池/进程退出时任务仍在跑。
    std::this_thread::sleep_for(std::chrono::milliseconds(1600));
}

// 验证静态预算注入：wrapper 在 Init 阶段把自身 timeout_config 注入
// SdkBackendConfig，后端（fake 观测面）读到的是同一份预算 —— 这是后端
// "从自身任务起点起算 deadline 并自律"的数据来源。
TEST_F(SdkWrapperTest, TestStaticBudgetInjection) {
    auto ctrl = std::make_shared<FakeSdkControl>();
    SdkWrapper sdk_wrapper;
    ClientErrorCode init_ec = ER_OK;
    auto fake_factory = InitWrapperWithFake(sdk_wrapper, ctrl, client_config_, init_params_, init_ec);
    ASSERT_EQ(ER_OK, init_ec);

    // fixture 的 client_config_：put_timeout_ms=2000, get_timeout_ms=2000。
    ASSERT_EQ(2000, ctrl->observed_get_budget_ms.load());
    ASSERT_EQ(2000, ctrl->observed_put_budget_ms.load());
}

// 用于直接调用受保护方法 SplitByPath 的最小实现。
class TestSplitSdk : public SdkInterface {
public:
    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &, const std::shared_ptr<StorageConfig> &) override {
        return ER_OK;
    }
    SdkType Type() override { return SdkType::LOCAL_FILE; }
    ClientErrorCode Get(const std::vector<DataStorageUri> &, const BlockBuffers &) override { return ER_OK; }
    ClientErrorCode Put(const std::vector<DataStorageUri> &,
                        const BlockBuffers &,
                        std::shared_ptr<std::vector<DataStorageUri>>) override {
        return ER_OK;
    }

protected:
    ClientErrorCode Alloc(const std::vector<DataStorageUri> &, std::vector<DataStorageUri> &) override { return ER_OK; }
};

// 验证保序：交错多 path 输入时，每组 indices 记录原始下标。
TEST_F(SdkWrapperTest, TestSplitByPathRecordsIndices) {
    TestSplitSdk sdk;
    std::vector<DataStorageUri> remote_uris = {
        DataStorageUri("file://nfs_test/shared?blkid=0&size=1024"),
        DataStorageUri("file://nfs_test/other?blkid=0&size=1024"),
        DataStorageUri("file://nfs_test/shared?blkid=1&size=1024"),
        DataStorageUri("file://nfs_test/other?blkid=1&size=1024"),
    };
    BlockBuffers local_buffers = {BlockBuffer(), BlockBuffer(), BlockBuffer(), BlockBuffer()};

    auto groups = sdk.SplitByPath(remote_uris, local_buffers);
    ASSERT_EQ(groups.size(), 2);

    const auto &shared = groups.at("/shared");
    ASSERT_EQ(shared.indices.size(), 2);
    ASSERT_EQ(shared.indices[0], 0);
    ASSERT_EQ(shared.indices[1], 2);
    ASSERT_EQ(shared.remote_uris.size(), 2);
    ASSERT_EQ(shared.remote_uris[0].ToUriString(), remote_uris[0].ToUriString());
    ASSERT_EQ(shared.remote_uris[1].ToUriString(), remote_uris[2].ToUriString());
    ASSERT_EQ(shared.local_buffers.size(), 2);

    const auto &other = groups.at("/other");
    ASSERT_EQ(other.indices.size(), 2);
    ASSERT_EQ(other.indices[0], 1);
    ASSERT_EQ(other.indices[1], 3);
    ASSERT_EQ(other.remote_uris[0].ToUriString(), remote_uris[1].ToUriString());
    ASSERT_EQ(other.remote_uris[1].ToUriString(), remote_uris[3].ToUriString());
}
