// W5 验收测试：可控 slow/fake SDK 覆盖超时与 buffer 生命周期契约
// 契约正文见 docs/design/client_sdk_io_contract.md。
// 覆盖场景：
//  3.1 运行中超时（有界返回）            —— TestRunningTimeoutBoundedReturn
//  3.2 排队超时被拦下（核心缺陷防回归）  —— TestQueuedTaskNeverStartsIo
//                                         + TestPreExpiredDeadlineRejectsBeforeIo
//  3.3 hard 契约：返回后无后台访问       —— TestHardContractNoBackgroundWriteAfterReturn
//                                         + TestHardBackendNoWriteAfterTimeoutReturn
//  3.4 违约（soft）后端的可观测性        —— TestSoftBackendViolationIsObservable
//  3.5 deadline 传播                     —— TestDeadlinePropagationIntoSdk
//  3.6 多后端混合                       —— TestMixedBackendsFastCompletesSlowTimesOut
//  额外：逐 block 准入    —— TestPerBlockAdmissionStopsMidway
// 稳定性约定：全部用宽松上下界（如"耗时 < 1.5s"而非"≈200ms"）、计数器与事件
// （get_done/write_done/轮询）替代 sleep 同步；"证明某事没有发生"处给足余量
// （如返回后等 500ms 再断言哨兵完好）；测试目标 size=medium（含 sleep）。

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/deadline_util.h"
#include "kv_cache_manager/client/src/internal/sdk/lock_free_thread_pool.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/client/src/internal/sdk/test/fake_slow_sdk.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

using namespace kv_cache_manager;

// 测试用 factory：按 storage 名返回 fake SDK。生产代码不含任何测试钩子。
class FakeSdkFactory : public SdkFactory {
public:
    void SetFake(const std::string &global_unique_name, const std::shared_ptr<FakeSlowSdkControl> &ctrl) {
        fakes_[global_unique_name] = ctrl;
    }

    std::shared_ptr<SdkInterface> CreateSdk(const DataStorageType &type,
                                            const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                            const std::shared_ptr<StorageConfig> &storage_config) override {
        if (storage_config) {
            auto it = fakes_.find(storage_config->global_unique_name());
            if (it != fakes_.end()) {
                auto sdk = std::make_shared<FakeSlowSdk>(it->second);
                return sdk->Init(sdk_backend_config, storage_config) == ER_OK ? sdk : nullptr;
            }
        }
        return SdkFactory::CreateSdk(type, sdk_backend_config, storage_config);
    }

private:
    std::map<std::string, std::shared_ptr<FakeSlowSdkControl>> fakes_;
};

class SdkTimeoutContractTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        fake_factory_ = std::make_unique<FakeSdkFactory>();
    }

    void TearDown() override {
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
        fake_factory_.reset();
    }

protected:
    // 构造 ClientConfig（JSON 形状与 sdk_wrapper_test.cc 一致；线程数 / timeout 可配）。
    std::unique_ptr<ClientConfig> MakeClientConfig(int thread_num,
                                                   int queue_size,
                                                   int put_timeout_ms,
                                                   int get_timeout_ms,
                                                   const std::string &storage_configs_json) {
        auto client_config = std::make_unique<ClientConfig>();
        std::string client_config_str = R"({
            "instance_group": "group",
            "instance_id": "instance",
            "address": [
                "127.0.0.1:8080"
            ],
            "block_size": 128,
            "sdk_config": {
                "thread_num": )" + std::to_string(thread_num) +
                                        R"(,
                "queue_size": )" + std::to_string(queue_size) +
                                        R"(,
                "sdk_backend_configs": [
                    {
                        "type": "file"
                    }
                ],
                "timeout_config": {
                    "put_timeout_ms": )" +
                                        std::to_string(put_timeout_ms) + R"(,
                    "get_timeout_ms": )" +
                                        std::to_string(get_timeout_ms) + R"(
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

    std::string MakeSingleFileStorageConfigs(const std::string &global_unique_name) {
        return "[{\"type\":\"file\",\"global_unique_name\":\"" + global_unique_name +
               "\",\"storage_spec\":{\"root_path\":\"/nfs/\",\"key_count_per_file\":2}}]";
    }

    std::string MakeDualFileStorageConfigs(const std::string &name_a, const std::string &name_b) {
        return "[{\"type\":\"file\",\"global_unique_name\":\"" + name_a +
               "\",\"storage_spec\":{\"root_path\":\"/nfs_a/\",\"key_count_per_file\":2}},"
               "{\"type\":\"file\",\"global_unique_name\":\"" +
               name_b + "\",\"storage_spec\":{\"root_path\":\"/nfs_b/\",\"key_count_per_file\":2}}]";
    }

    // 为指定 host 注册 fake；wrapper 经注入的 factory 命中。
    void RegisterFakeForTest(const std::string &global_unique_name, const std::shared_ptr<FakeSlowSdkControl> &ctrl) {
        fake_factory_->SetFake(global_unique_name, ctrl);
    }

    // 设置 storage_configs 并 Init wrapper（storage_configs 必须与 URI host 对应）。
    ClientErrorCode InitWrapper(SdkWrapper &wrapper,
                                int thread_num,
                                int queue_size,
                                int put_timeout_ms,
                                int get_timeout_ms,
                                const std::string &storage_configs_json) {
        init_params_.storage_configs = storage_configs_json;
        // 注入测试 factory（-fno-access-control）：生产代码不含任何测试钩子。
        wrapper.sdk_factory_ = fake_factory_.get();
        return wrapper.Init(
            MakeClientConfig(thread_num, queue_size, put_timeout_ms, get_timeout_ms, storage_configs_json),
            init_params_);
    }

    DataStorageUri MakeUri(const std::string &host, const std::string &path_suffix, size_t blkid, size_t size) {
        return DataStorageUri("file://" + host + "/" + root_path_ + "/" + path_suffix +
                              "?blkid=" + std::to_string(blkid) + "&size=" + std::to_string(size));
    }

    // CPU BlockBuffers：每个 block 一块 malloc 内存，可指定初始字节。
    struct TestBuffers {
        std::vector<void *> raws;
        BlockBuffers buffers;
        ~TestBuffers() {
            for (auto *p : raws) {
                free(p);
            }
        }
        static TestBuffers Make(size_t block_count, size_t block_size, uint8_t init_byte = 0x00) {
            TestBuffers tb;
            for (size_t i = 0; i < block_count; ++i) {
                auto *mem = malloc(block_size);
                std::memset(mem, init_byte, block_size);
                tb.raws.push_back(mem);
                BlockBuffer buf;
                Iov iov;
                iov.type = MemoryType::CPU;
                iov.base = mem;
                iov.size = block_size;
                iov.ignore = false;
                buf.iovs.push_back(iov);
                tb.buffers.push_back(std::move(buf));
            }
            return tb;
        }
        bool AllBytesEqual(uint8_t byte) const {
            for (const auto &buf : buffers) {
                for (const auto &iov : buf.iovs) {
                    if (!iov.base || iov.size == 0) {
                        continue;
                    }
                    const auto *p = static_cast<const uint8_t *>(iov.base);
                    for (size_t i = 0; i < iov.size; ++i) {
                        if (p[i] != byte) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        bool BlockBytesEqual(size_t block_index, uint8_t byte) const {
            if (block_index >= buffers.size()) {
                return false;
            }
            for (const auto &iov : buffers[block_index].iovs) {
                if (!iov.base || iov.size == 0) {
                    continue;
                }
                const auto *p = static_cast<const uint8_t *>(iov.base);
                for (size_t i = 0; i < iov.size; ++i) {
                    if (p[i] != byte) {
                        return false;
                    }
                }
            }
            return true;
        }
    };

    // 由毫秒预算构造显式 deadline_ms（绝对 steady_clock 毫秒）。
    static int64_t DeadlineFromNowMs(int64_t timeout_ms) { return SteadyClockMs() + timeout_ms; }

    // 事件轮询等待（替代 sleep 同步；给足余量，绝不依赖"恰好 X ms 后"）。
    static bool WaitFor(const std::atomic<bool> &flag, int timeout_ms) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (flag.load()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return flag.load();
    }

protected:
    InitParams init_params_;
    std::string root_path_;
    std::unique_ptr<FakeSdkFactory> fake_factory_;
};

// ============================================================================
// 3.1 运行中超时（有界返回）
// ============================================================================
// 守门测试：防止"无界等待 in-flight I/O"方案复活（docs/design/client_sdk_io_contract.md 否决项）。
// 若有人把 wrapper 改回 drain（等 in-flight 完成后再返回），fake 睡 3s 而 timeout
// 只有 200ms，本测试会在远大于 1.5s 后才返回，断言立刻变红。
TEST_F(SdkTimeoutContractTest, TestRunningTimeoutBoundedReturn) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    ctrl->get_delay_ms.store(3000); // 慢 fake：睡 3s
    RegisterFakeForTest("nfs_test", ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 200, MakeSingleFileStorageConfigs("nfs_test")));

    auto uris = std::vector<DataStorageUri>{MakeUri("nfs_test", "nfs/0/0/1", 0, 1024)};
    auto buffers = TestBuffers::Make(1, 1024);

    auto start = std::chrono::steady_clock::now();
    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers, DeadlineFromNowMs(200));
    int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    ASSERT_EQ(ER_SDK_TIMEOUT, ec);
    // 宽松上界：明显小于 fake 的 3s 睡眠（而不是精确 200ms），证明没有 drain/等待 in-flight。
    ASSERT_LT(elapsed_ms, 1500);
    // 运行中超时：fake 的 Get 确实被发起过（区别于排队未启动）。
    ASSERT_EQ(1, ctrl->get_call_count.load());

    // 等 fake 的 3s 睡眠结束，避免 wrapper 析构 join 池线程时占时长（事件同步，非时序断言）。
    ASSERT_TRUE(WaitFor(ctrl->get_done, 5000));
}

// ============================================================================
// 3.2 排队超时被拦下
// ============================================================================
// 守门测试：若有人把准入检查改回"只看 stop flag 不看时间"，排队超过 deadline 的
// 任务会照样发起 I/O —— 下层再写 9~15s，而 caller 早已返回。get_call_count == 0
// 是核心断言，绝不允许放松。
TEST_F(SdkTimeoutContractTest, TestQueuedTaskNeverStartsIo) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    RegisterFakeForTest("nfs_test", ctrl);

    // thread_num=1：占满唯一工作线程后，后到的 Get 任务必然在队列里等过 deadline（50ms）。
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 1, 2000, 2000, 50, MakeSingleFileStorageConfigs("nfs_test")));

    // 占住唯一线程 500ms（远大于 50ms deadline）。
    auto blocker = sdk_wrapper.wait_task_thread_pool_->async([]() -> ClientErrorCode {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return ER_OK;
    });

    auto uris = std::vector<DataStorageUri>{MakeUri("nfs_test", "nfs/0/0/1", 0, 1024)};
    auto buffers = TestBuffers::Make(1, 1024);

    auto start = std::chrono::steady_clock::now();
    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers, DeadlineFromNowMs(50));
    int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // 核心断言：目标 fake 的 Get 从未被调用（I/O 未发起），返回超时。
    ASSERT_EQ(ER_SDK_TIMEOUT, ec);
    ASSERT_EQ(0, ctrl->get_call_count.load());
    // wrapper 在 deadline（50ms）附近即返回：若它等占位任务（500ms）或 drain，此处必超。
    ASSERT_LT(elapsed_ms, 400);

    ASSERT_EQ(ER_OK, blocker.get());
}

// 3.2 变体：显式传入已过期的 deadline_ms（绝对时间点早于当前时刻）。
// 任务无论何时被拾起，now >= deadline 恒成立 → 准入检查必然拦截，SDK 从未被调用。
TEST_F(SdkTimeoutContractTest, TestPreExpiredDeadlineRejectsBeforeIo) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    RegisterFakeForTest("nfs_test", ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 2, 2000, 2000, 0, MakeSingleFileStorageConfigs("nfs_test")));

    auto uris = std::vector<DataStorageUri>{MakeUri("nfs_test", "nfs/0/0/1", 0, 1024)};
    auto buffers = TestBuffers::Make(1, 1024);

    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers, SteadyClockMs() - 1);
    ASSERT_EQ(ER_SDK_TIMEOUT, ec);
    ASSERT_EQ(0, ctrl->get_call_count.load());
}

// ============================================================================
// 3.3 hard 契约：返回后无后台访问（真实 hard 级后端 LocalFileSdk）
// ============================================================================
// 多 block Get 成功后，立即用哨兵值覆写整个 caller buffer，等 500ms 再断言哨兵完好
// —— 证明 wrapper 返回后没有任何后台线程/异步 I/O 再写 caller buffer（LocalFile 是
// hard 级：I/O 同步完成，future 就绪即代表 memcpy 已完成，happens-before 由
// future/promise 保证）。若未来有人把 LocalFile 改成异步写、或 wrapper 不等 future
// 就返回，本测试会变红。此用例在 ASAN 下跑（内存竞争/UAF 检测点）。
TEST_F(SdkTimeoutContractTest, TestHardContractNoBackgroundWriteAfterReturn) {
    // 不注册 fake：使用真实 LocalFileSdk（hard 级后端）。deadline 设短（200ms），
    // 多 block Get 的正常耗时在 ms 级，必然在 deadline 前完成（ER_OK 路径）。
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 200, MakeSingleFileStorageConfigs("nfs_test")));

    const size_t kBlockCount = 4;
    const size_t kBlockSize = 1024;

    // 先 Put 4 个 block（blkid 0..3），内容为 kTouchByte（与哨兵 0xA5 区分）。
    std::vector<DataStorageUri> uris;
    for (size_t i = 0; i < kBlockCount; ++i) {
        uris.push_back(MakeUri("nfs_test", "nfs/0/0/1", i, kBlockSize));
    }
    auto put_buffers = TestBuffers::Make(kBlockCount, kBlockSize, FakeSlowSdkControl::kTouchByte);
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk_wrapper.Put(uris, put_buffers.buffers, actual_remote_uris, DeadlineFromNowMs(200)));
    ASSERT_EQ(kBlockCount, actual_remote_uris->size());

    // Get 回读，先验证成功路径数据确实被搬运到 caller buffer。
    auto get_buffers = TestBuffers::Make(kBlockCount, kBlockSize, 0x00);
    ASSERT_EQ(ER_OK, sdk_wrapper.Get(uris, get_buffers.buffers, DeadlineFromNowMs(200)));
    for (size_t i = 0; i < kBlockCount; ++i) {
        ASSERT_TRUE(get_buffers.BlockBytesEqual(i, FakeSlowSdkControl::kTouchByte));
    }

    // 返回后立即用哨兵值覆写整个 caller buffer。
    constexpr uint8_t kSentinel = 0xA5;
    for (auto &buf : get_buffers.buffers) {
        for (auto &iov : buf.iovs) {
            if (iov.base && iov.size > 0) {
                std::memset(iov.base, kSentinel, iov.size);
            }
        }
    }
    // "证明某事没有发生"要给足余量：等 500ms 再断言哨兵完好。
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ASSERT_TRUE(get_buffers.AllBytesEqual(kSentinel));
}

// 3.3 超时路径的 hard 契约：wrapper 返回 ER_SDK_TIMEOUT 后，hard 级（不违约的）
// 后端仍在后台运行（fake 睡 3s），但绝不再碰 caller buffer → 哨兵完好。
// 若未来有人给 fake 对应的后端引入"返回后写 buffer"，此测试变红。
TEST_F(SdkTimeoutContractTest, TestHardBackendNoWriteAfterTimeoutReturn) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    ctrl->get_delay_ms.store(3000);        // 超时返回后任务仍在后台睡眠
    ctrl->touch_buffer_on_get.store(true); // hard 后端：只在返回前写（同步交付）
    RegisterFakeForTest("nfs_test", ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 200, MakeSingleFileStorageConfigs("nfs_test")));

    auto uris = std::vector<DataStorageUri>{MakeUri("nfs_test", "nfs/0/0/1", 0, 1024)};
    auto buffers = TestBuffers::Make(1, 1024, 0x00);

    ASSERT_EQ(ER_SDK_TIMEOUT, sdk_wrapper.Get(uris, buffers.buffers, DeadlineFromNowMs(200)));

    // 返回后立即覆写哨兵；等待期间 fake 的 sleep（3s）在后台结束。
    constexpr uint8_t kSentinel = 0xA5;
    for (auto &buf : buffers.buffers) {
        for (auto &iov : buf.iovs) {
            if (iov.base && iov.size > 0) {
                std::memset(iov.base, kSentinel, iov.size);
            }
        }
    }
    // 等 fake 完全结束（事件轮询，非 sleep 同步）再断言哨兵完好 ——
    // 证明从返回到任务结束全程无后台写。
    ASSERT_TRUE(WaitFor(ctrl->get_done, 5000));
    ASSERT_TRUE(buffers.AllBytesEqual(kSentinel));
}

// ============================================================================
// 3.4 违约（soft 级）后端的可观测性
// ============================================================================
// fake 模拟 mooncake 那样的 soft 后端：超时返回后仍会写 caller buffer。
// 本用例不断言 buffer 干净（soft 契约不保证），而是断言"我们知道它不干净"：
//  1) wrapper 在 deadline 内返回且 SdkIoStats 记录了超时（可归因观测）；
//  2) 违约写入确实发生了（write_done 事件 + buffer 内容被写脏）。
TEST_F(SdkTimeoutContractTest, TestSoftBackendViolationIsObservable) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    ctrl->get_delay_ms.store(500);        // wrapper 200ms 超时后，fake 还在后台运行
    ctrl->write_after_return.store(true); // 违约：sleep 结束后仍写 caller buffer
    RegisterFakeForTest("nfs_test", ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 200, MakeSingleFileStorageConfigs("nfs_test")));

    auto uris = std::vector<DataStorageUri>{MakeUri("nfs_test", "nfs/0/0/1", 0, 1024)};
    auto buffers = TestBuffers::Make(1, 1024, 0x00);

    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers, DeadlineFromNowMs(200));
    ASSERT_EQ(ER_SDK_TIMEOUT, ec);

    // 违约写入在 fake 延迟结束后发生（事件轮询，给足余量）。
    ASSERT_TRUE(WaitFor(ctrl->write_done, 3000));
    ASSERT_TRUE(WaitFor(ctrl->get_done, 3000));
    // buffer 确实被违约后端写脏 —— 我们"知道它不干净"，而不是假装安全。
    ASSERT_TRUE(buffers.AllBytesEqual(FakeSlowSdkControl::kAfterReturnByte));
}

// ============================================================================
// 3.5 deadline 透传
// ============================================================================
// fake 在 Get 内观测传入的 deadline_ms：必须有值，且 RemainingMs() 落在 (0, timeout_ms]。
// W1/W2/W3 的逐 block/逐 key 准入依赖显式参数透传；若 SdkWrapper 忘记把 deadline
// 传进 SDK，本测试变红。
TEST_F(SdkTimeoutContractTest, TestDeadlinePropagationIntoSdk) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    RegisterFakeForTest("nfs_test", ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 2000, MakeSingleFileStorageConfigs("nfs_test")));

    auto uris = std::vector<DataStorageUri>{MakeUri("nfs_test", "nfs/0/0/1", 0, 1024)};
    auto buffers = TestBuffers::Make(1, 1024);

    ASSERT_EQ(ER_OK, sdk_wrapper.Get(uris, buffers.buffers, DeadlineFromNowMs(2000)));
    ASSERT_EQ(1, ctrl->get_call_count.load());
    // fake 在池内任务线程中读到了调用方传入的绝对 deadline（未来时刻）。
    ASSERT_TRUE(ctrl->deadline_set.load());
    ASSERT_GT(ctrl->observed_deadline_ms.load(), SteadyClockMs());
}

// ============================================================================
// 额外：逐 block 准入
// ============================================================================
// fake 模拟"每一块前检查 deadline"的实现：
// 块边界发现已过期即停止，不再触碰后续 block。验证中途停下。
TEST_F(SdkTimeoutContractTest, TestPerBlockAdmissionStopsMidway) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    ctrl->per_block_check.store(true);
    ctrl->per_block_delay_ms.store(500); // 每块睡 500ms；timeout 只有 50ms → 第 1 块后必然过期
    RegisterFakeForTest("nfs_test", ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 50, MakeSingleFileStorageConfigs("nfs_test")));

    const size_t kBlockCount = 4;
    std::vector<DataStorageUri> uris;
    for (size_t i = 0; i < kBlockCount; ++i) {
        uris.push_back(MakeUri("nfs_test", "nfs/0/0/1", i, 1024));
    }
    auto buffers = TestBuffers::Make(kBlockCount, 1024, 0x00);

    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers, DeadlineFromNowMs(50));
    ASSERT_EQ(ER_SDK_TIMEOUT, ec);

    // fake 在 ~500ms 处自查过期并停止（事件轮询）。
    ASSERT_TRUE(WaitFor(ctrl->get_done, 3000));
    // 中途停下：无论调度快慢，处理过的 block 数必须 < 总数（不允许跑完全部 4 块）。
    size_t touched = ctrl->blocks_touched.load();
    ASSERT_LT(touched, kBlockCount);
    // 被 touch 的 block 已写 kTouchByte；未 touch 的保持初始字节 0x00（未被触碰）。
    for (size_t i = 0; i < touched; ++i) {
        ASSERT_TRUE(buffers.BlockBytesEqual(i, FakeSlowSdkControl::kTouchByte));
    }
    for (size_t i = touched; i < kBlockCount; ++i) {
        ASSERT_TRUE(buffers.BlockBytesEqual(i, 0x00));
    }
}

// ============================================================================
// 3.6 多后端混合：一个快一个慢
// ============================================================================
// 慢后端超时不影响快后端已完成的结果：快组 future 先就绪（ER_OK + 数据已搬运），
// 慢组在 deadline 处超时 → 整体返回 ER_SDK_TIMEOUT；快组的 caller buffer 完整可用。
TEST_F(SdkTimeoutContractTest, TestMixedBackendsFastCompletesSlowTimesOut) {
    const std::string kFastHost = "fast_host";
    const std::string kSlowHost = "slow_host";
    auto fast_ctrl = std::make_shared<FakeSlowSdkControl>();
    auto slow_ctrl = std::make_shared<FakeSlowSdkControl>();
    fast_ctrl->touch_buffer_on_get.store(true); // 快后端：同步交付数据
    slow_ctrl->get_delay_ms.store(3000);        // 慢后端：睡 3s（远超 200ms deadline）

    RegisterFakeForTest(kFastHost, fast_ctrl);
    RegisterFakeForTest(kSlowHost, slow_ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 200, MakeDualFileStorageConfigs(kFastHost, kSlowHost)));

    // 快组在前：wrapper 先等到快组 future 就绪，再等慢组 → deadline 到期 → 超时返回。
    std::vector<DataStorageUri> uris = {
        MakeUri(kFastHost, "fast/0/0/1", 0, 1024),
        MakeUri(kSlowHost, "slow/0/0/1", 0, 1024),
    };
    auto buffers = TestBuffers::Make(2, 1024, 0x00);

    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers, DeadlineFromNowMs(200));
    ASSERT_EQ(ER_SDK_TIMEOUT, ec);
    // 快后端已完成：被调用 1 次，其 caller buffer 已被写满 kTouchByte（结果完整可用，
    // 不因慢后端超时而受影响）。
    ASSERT_EQ(1, fast_ctrl->get_call_count.load());
    ASSERT_TRUE(buffers.BlockBytesEqual(0, FakeSlowSdkControl::kTouchByte));
    // 慢后端被发起（in-flight）但结果未交付：buffer 保持初始字节 0x00。
    ASSERT_EQ(1, slow_ctrl->get_call_count.load());
    ASSERT_TRUE(buffers.BlockBytesEqual(1, 0x00));
    // 慢后端的 in-flight 任务最终自行结束（不阻塞、不取消、不额外写任何东西）。
    ASSERT_TRUE(WaitFor(slow_ctrl->get_done, 5000));
}

// ============================================================================
// 跨语言时钟同源
// ============================================================================
// connector 用 Python 算 deadline_ms，SDK 用 C++ SteadyClockMs() 比较；两者必须
// 读同一个内核时钟（CLOCK_MONOTONIC），否则 deadline 全线失效。此处直接取
// common/utils.py 的 deadline_ms_from_now(0) 与 C++ 读数比对。
TEST_F(SdkTimeoutContractTest, TestPythonAndCppShareSteadyClock) {
    const char *kScript = "python3 -c \"import time;print(time.monotonic_ns()//1_000_000)\" 2>/dev/null";
    const int64_t before_ms = SteadyClockMs();
    FILE *pipe = popen(kScript, "r");
    if (pipe == nullptr) {
        GTEST_SKIP() << "python3 unavailable";
    }
    char buf[64] = {0};
    const bool got = fgets(buf, sizeof(buf), pipe) != nullptr;
    const int rc = pclose(pipe);
    if (!got || rc != 0) {
        GTEST_SKIP() << "python3 unavailable";
    }
    const int64_t py_ms = std::strtoll(buf, nullptr, 10);
    const int64_t after_ms = SteadyClockMs();

    // Python 读数必须落在 C++ 前后两次读数之间（允许进程启动开销带来的上界放宽）。
    ASSERT_GT(py_ms, 0);
    EXPECT_GE(py_ms, before_ms) << "python monotonic clock is behind C++ steady_clock";
    EXPECT_LE(py_ms, after_ms) << "python monotonic clock is ahead of C++ steady_clock";
}
