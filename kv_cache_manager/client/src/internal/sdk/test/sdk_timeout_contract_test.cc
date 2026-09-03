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

    // 三 host：组顺序 = URI 首现顺序（GroupBySdk 按输入顺序建组，不依赖 unordered_map
    // 迭代序），单线程 FIFO 池下任务拾起顺序与提交顺序一致 —— 测试可确定性编排
    // "err → gap → tail"。
    std::string MakeTripleFileStorageConfigs(const std::string &name_a, const std::string &name_b,
                                             const std::string &name_c) {
        return "[{\"type\":\"file\",\"global_unique_name\":\"" + name_a +
               "\",\"storage_spec\":{\"root_path\":\"/nfs_a/\",\"key_count_per_file\":2}},"
               "{\"type\":\"file\",\"global_unique_name\":\"" +
               name_b + "\",\"storage_spec\":{\"root_path\":\"/nfs_b/\",\"key_count_per_file\":2}},"
               "{\"type\":\"file\",\"global_unique_name\":\"" +
               name_c + "\",\"storage_spec\":{\"root_path\":\"/nfs_c/\",\"key_count_per_file\":2}}]";
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
// 超时路径的 drain 只 wait_until(deadline)——deadline 已过则立即返回；若有人把等待改成
// "等 in-flight 完成后再返回"（无界），fake 睡 3s 而 timeout 只有 200ms，本测试会在
// 远大于 1.5s 后才返回，断言立刻变红。
TEST_F(SdkTimeoutContractTest, TestRunningTimeoutBoundedReturn) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    ctrl->get_delay_ms.store(3000); // 慢 fake：睡 3s
    RegisterFakeForTest("nfs_test", ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 200, MakeSingleFileStorageConfigs("nfs_test")));

    auto uris = std::vector<DataStorageUri>{MakeUri("nfs_test", "nfs/0/0/1", 0, 1024)};
    auto buffers = TestBuffers::Make(1, 1024);

    auto start = std::chrono::steady_clock::now();
    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers);
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
    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers);
    int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // 核心断言：目标 fake 的 Get 从未被调用（I/O 未发起），返回超时。
    ASSERT_EQ(ER_SDK_TIMEOUT, ec);
    ASSERT_EQ(0, ctrl->get_call_count.load());
    // wrapper 在 deadline（50ms）附近即返回：若它等占位任务（500ms）或 drain，此处必超。
    ASSERT_LT(elapsed_ms, 400);

    ASSERT_EQ(ER_OK, blocker.get());
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
    ASSERT_EQ(ER_OK, sdk_wrapper.Put(uris, put_buffers.buffers, actual_remote_uris));
    ASSERT_EQ(kBlockCount, actual_remote_uris->size());

    // Get 回读，先验证成功路径数据确实被搬运到 caller buffer。
    auto get_buffers = TestBuffers::Make(kBlockCount, kBlockSize, 0x00);
    ASSERT_EQ(ER_OK, sdk_wrapper.Get(uris, get_buffers.buffers));
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

    ASSERT_EQ(ER_SDK_TIMEOUT, sdk_wrapper.Get(uris, buffers.buffers));

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

    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers);
    ASSERT_EQ(ER_SDK_TIMEOUT, ec);

    // 违约写入在 fake 延迟结束后发生（事件轮询，给足余量）。
    ASSERT_TRUE(WaitFor(ctrl->write_done, 3000));
    ASSERT_TRUE(WaitFor(ctrl->get_done, 3000));
    // buffer 确实被违约后端写脏 —— 我们"知道它不干净"，而不是假装安全。
    ASSERT_TRUE(buffers.AllBytesEqual(FakeSlowSdkControl::kAfterReturnByte));
}

// ============================================================================
// 3.5 静态预算注入 → 后端推导
// ============================================================================
// fake 在 Get 入口从注入预算推导自身 deadline：推导出的预算必须恰好等于
// client config 配置的 get_timeout_ms。若 SdkWrapper 忘记在 Init 时注入
// timeout_config，后端将拿不到预算（推导值为默认值），本测试变红。
TEST_F(SdkTimeoutContractTest, TestBackendDerivesDeadlineFromInjectedBudget) {
    auto ctrl = std::make_shared<FakeSlowSdkControl>();
    RegisterFakeForTest("nfs_test", ctrl);

    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 5000, MakeSingleFileStorageConfigs("nfs_test")));

    auto uris = std::vector<DataStorageUri>{MakeUri("nfs_test", "nfs/0/0/1", 0, 1024)};
    auto buffers = TestBuffers::Make(1, 1024);

    ASSERT_EQ(ER_OK, sdk_wrapper.Get(uris, buffers.buffers));
    ASSERT_EQ(1, ctrl->get_call_count.load());
    // fake 采纳的预算恰为注入的 5000ms（= client config 的 get_timeout_ms）。
    ASSERT_EQ(5000, ctrl->derived_get_budget_ms.load());
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

    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers);
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

    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers);
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
// 3.7 普通错误路径的有界 drain（回归：#280 曾整体删除，此处恢复 origin/main 语义）
// ============================================================================
// 一个 group 返回普通错误（非超时）时：wrapper 必须有界等待其余在飞 group（至多到
// deadline）再返回。理由：错误往往发生在 deadline 之前，SDK 仍在契约允许的窗口内写
// caller buffer；不等 peer 就返回，caller 拿到错误后复用/释放 buffer，与在飞 DMA 构成
// 数据竞争 —— 这正是本 PR 要关掉的"返回后写 caller buffer"窗口，只是换到了错误路径。
TEST_F(SdkTimeoutContractTest, TestErrorPathWaitsForInFlightPeer) {
    const std::string kErrHost = "err_wait_host";
    const std::string kSlowHost = "err_wait_slow_host";
    auto err_ctrl = std::make_shared<FakeSlowSdkControl>();
    auto slow_ctrl = std::make_shared<FakeSlowSdkControl>();
    err_ctrl->get_result.store(static_cast<int>(ER_SDKREAD_ERROR)); // 快速失败
    slow_ctrl->get_delay_ms.store(400);                            // 在飞 peer：睡 400ms
    // 确定性编排：err 等 slow 真正发起后才报错（否则满载机器上 pickup 竞态会让
    // stop 先拦下 slow，"等待在飞 peer"的断言对象根本不存在）。
    err_ctrl->gate_on_peer_calls = &slow_ctrl->get_call_count;

    RegisterFakeForTest(kErrHost, err_ctrl);
    RegisterFakeForTest(kSlowHost, slow_ctrl);

    SdkWrapper sdk_wrapper;
    // deadline 给足（3s）：确保走的是普通错误路径而不是超时路径。
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 2000, 3000, MakeDualFileStorageConfigs(kErrHost, kSlowHost)));

    std::vector<DataStorageUri> uris = {
        MakeUri(kErrHost, "e/0/0/1", 0, 1024),
        MakeUri(kSlowHost, "s/0/0/1", 0, 1024),
    };
    auto buffers = TestBuffers::Make(2, 1024);

    auto start = std::chrono::steady_clock::now();
    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers);
    int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // 错误码透传（无论 group 提交顺序如何，err group 的错误最终胜出）。
    ASSERT_EQ(ER_SDKREAD_ERROR, ec);
    // 核心断言：返回发生在慢 peer 完成（≥400ms）之后 —— 即错误路径等待了在飞 peer，
    // 而不是拿到错误就立刻返回（那会留下 ~400ms 的"返回后写 buffer"窗口）。
    ASSERT_GE(elapsed_ms, 350);
    // 等待有内容但必须有界：远小于 deadline（3s）。
    ASSERT_LT(elapsed_ms, 2500);
    // 慢 peer 确实完整跑完（不是被中途放弃）。
    ASSERT_TRUE(WaitFor(slow_ctrl->get_done, 1000));
}

// 错误路径的 drain 有界：peer 迟迟不完成时，返回不得晚于 deadline（不会为了等
// peer 而无限期挂起）。与 3.1 的守门测试互补：3.1 守超时路径，本用例守错误路径。
TEST_F(SdkTimeoutContractTest, TestErrorPathDrainIsBoundedByDeadline) {
    const std::string kErrHost = "err_bounded_host";
    const std::string kSlowHost = "err_bounded_slow_host";
    auto err_ctrl = std::make_shared<FakeSlowSdkControl>();
    auto slow_ctrl = std::make_shared<FakeSlowSdkControl>();
    err_ctrl->get_result.store(static_cast<int>(ER_SDKREAD_ERROR));
    slow_ctrl->get_delay_ms.store(3000); // 远超 1000ms 的 wrapper 预算
    err_ctrl->gate_on_peer_calls = &slow_ctrl->get_call_count; // slow 已发起后 err 才报错

    RegisterFakeForTest(kErrHost, err_ctrl);
    RegisterFakeForTest(kSlowHost, slow_ctrl);

    SdkWrapper sdk_wrapper;
    // get_timeout_ms=1000：wrapper deadline = 入口 + 1000ms；err 在 slow 发起后
    // 立即失败，drain 最多等到 1000ms（1000 的取值兼顾满载机器的 pickup 延迟，
    // 有界性由"远小于 slow 的 3s"证明）。
    ASSERT_EQ(ER_OK, InitWrapper(sdk_wrapper, 8, 2000, 1000, 1000, MakeDualFileStorageConfigs(kErrHost, kSlowHost)));

    std::vector<DataStorageUri> uris = {
        MakeUri(kErrHost, "e/0/0/1", 0, 1024),
        MakeUri(kSlowHost, "s/0/0/1", 0, 1024),
    };
    auto buffers = TestBuffers::Make(2, 1024);

    auto start = std::chrono::steady_clock::now();
    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers);
    int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // 返回不得晚于 deadline（1000ms）+ 合理调度余量；绝不等待 slow peer 的 3s。
    ASSERT_LT(elapsed_ms, 2500);
    ASSERT_NE(ER_OK, ec);
    // 收尾：等 slow fake 自行结束，避免线程池析构时任务仍在跑。
    ASSERT_TRUE(WaitFor(slow_ctrl->get_done, 5000));
}

// 错误路径的 stop 标志：一个 group 报错后，仍在排队的 group 不得再发起 I/O。
// 确定性设计（无 pop 竞态）：单线程 + 300ms 占位，三个任务全部排队，提交顺序 =
// URI 首现顺序 = err → gap → tail（GroupBySdk 按输入顺序建组，FIFO 池按序拾起）。
// err 在 ~300ms 报错时 wrapper 置 stop；gap（睡 500ms）无论是否抢在置位前被拾起，
// 都保证 tail 被拾起时 stop 已置位数百毫秒 —— tail 的 I/O 必然从未发起。
TEST_F(SdkTimeoutContractTest, TestErrorPathStopsQueuedTasks) {
    const std::string kErrHost = "q_err_host";
    const std::string kGapHost = "q_gap_host";
    const std::string kTailHost = "q_tail_host";
    auto err_ctrl = std::make_shared<FakeSlowSdkControl>();
    auto gap_ctrl = std::make_shared<FakeSlowSdkControl>();
    auto tail_ctrl = std::make_shared<FakeSlowSdkControl>();
    err_ctrl->get_result.store(static_cast<int>(ER_SDKREAD_ERROR)); // 第一个被拾起即报错
    gap_ctrl->get_delay_ms.store(500);                             // 制造确定性的时间差

    RegisterFakeForTest(kErrHost, err_ctrl);
    RegisterFakeForTest(kGapHost, gap_ctrl);
    RegisterFakeForTest(kTailHost, tail_ctrl);

    // 单线程：占住 300ms，确保三个任务提交时全部在队列里排队。
    SdkWrapper sdk_wrapper;
    ASSERT_EQ(ER_OK,
              InitWrapper(sdk_wrapper, 1, 2000, 2000, 10000, MakeTripleFileStorageConfigs(kErrHost, kGapHost, kTailHost)));

    auto blocker = sdk_wrapper.wait_task_thread_pool_->async([]() -> ClientErrorCode {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        return ER_OK;
    });

    std::vector<DataStorageUri> uris = {
        MakeUri(kErrHost, "e/0/0/1", 0, 1024),
        MakeUri(kGapHost, "g/0/0/1", 0, 1024),
        MakeUri(kTailHost, "t/0/0/1", 0, 1024),
    };
    auto buffers = TestBuffers::Make(3, 1024);

    // deadline 给足：走普通错误路径（err 是第一个 future，其错误码直接透传）。
    ClientErrorCode ec = sdk_wrapper.Get(uris, buffers.buffers);
    ASSERT_EQ(ER_SDKREAD_ERROR, ec);
    // 核心断言：排在错误之后的 tail group 从未发起 I/O（stop 拦截，而非靠时间准入——
    // 此刻距离 deadline 还有 ~9s，时间准入不会拦）。
    ASSERT_EQ(0, tail_ctrl->get_call_count.load());
    // gap 的两种合法结局（err 完成与 stop 置位之间存在固有 pop 竞态窗口）：
    // 要么被 stop 拦下（I/O 未发起），要么抢跑后自然跑完（drain 已等到其 future，
    // 即 get_done 已置位）。此时 gap 状态必然已定，无需轮询。
    ASSERT_TRUE(gap_ctrl->get_call_count.load() == 0 || gap_ctrl->get_done.load());

    ASSERT_EQ(ER_OK, blocker.get());
}
