#pragma once

// 可控 slow/fake SDK（W5 验收测试替身，仅测试使用）。
// 能力：
//  - get_delay_ms / set_delay(ms)          ：模拟慢 I/O（运行中超时场景）
//  - per_block_check + per_block_delay_ms  ：模拟逐 block 慢 + 每块前准入检查，
//                                            验证"中途停下"
//  - write_after_return                    ：模拟"返回后仍写 caller buffer"的违约
//                                            （soft 级，如 mooncake）后端
//  - get_call_count() / put_call_count()   ：断言超时任务从未发起 I/O（准入检查核心验证）
//  - derived_get_budget_ms                 ：Get 从注入预算推导的 deadline 与自身
//                                            起点的差值（验证静态预算注入→推导）
//  - touch_buffer_on_get                   ：正常路径写 kTouchByte，验证成功时数据确实被搬运
// 稳定性约定：所有跨线程控制字段都是 std::atomic（池内工作线程与测试线程共享）；
// 完成事件用 get_done / write_done / blocks_touched 暴露，测试用轮询替代 sleep 同步，
// 避免紧耦合时序断言（main 7f7f20c2 之后不允许引入 flaky 测试）。

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/deadline_util.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// fake 的控制面与观测面。同一实例可被多个池内任务并发访问，全部用原子量。
struct FakeSlowSdkControl {
    std::atomic<int> get_call_count{0};           // Get 被发起的次数（准入检查的核心断言对象）
    std::atomic<int> put_call_count{0};           // Put 被发起的次数
    std::atomic<int> get_delay_ms{0};             // Get 内的睡眠时长（模拟慢 I/O）
    std::atomic<int> per_block_delay_ms{0};       // 逐 block 睡眠时长（模拟逐 block 慢）
    std::atomic<bool> per_block_check{false};     // 开启逐 block 前置准入检查
    std::atomic<bool> write_after_return{false};  // 违约：sleep 结束后仍写 caller buffer
    std::atomic<bool> touch_buffer_on_get{false}; // 正常路径：Get 返回前写 kTouchByte
    std::atomic<int> get_result{static_cast<int>(ER_OK)};

    // Get 入口对静态预算的推导观测：deadline（= 自身起点 + 注入预算）与自身起点
    // 的差值，即后端实际采纳的预算（3.5 静态预算注入→推导验证）。
    std::atomic<int64_t> derived_get_budget_ms{0};

    // 确定性门控：非空时 Get 在返回结果前等待 *gate_on_peer_calls >= 1（最多 5s）。
    // 用于"错误发生时 peer 已在飞"的编排，消除线程池 pickup 时序竞态（满载机器上
    // pickup 可晚于错误返回，导致 peer 被 stop 拦下、测试断言失效）。
    std::atomic<int> *gate_on_peer_calls{nullptr};

    // 事件（原子标志，测试轮询等待；替代"睡 X ms 后断言"的紧耦合时序）。
    std::atomic<bool> get_done{false};     // Get 完全结束（含延迟与违约写入）
    std::atomic<bool> write_done{false};   // 违约写入完成后置位
    std::atomic<size_t> blocks_touched{0}; // 逐 block 模式：已处理（touch）的 block 数

    static constexpr uint8_t kTouchByte = 0x3C;       // 正常搬运（成功交付）的哨兵字节
    static constexpr uint8_t kAfterReturnByte = 0x5A; // 违约后端返回后写入的字节
    std::atomic<uint8_t> after_return_byte{kAfterReturnByte};
};

// 实现 SdkInterface 的测试替身。Alloc 是 protected —— 按任务卡要求实现为恒等回显。
class FakeSlowSdk : public SdkInterface {
public:
    explicit FakeSlowSdk(std::shared_ptr<FakeSlowSdkControl> ctrl) : ctrl_(std::move(ctrl)) {}

    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                         const std::shared_ptr<StorageConfig> &) override {
        if (sdk_backend_config) {
            timeout_config_ = sdk_backend_config->timeout_config();
        }
        return ER_OK;
    }
    SdkType Type() override { return SdkType::LOCAL_FILE; }

    ClientErrorCode Get(const std::vector<DataStorageUri> &, const BlockBuffers &local_buffers) override {
        ctrl_->get_call_count.fetch_add(1);
        // 与生产后端一致：从注入预算推导自身 deadline（任务起点起算）。
        const int64_t start_ms = SteadyClockMs();
        const int64_t deadline_ms = start_ms + timeout_config_.get_timeout_ms();
        ctrl_->derived_get_budget_ms.store(deadline_ms - start_ms);
        ctrl_->get_done.store(false);
        ctrl_->write_done.store(false);
        ctrl_->blocks_touched.store(0);

        // SdkInterface::Get 的 buffers 是 const 引用，但真实 SDK 本来就要写 caller buffer；
        // 这里 const_cast 只是模拟"后端向 caller buffer 搬运/写入"这一事实行为。
        auto *buffers = const_cast<BlockBuffers *>(&local_buffers);

        // 逐 block 准入模式：模拟"每一块前检查 deadline"。
        // 块边界发现已过期即停止，不再触碰后续 block。
        if (ctrl_->per_block_check.load()) {
            size_t touched = 0;
            for (size_t i = 0; i < buffers->size(); ++i) {
                if (DeadlineExpired(deadline_ms)) {
                    ctrl_->blocks_touched.store(touched);
                    ctrl_->get_done.store(true);
                    return ER_SDK_TIMEOUT;
                }
                TouchBlock(*buffers, i, FakeSlowSdkControl::kTouchByte);
                ++touched;
                int per_block_delay_ms = ctrl_->per_block_delay_ms.load();
                if (per_block_delay_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(per_block_delay_ms));
                }
            }
            ctrl_->blocks_touched.store(touched);
            ctrl_->get_done.store(true);
            return static_cast<ClientErrorCode>(ctrl_->get_result.load());
        }

        // 门控：等待 peer 发起（错误路径测试的确定性编排）。
        if (ctrl_->gate_on_peer_calls != nullptr) {
            for (int i = 0; i < 500 && ctrl_->gate_on_peer_calls->load() == 0; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // 正常路径：同步搬运（模拟 hard 后端的同步 I/O —— 返回前数据已交付）。
        if (ctrl_->touch_buffer_on_get.load()) {
            for (size_t i = 0; i < buffers->size(); ++i) {
                TouchBlock(*buffers, i, FakeSlowSdkControl::kTouchByte);
            }
        }

        int delay_ms = ctrl_->get_delay_ms.load();
        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }

        // 违约路径：sleep 结束后仍写 caller buffer（模拟 mooncake 的 DMA 在飞）。
        if (ctrl_->write_after_return.load()) {
            for (size_t i = 0; i < buffers->size(); ++i) {
                TouchBlock(*buffers, i, ctrl_->after_return_byte.load());
            }
            ctrl_->write_done.store(true);
        }

        ctrl_->get_done.store(true);
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
    static void TouchBlock(BlockBuffers &buffers, size_t block_index, uint8_t byte) {
        if (block_index >= buffers.size()) {
            return;
        }
        for (auto &iov : buffers[block_index].iovs) {
            if (iov.base && iov.size > 0) {
                std::memset(iov.base, byte, iov.size);
            }
        }
    }

    std::shared_ptr<FakeSlowSdkControl> ctrl_;
    SdkTimeoutConfig timeout_config_; // Init 时由 wrapper 注入的静态超时预算
};

} // namespace kv_cache_manager
