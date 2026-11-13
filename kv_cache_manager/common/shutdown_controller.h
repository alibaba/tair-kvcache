#pragma once

#include <signal.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace kv_cache_manager {

// 进程级退出信号控制器：把「收到退出信号」收敛成一个只能由固定控制线程观察的状态。
//
// 使用约定：
// - BlockExitSignals() 必须在创建任何其它线程之前调用（CommandLine::Run() 的最前面）。
//   之后创建的线程都继承该阻塞掩码，退出信号不会在任何业务线程里执行 handler，
//   只会成为进程 pending，由控制线程通过 sigtimedwait 消费。
// - 控制线程只做「记录 + 唤醒」，不执行任何组件的 Stop/join/清理；完整关闭流程由
//   调用 WaitForShutdownRequest() 的线程（主线程）在自己的普通执行上下文里执行，
//   以免组件线程 self-join、在持锁区重入清理，或与请求排空形成循环等待。
// - Stop() 必须早于 logger 销毁调用：控制线程在强制退出前会打日志。
//
// 测试注意：若在同一进程内已经阻塞了退出信号，又希望用 raise() 让进程立刻终止，
// 必须先 pthread_sigmask(SIG_UNBLOCK, &ExitSignalSet(), nullptr) 再 raise()，
// 否则信号只会挂起，用例会超时。
class ShutdownController {
public:
    // 只设置阻塞掩码，不启线程。
    static bool BlockExitSignals();
    static const sigset_t &ExitSignalSet();

    ShutdownController() = default;
    ~ShutdownController();

    ShutdownController(const ShutdownController &) = delete;
    ShutdownController &operator=(const ShutdownController &) = delete;

    // 启动控制线程；幂等。失败时解除阻塞掩码，恢复「SIGTERM 直接终止进程」的默认语义。
    bool Start();
    // 阻塞直到收到退出信号（或被 Stop() 打断）；返回是否为信号驱动。
    bool WaitForShutdownRequest();
    // 置停止标志并 join 控制线程；幂等，无信号时最多等待 kPollTimeoutMs。
    void Stop();

    bool IsShutdownRequested() const;
    int ReceivedSignal() const;
    int ForceExitSignal() const;

private:
    void SignalLoop();
    static void UnblockExitSignals();

    enum class State { kIdle, kRequested };

    // sigtimedwait 轮询间隔，同时是 Stop() join 的延迟上限。
    static constexpr int kPollTimeoutMs = 100;

    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    State state_ = State::kIdle;
    std::atomic<bool> stop_{false};
    std::atomic<bool> started_{false};
    int received_signal_ = 0;
    int force_exit_signal_ = 0;
};

} // namespace kv_cache_manager
