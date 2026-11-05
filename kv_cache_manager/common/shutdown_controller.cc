#include "kv_cache_manager/common/shutdown_controller.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <chrono>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

ShutdownController::~ShutdownController() { Stop(); }

const sigset_t &ShutdownController::ExitSignalSet() {
    static const sigset_t kExitSignalSet = [] {
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGTERM);
        sigaddset(&set, SIGINT);
        sigaddset(&set, SIGUSR1);
        sigaddset(&set, SIGUSR2);
        return set;
    }();
    return kExitSignalSet;
}

void ShutdownController::UnblockExitSignals() {
    int ret = pthread_sigmask(SIG_UNBLOCK, &ExitSignalSet(), nullptr);
    if (ret != 0) {
        fprintf(stderr, "unblock exit signals failed: %s\n", strerror(ret));
    }
}

bool ShutdownController::BlockExitSignals() {
    // 早于 logger 初始化调用，这里只能走 stderr。
    int ret = pthread_sigmask(SIG_BLOCK, &ExitSignalSet(), nullptr);
    if (ret != 0) {
        fprintf(stderr, "block exit signals failed: %s\n", strerror(ret));
        return false;
    }
    return true;
}

bool ShutdownController::Start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) {
        return true;
    }

    stop_.store(false);
    try {
        thread_ = std::thread(&ShutdownController::SignalLoop, this);
    } catch (const std::system_error &e) {
        started_.store(false);
        // fail-safe：控制线程起不来时不能把进程变成只能被 SIGKILL 杀掉。
        fprintf(stderr, "create shutdown controller thread failed: %s, unblock exit signals\n", e.what());
        UnblockExitSignals();
        return false;
    }
    return true;
}

bool ShutdownController::WaitForShutdownRequest() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return state_ == State::kRequested || stop_.load(); });
    return state_ == State::kRequested;
}

void ShutdownController::Stop() {
    bool expected = true;
    if (!started_.compare_exchange_strong(expected, false)) {
        return;
    }
    stop_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool ShutdownController::IsShutdownRequested() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ == State::kRequested;
}

int ShutdownController::ReceivedSignal() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_signal_;
}

int ShutdownController::ForceExitSignal() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return force_exit_signal_;
}

void ShutdownController::SignalLoop() {
    // 控制线程自己也阻塞一遍：即使调用方漏掉 BlockExitSignals()，退出信号也不会在
    // 本线程触发默认终止动作，而是被 sigtimedwait 取走。
    pthread_sigmask(SIG_BLOCK, &ExitSignalSet(), nullptr);

    const timespec poll_timeout = {kPollTimeoutMs / 1000, (kPollTimeoutMs % 1000) * 1000 * 1000};

    while (!stop_.load()) {
        siginfo_t info;
        memset(&info, 0, sizeof(info));
        int sig = sigtimedwait(&ExitSignalSet(), &info, &poll_timeout);
        if (sig <= 0) {
            if (errno != EAGAIN && errno != EINTR) {
                KVCM_LOG_ERROR("sigtimedwait failed, errno=%d", errno);
            }
            continue;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        if (state_ == State::kRequested) {
            // 优雅关闭已在进行中却又收到退出信号：视为运维要求强制结束。
            force_exit_signal_ = sig;
            KVCM_LOG_WARN("received exit signal %d again while shutting down, force exit with code %d",
                          sig,
                          128 + sig);
            lock.unlock();
            // 与 CrashHandler 的兜底一致：跳过 atexit 与静态析构，避免在关闭中途再触发一轮清理。
            _exit(128 + sig);
        }
        state_ = State::kRequested;
        received_signal_ = sig;
        lock.unlock();
        cv_.notify_all();
        // 继续循环，这样关闭流程进行中再来的信号能被立刻捕获（见上面的强退分支）。
    }
}

} // namespace kv_cache_manager
