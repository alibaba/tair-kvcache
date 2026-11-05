#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/shutdown_controller.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

namespace {

const std::vector<int> kExitSignals = {SIGTERM, SIGINT, SIGUSR1, SIGUSR2};
const int64_t kWaitTimeoutMs = 5000;
// Stop() 的 join 延迟上限就是 sigtimedwait 的 100ms 轮询间隔，这里留出余量。
const int64_t kStopTimeoutMs = 200;

int64_t ElapsedMsSince(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
}

// 在超时上限内等待退出请求；超时则先 Stop() 唤醒等待者再判定失败，避免测试挂死。
bool WaitForRequestWithTimeout(ShutdownController &controller, int64_t timeout_ms) {
    std::atomic<bool> requested{false};
    std::atomic<bool> finished{false};
    std::thread waiter([&] {
        requested.store(controller.WaitForShutdownRequest());
        finished.store(true);
    });

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!finished.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!finished.load()) {
        controller.Stop();
    }
    waiter.join();
    return finished.load() && requested.load();
}

// 带超时地回收子进程；超时则 SIGKILL 并判定失败。
bool WaitChildWithTimeout(pid_t pid, int &status, int64_t timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        pid_t ret = waitpid(pid, &status, WNOHANG);
        if (ret == pid) {
            return true;
        }
        if (ret < 0) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return false;
}

} // namespace

class ShutdownControllerTest : public TESTBASE {
protected:
    void SetUp() override {
        // 进程级掩码，一次设置对本进程后续所有测试与线程都生效。
        ASSERT_TRUE(ShutdownController::BlockExitSignals());
    }
    void TearDown() override {}
};

// AC-1: 掩码在创建线程之前设置，之后创建的线程一律继承。
TEST_F(ShutdownControllerTest, NewThreadsInheritBlockedExitSignals) {
    std::promise<sigset_t> promise;
    std::thread worker([&promise] {
        sigset_t current;
        sigemptyset(&current);
        pthread_sigmask(SIG_SETMASK, nullptr, &current);
        promise.set_value(current);
    });
    sigset_t inherited = promise.get_future().get();
    worker.join();

    for (int sig : kExitSignals) {
        EXPECT_EQ(1, sigismember(&inherited, sig)) << "signal " << sig << " is not blocked in new thread";
    }
}

// AC-1: 退出信号不再安装任何 handler，信号上下文不执行加锁、日志与清理。
TEST_F(ShutdownControllerTest, NoExitSignalHandlerIsInstalled) {
    ShutdownController controller;
    ASSERT_TRUE(controller.Start());

    for (int sig : kExitSignals) {
        struct sigaction current {};
        ASSERT_EQ(0, sigaction(sig, nullptr, &current));
        EXPECT_EQ(SIG_DFL, current.sa_handler) << "signal " << sig << " still has a handler installed";
    }

    controller.Stop();
}

// AC-1 + AC-4: 定向投递到工作线程的退出信号不会让该线程进入关闭路径；
// 运维实际发送的进程级信号（kill -TERM <pid>）由控制线程统一消费。
//
// 说明：POSIX 下 pthread_kill 到某个已阻塞该信号的线程，信号只挂在该线程自己的
// pending 队列里，其它线程的 sigtimedwait 取不到（已实测确认）。因此分两步验证：
// 先用 pthread_kill 证明「落在组件线程上的信号不会触发任何关闭逻辑」，
// 再用进程级 kill 证明控制线程能在超时上限内感知并唤醒主线程。
TEST_F(ShutdownControllerTest, SignalDirectedAtWorkerThreadDoesNotEnterShutdownPath) {
    for (int sig : kExitSignals) {
        ShutdownController controller;
        ASSERT_TRUE(controller.Start());

        std::atomic<bool> worker_running{true};
        std::atomic<int> worker_ticks{0};
        std::thread worker([&] {
            while (worker_running.load()) {
                worker_ticks.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });

        ASSERT_EQ(0, pthread_kill(worker.native_handle(), sig));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // 工作线程被打断后继续跑自己的逻辑，既没有 handler 也没有触发关闭状态。
        EXPECT_FALSE(controller.IsShutdownRequested()) << "signal " << sig << " reached controller from worker";
        int ticks_after_signal = worker_ticks.load();
        EXPECT_GT(ticks_after_signal, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_GT(worker_ticks.load(), ticks_after_signal) << "worker thread stuck after signal " << sig;

        // 进程级信号才是线上真实路径（运维与测试侧只发 kill -TERM <pid>）。
        ASSERT_EQ(0, kill(getpid(), sig));
        EXPECT_TRUE(WaitForRequestWithTimeout(controller, kWaitTimeoutMs))
            << "controller did not observe process-directed signal " << sig;
        EXPECT_EQ(sig, controller.ReceivedSignal());

        worker_running.store(false);
        worker.join();

        auto stop_begin = std::chrono::steady_clock::now();
        controller.Stop();
        EXPECT_LE(ElapsedMsSince(stop_begin), kStopTimeoutMs);
    }
}

// AC-3: 优雅关闭进行中再次收到退出信号 → 立即以 128+sig 终止。
TEST_F(ShutdownControllerTest, RepeatedExitSignalForcesImmediateExit) {
    for (int sig : kExitSignals) {
        pid_t pid = fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            // fork 只保留调用线程，子进程自己重建控制线程；阻塞掩码已从父进程继承。
            ShutdownController controller;
            if (!controller.Start()) {
                _exit(200);
            }
            kill(getpid(), sig);
            for (int i = 0; i < 2000 && !controller.IsShutdownRequested(); ++i) {
                usleep(1000);
            }
            if (!controller.IsShutdownRequested()) {
                _exit(201);
            }
            // 关闭流程进行中重复收到信号：控制线程应立即 _exit(128+sig)。
            kill(getpid(), sig);
            for (int i = 0; i < 2000; ++i) {
                usleep(1000);
            }
            _exit(202);
        }

        int status = 0;
        ASSERT_TRUE(WaitChildWithTimeout(pid, status, kWaitTimeoutMs)) << "child hung for signal " << sig;
        ASSERT_TRUE(WIFEXITED(status)) << "signal " << sig << ": child did not exit normally";
        EXPECT_EQ(128 + sig, WEXITSTATUS(status));
    }
}

// AC-5: Start()/Stop() 幂等，Stop() 在无信号时也能在超时上限内返回。
TEST_F(ShutdownControllerTest, StartAndStopAreIdempotent) {
    ShutdownController controller;
    EXPECT_TRUE(controller.Start());
    EXPECT_TRUE(controller.Start());

    auto stop_begin = std::chrono::steady_clock::now();
    controller.Stop();
    EXPECT_LE(ElapsedMsSince(stop_begin), kStopTimeoutMs);
    controller.Stop();
    controller.Stop();

    // 未启动过的实例同样可以安全 Stop()。
    ShutdownController never_started;
    never_started.Stop();
}

// WaitForShutdownRequest() 在未收到信号时阻塞，收到信号后被唤醒并返回 true。
TEST_F(ShutdownControllerTest, WaitForShutdownRequestWakesUpOnSignal) {
    ShutdownController controller;
    ASSERT_TRUE(controller.Start());

    ASSERT_EQ(0, kill(getpid(), SIGTERM));
    EXPECT_TRUE(WaitForRequestWithTimeout(controller, kWaitTimeoutMs));
    EXPECT_EQ(SIGTERM, controller.ReceivedSignal());
    controller.Stop();
}

// WaitForShutdownRequest() 确实会阻塞：没有信号时不会提前返回。
TEST_F(ShutdownControllerTest, WaitForShutdownRequestBlocksWithoutSignal) {
    ShutdownController controller;
    ASSERT_TRUE(controller.Start());

    std::atomic<bool> requested{true};
    std::atomic<bool> finished{false};
    std::thread waiter([&] {
        requested.store(controller.WaitForShutdownRequest());
        finished.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(finished.load()) << "WaitForShutdownRequest returned without any signal";

    // 用 Stop() 解除阻塞，此时返回 false（非信号驱动）。
    controller.Stop();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kWaitTimeoutMs);
    while (!finished.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    waiter.join();
    EXPECT_TRUE(finished.load());
    EXPECT_FALSE(requested.load());
}
