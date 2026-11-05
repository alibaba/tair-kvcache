#include <gtest/gtest.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/common/shutdown_controller.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/coordination_memory_backend.h"
#include "kv_cache_manager/config/leader_elector.h"

using namespace kv_cache_manager;

namespace {

const int64_t kWaitTimeoutMs = 20000;

// 可重复使用的 barrier：handler 进入后阻塞，直到测试放行；
// Entered/Passed 用来断言「清理已经开始」和「清理没有被跳过」。
class Barrier {
public:
    void EnterAndWait() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++entered_;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return released_; });
        ++passed_;
    }

    void Release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

    int Entered() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entered_;
    }

    int Passed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return passed_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int entered_ = 0;
    int passed_ = 0;
    bool released_ = false;
};

template <typename Pred>
bool WaitUntil(Pred pred, int64_t timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return pred();
}

// 在超时上限内执行关闭动作；超时说明关闭卡住，直接失败退出，避免整个 CI 被挂住。
template <typename Fn>
void RunWithTimeout(Fn &&fn, int64_t timeout_ms) {
    auto future = std::async(std::launch::async, std::forward<Fn>(fn));
    if (future.wait_for(std::chrono::milliseconds(timeout_ms)) != std::future_status::ready) {
        ADD_FAILURE() << "shutdown did not finish within " << timeout_ms << " ms";
        _exit(1);
    }
    future.get();
}

} // namespace

// 定向信号回归：退出信号落在选主/租约检查/状态转换线程上时，这些线程不执行关闭逻辑，
// 关闭流程始终由非组件线程（这里是测试主线程）在普通执行上下文里完成。
//
// 说明：POSIX 下 pthread_kill 到已阻塞该信号的线程，信号只挂在该线程自己的 pending
// 队列里，其它线程的 sigtimedwait 取不到（已实测确认）。因此每个用例分两步：
// 先用 pthread_kill 证明组件线程不会因此进入关闭路径，再用进程级 kill（线上运维
// 实际发送的 kill -TERM <pid>）驱动控制线程唤醒主线程。
class LeaderElectorShutdownTest : public TESTBASE {
protected:
    void SetUp() override {
        ASSERT_TRUE(ShutdownController::BlockExitSignals());
        backend_ = std::make_shared<CoordinationMemoryBackend>();
        ASSERT_EQ(EC_OK, backend_->Init(StandardUri("memory://")));
        lock_key_ = "shutdown_lock_key";
        lock_value_ = "self-" + std::to_string(getpid());
    }

    void TearDown() override { controller_.Stop(); }

    std::shared_ptr<LeaderElector> CreateElector() {
        return std::make_shared<LeaderElector>(backend_, lock_key_, lock_value_, /*lease_ms=*/60000,
                                               /*loop_interval_ms=*/5);
    }

    // 控制线程感知退出请求（带超时上限）。
    bool WaitForShutdownRequest() {
        return WaitUntil([this] { return controller_.IsShutdownRequested(); }, kWaitTimeoutMs);
    }

protected:
    std::shared_ptr<CoordinationBackend> backend_;
    std::string lock_key_;
    std::string lock_value_;
    ShutdownController controller_;
};

TEST_F(LeaderElectorShutdownTest, DirectedSignalOnElectorThreadsDoesNotMakeThemRunShutdown) {
    ASSERT_TRUE(controller_.Start());

    std::atomic<int> recoveries{0};
    std::atomic<int> cleanups{0};
    auto elector = CreateElector();
    elector->SetBecomeLeaderHandler([&recoveries] { recoveries.fetch_add(1); });
    elector->SetNoLongerLeaderHandler([&cleanups] { cleanups.fetch_add(1); });
    elector->SetLeadershipLostHandler([] {});
    ASSERT_TRUE(elector->Start());
    ASSERT_TRUE(WaitUntil([&elector] { return elector->IsLeader(); }, kWaitTimeoutMs));

    auto leader_lock_thread = elector->leader_lock_thread_ptr_;
    auto lease_check_thread = elector->lease_check_thread_ptr_;
    ASSERT_NE(leader_lock_thread, nullptr);
    ASSERT_NE(lease_check_thread, nullptr);
    ASSERT_NE(elector->state_transition_thread_, nullptr);

    ASSERT_EQ(0, pthread_kill(leader_lock_thread->thread_.native_handle(), SIGTERM));
    ASSERT_EQ(0, pthread_kill(lease_check_thread->thread_.native_handle(), SIGINT));
    ASSERT_EQ(0, pthread_kill(elector->state_transition_thread_->native_handle(), SIGUSR1));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // 组件线程被打断后继续做自己的事，既没有触发关闭状态，也没有 self-join。
    EXPECT_FALSE(controller_.IsShutdownRequested());
    EXPECT_TRUE(elector->IsLeader());
    EXPECT_FALSE(leader_lock_thread->SelfJoinDetected());
    EXPECT_FALSE(lease_check_thread->SelfJoinDetected());

    ASSERT_EQ(0, kill(getpid(), SIGUSR2));
    ASSERT_TRUE(WaitForShutdownRequest()) << "controller did not observe the process-directed signal";
    EXPECT_EQ(SIGUSR2, controller_.ReceivedSignal());

    // 关闭由测试主线程执行：不抛 std::system_error，且在超时上限内完成。
    EXPECT_NO_THROW(RunWithTimeout([&elector] { elector->Stop(); }, kWaitTimeoutMs));

    EXPECT_FALSE(leader_lock_thread->SelfJoinDetected());
    EXPECT_FALSE(lease_check_thread->SelfJoinDetected());
    EXPECT_EQ(elector->state_transition_thread_, nullptr);
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_GE(recoveries.load(), 1);
    EXPECT_GE(cleanups.load(), 1) << "leadership revocation skipped the cleanup callback";
}

TEST_F(LeaderElectorShutdownTest, ShutdownWaitsForLeadershipLostHandlerHoldingTransitionMutex) {
    ASSERT_TRUE(controller_.Start());

    Barrier gate_barrier;
    std::atomic<int> gates{0};
    std::atomic<int> cleanups{0};
    auto elector = CreateElector();
    elector->SetBecomeLeaderHandler([] {});
    elector->SetNoLongerLeaderHandler([&cleanups] { cleanups.fetch_add(1); });
    // leadership_lost_handler_ 在持有 transition_mutex_ 时被调用。
    elector->SetLeadershipLostHandler([&] {
        gate_barrier.EnterAndWait();
        gates.fetch_add(1);
    });
    ASSERT_TRUE(elector->Start());
    ASSERT_TRUE(WaitUntil([&elector] { return elector->IsLeader(); }, kWaitTimeoutMs));

    auto leader_lock_thread = elector->leader_lock_thread_ptr_;
    auto lease_check_thread = elector->lease_check_thread_ptr_;

    // 失锁：租约检查/选主线程在持有 transition_mutex_ 的情况下卡进 barrier。
    ASSERT_EQ(EC_OK, backend_->Unlock(lock_key_, lock_value_));
    ASSERT_TRUE(WaitUntil([&gate_barrier] { return gate_barrier.Entered() > 0; }, kWaitTimeoutMs));

    ASSERT_EQ(0, pthread_kill(lease_check_thread->thread_.native_handle(), SIGTERM));
    ASSERT_EQ(0, pthread_kill(leader_lock_thread->thread_.native_handle(), SIGTERM));
    ASSERT_EQ(0, kill(getpid(), SIGTERM));
    ASSERT_TRUE(WaitForShutdownRequest());

    std::atomic<bool> stopped{false};
    std::thread shutdown_thread([&] {
        elector->Stop();
        stopped.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(stopped.load()) << "shutdown skipped a handler that still holds transition_mutex_";

    gate_barrier.Release();
    ASSERT_TRUE(WaitUntil([&stopped] { return stopped.load(); }, kWaitTimeoutMs)) << "shutdown hung after release";
    shutdown_thread.join();

    EXPECT_GE(gates.load(), 1);
    EXPECT_GE(cleanups.load(), 1);
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_FALSE(leader_lock_thread->SelfJoinDetected());
    EXPECT_FALSE(lease_check_thread->SelfJoinDetected());
}

TEST_F(LeaderElectorShutdownTest, ShutdownWaitsForRecoveryInProgress) {
    ASSERT_TRUE(controller_.Start());

    Barrier recovery_barrier;
    std::atomic<int> recoveries{0};
    std::atomic<int> cleanups{0};
    auto elector = CreateElector();
    elector->SetBecomeLeaderHandler([&] {
        recovery_barrier.EnterAndWait();
        recoveries.fetch_add(1);
    });
    elector->SetNoLongerLeaderHandler([&cleanups] { cleanups.fetch_add(1); });
    elector->SetLeadershipLostHandler([] {});
    ASSERT_TRUE(elector->Start());
    // 状态转换线程进入 PROMOTING 并卡在恢复 barrier 上。
    ASSERT_TRUE(WaitUntil([&recovery_barrier] { return recovery_barrier.Entered() > 0; }, kWaitTimeoutMs));

    ASSERT_EQ(0, pthread_kill(elector->state_transition_thread_->native_handle(), SIGUSR1));
    ASSERT_EQ(0, kill(getpid(), SIGTERM));
    ASSERT_TRUE(WaitForShutdownRequest());

    std::atomic<bool> stopped{false};
    std::thread shutdown_thread([&] {
        elector->Stop();
        stopped.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(stopped.load()) << "shutdown skipped an in-flight recovery";

    recovery_barrier.Release();
    ASSERT_TRUE(WaitUntil([&stopped] { return stopped.load(); }, kWaitTimeoutMs)) << "shutdown hung after release";
    shutdown_thread.join();

    EXPECT_EQ(1, recoveries.load()) << "started recovery did not run to completion";
    EXPECT_GE(cleanups.load(), 1) << "cleanup after a discarded recovery was skipped";
    EXPECT_FALSE(elector->IsLeader());
}

TEST_F(LeaderElectorShutdownTest, ShutdownWaitsForDemotionCleanupInProgress) {
    ASSERT_TRUE(controller_.Start());

    Barrier cleanup_barrier;
    std::atomic<int> cleanups{0};
    auto elector = CreateElector();
    elector->SetBecomeLeaderHandler([] {});
    elector->SetNoLongerLeaderHandler([&] {
        cleanup_barrier.EnterAndWait();
        cleanups.fetch_add(1);
    });
    elector->SetLeadershipLostHandler([] {});
    ASSERT_TRUE(elector->Start());
    ASSERT_TRUE(WaitUntil([&elector] { return elector->IsLeader(); }, kWaitTimeoutMs));

    auto lease_check_thread = elector->lease_check_thread_ptr_;

    ASSERT_EQ(EC_OK, backend_->Unlock(lock_key_, lock_value_));
    // 状态转换线程进入 DEMOTING 并卡在降级清理 barrier 上。
    ASSERT_TRUE(WaitUntil([&cleanup_barrier] { return cleanup_barrier.Entered() > 0; }, kWaitTimeoutMs));

    ASSERT_EQ(0, pthread_kill(elector->state_transition_thread_->native_handle(), SIGTERM));
    ASSERT_EQ(0, pthread_kill(lease_check_thread->thread_.native_handle(), SIGTERM));
    ASSERT_EQ(0, kill(getpid(), SIGTERM));
    ASSERT_TRUE(WaitForShutdownRequest());

    std::atomic<bool> stopped{false};
    std::thread shutdown_thread([&] {
        elector->Stop();
        stopped.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(stopped.load()) << "shutdown skipped an in-flight demotion cleanup";

    cleanup_barrier.Release();
    ASSERT_TRUE(WaitUntil([&stopped] { return stopped.load(); }, kWaitTimeoutMs)) << "shutdown hung after release";
    shutdown_thread.join();

    EXPECT_EQ(1, cleanups.load());
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_FALSE(lease_check_thread->SelfJoinDetected());
}
