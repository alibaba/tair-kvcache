#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/shutdown_controller.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/coordination_memory_backend.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/service/service_impl_base.h"

using namespace kv_cache_manager;

class RequestGateForTest : public ServiceImplBase {
public:
    bool Begin(bool leader_only = true) { return CheckAndIncrementRequestCount(leader_only); }
    void Finish(bool leader_only = true) { DecrementRequestCount(leader_only); }
};

class ServiceImplBaseTest : public TESTBASE {};

TEST_F(ServiceImplBaseTest, ClosingGateRejectsNewRequestsAndDrainsAcceptedRequests) {
    RequestGateForTest service;
    ASSERT_TRUE(service.Begin());
    service.DisableLeaderOnlyRequests();
    EXPECT_FALSE(service.Begin());
    EXPECT_TRUE(service.Begin(false));
    service.Finish(false);
    auto drain = std::async(std::launch::async, [&] { service.WaitForAllLeaderOnlyRequestsToComplete(); });
    EXPECT_EQ(std::future_status::timeout, drain.wait_for(std::chrono::milliseconds(50)));
    service.Finish();
    auto status = drain.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        service.leader_only_request_done_cv_.notify_all();
    }
    drain.get();
    EXPECT_EQ(std::future_status::ready, status);
}

TEST_F(ServiceImplBaseTest, ConcurrentFinalRequestCompletionDoesNotLoseDrainNotification) {
    RequestGateForTest service;
    for (int round = 0; round < 3000; ++round) {
        service.EnableLeaderOnlyRequests();
        ASSERT_TRUE(service.Begin());
        service.DisableLeaderOnlyRequests();
        std::promise<void> start;
        auto released = start.get_future().share();
        auto drain = std::async(std::launch::async, [&] {
            released.wait();
            service.WaitForAllLeaderOnlyRequestsToComplete();
        });
        auto finish = std::async(std::launch::async, [&] {
            released.wait();
            service.Finish();
        });
        start.set_value();
        finish.get();
        auto status = drain.wait_for(std::chrono::milliseconds(500));
        if (status != std::future_status::ready) {
            EXPECT_EQ(0, service.active_leader_only_request_count_.load());
            service.leader_only_request_done_cv_.notify_all();
        }
        drain.get();
        ASSERT_EQ(std::future_status::ready, status) << "drain stranded after last request completed, round=" << round;
    }
}

TEST_F(ServiceImplBaseTest, LeadershipLossDrainsBothServicesBeforeCampaignResumes) {
    auto backend = std::make_shared<CoordinationMemoryBackend>();
    ASSERT_EQ(EC_OK, backend->Init(StandardUri("memory://")));
    LeaderElector elector(backend, "lock", "self", 60000, 10);
    RequestGateForTest meta;
    RequestGateForTest admin;
    meta.DisableLeaderOnlyRequests();
    admin.DisableLeaderOnlyRequests();
    int recoveries = 0;
    int cleanups = 0;
    std::promise<void> cleanup_started;
    std::promise<void> meta_drained;
    elector.SetLeadershipLostHandler([&] {
        meta.DisableLeaderOnlyRequests();
        admin.DisableLeaderOnlyRequests();
    });
    elector.SetBecomeLeaderHandler([&] {
        ++recoveries;
        EXPECT_TRUE(elector.RunIfCurrentPromotionValid([&] {
            meta.EnableLeaderOnlyRequests();
            admin.EnableLeaderOnlyRequests();
        }));
    });
    elector.SetNoLongerLeaderHandler([&] {
        cleanup_started.set_value();
        meta.WaitForAllLeaderOnlyRequestsToComplete();
        meta_drained.set_value();
        admin.WaitForAllLeaderOnlyRequestsToComplete();
        ++cleanups;
    });
    elector.WorkLoop();
    elector.ProcessStateTransitionsForTest();
    ASSERT_TRUE(meta.Begin());
    ASSERT_TRUE(admin.Begin());

    ASSERT_EQ(EC_OK, backend->Unlock("lock", "self"));
    elector.HoldLeader(TimestampUtil::GetCurrentTimeUs());
    // Both business gates close before the cleanup worker starts. Discovery
    // requests remain available, and previously accepted requests still drain.
    EXPECT_FALSE(meta.Begin());
    EXPECT_FALSE(admin.Begin());
    EXPECT_TRUE(meta.Begin(false));
    meta.Finish(false);
    auto cleanup = std::async(std::launch::async, [&] { elector.ProcessStateTransitionsForTest(); });
    EXPECT_EQ(std::future_status::ready, cleanup_started.get_future().wait_for(std::chrono::seconds(5)));
    for (int round = 0; round < 100; ++round) {
        elector.WorkLoop();
    }
    EXPECT_FALSE(elector.IsLeader());
    EXPECT_EQ(RoleState::DEMOTING, elector.GetRoleState());
    EXPECT_EQ(std::future_status::timeout, cleanup.wait_for(std::chrono::milliseconds(50)));

    meta.Finish();
    EXPECT_EQ(std::future_status::ready, meta_drained.get_future().wait_for(std::chrono::seconds(5)));
    EXPECT_EQ(std::future_status::timeout, cleanup.wait_for(std::chrono::milliseconds(50)));
    admin.Finish();
    auto status = cleanup.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        meta.leader_only_request_done_cv_.notify_all();
        admin.leader_only_request_done_cv_.notify_all();
    }
    cleanup.get();
    EXPECT_EQ(std::future_status::ready, status);
    EXPECT_EQ(1, cleanups);
    EXPECT_EQ(RoleState::FOLLOWER, elector.GetRoleState());

    elector.WorkLoop();
    elector.ProcessStateTransitionsForTest();
    EXPECT_EQ(2, recoveries);
    EXPECT_TRUE(meta.Begin());
    EXPECT_TRUE(admin.Begin());
    meta.Finish();
    admin.Finish();
}

namespace {

const int64_t kSignalWaitTimeoutMs = 20000;

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

} // namespace

// 线程 X 持有一个 in-flight leader-only 请求时收到定向退出信号。
// 关闭由主线程执行：先关门禁（新请求拿到拒绝码而不是传输错误），再等 X 完成请求后排空。
//
// 说明：POSIX 下 pthread_kill 到已阻塞该信号的线程，信号只挂在该线程自己的 pending
// 队列里，其它线程的 sigtimedwait 取不到（已实测确认）。因此先用 pthread_kill 证明
// 请求线程不会因此中断请求或进入关闭路径，再用进程级 kill（线上运维实际发送的
// kill -TERM <pid>）驱动控制线程唤醒主线程。
TEST_F(ServiceImplBaseTest, SignalledRequestThreadDoesNotDeadlockGateClosingAndDrain) {
    ASSERT_TRUE(ShutdownController::BlockExitSignals());
    ShutdownController controller;
    ASSERT_TRUE(controller.Start());

    auto backend = std::make_shared<CoordinationMemoryBackend>();
    ASSERT_EQ(EC_OK, backend->Init(StandardUri("memory://")));
    LeaderElector elector(backend, "lock", "self", 60000, 5);
    RequestGateForTest meta;
    std::atomic<int> cleanups{0};
    std::atomic<bool> drained{false};
    elector.SetBecomeLeaderHandler([] {});
    elector.SetLeadershipLostHandler([&meta] { meta.DisableLeaderOnlyRequests(); });
    elector.SetNoLongerLeaderHandler([&] {
        meta.WaitForAllLeaderOnlyRequestsToComplete();
        drained.store(true);
        cleanups.fetch_add(1);
    });
    ASSERT_TRUE(elector.Start());
    ASSERT_TRUE(WaitUntil([&elector] { return elector.IsLeader(); }, kSignalWaitTimeoutMs));

    // 线程 X 接下一个 leader-only 请求并停在处理中。
    ASSERT_TRUE(meta.Begin());
    std::promise<void> request_reached;
    std::promise<void> request_release;
    auto released = request_release.get_future().share();
    std::atomic<bool> request_finished{false};
    std::thread request_thread([&] {
        request_reached.set_value();
        released.wait();
        meta.Finish();
        request_finished.store(true);
    });
    ASSERT_EQ(std::future_status::ready, request_reached.get_future().wait_for(std::chrono::seconds(5)));

    // 定向投递到正在处理请求的线程：请求既没有被打断，也没有触发关闭状态。
    ASSERT_EQ(0, pthread_kill(request_thread.native_handle(), SIGTERM));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(request_finished.load());
    EXPECT_FALSE(controller.IsShutdownRequested());

    ASSERT_EQ(0, kill(getpid(), SIGTERM));
    ASSERT_TRUE(WaitUntil([&controller] { return controller.IsShutdownRequested(); }, kSignalWaitTimeoutMs));

    // 关闭由主线程侧发起，绝不由请求线程执行。
    std::atomic<bool> stopped{false};
    std::thread shutdown_thread([&] {
        elector.Stop();
        stopped.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(drained.load()) << "drain finished while a request was still in flight";
    EXPECT_FALSE(stopped.load()) << "shutdown did not wait for the in-flight request";
    EXPECT_FALSE(meta.Begin()) << "gate stayed open during shutdown";

    request_release.set_value();
    request_thread.join();
    ASSERT_TRUE(WaitUntil([&stopped] { return stopped.load(); }, kSignalWaitTimeoutMs))
        << "drain did not finish after the in-flight request completed";
    shutdown_thread.join();

    EXPECT_TRUE(drained.load());
    EXPECT_EQ(1, cleanups.load());
    EXPECT_FALSE(elector.IsLeader());
    controller.Stop();
}
