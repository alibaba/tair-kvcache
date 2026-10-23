#include <chrono>
#include <future>
#include <gtest/gtest.h>

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
