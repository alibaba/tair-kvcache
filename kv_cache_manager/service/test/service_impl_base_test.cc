#include <chrono>
#include <future>
#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
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
