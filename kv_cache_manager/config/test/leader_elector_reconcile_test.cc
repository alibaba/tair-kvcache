#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/coordination_memory_backend.h"
#include "kv_cache_manager/config/leader_elector.h"

using namespace kv_cache_manager;

namespace {

// Control callback/RPC completion without relying on scheduling sleeps.
class BlockingPoint {
public:
    void Block() {
        std::unique_lock lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        EXPECT_TRUE(cv_.wait_for(lock, std::chrono::seconds(10), [this] { return released_; }));
    }
    bool WaitEntered() {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [this] { return entered_; });
    }
    void Release() {
        std::lock_guard lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false;
    bool released_ = false;
};

class ControlledCoordinationBackend : public CoordinationMemoryBackend {
public:
    ErrorCode TryLock(const std::string &key, const std::string &value, int64_t ttl) override {
        ++campaigns;
        if (before_try) {
            before_try();
        }
        return CoordinationMemoryBackend::TryLock(key, value, ttl);
    }
    ErrorCode RenewLock(const std::string &key, const std::string &value, int64_t ttl) override {
        ++renewals;
        if (renew_error != EC_OK) {
            return renew_error;
        }
        ErrorCode ec = CoordinationMemoryBackend::RenewLock(key, value, ttl);
        if (after_renew) {
            after_renew();
        }
        return ec;
    }
    ErrorCode GetLockHolder(const std::string &key, std::string &value, int64_t &expiration) override {
        if (before_get) {
            before_get();
        }
        ErrorCode ec = CoordinationMemoryBackend::GetLockHolder(key, value, expiration);
        if (after_get) {
            after_get();
        }
        return ec;
    }
    ErrorCode Unlock(const std::string &key, const std::string &value) override {
        if (before_unlock) {
            before_unlock();
        }
        if (unlock_error != EC_OK) {
            return unlock_error;
        }
        return CoordinationMemoryBackend::Unlock(key, value);
    }
    std::atomic<int> campaigns{0};
    std::atomic<int> renewals{0};
    ErrorCode renew_error = EC_OK;
    ErrorCode unlock_error = EC_OK;
    std::function<void()> before_try;
    std::function<void()> after_renew;
    std::function<void()> before_get;
    std::function<void()> after_get;
    std::function<void()> before_unlock;
};

} // namespace

class LeaderReconcileTest : public TESTBASE {
protected:
    void SetUp() override {
        backend = std::make_shared<ControlledCoordinationBackend>();
        ASSERT_EQ(EC_OK, backend->Init(StandardUri("memory://")));
        elector = std::make_shared<LeaderElector>(backend, "lock", "self", 60000, 10);
        elector->SetLeadershipLostHandler([this] { ready = false; });
        elector->SetBecomeLeaderHandler([this] {
            ++recoveries;
            elector->RunIfCurrentPromotionValid([this] { ready = true; });
        });
        elector->SetNoLongerLeaderHandler([this] { ++cleanups; });
    }
    void TearDown() override { elector->Stop(); }
    void Acquire() { elector->WorkLoop(); }
    void Reconcile() { elector->ProcessStateTransitionsForTest(); }
    void LoseLock() {
        EXPECT_EQ(EC_OK, backend->Unlock("lock", "self"));
        elector->HoldLeader(TimestampUtil::GetCurrentTimeUs());
        EXPECT_FALSE(elector->IsLeader());
        EXPECT_FALSE(ready);
    }
    void ExpectBackendUnlocked() {
        std::string holder;
        int64_t expiration;
        EXPECT_EQ(EC_NOENT, backend->GetLockHolder("lock", holder, expiration));
    }
    std::shared_ptr<ControlledCoordinationBackend> backend;
    std::shared_ptr<LeaderElector> elector;
    std::atomic<bool> ready{false};
    std::atomic<int> recoveries{0};
    std::atomic<int> cleanups{0};
};

TEST_F(LeaderReconcileTest, LockChurnBeforeRecoveryCoalescesWithoutStaleCallbacks) {
    for (int i = 0; i < 200; ++i) {
        Acquire();
        EXPECT_FALSE(elector->IsStableState());
        LoseLock();
    }
    Reconcile();
    EXPECT_EQ(0, recoveries);
    EXPECT_EQ(0, cleanups);
    EXPECT_TRUE(elector->IsStableState());
    Acquire();
    Reconcile();
    EXPECT_TRUE(ready);
    EXPECT_EQ(RoleState::LEADER, elector->GetRoleState());
    EXPECT_EQ(1, recoveries);
}

TEST_F(LeaderReconcileTest, LossDuringRecoveryBlocksCampaignThroughCleanupAndRecoversAgain) {
    BlockingPoint recovery;
    BlockingPoint cleanup;
    elector->SetBecomeLeaderHandler([&] {
        ++recoveries;
        recovery.Block();
        EXPECT_FALSE(elector->RunIfCurrentPromotionValid([&] { ready = true; }));
    });
    elector->SetNoLongerLeaderHandler([&] {
        ++cleanups;
        cleanup.Block();
    });
    Acquire();
    auto worker = std::async(std::launch::async, [&] { Reconcile(); });
    EXPECT_TRUE(recovery.WaitEntered());
    EXPECT_EQ(RoleState::PROMOTING, elector->GetRoleState());
    LoseLock();
    // Another node can acquire and be discovered before the old recovery returns.
    EXPECT_EQ(EC_OK, backend->TryLock("lock", "other", 60000));
    int campaigns = backend->campaigns;
    for (int i = 0; i < 100; ++i) {
        Acquire();
    }
    EXPECT_EQ(campaigns, backend->campaigns);
    EXPECT_EQ("other", elector->GetLeaderNodeID());
    recovery.Release();
    EXPECT_TRUE(cleanup.WaitEntered());
    EXPECT_EQ(RoleState::DEMOTING, elector->GetRoleState());
    for (int i = 0; i < 100; ++i) {
        Acquire();
    }
    EXPECT_EQ(campaigns, backend->campaigns);
    EXPECT_EQ("other", elector->GetLeaderNodeID());
    cleanup.Release();
    worker.get();
    EXPECT_EQ(1, recoveries);
    EXPECT_EQ(1, cleanups);
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    EXPECT_EQ(EC_OK, backend->Unlock("lock", "other"));
    elector->SetBecomeLeaderHandler([&] {
        ++recoveries;
        EXPECT_TRUE(elector->RunIfCurrentPromotionValid([&] { ready = true; }));
    });
    Acquire();
    Reconcile();
    EXPECT_TRUE(ready);
    EXPECT_EQ(2, recoveries);
}

TEST_F(LeaderReconcileTest, ServingLossClosesGateBeforeCleanupAndDoesNotCampaignWhileCleaning) {
    Acquire();
    Reconcile();
    ASSERT_TRUE(ready);
    LoseLock();
    EXPECT_FALSE(elector->WaitForStableState(0));
    EXPECT_EQ(EC_OK, backend->TryLock("lock", "other", 60000));
    int campaigns = backend->campaigns;
    Acquire();
    EXPECT_EQ(RoleState::LEADER, elector->GetRoleState()); // Cleanup has not started yet.
    EXPECT_EQ("other", elector->GetLeaderNodeID());
    EXPECT_EQ(campaigns, backend->campaigns);
    BlockingPoint cleanup;
    elector->SetNoLongerLeaderHandler([&] {
        ++cleanups;
        cleanup.Block();
    });
    auto worker = std::async(std::launch::async, [&] { Reconcile(); });
    EXPECT_TRUE(cleanup.WaitEntered());
    EXPECT_EQ(EC_OK, backend->Unlock("lock", "other"));
    Acquire();
    EXPECT_TRUE(elector->GetLeaderNodeID().empty());
    EXPECT_EQ(campaigns, backend->campaigns);
    EXPECT_EQ(EC_OK, backend->TryLock("lock", "other", 60000));
    campaigns = backend->campaigns;
    for (int i = 0; i < 100; ++i) {
        Acquire();
    }
    EXPECT_EQ(campaigns, backend->campaigns);
    EXPECT_EQ("other", elector->GetLeaderNodeID());
    cleanup.Release();
    worker.get();
    EXPECT_EQ(EC_OK, backend->Unlock("lock", "other"));
    Acquire();
    Reconcile();
    EXPECT_TRUE(ready);
    EXPECT_EQ(2, recoveries);
    EXPECT_EQ(1, cleanups);
}

TEST_F(LeaderReconcileTest, DelayedAcquisitionAfterDemoteIsReleasedWithoutRecovery) {
    BlockingPoint acquisition;
    backend->before_try = [&] { acquisition.Block(); };
    auto rpc = std::async(std::launch::async, [&] { Acquire(); });
    EXPECT_TRUE(acquisition.WaitEntered());
    elector->Demote();
    acquisition.Release();
    rpc.get();
    Reconcile();
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_EQ(0, recoveries);
    ExpectBackendUnlocked();
    backend->before_try = {};
    Acquire();
    Reconcile();
    EXPECT_TRUE(ready);
}

TEST_F(LeaderReconcileTest, SuccessfulLateRenewalCannotResurrectExpiredRound) {
    Acquire();
    Reconcile();
    int64_t expiration = elector->GetLeaseExpirationTime();
    BlockingPoint renewal;
    backend->after_renew = [&] { renewal.Block(); };
    auto rpc = std::async(std::launch::async, [&] { elector->HoldLeader(TimestampUtil::GetCurrentTimeUs()); });
    EXPECT_TRUE(renewal.WaitEntered());
    EXPECT_FALSE(elector->DoCheckLeaseTimeout(expiration + 1));
    EXPECT_FALSE(ready);
    Reconcile();
    renewal.Release();
    rpc.get();
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_EQ(-1, elector->GetLeaseExpirationTime());
    ExpectBackendUnlocked();
    backend->after_renew = {};
    Acquire();
    Reconcile();
    EXPECT_TRUE(ready);
    EXPECT_EQ(2, recoveries);
}

TEST_F(LeaderReconcileTest, PublicationChecksExpiryWithoutWaitingForLeaseCheckThread) {
    elector->SetBecomeLeaderHandler([&] {
        ++recoveries;
        // Simulate expiry immediately before opening the request gate.
        elector->lease_expiration_time_us_.store(TimestampUtil::GetCurrentTimeUs() - 1);
        EXPECT_FALSE(elector->RunIfCurrentPromotionValid([&] { ready = true; }));
    });
    Acquire();
    Reconcile();
    EXPECT_FALSE(ready);
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    EXPECT_EQ(1, cleanups);
}

TEST_F(LeaderReconcileTest, GatePublicationAndLossAreSerialized) {
    BlockingPoint publication;
    elector->SetBecomeLeaderHandler([&] {
        elector->RunIfCurrentPromotionValid([&] {
            publication.Block();
            ready = true;
        });
    });
    Acquire();
    int64_t expiration = elector->GetLeaseExpirationTime();
    auto worker = std::async(std::launch::async, [&] { Reconcile(); });
    EXPECT_TRUE(publication.WaitEntered());
    // Loss must close the gate after the publication's critical section.
    std::promise<void> checking;
    auto lease_check = std::async(std::launch::async, [&] {
        checking.set_value();
        elector->DoCheckLeaseTimeout(expiration + 1);
    });
    EXPECT_EQ(std::future_status::ready, checking.get_future().wait_for(std::chrono::seconds(5)));
    EXPECT_EQ(std::future_status::timeout, lease_check.wait_for(std::chrono::milliseconds(50)));
    publication.Release();
    lease_check.get();
    worker.get();
    EXPECT_FALSE(ready);
    EXPECT_FALSE(elector->RunIfCurrentPromotionValid([&] { ready = true; }));
    Reconcile();
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    EXPECT_EQ(1, cleanups);
}

TEST_F(LeaderReconcileTest, StopWaitsForRecoveryCleanupAndNeverPublishesOldReadiness) {
    BlockingPoint recovery;
    std::promise<void> revoked;
    elector->SetLeadershipLostHandler([&] {
        ready = false;
        revoked.set_value();
    });
    elector->SetBecomeLeaderHandler([&] {
        ++recoveries;
        recovery.Block();
        EXPECT_FALSE(elector->RunIfCurrentPromotionValid([&] { ready = true; }));
    });
    ASSERT_TRUE(elector->Start());
    EXPECT_TRUE(recovery.WaitEntered());
    auto stopped = std::async(std::launch::async, [&] { elector->Stop(); });
    EXPECT_EQ(std::future_status::ready, revoked.get_future().wait_for(std::chrono::seconds(5)));
    recovery.Release();
    stopped.get();
    EXPECT_FALSE(ready);
    EXPECT_EQ(1, cleanups);
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    ExpectBackendUnlocked();
}

TEST_F(LeaderReconcileTest, StopRejectsAcquisitionStillInFlight) {
    BlockingPoint acquisition;
    backend->before_try = [&] { acquisition.Block(); };
    ASSERT_TRUE(elector->Start());
    EXPECT_TRUE(acquisition.WaitEntered());
    auto stopped = std::async(std::launch::async, [&] { elector->Stop(); });
    {
        std::unique_lock lock(elector->transition_mutex_);
        EXPECT_TRUE(elector->transition_cv_.wait_for(
            lock, std::chrono::seconds(5), [&] { return elector->stop_flag_.load(); }));
    }
    acquisition.Release();
    stopped.get();
    EXPECT_EQ(0, recoveries);
    EXPECT_EQ(0, cleanups);
    EXPECT_FALSE(elector->IsLeader());
    ExpectBackendUnlocked();
}

TEST_F(LeaderReconcileTest, CallbackExceptionsKeepExistingBehavior) {
    elector->SetBecomeLeaderHandler([] { throw std::runtime_error("recover failed"); });
    elector->SetNoLongerLeaderHandler([] { throw std::runtime_error("cleanup failed"); });
    Acquire();
    Reconcile();
    // Callback failures still only log; readiness remains closed and the role
    // reaches LEADER while the lease is valid, just as before this change.
    EXPECT_EQ(RoleState::LEADER, elector->GetRoleState());
    EXPECT_TRUE(elector->IsLeader());
    EXPECT_FALSE(ready);
    LoseLock();
    Reconcile();
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    Acquire();
    EXPECT_TRUE(elector->IsLeader());
}

TEST_F(LeaderReconcileTest, TransientRenewErrorWithinLeaseDoesNotInvalidateRecovery) {
    Acquire();
    backend->renew_error = EC_ERROR;
    elector->HoldLeader(TimestampUtil::GetCurrentTimeUs());
    Reconcile();
    EXPECT_TRUE(ready);
    EXPECT_TRUE(elector->IsLeader());
    EXPECT_EQ(0, cleanups);
}

TEST_F(LeaderReconcileTest, RenewalWaitingForStateMutexCannotCommitExpiredLease) {
    elector->lease_timeout_us_ = 200000;
    Acquire();
    Reconcile();
    ASSERT_TRUE(ready);
    BlockingPoint renewal;
    backend->after_renew = [&] { renewal.Block(); };
    auto rpc = std::async(std::launch::async, [&] { elector->HoldLeader(TimestampUtil::GetCurrentTimeUs()); });
    EXPECT_TRUE(renewal.WaitEntered());
    {
        std::unique_lock lock(elector->transition_mutex_);
        renewal.Release();
        // The RPC has succeeded, but committing waits for the state mutex.
        // Keep it blocked across the actual local deadline; no LeaseCheck
        // thread is running to revoke the lease on the RPC's behalf.
        EXPECT_EQ(std::future_status::timeout, rpc.wait_for(std::chrono::milliseconds(300)));
    }
    rpc.get();
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_FALSE(ready);
    Reconcile();
    EXPECT_EQ(1, cleanups);
    ExpectBackendUnlocked();
}

TEST_F(LeaderReconcileTest, CleanupCompletionDoesNotPermitCampaignBeforeUnlockReturns) {
    Acquire();
    Reconcile();
    BlockingPoint unlock;
    backend->before_unlock = [&] { unlock.Block(); };
    auto demote = std::async(std::launch::async, [&] { elector->Demote(); });
    EXPECT_TRUE(unlock.WaitEntered());
    EXPECT_FALSE(ready);
    Reconcile();
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    int campaigns = backend->campaigns;
    for (int i = 0; i < 100; ++i) {
        Acquire();
    }
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_EQ(campaigns, backend->campaigns);
    unlock.Release();
    demote.get();
    backend->before_unlock = {};
    Acquire();
    Reconcile();
    EXPECT_TRUE(ready);
    EXPECT_EQ(2, recoveries);
    EXPECT_EQ(1, cleanups);
}

TEST_F(LeaderReconcileTest, FailedUnlockStillAllowsFreshRecoveryOfSelfHeldKey) {
    Acquire();
    Reconcile();
    backend->unlock_error = EC_ERROR;
    elector->Demote();
    Reconcile();
    EXPECT_FALSE(ready);
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    backend->unlock_error = EC_OK;
    int campaigns = backend->campaigns;
    Acquire();
    Reconcile();
    EXPECT_EQ(campaigns, backend->campaigns); // GetLockHolder found our existing key.
    EXPECT_TRUE(ready);
    EXPECT_EQ(2, recoveries);
    EXPECT_EQ(1, cleanups);
}

TEST_F(LeaderReconcileTest, DiscoveryRefreshContinuesDuringCooldownWithoutCampaigning) {
    Acquire();
    Reconcile();
    elector->SetForbidCampaignLeaderTimeMs(60000);
    elector->Demote();
    Reconcile();
    ASSERT_EQ(RoleState::FOLLOWER, elector->GetRoleState());

    ASSERT_EQ(EC_OK, backend->TryLock("lock", "other", 60000));
    int campaigns = backend->campaigns;
    Acquire();
    EXPECT_EQ("other", elector->GetLeaderNodeID());
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_FALSE(ready);
    EXPECT_EQ(campaigns, backend->campaigns);

    ASSERT_EQ(EC_OK, backend->Unlock("lock", "other"));
    Acquire();
    EXPECT_TRUE(elector->GetLeaderNodeID().empty());
    EXPECT_EQ(campaigns, backend->campaigns);
    EXPECT_EQ(1, recoveries);
}

TEST_F(LeaderReconcileTest, SelfHeldKeyDuringCooldownIsDiscoveryOnly) {
    Acquire();
    Reconcile();
    elector->SetForbidCampaignLeaderTimeMs(60000);
    backend->unlock_error = EC_ERROR;
    elector->Demote();
    Reconcile();
    int campaigns = backend->campaigns;
    int renewals = backend->renewals;

    Acquire();
    EXPECT_EQ("self", elector->GetLeaderNodeID());
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_FALSE(ready);
    EXPECT_EQ(campaigns, backend->campaigns);
    EXPECT_EQ(renewals, backend->renewals);
    EXPECT_EQ(1, recoveries);
    backend->unlock_error = EC_OK;
}

TEST_F(LeaderReconcileTest, DiscoveryReadDuringReleaseCannotAdoptReleasedLease) {
    Acquire();
    Reconcile();
    BlockingPoint unlock;
    backend->before_unlock = [&] { unlock.Block(); };
    auto demote = std::async(std::launch::async, [&] { elector->Demote(); });
    EXPECT_TRUE(unlock.WaitEntered());
    Reconcile();
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());

    // Read the old self-held key while Unlock is blocked, then return that
    // snapshot only after Unlock and cleanup have both finished.
    BlockingPoint query;
    backend->after_get = [&] { query.Block(); };
    int campaigns = backend->campaigns;
    int renewals = backend->renewals;
    auto rpc = std::async(std::launch::async, [&] { Acquire(); });
    EXPECT_TRUE(query.WaitEntered());
    unlock.Release();
    demote.get();
    backend->before_unlock = {};
    query.Release();
    rpc.get();
    backend->after_get = {};

    EXPECT_FALSE(elector->IsLeader());
    EXPECT_FALSE(ready);
    EXPECT_EQ(campaigns, backend->campaigns);
    EXPECT_EQ(renewals, backend->renewals); // The snapshot was never adopted as a lease.
    EXPECT_EQ(1, recoveries);
    ExpectBackendUnlocked();
    Acquire();
    Reconcile();
    EXPECT_TRUE(ready);
    EXPECT_EQ(2, recoveries);
}

TEST_F(LeaderReconcileTest, DemoteWhileCheckingHolderCancelsAcquisitionBeforeTryLock) {
    BlockingPoint query;
    backend->before_get = [&] { query.Block(); };
    auto rpc = std::async(std::launch::async, [&] { Acquire(); });
    EXPECT_TRUE(query.WaitEntered());
    elector->Demote();
    query.Release();
    rpc.get();
    Reconcile();
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_EQ(0, recoveries);
    EXPECT_EQ(0, backend->campaigns);
    ExpectBackendUnlocked();
}

TEST_F(LeaderReconcileTest, LateRenewalReleaseDoesNotDeleteAnotherNodesLock) {
    Acquire();
    Reconcile();
    BlockingPoint renewal;
    backend->after_renew = [&] { renewal.Block(); };
    auto rpc = std::async(std::launch::async, [&] { elector->HoldLeader(TimestampUtil::GetCurrentTimeUs()); });
    EXPECT_TRUE(renewal.WaitEntered());
    EXPECT_EQ(EC_OK, backend->Unlock("lock", "self"));
    EXPECT_EQ(EC_OK, backend->TryLock("lock", "other", 60000));
    elector->DoCheckLeaseTimeout(elector->GetLeaseExpirationTime() + 1);
    Reconcile();
    renewal.Release();
    rpc.get();
    std::string holder;
    int64_t expiration;
    EXPECT_EQ(EC_OK, backend->GetLockHolder("lock", holder, expiration));
    EXPECT_EQ("other", holder);
    backend->after_renew = {};
    Acquire();
    Reconcile();
    EXPECT_EQ("other", elector->GetLeaderNodeID());
    EXPECT_FALSE(ready);
    EXPECT_EQ(1, recoveries);
}

TEST_F(LeaderReconcileTest, ExpiredAcquisitionResponseNeverStartsRecovery) {
    // Model a request whose successful response took longer than its lease.
    elector->CampaignLeader(TimestampUtil::GetCurrentTimeUs() - elector->lease_timeout_us_);
    Reconcile();
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_FALSE(ready);
    EXPECT_EQ(0, recoveries);
    ExpectBackendUnlocked();
}

TEST_F(LeaderReconcileTest, ExpiryBeforeRecoveryStartsRequiresNoCleanup) {
    Acquire();
    elector->lease_expiration_time_us_.store(TimestampUtil::GetCurrentTimeUs() - 1);
    Reconcile();
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_EQ(0, recoveries);
    EXPECT_EQ(0, cleanups);
    EXPECT_TRUE(elector->IsStableState());
}

TEST_F(LeaderReconcileTest, RecoveryCompletionChecksExpiryEvenWithoutPublicationAttempt) {
    elector->SetBecomeLeaderHandler([&] {
        ++recoveries;
        elector->lease_expiration_time_us_.store(TimestampUtil::GetCurrentTimeUs() - 1);
    });
    Acquire();
    Reconcile();
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_FALSE(ready);
    EXPECT_EQ(1, recoveries);
    EXPECT_EQ(1, cleanups);
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
}

TEST_F(LeaderReconcileTest, CooldownElapsedDoesNotPermitCampaignDuringCleanup) {
    Acquire();
    Reconcile();
    elector->SetForbidCampaignLeaderTimeMs(500);
    int64_t base_time = TimestampUtil::GetCurrentTimeUs();
    elector->DoDemote(base_time);
    int campaigns = backend->campaigns;
    elector->SetNoLongerLeaderHandler([&] {
        ++cleanups;
        elector->DoWorkLoop(base_time + 600000);
        EXPECT_EQ(campaigns, backend->campaigns);
    });
    Reconcile();
    elector->DoWorkLoop(base_time + 499000);
    EXPECT_FALSE(elector->IsLeader());
    elector->DoWorkLoop(base_time + 600000);
    Reconcile();
    EXPECT_TRUE(ready);
    EXPECT_EQ(2, recoveries);
}

TEST_F(LeaderReconcileTest, StopDuringCleanupWaitsForAlreadyRunningCleanup) {
    BlockingPoint cleanup;
    elector->SetNoLongerLeaderHandler([&] {
        ++cleanups;
        cleanup.Block();
    });
    ASSERT_TRUE(elector->Start());
    {
        std::unique_lock lock(elector->transition_mutex_);
        EXPECT_TRUE(elector->transition_cv_.wait_for(
            lock, std::chrono::seconds(5), [&] { return elector->GetRoleState() == RoleState::LEADER && ready; }));
    }
    elector->Demote();
    EXPECT_TRUE(cleanup.WaitEntered());
    auto stopped = std::async(std::launch::async, [&] { elector->Stop(); });
    EXPECT_EQ(std::future_status::timeout, stopped.wait_for(std::chrono::milliseconds(50)));
    cleanup.Release();
    stopped.get();
    EXPECT_FALSE(ready);
    EXPECT_EQ(RoleState::FOLLOWER, elector->GetRoleState());
    EXPECT_EQ(1, cleanups);
    ExpectBackendUnlocked();
}
