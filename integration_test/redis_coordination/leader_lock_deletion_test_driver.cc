#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <string>

#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/coordination_redis_backend.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/node_endpoint_info.h"

namespace kv_cache_manager {
namespace {

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

// Count real backend calls; every operation still executes against Redis.
class CountingRedisBackend : public CoordinationRedisBackend {
public:
    ErrorCode TryLock(const std::string &key, const std::string &value, int64_t ttl_ms) override {
        ++campaigns;
        return CoordinationRedisBackend::TryLock(key, value, ttl_ms);
    }

    ErrorCode GetLockHolder(const std::string &key, std::string &holder, int64_t &expiration) override {
        ++holder_checks;
        return CoordinationRedisBackend::GetLockHolder(key, holder, expiration);
    }

    ErrorCode RenewLock(const std::string &key, const std::string &value, int64_t ttl_ms) override {
        ErrorCode ec = CoordinationRedisBackend::RenewLock(key, value, ttl_ms);
        if (ec == EC_NOENT) {
            ++missing_renewals;
        }
        return ec;
    }

    std::atomic<int> campaigns{0};
    std::atomic<int> holder_checks{0};
    std::atomic<int> missing_renewals{0};
};

class RedisLeaderLockDeletionTest : public TESTBASE {
protected:
    void SetUp() override {
        // The Python integration fixture owns an isolated Redis process/port.
        const char *uri = std::getenv("KVCM_TEST_REDIS_URI");
        ASSERT_NE(nullptr, uri);
        StandardUri redis_uri(uri);
        backend_ = std::make_shared<CountingRedisBackend>();
        ASSERT_EQ(EC_OK, backend_->Init(redis_uri));
        control_ = std::make_unique<RedisClientExt>(redis_uri);
        ASSERT_TRUE(control_->Open());
        redis_key_ = backend_->GetRedisLockKey("leader");
        elector_ = std::make_unique<LeaderElector>(backend_, "leader", "self", 60000, 10);
        elector_->SetForbidCampaignLeaderTimeMs(0);
        elector_->SetLeadershipLostHandler([this] { ready_ = false; });
        elector_->SetBecomeLeaderHandler([this] {
            int round = ++recoveries_;
            if (round == 1) {
                recovery_.Block();
            }
            bool published = elector_->RunIfCurrentPromotionValid([this] { ready_ = true; });
            EXPECT_EQ(round != 1, published);
        });
        elector_->SetNoLongerLeaderHandler([this] {
            ++cleanups_;
            cleanup_.Block();
        });
    }

    void TearDown() override {
        // Release callbacks before joining even when an assertion aborted the test.
        recovery_.Release();
        cleanup_.Release();
        if (worker_.valid()) {
            worker_.get();
        }
        if (elector_) {
            elector_->Stop();
        }
        if (control_) {
            if (!redis_key_.empty()) {
                control_->Del(redis_key_);
            }
            control_->Close();
        }
    }

    void ExpectNoCampaignWhileKeyIsMissing() {
        int campaigns = backend_->campaigns;
        int holder_checks = backend_->holder_checks;
        for (int i = 0; i < 100; ++i) {
            elector_->WorkLoop();
        }
        EXPECT_EQ(campaigns, backend_->campaigns);
        EXPECT_EQ(holder_checks + 100, backend_->holder_checks);
        EXPECT_TRUE(elector_->GetLeaderNodeID().empty());
        EXPECT_FALSE(elector_->IsLeader());
        EXPECT_FALSE(ready_);
        std::string value;
        EXPECT_EQ(EC_NOENT, control_->Get(redis_key_, value));
    }

    std::shared_ptr<CountingRedisBackend> backend_;
    std::unique_ptr<RedisClientExt> control_;
    std::unique_ptr<LeaderElector> elector_;
    std::string redis_key_;
    BlockingPoint recovery_;
    BlockingPoint cleanup_;
    std::future<void> worker_;
    std::atomic<bool> ready_{false};
    std::atomic<int> recoveries_{0};
    std::atomic<int> cleanups_{0};
};

TEST_F(RedisLeaderLockDeletionTest, ExternalDeletionDuringRecoveryRequiresCleanupBeforeReacquisition) {
    elector_->WorkLoop();
    worker_ = std::async(std::launch::async, [this] { elector_->ProcessStateTransitionsForTest(); });
    ASSERT_TRUE(recovery_.WaitEntered());
    ASSERT_EQ(RoleState::PROMOTING, elector_->GetRoleState());
    int64_t ttl_ms;
    ASSERT_EQ(EC_OK, control_->Pttl(redis_key_, ttl_ms));
    ASSERT_GT(ttl_ms, 0);

    // A separate Redis client deletes a live key; the elector is not demoted
    // directly. Its next real Lua renewal must discover EC_NOENT.
    ASSERT_EQ(EC_OK, control_->Del(redis_key_));
    elector_->WorkLoop();
    EXPECT_EQ(1, backend_->missing_renewals);
    EXPECT_EQ(RoleState::PROMOTING, elector_->GetRoleState());
    ExpectNoCampaignWhileKeyIsMissing();

    // A second elector becomes leader while the first recovery is blocked.
    // The old node must still discover its real Redis endpoint record.
    auto probe_backend = std::make_shared<CoordinationRedisBackend>();
    ASSERT_EQ(EC_OK, probe_backend->Init(StandardUri(std::getenv("KVCM_TEST_REDIS_URI"))));
    LeaderElector probe(probe_backend, "leader", "probe", 60000, 10);
    NodeEndpointInfo endpoint("probe", "127.0.0.1", 12345, 12346, 12345, 12347);
    ASSERT_EQ(EC_OK, probe.SetSelfNodeInfo(endpoint));
    bool probe_ready = false;
    probe.SetBecomeLeaderHandler([&] { probe.RunIfCurrentPromotionValid([&] { probe_ready = true; }); });
    probe.SetNoLongerLeaderHandler([&] { probe_ready = false; });
    probe.WorkLoop();
    probe.ProcessStateTransitionsForTest();
    ASSERT_TRUE(probe_ready);
    int campaigns = backend_->campaigns;
    elector_->WorkLoop();
    EXPECT_EQ("probe", elector_->GetLeaderNodeID());
    EXPECT_EQ(campaigns, backend_->campaigns);

    recovery_.Release();
    ASSERT_TRUE(cleanup_.WaitEntered());
    ASSERT_EQ(RoleState::DEMOTING, elector_->GetRoleState());
    int holder_checks = backend_->holder_checks;
    elector_->WorkLoop();
    EXPECT_GT(backend_->holder_checks, holder_checks);
    EXPECT_EQ("probe", elector_->GetLeaderNodeID());
    EXPECT_EQ(campaigns, backend_->campaigns);
    EXPECT_FALSE(elector_->IsLeader());
    EXPECT_FALSE(ready_);
    NodeEndpointInfo discovered;
    EXPECT_EQ(EC_OK, elector_->GetLeaderNodeInfo(discovered));
    EXPECT_EQ(endpoint.node_id(), discovered.node_id());
    EXPECT_EQ(endpoint.host(), discovered.host());
    EXPECT_EQ(endpoint.meta_http_port(), discovered.meta_http_port());
    probe.Demote();
    probe.ProcessStateTransitionsForTest();
    EXPECT_FALSE(probe_ready);
    ExpectNoCampaignWhileKeyIsMissing();
    cleanup_.Release();
    worker_.get();
    EXPECT_EQ(RoleState::FOLLOWER, elector_->GetRoleState());
    EXPECT_FALSE(elector_->IsLeader());
    EXPECT_FALSE(ready_);

    elector_->WorkLoop();
    elector_->ProcessStateTransitionsForTest();
    EXPECT_EQ(RoleState::LEADER, elector_->GetRoleState());
    EXPECT_TRUE(elector_->IsLeader());
    EXPECT_TRUE(ready_);
    EXPECT_EQ(2, recoveries_);
    EXPECT_EQ(1, cleanups_);
    EXPECT_EQ(2, backend_->campaigns);
    std::string holder;
    EXPECT_EQ(EC_OK, control_->Get(redis_key_, holder));
    EXPECT_EQ("self", holder);
}

} // namespace
} // namespace kv_cache_manager
