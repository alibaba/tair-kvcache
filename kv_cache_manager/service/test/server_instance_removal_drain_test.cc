#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/admin_service_impl.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/server.h"
#include "stub.h"

using namespace kv_cache_manager;
using namespace std::chrono_literals;

namespace {

class RequestStageGate {
public:
    void Enter() {
        std::unique_lock lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this] { return released_; });
    }

    bool WaitForEntered() {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, 2s, [this] { return entered_; });
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

struct OwnerWaitControl {
    MetaSearcherManager *searchers;
    const std::string &instance_id;
    RequestContext *context;
    RequestStageGate owner_wait;
    RequestStageGate response_materialization;
    bool deadline_started = false;
};

thread_local OwnerWaitControl *owner_wait_control = nullptr;

// Keep the real RemoveInstance/WaitForOtherOwners/Restore path. Only the removal
// thread's owner-wait clock advances, avoiding the production five-second wait.
// Background threads and the test's timed waits retain the normal monotonic clock.
std::chrono::steady_clock::time_point ControlledSteadyNow() noexcept {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    const std::chrono::steady_clock::time_point real_now(
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(now.tv_sec) +
                                                                        std::chrono::nanoseconds(now.tv_nsec)));
    auto *control = owner_wait_control;
    if (!control) {
        return real_now;
    }
    if (control->deadline_started) {
        return real_now + 1min;
    }

    // Earlier calls in admission/removal must see real time. The first clock
    // call after extraction is WaitForOtherOwners constructing its deadline.
    std::shared_lock lock(control->searchers->mutex_, std::try_to_lock);
    if (!lock.owns_lock() || control->searchers->meta_searcher_map_.count(control->instance_id) != 0) {
        return real_now;
    }
    lock.unlock();
    control->deadline_started = true;
    control->context->set_response_debug_json_generator([control] {
        control->response_materialization.Enter();
        RequestContext::JsonFragment fragment;
        fragment.json = "{}";
        fragment.valid = true;
        return fragment;
    });
    control->owner_wait.Enter();
    return real_now;
}

} // namespace

class ServerInstanceRemovalDrainTest : public TESTBASE {
public:
    void SetUp() override {
        // Install before Init starts maintenance threads; restore only after
        // server_ has destroyed and joined them (member destruction order).
        clock_stub_.set(&std::chrono::steady_clock::now, ControlledSteadyNow);
        server_.metrics_registry_ = std::make_shared<MetricsRegistry>();
        server_.registry_manager_ = std::make_shared<RegistryManager>("local://", server_.metrics_registry_);
        ASSERT_TRUE(server_.registry_manager_->Init());
        auto group = std::make_shared<InstanceGroup>();
        group->cache_config_ = std::make_shared<CacheConfig>();
        group->cache_config_->meta_indexer_config_ = std::make_shared<MetaIndexerConfig>();
        group->cache_config_->meta_indexer_config_->meta_storage_backend_config_->storage_type_ =
            META_LOCAL_BACKEND_TYPE_STR;
        server_.registry_manager_->instance_group_configs_[kGroup] = group;
        server_.cache_manager_ = std::make_shared<CacheManager>(server_.metrics_registry_, server_.registry_manager_);
        ASSERT_TRUE(server_.cache_manager_->Init());
        SeedInstance(kInstance);
        SeedInstance(kSentinel);
        server_.meta_impl_ = std::make_shared<MetaServiceImpl>(server_.cache_manager_, nullptr, nullptr);
        server_.admin_impl_ = std::make_shared<AdminServiceImpl>(
            server_.cache_manager_, nullptr, server_.metrics_registry_, server_.registry_manager_, nullptr);
        server_.meta_impl_->EnableLeaderOnlyRequests();
        server_.admin_impl_->EnableLeaderOnlyRequests();
    }

    void SeedInstance(const std::string &instance_id) {
        auto info = std::make_shared<InstanceInfo>("test_quota_group",
                                                   kGroup,
                                                   instance_id,
                                                   64,
                                                   std::vector<LocationSpecInfo>{LocationSpecInfo("tp0", 512)},
                                                   ModelDeployment());
        server_.registry_manager_->instance_infos_[instance_id] = info;
        RequestContext context("seed-" + instance_id);
        ASSERT_EQ(EC_OK, server_.cache_manager_->TryCreateMetaSearcher(&context, instance_id));
    }

    bool WaitForAdmissionToClose() {
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (server_.admin_impl_->is_accepting_leader_only_requests_.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    }

    Stub clock_stub_;
    Server server_;
    const std::string kGroup = "test_group";
    const std::string kInstance = "removing_instance";
    const std::string kSentinel = "other_instance";
};

TEST_F(ServerInstanceRemovalDrainTest, DemotionDrainsRemoveTimeoutAndResponseBeforeCleanup) {
    auto searchers = server_.cache_manager_->meta_searcher_manager_;
    auto other_owner = searchers->GetMetaSearcher(kInstance);
    ASSERT_NE(nullptr, other_owner);
    std::weak_ptr<MetaSearcher> removed_searcher = other_owner;
    RequestContext context("remove-during-demotion");
    proto::admin::RemoveInstanceRequest request;
    request.set_instance_group(kGroup);
    request.set_instance_id(kInstance);
    proto::admin::CommonResponse response;
    OwnerWaitControl control{searchers.get(), kInstance, &context};

    auto removal = std::async(std::launch::async, [&] {
        owner_wait_control = &control;
        server_.admin_impl_->RemoveInstance(&context, &request, &response);
        owner_wait_control = nullptr;
    });
    // Do not use fatal assertions after starting the request: both controlled
    // stages must be released even when an expectation fails.
    EXPECT_TRUE(control.owner_wait.WaitForEntered());
    EXPECT_EQ(1, server_.admin_impl_->active_leader_only_request_count_.load());
    auto demotion = std::async(std::launch::async, [&] { server_.OnNoLongerLeader(); });
    EXPECT_TRUE(WaitForAdmissionToClose());
    EXPECT_EQ(std::future_status::timeout, demotion.wait_for(50ms));
    EXPECT_NE(nullptr, searchers->GetMetaSearcher(kSentinel));
    EXPECT_EQ(nullptr, searchers->GetMetaSearcher(kInstance));

    // Rejected requests must not join the in-flight count or restart removal.
    RequestContext rejected_context("remove-after-demotion");
    proto::admin::CommonResponse rejected;
    server_.admin_impl_->RemoveInstance(&rejected_context, &request, &rejected);
    EXPECT_EQ(proto::admin::SERVER_NOT_LEADER, rejected.header().status().code());
    EXPECT_EQ(1, server_.admin_impl_->active_leader_only_request_count_.load());

    control.owner_wait.Release();
    EXPECT_TRUE(control.response_materialization.WaitForEntered());
    // The actual timeout branch has restored the old searcher, but the Admin
    // guard has not completed admission accounting during response generation.
    EXPECT_EQ(1, server_.admin_impl_->active_leader_only_request_count_.load());
    {
        std::shared_lock lock(searchers->mutex_);
        const auto retiring = searchers->retiring_meta_searcher_map_.find(kInstance);
        EXPECT_NE(searchers->retiring_meta_searcher_map_.end(), retiring);
        if (retiring != searchers->retiring_meta_searcher_map_.end()) {
            EXPECT_EQ(other_owner, retiring->second);
        }
    }
    EXPECT_EQ(std::future_status::timeout, demotion.wait_for(50ms));
    EXPECT_NE(nullptr, searchers->GetMetaSearcher(kSentinel));

    control.response_materialization.Release();
    removal.get();
    demotion.get();
    EXPECT_EQ(proto::admin::INTERNAL_ERROR, response.header().status().code());
    EXPECT_EQ(0, server_.admin_impl_->active_leader_only_request_count_.load());
    {
        std::shared_lock lock(searchers->mutex_);
        EXPECT_TRUE(searchers->meta_searcher_map_.empty());
        EXPECT_TRUE(searchers->retiring_meta_searcher_map_.empty());
    }
    other_owner.reset();
    EXPECT_TRUE(removed_searcher.expired());
}
