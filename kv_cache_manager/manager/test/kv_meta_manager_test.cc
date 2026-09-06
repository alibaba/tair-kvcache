#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/quota_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/kv_meta_instance.h"
#include "kv_cache_manager/manager/kv_meta_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class KvMetaManagerTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        ASSERT_TRUE(registry_manager_->Init());

        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        ASSERT_TRUE(cache_manager_->Init());

        StartupConfigLoader loader;
        ASSERT_TRUE(loader.Init(registry_manager_));
        ASSERT_TRUE(loader.Load(""));

        manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
        ASSERT_TRUE(manager_->Init());
        ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, "default", kInstanceId, "emb-test").first);
    }

    void TearDown() override {
        manager_->Shutdown();
        manager_.reset();
        cache_manager_.reset();
        registry_manager_.reset();
        metrics_registry_.reset();
    }

    static constexpr const char *kInstanceId = "embedding-instance";
    RequestContext request_context_{"kv_meta_manager_test"};
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<KvMetaManager> manager_;
};

TEST_F(KvMetaManagerTest, DynamicSizesAreIndependentAndInvisibleUntilFinish) {
    const std::vector<std::string> keys{"emb-a", "emb-b"};
    const std::vector<std::uint64_t> sizes{17, 33};
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, keys, sizes, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ((std::vector<bool>{false, false}), start.key_mask);
    ASSERT_EQ(2, start.locations.size());
    ASSERT_FALSE(start.write_session_id.empty());
    EXPECT_EQ(17, start.locations[0].value_size);
    EXPECT_EQ(33, start.locations[1].value_size);
    ASSERT_EQ(1, start.locations[0].specs.size());
    ASSERT_EQ(1, start.locations[1].specs.size());

    const DataStorageUri first_uri(start.locations[0].specs[0].second);
    const DataStorageUri second_uri(start.locations[1].specs[0].second);
    ASSERT_TRUE(first_uri.Valid());
    ASSERT_TRUE(second_uri.Valid());
    // The default NFS backend is configured to pack up to eight keys. The
    // generic path deliberately uses singleton Create calls, so these values
    // still have different physical deletion boundaries.
    EXPECT_NE(first_uri.GetPath(), second_uri.GetPath());
    std::uint64_t first_size = 0;
    std::uint64_t second_size = 0;
    first_uri.GetParamAs<std::uint64_t>("size", first_size);
    second_uri.GetParamAs<std::uint64_t>("size", second_size);
    EXPECT_EQ(17, first_size);
    EXPECT_EQ(33, second_size);

    auto [before_finish_ec, before_finish] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, before_finish_ec);
    ASSERT_EQ(2, before_finish.size());
    EXPECT_FALSE(before_finish[0].found);
    EXPECT_FALSE(before_finish[1].found);

    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    ASSERT_TRUE(values[0].found);
    ASSERT_TRUE(values[1].found);
    EXPECT_EQ(17, values[0].location.value_size);
    EXPECT_EQ(33, values[1].location.value_size);

    // Committed generic objects deliberately remain CLS_NEW. The negative
    // timestamp is private to KVMeta and keeps them out of the existing
    // CLS_SERVING reclaimer/migration path.
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(
        KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    KeyVector internal_keys;
    LocationIdsPerKey location_ids;
    for (const auto &key : keys) {
        internal_keys.push_back(KvMetaManager::InternalKey(key));
        location_ids.push_back({KvMetaManager::StableLocationId(key)});
    }
    LocationsPerKey exact;
    const auto exact_result = indexer->GetLocations(&request_context_, internal_keys, location_ids, exact);
    ASSERT_EQ(2, exact_result.per_location_error_codes.size());
    ASSERT_EQ(2, exact.size());
    for (std::size_t i = 0; i < exact.size(); ++i) {
        ASSERT_EQ(EC_OK, exact_result.per_location_error_codes[i][0]);
        ASSERT_TRUE(exact[i][0]);
        EXPECT_EQ(CLS_NEW, exact[i][0]->status());
        EXPECT_LT(exact[i][0]->create_time(), 0);
    }

    ASSERT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {keys[0]}));
    auto [after_remove_ec, after_remove] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, after_remove_ec);
    EXPECT_FALSE(after_remove[0].found);
    EXPECT_TRUE(after_remove[1].found);
    EXPECT_EQ(33, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, AtomicFinishFailureRollsBackEveryValue) {
    const std::vector<std::string> keys{"atomic-a", "atomic-b"};
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, keys, {64, 128}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(2, start.locations.size());

    // One failed item aborts the complete generic-object transaction.
    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, false}));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);

    // Exact metadata was removed, so the same keys can be admitted again.
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, keys, {7, 9}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ((std::vector<bool>{false, false}), retry.key_mask);
    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false, false}));
}

TEST_F(KvMetaManagerTest, RecoveryRebuildsExactDynamicByteUsage) {
    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {"recover-a", "recover-b"}, {17, 33}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(2, start.locations.size());
    ASSERT_EQ(start.locations[0].type, start.locations[1].type);
    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(
        KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    indexer->SetStorageUsageByType(start.locations[0].type, 1);
    ASSERT_EQ(1, indexer->GetStorageUsage());

    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(50, indexer->GetStorageUsage());
    EXPECT_EQ(50, indexer->GetStorageUsageByType(start.locations[0].type));
}

TEST_F(KvMetaManagerTest, TrimUsesBoundedMaintenanceBatches) {
    constexpr std::size_t kObjectCount = 257;
    for (std::size_t begin = 0; begin < kObjectCount; begin += manager_->limits().max_batch_items) {
        const std::size_t end = std::min(kObjectCount, begin + manager_->limits().max_batch_items);
        std::vector<std::string> keys;
        std::vector<std::uint64_t> sizes;
        for (std::size_t i = begin; i < end; ++i) {
            keys.push_back("trim-" + std::to_string(i));
            sizes.push_back(1);
        }
        auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, keys, sizes, 30);
        ASSERT_EQ(EC_OK, start_ec);
        ASSERT_EQ(EC_OK,
                  manager_->FinishWrite(&request_context_,
                                        kInstanceId,
                                        start.write_session_id,
                                        std::vector<bool>(start.locations.size(), true)));
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(
        KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(kObjectCount, indexer->GetStorageUsage());

    // A demotion cancels an unbounded namespace walk before the server waits
    // for KVMeta RPCs. Cancellation is sticky until the next successful
    // leader recovery explicitly resumes maintenance.
    manager_->CancelMaintenance();
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->TrimAll(&request_context_, kInstanceId, false));
    auto [before_resume_ec, before_resume] =
        manager_->Get(&request_context_, kInstanceId, {"trim-0", "trim-256"});
    ASSERT_EQ(EC_OK, before_resume_ec);
    ASSERT_EQ(2, before_resume.size());
    EXPECT_TRUE(before_resume[0].found);
    EXPECT_TRUE(before_resume[1].found);

    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_EQ(0, indexer->GetStorageUsage());

    auto [get_ec, values] = manager_->Get(
        &request_context_, kInstanceId, {"trim-0", "trim-128", "trim-256"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(3, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    EXPECT_FALSE(values[2].found);
}

TEST_F(KvMetaManagerTest, ExistingAndInflightKeysAreMasked) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {21}, 30);
    ASSERT_EQ(EC_OK, first_ec);
    ASSERT_EQ((std::vector<bool>{false}), first.key_mask);

    auto [wrong_active_ec, wrong_active] =
        manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {999}, 30);
    EXPECT_EQ(EC_MISMATCH, wrong_active_ec);
    EXPECT_TRUE(wrong_active.key_mask.empty());
    EXPECT_TRUE(wrong_active.locations.empty());

    auto [second_ec, second] = manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {21}, 30);
    ASSERT_EQ(EC_OK, second_ec);
    EXPECT_EQ((std::vector<bool>{true}), second.key_mask);
    EXPECT_TRUE(second.locations.empty());
    EXPECT_TRUE(second.write_session_id.empty());

    // A malformed finish request must not consume the valid session.
    EXPECT_EQ(EC_BADARGS, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {}));
    EXPECT_EQ(EC_MISMATCH,
              manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {true, true}));
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {true}));

    auto [wrong_committed_ec, wrong_committed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {42}, 30);
    EXPECT_EQ(EC_MISMATCH, wrong_committed_ec);
    EXPECT_TRUE(wrong_committed.key_mask.empty());
    EXPECT_TRUE(wrong_committed.locations.empty());

    auto [third_ec, third] = manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {21}, 30);
    ASSERT_EQ(EC_OK, third_ec);
    EXPECT_EQ((std::vector<bool>{true}), third.key_mask);
    EXPECT_TRUE(third.locations.empty());
}

TEST_F(KvMetaManagerTest, ExactIdentityAndStorageSchemeAreValidated) {
    const std::string key = "owned-key";
    const auto internal_key = KvMetaManager::InternalKey(key);
    const std::string location_id = KvMetaManager::StableLocationId(key);
    EXPECT_TRUE(manager_->IsOwnedLocation(internal_key, location_id));
    EXPECT_FALSE(manager_->IsOwnedLocation(KvMetaManager::InternalKey("another-key"), location_id));
    EXPECT_FALSE(manager_->IsOwnedLocation(internal_key, "kvmeta:v1:6F"));
    EXPECT_FALSE(manager_->IsOwnedLocation(internal_key, "kvmeta:v1:0"));

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {key}, {31}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(
        KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    auto corrupt_scheme = [](const std::vector<ErrorCode> &get_ecs,
                             const LocationIdVector &,
                             std::size_t,
                             CacheLocationVector &locations,
                             PropertyMap &) -> LocationModifierResult {
        if (get_ecs.size() != 1 || get_ecs[0] != EC_OK || locations.size() != 1 || !locations[0]) {
            return {MA_FAIL, {EC_CORRUPTION}};
        }
        auto replacement = std::make_shared<CacheLocation>(*locations[0]);
        DataStorageUri uri(replacement->location_specs().front().uri());
        uri.SetProtocol("dummy");
        replacement->mutable_location_specs().front().set_uri(uri.ToUriString());
        locations[0] = std::move(replacement);
        return {MA_OK, {EC_OK}};
    };
    const auto rmw = indexer->ReadModifyWriteTargetLocations(
        &request_context_, {internal_key}, {{location_id}}, corrupt_scheme);
    ASSERT_EQ(EC_OK, rmw.ec);
    ASSERT_EQ(1, rmw.per_location_error_codes.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), rmw.per_location_error_codes[0]);

    EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstanceId, {key}).first);
    EXPECT_EQ(EC_CORRUPTION, manager_->Remove(&request_context_, kInstanceId, {key}));
}

TEST_F(KvMetaManagerTest, RejectsAmbiguousOrUnboundedRequestsBeforeAllocation) {
    auto [duplicate_ec, duplicate] =
        manager_->StartWrite(&request_context_, kInstanceId, {"dup", "dup"}, {1, 2}, 30);
    EXPECT_EQ(EC_DUPLICATE_ENTITY, duplicate_ec);
    EXPECT_TRUE(duplicate.locations.empty());

    auto [size_count_ec, size_count] = manager_->StartWrite(&request_context_, kInstanceId, {"key"}, {}, 30);
    EXPECT_EQ(EC_BADARGS, size_count_ec);
    EXPECT_TRUE(size_count.locations.empty());

    auto [zero_size_ec, zero_size] = manager_->StartWrite(&request_context_, kInstanceId, {"key"}, {0}, 30);
    EXPECT_EQ(EC_OUT_OF_LIMIT, zero_size_ec);
    EXPECT_TRUE(zero_size.locations.empty());

    auto [timeout_ec, timeout] = manager_->StartWrite(&request_context_, kInstanceId, {"key"}, {1}, 0);
    EXPECT_EQ(EC_BADARGS, timeout_ec);
    EXPECT_TRUE(timeout.locations.empty());

    // Keep the overflow check correct for a custom configuration where one
    // value may fit max_value_bytes but cannot fit max_batch_bytes.
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_value_bytes = 16;
    limits.max_batch_bytes = 8;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());
    auto [batch_limit_ec, batch_limit] =
        manager_->StartWrite(&request_context_, kInstanceId, {"too-large-for-batch"}, {9}, 30);
    EXPECT_EQ(EC_OUT_OF_LIMIT, batch_limit_ec);
    EXPECT_TRUE(batch_limit.locations.empty());
}

TEST_F(KvMetaManagerTest, RecoveryCanBeCancelledWithoutTouchingMetadata) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"committed"}, {23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->DoRecover([]() { return true; }));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"committed"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values[0].found);
}

TEST_F(KvMetaManagerTest, DemotionDefersUnboundedSessionCleanupToRecovery) {
    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {"demoted-active"}, {19}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    manager_->DoCleanup();
    EXPECT_EQ(EC_NOENT,
              manager_->FinishWrite(
                  &request_context_, kInstanceId, start.write_session_id, {true}));
    auto [hidden_ec, hidden] =
        manager_->Get(&request_context_, kInstanceId, {"demoted-active"});
    ASSERT_EQ(EC_OK, hidden_ec);
    ASSERT_EQ(1, hidden.size());
    EXPECT_FALSE(hidden[0].found);

    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] =
        manager_->StartWrite(&request_context_, kInstanceId, {"demoted-active"}, {19}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ((std::vector<bool>{false}), retry.key_mask);
    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(
                  &request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ActiveSessionCountIsBoundedBeforeAllocation) {
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_active_write_sessions = 1;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());

    auto [first_ec, first] =
        manager_->StartWrite(&request_context_, kInstanceId, {"session-a"}, {7}, 30);
    ASSERT_EQ(EC_OK, first_ec);
    auto [second_ec, second] =
        manager_->StartWrite(&request_context_, kInstanceId, {"session-b"}, {9}, 30);
    EXPECT_EQ(EC_NOSPC, second_ec);
    EXPECT_TRUE(second.key_mask.empty());
    EXPECT_TRUE(second.locations.empty());

    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(
                  &request_context_, kInstanceId, first.write_session_id, {false}));
    auto [retry_ec, retry] =
        manager_->StartWrite(&request_context_, kInstanceId, {"session-b"}, {9}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(
                  &request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RejectsAnInstanceGroupAlreadyUsedByKvCache) {
    ModelDeployment deployment;
    deployment.set_model_name("ordinary-kv-cache");
    deployment.set_dtype("fp16");
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);
    ASSERT_EQ(EC_OK,
              cache_manager_
                  ->RegisterInstance(&request_context_,
                                     "default",
                                     "ordinary-instance",
                                     1,
                                     {LocationSpecInfo("value", 1)},
                                     deployment,
                                     {},
                                     CacheManager::QueryType::QT_BATCH_GET)
                  .first);

    EXPECT_EQ(EC_BADARGS,
              manager_->RegisterInstance(&request_context_, "default", "another-object-instance", "").first);
    EXPECT_EQ(EC_BADARGS,
              manager_->StartWrite(&request_context_, kInstanceId, {"must-not-share-quota"}, {1}, 30).first);
}

TEST_F(KvMetaManagerTest, ExactValueSizesAreIncludedInByteAdmission) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup object_group(*default_group);
    object_group.set_name("small-object-group");
    object_group.set_global_quota_group_name("small-object-quota");
    object_group.set_version(1);
    object_group.set_quota(InstanceGroupQuota(
        20, {QuotaConfig(20, DataStorageType::DATA_STORAGE_TYPE_NFS)}));
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));
    ASSERT_EQ(EC_OK,
              manager_->RegisterInstance(&request_context_, "small-object-group", "small-object-instance", "").first);

    auto [oversized_ec, oversized] = manager_->StartWrite(
        &request_context_, "small-object-instance", {"a", "b"}, {17, 4}, 30);
    EXPECT_EQ(EC_NOSPC, oversized_ec);
    EXPECT_TRUE(oversized.locations.empty());

    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, "small-object-instance", {"a"}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(
                  &request_context_, "small-object-instance", start.write_session_id, {true}));
    auto [remaining_ec, remaining] =
        manager_->StartWrite(&request_context_, "small-object-instance", {"b"}, {4}, 30);
    EXPECT_EQ(EC_NOSPC, remaining_ec);
    EXPECT_TRUE(remaining.locations.empty());

    auto [fill_ec, fill] =
        manager_->StartWrite(&request_context_, "small-object-instance", {"b"}, {3}, 30);
    ASSERT_EQ(EC_OK, fill_ec);
    ASSERT_EQ(EC_OK,
              manager_->FinishWrite(
                  &request_context_, "small-object-instance", fill.write_session_id, {true}));
    auto [full_ec, full] =
        manager_->StartWrite(&request_context_, "small-object-instance", {"c"}, {1}, 30);
    EXPECT_EQ(EC_NOSPC, full_ec);
    EXPECT_TRUE(full.locations.empty());
}

TEST_F(KvMetaManagerTest, ConcurrentStartsCannotOvershootExactByteQuota) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup object_group(*default_group);
    object_group.set_name("concurrent-object-group");
    object_group.set_global_quota_group_name("concurrent-object-quota");
    object_group.set_version(1);
    object_group.set_quota(InstanceGroupQuota(
        20, {QuotaConfig(20, DataStorageType::DATA_STORAGE_TYPE_NFS)}));
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));
    ASSERT_EQ(EC_OK,
              manager_->RegisterInstance(
                  &request_context_, "concurrent-object-group", "concurrent-object-instance", "")
                  .first);

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::array<ErrorCode, 2> errors{EC_UNKNOWN, EC_UNKNOWN};
    std::array<KvMetaManager::StartWriteResult, 2> results;
    std::array<std::thread, 2> workers;
    for (std::size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i]() {
            RequestContext context("kv_meta_concurrent_admission_" + std::to_string(i));
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto [ec, result] = manager_->StartWrite(&context,
                                                      "concurrent-object-instance",
                                                      {"key-" + std::to_string(i)},
                                                      {15},
                                                      30);
            errors[i] = ec;
            results[i] = std::move(result);
        });
    }
    while (ready.load(std::memory_order_acquire) != static_cast<int>(workers.size())) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto &worker : workers) {
        worker.join();
    }

    const std::size_t success_count =
        static_cast<std::size_t>(std::count(errors.begin(), errors.end(), EC_OK));
    const std::size_t quota_failure_count =
        static_cast<std::size_t>(std::count(errors.begin(), errors.end(), EC_NOSPC));
    EXPECT_EQ(1, success_count);
    EXPECT_EQ(1, quota_failure_count);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(
        KvMetaManager::InternalInstanceId("concurrent-object-instance"));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(15, indexer->GetStorageUsage());
    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (errors[i] == EC_OK) {
            ASSERT_EQ((std::vector<bool>{false}), results[i].key_mask);
            ASSERT_EQ(1, results[i].locations.size());
            EXPECT_EQ(15, results[i].locations.front().value_size);
            ASSERT_EQ(EC_OK,
                      manager_->FinishWrite(&request_context_,
                                            "concurrent-object-instance",
                                            results[i].write_session_id,
                                            {false}));
        } else {
            EXPECT_TRUE(results[i].key_mask.empty());
            EXPECT_TRUE(results[i].locations.empty());
        }
    }
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST(KvMetaInstanceMarkerTest, RequiresTheCompleteReservedSchema) {
    ModelDeployment deployment;
    deployment.set_model_name(std::string(kKvMetaModelName));
    deployment.set_dtype(std::string(kKvMetaDtype));
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);
    deployment.set_extra(std::string(kKvMetaDeploymentExtra));
    InstanceInfo instance("quota",
                          "objects",
                          std::string(kKvMetaInternalInstancePrefix) + "6964",
                          1,
                          {LocationSpecInfo(std::string(kKvMetaValueSpecName), 1)},
                          deployment,
                          {},
                          1);
    EXPECT_TRUE(IsKvMetaInstance(instance));

    instance.set_block_size(2);
    EXPECT_FALSE(IsKvMetaInstance(instance));
    instance.set_block_size(1);
    instance.set_instance_id(std::string(kKvMetaInternalInstancePrefix) + "not-hex");
    EXPECT_FALSE(IsKvMetaInstance(instance));
}

} // namespace kv_cache_manager
