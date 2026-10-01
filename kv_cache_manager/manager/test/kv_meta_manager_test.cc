#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/config/trigger_strategy.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/kv_meta_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

class TestPaceBackend : public DataStorageBackend {
public:
    explicit TestPaceBackend(std::shared_ptr<MetricsRegistry> metrics_registry,
                             DataStorageType type = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL)
        : DataStorageBackend(std::move(metrics_registry)), type_(type) {}

    DataStorageType GetType() override { return type_; }
    bool Available() override { return IsOpen() && IsAvailable(); }
    double GetStorageUsageRatio(const std::string &) const override { return 0.0; }
    ErrorCode DoOpen(const StorageConfig &, const std::string &) override {
        SetOpen(true);
        SetAvailable(true);
        return EC_OK;
    }
    ErrorCode Close() override {
        SetOpen(false);
        SetAvailable(false);
        return EC_OK;
    }
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             std::size_t size,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        std::vector<std::pair<ErrorCode, DataStorageUri>> result;
        result.reserve(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            DataStorageUri uri;
            uri.SetProtocol(kTairMempoolUriScheme);
            uri.SetPath("/" + std::to_string(next_offset_.fetch_add(1)));
            uri.SetParam("media_type", "0");
            uri.SetParam("node_id", "1");
            uri.SetParam("range_id", "0");
            uri.SetParam("size", std::to_string(size));
            result.emplace_back(EC_OK, std::move(uri));
        }
        if (cb) {
            cb();
        }
        return result;
    }
    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &uris, const std::string &, std::function<void()> cb) override {
        delete_count_.fetch_add(uris.size(), std::memory_order_release);
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(uris.size(), EC_OK);
    }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &uris) override {
        return std::vector<bool>(uris.size(), true);
    }
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &uris) override {
        return std::vector<ErrorCode>(uris.size(), EC_OK);
    }
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &uris) override {
        return std::vector<ErrorCode>(uris.size(), EC_OK);
    }

    std::size_t delete_count() const { return delete_count_.load(std::memory_order_acquire); }

private:
    DataStorageType type_;
    std::atomic<std::uint64_t> next_offset_{1};
    std::atomic<std::size_t> delete_count_{0};
};

} // namespace

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

        auto pace_spec = std::make_shared<TairMemPoolStorageSpec>();
        pace_spec->set_domain("test-pace");
        pace_backend_ = std::make_shared<TestPaceBackend>(metrics_registry_);
        ASSERT_EQ(
            EC_OK,
            pace_backend_->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, kStorageName, pace_spec),
                                context_.trace_id()));
        registry_manager_->data_storage_manager()->storage_map_[kStorageName] = pace_backend_;
        const auto [group_ec, group] = registry_manager_->GetInstanceGroup(&context_, "default");
        ASSERT_EQ(EC_OK, group_ec);
        ASSERT_TRUE(group);
        InstanceGroup updated(*group);
        updated.set_storage_candidates({kStorageName});
        updated.set_version(group->version() + 1);
        ASSERT_EQ(EC_OK, registry_manager_->UpdateInstanceGroup(&context_, updated, group->version()));

        manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
        ASSERT_TRUE(manager_->Init());
        ASSERT_EQ(EC_OK, manager_->DoRecover());
        ASSERT_EQ(EC_OK, manager_->RegisterInstance(&context_, "default", kDefaultInstance, "emb-test").first);
    }

    void TearDown() override {
        manager_->Shutdown();
        manager_.reset();
        cache_manager_.reset();
        registry_manager_.reset();
        metrics_registry_.reset();
    }

    void CreateGroup(const std::string &group_name,
                     const std::string &instance_id,
                     std::int64_t capacity,
                     double threshold,
                     std::int32_t delete_delay_ms = 0) {
        const auto [ec, default_group] = registry_manager_->GetInstanceGroup(&context_, "default");
        ASSERT_EQ(EC_OK, ec);
        ASSERT_TRUE(default_group && default_group->cache_config() &&
                    default_group->cache_config()->reclaim_strategy());

        auto cache_config = std::make_shared<CacheConfig>();
        ASSERT_TRUE(cache_config->FromJsonString(default_group->cache_config()->ToJsonString()));
        auto strategy = std::make_shared<CacheReclaimStrategy>(*cache_config->reclaim_strategy());
        TriggerStrategy trigger = strategy->trigger_strategy();
        trigger.set_used_percentage(threshold);
        strategy->set_trigger_strategy(trigger);
        strategy->set_reclaim_policy(ReclaimPolicy::POLICY_LRU);
        strategy->set_delay_before_delete_ms(delete_delay_ms);
        cache_config->set_reclaim_strategy(strategy);

        InstanceGroup group(*default_group);
        group.set_name(group_name);
        group.set_global_quota_group_name(group_name + "-quota");
        group.set_storage_candidates({kStorageName});
        group.set_cache_config(cache_config);
        group.set_quota(
            InstanceGroupQuota(capacity, {QuotaConfig(capacity, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL)}));
        group.set_version(1);
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&context_, group));
        ASSERT_EQ(EC_OK, manager_->RegisterInstance(&context_, group_name, instance_id, "emb-test").first);
    }

    KvMetaManager::StartWriteResult Start(const std::string &instance_id,
                                          const std::vector<std::string> &keys,
                                          const std::vector<std::uint64_t> &sizes,
                                          std::int64_t timeout_seconds = 30) {
        auto [ec, result] = manager_->StartWrite(&context_, instance_id, keys, sizes, timeout_seconds);
        EXPECT_EQ(EC_OK, ec);
        return result;
    }

    void Commit(const std::string &instance_id, const std::string &key, std::uint64_t size) {
        auto start = Start(instance_id, {key}, {size});
        ASSERT_EQ(1u, start.locations.size());
        ASSERT_EQ(EC_OK, manager_->FinishWrite(&context_, instance_id, start.write_session_id, {true}));
    }

    std::shared_ptr<MetaIndexer> Indexer(const std::string &instance_id) {
        return cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
    }

    static bool WaitUntil(const std::function<bool()> &predicate, std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return predicate();
    }

    static constexpr const char *kDefaultInstance = "embedding-instance";
    static constexpr const char *kStorageName = "pace_test";
    RequestContext context_{"kv_meta_manager_test"};
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<TestPaceBackend> pace_backend_;
    std::unique_ptr<KvMetaManager> manager_;
};

TEST_F(KvMetaManagerTest, VariableSizesBecomeVisibleOnlyAfterFinish) {
    const std::vector<std::string> keys{"emb-a", "emb-b"};
    auto start = Start(kDefaultInstance, keys, {17, 33});
    ASSERT_EQ((std::vector<bool>{false, false}), start.key_mask);
    ASSERT_EQ(2u, start.locations.size());
    EXPECT_EQ(17u, start.locations[0].value_size);
    EXPECT_EQ(33u, start.locations[1].value_size);

    auto [before_ec, before] = manager_->Get(&context_, kDefaultInstance, keys);
    ASSERT_EQ(EC_OK, before_ec);
    ASSERT_EQ(2u, before.size());
    EXPECT_FALSE(before[0].found);
    EXPECT_FALSE(before[1].found);
    EXPECT_EQ(0u, Indexer(kDefaultInstance)->GetStorageUsage());

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&context_, kDefaultInstance, start.write_session_id, {true, true}));
    auto [after_ec, after] = manager_->Get(&context_, kDefaultInstance, keys);
    ASSERT_EQ(EC_OK, after_ec);
    ASSERT_EQ(2u, after.size());
    ASSERT_TRUE(after[0].found);
    ASSERT_TRUE(after[1].found);
    EXPECT_EQ(17u, after[0].location.value_size);
    EXPECT_EQ(33u, after[1].location.value_size);
    EXPECT_EQ(50u, Indexer(kDefaultInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, FailedFinishAbortsTheWholeSession) {
    auto start = Start(kDefaultInstance, {"a", "b"}, {11, 19});
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&context_, kDefaultInstance, start.write_session_id, {true, false}));

    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"a", "b"});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(2u, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    EXPECT_EQ(0u, Indexer(kDefaultInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, WriteInProgressHitAndSizeMismatchAreDistinct) {
    auto first = Start(kDefaultInstance, {"same"}, {23});
    EXPECT_EQ(EC_EXIST, manager_->StartWrite(&context_, kDefaultInstance, {"same"}, {23}, 30).first);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&context_, kDefaultInstance, first.write_session_id, {true}));

    auto [hit_ec, hit] = manager_->StartWrite(&context_, kDefaultInstance, {"same"}, {23}, 30);
    ASSERT_EQ(EC_OK, hit_ec);
    EXPECT_EQ((std::vector<bool>{true}), hit.key_mask);
    EXPECT_TRUE(hit.locations.empty());
    EXPECT_TRUE(hit.write_session_id.empty());
    EXPECT_EQ(EC_MISMATCH, manager_->StartWrite(&context_, kDefaultInstance, {"same"}, {24}, 30).first);
}

TEST_F(KvMetaManagerTest, ExistingInstanceCanReregisterAfterPolicyBecomesInvalid) {
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(&context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(group && group->cache_config() && group->cache_config()->reclaim_strategy());

    auto cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(cache_config->FromJsonString(group->cache_config()->ToJsonString()));
    auto strategy = std::make_shared<CacheReclaimStrategy>(*cache_config->reclaim_strategy());
    strategy->set_reclaim_policy(ReclaimPolicy::POLICY_TTL);
    cache_config->set_reclaim_strategy(strategy);
    InstanceGroup updated(*group);
    updated.set_cache_config(cache_config);
    updated.set_version(group->version() + 1);
    ASSERT_EQ(EC_OK, registry_manager_->UpdateInstanceGroup(&context_, updated, group->version()));

    // Registration is idempotent and must remain available to restarting
    // clients. Only a new allocation depends on the current write policy.
    EXPECT_EQ(EC_OK, manager_->RegisterInstance(&context_, "default", kDefaultInstance, "emb-test").first);
    EXPECT_EQ(EC_CONFIG_ERROR, manager_->StartWrite(&context_, kDefaultInstance, {"new"}, {1}, 30).first);
}

TEST_F(KvMetaManagerTest, MixedPaceTiersAreRejectedInsteadOfStrandingAFullTier) {
    constexpr const char *kSsdStorage = "pace_ssd_test";
    auto ssd_spec = std::make_shared<TairMemPoolStorageSpec>();
    ssd_spec->set_domain("test-pace-ssd");
    ssd_spec->set_media_type(kTairMemPoolMediaTypeSsd);
    auto ssd_backend =
        std::make_shared<TestPaceBackend>(metrics_registry_, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD);
    ASSERT_EQ(
        EC_OK,
        ssd_backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, kSsdStorage, ssd_spec),
                          context_.trace_id()));
    registry_manager_->data_storage_manager()->storage_map_[kSsdStorage] = std::move(ssd_backend);

    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(&context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(group);
    InstanceGroup updated(*group);
    updated.set_storage_candidates({kStorageName, kSsdStorage});
    updated.set_version(group->version() + 1);
    ASSERT_EQ(EC_OK, registry_manager_->UpdateInstanceGroup(&context_, updated, group->version()));

    EXPECT_EQ(EC_CONFIG_ERROR, manager_->StartWrite(&context_, kDefaultInstance, {"new"}, {1}, 30).first);
}

TEST_F(KvMetaManagerTest, RemoveIsIdempotentAndUpdatesActualBytes) {
    Commit(kDefaultInstance, "remove", 37);
    ASSERT_EQ(37u, Indexer(kDefaultInstance)->GetStorageUsage());
    EXPECT_EQ(EC_OK, manager_->Remove(&context_, kDefaultInstance, {"remove"}));
    EXPECT_EQ(0u, Indexer(kDefaultInstance)->GetStorageUsage());
    EXPECT_EQ(EC_OK, manager_->Remove(&context_, kDefaultInstance, {"remove"}));

    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"remove"});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, RemoveHonorsTheConfiguredReaderGracePeriod) {
    constexpr const char *kGroup = "remove-delay-group";
    constexpr const char *kInstance = "remove-delay-instance";
    CreateGroup(kGroup, kInstance, 100, 0.8, 100);
    Commit(kInstance, "remove", 17);
    const std::size_t deletes_before = pace_backend_->delete_count();

    const auto begin = std::chrono::steady_clock::now();
    EXPECT_EQ(EC_OK, manager_->Remove(&context_, kInstance, {"remove"}));
    const auto elapsed = std::chrono::steady_clock::now() - begin;

    EXPECT_GE(elapsed, std::chrono::milliseconds(80));
    EXPECT_GT(pace_backend_->delete_count(), deletes_before);
    EXPECT_EQ(0u, Indexer(kInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, RemoveDoesNotFreeAnActiveWrite) {
    auto start = Start(kDefaultInstance, {"active"}, {23});
    EXPECT_EQ(EC_EXIST, manager_->Remove(&context_, kDefaultInstance, {"active"}));
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&context_, kDefaultInstance, start.write_session_id, {true}));

    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"active"});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_EQ(23u, Indexer(kDefaultInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, GenerationComparisonIncludesCreationTime) {
    auto start = Start(kDefaultInstance, {"generation"}, {31});
    const auto internal_key = KvMetaManager::InternalKey("generation");
    const auto location_id = KvMetaManager::StableLocationId("generation");
    LocationsPerKey locations;
    const auto writing_result =
        Indexer(kDefaultInstance)->GetLocations(&context_, {internal_key}, {{location_id}}, locations);
    ASSERT_EQ(EC_OK, writing_result.ec);
    ASSERT_EQ(1u, locations.size());
    ASSERT_EQ(1u, locations[0].size());
    ASSERT_TRUE(locations[0][0]);
    const auto writing_generation = locations[0][0];
    ASSERT_EQ(CLS_WRITING, writing_generation->status());

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&context_, kDefaultInstance, start.write_session_id, {true}));
    locations.clear();
    const auto serving_result =
        Indexer(kDefaultInstance)->GetLocations(&context_, {internal_key}, {{location_id}}, locations);
    ASSERT_EQ(EC_OK, serving_result.ec);
    ASSERT_EQ(1u, locations.size());
    ASSERT_EQ(1u, locations[0].size());
    ASSERT_TRUE(locations[0][0]);
    const auto serving_generation = locations[0][0];
    EXPECT_EQ(CLS_SERVING, serving_generation->status());
    EXPECT_TRUE(KvMetaManager::SameGeneration(*writing_generation, *serving_generation));

    auto replacement = std::make_shared<CacheLocation>(*serving_generation);
    replacement->set_create_time(serving_generation->create_time() + 1);
    EXPECT_FALSE(KvMetaManager::SameGeneration(*serving_generation, *replacement));
}

TEST_F(KvMetaManagerTest, CapacityCheckIsSoftAndAllowsOneWriteToCrossTheLimit) {
    constexpr const char *kGroup = "soft-capacity-group";
    constexpr const char *kInstance = "soft-capacity-instance";
    CreateGroup(kGroup, kInstance, 100, 0.8);

    Commit(kInstance, "first", 99);
    // Current usage is below 100, so PutStart is admitted without reserving
    // the requested 4 bytes. The committed usage may therefore become 103.
    Commit(kInstance, "second", 4);
    EXPECT_EQ(103u, Indexer(kInstance)->GetStorageUsage());

    // The next selector snapshot sees the already-crossed limit and rejects.
    EXPECT_EQ(EC_NOSPC, manager_->StartWrite(&context_, kInstance, {"third"}, {1}, 30).first);
}

TEST_F(KvMetaManagerTest, LruReclaimerConvergesActualUsageBelowWatermark) {
    constexpr const char *kGroup = "reclaim-group";
    constexpr const char *kInstance = "reclaim-instance";
    CreateGroup(kGroup, kInstance, 100, 0.8);
    Commit(kInstance, "old", 45);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Commit(kInstance, "new", 45);
    ASSERT_EQ(90u, Indexer(kInstance)->GetStorageUsage());

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&context_, 10);
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&context_, 32));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&context_, 8));
    ASSERT_TRUE(manager_->ResumeMaintenance());

    ASSERT_TRUE(WaitUntil([&]() { return Indexer(kInstance)->GetStorageUsage() <= 80; }, std::chrono::seconds(3)));
    EXPECT_EQ(45u, Indexer(kInstance)->GetStorageUsage());

    auto [ec, values] = manager_->Get(&context_, kInstance, {"old", "new"});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(2u, values.size());
    EXPECT_EQ(1, static_cast<int>(values[0].found) + static_cast<int>(values[1].found));
}

TEST_F(KvMetaManagerTest, ReclaimerUsesTheSmallerStorageTierQuota) {
    constexpr const char *kGroup = "tier-quota-group";
    constexpr const char *kInstance = "tier-quota-instance";
    CreateGroup(kGroup, kInstance, 100, 0.8);
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(&context_, kGroup);
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(group);
    InstanceGroup updated(*group);
    updated.set_quota(InstanceGroupQuota(100, {QuotaConfig(50, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL)}));
    updated.set_version(group->version() + 1);
    ASSERT_EQ(EC_OK, registry_manager_->UpdateInstanceGroup(&context_, updated, group->version()));
    Commit(kInstance, "object", 45);

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&context_, 10);
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&context_, 32));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&context_, 8));
    ASSERT_TRUE(manager_->ResumeMaintenance());

    ASSERT_TRUE(WaitUntil([&]() { return Indexer(kInstance)->GetStorageUsage() <= 40; }, std::chrono::seconds(3)));
    EXPECT_EQ(0u, Indexer(kInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, ReclaimerHonorsPhysicalDeleteDelay) {
    constexpr const char *kGroup = "delayed-reclaim-group";
    constexpr const char *kInstance = "delayed-reclaim-instance";
    CreateGroup(kGroup, kInstance, 100, 0.8, 200);
    Commit(kInstance, "old", 45);
    Commit(kInstance, "new", 45);

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&context_, 10);
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&context_, 32));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&context_, 8));
    ASSERT_TRUE(manager_->ResumeMaintenance());

    ASSERT_TRUE(WaitUntil([&]() { return Indexer(kInstance)->GetStorageUsage() <= 80; }, std::chrono::seconds(3)));
    EXPECT_EQ(0u, pace_backend_->delete_count());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(0u, pace_backend_->delete_count());
    EXPECT_TRUE(WaitUntil([&]() { return pace_backend_->delete_count() > 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ExpiredSessionIsInvisibleAndCleaned) {
    auto start = Start(kDefaultInstance, {"expires"}, {29}, 1);
    ASSERT_FALSE(start.write_session_id.empty());
    ASSERT_TRUE(WaitUntil(
        [&]() {
            LocationsPerKey locations;
            const auto result = Indexer(kDefaultInstance)
                                    ->GetLocations(&context_,
                                                   {KvMetaManager::InternalKey("expires")},
                                                   {{KvMetaManager::StableLocationId("expires")}},
                                                   locations);
            return result.per_location_error_codes.size() == 1 && result.per_location_error_codes[0].size() == 1 &&
                   result.per_location_error_codes[0][0] == EC_NOENT;
        },
        std::chrono::seconds(3)));
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&context_, kDefaultInstance, start.write_session_id, {true}));
    EXPECT_EQ(0u, Indexer(kDefaultInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, RecoveryDropsIncompleteWritesAndRebuildsUsage) {
    Commit(kDefaultInstance, "committed", 31);
    auto incomplete = Start(kDefaultInstance, {"incomplete"}, {47});
    ASSERT_FALSE(incomplete.write_session_id.empty());
    Indexer(kDefaultInstance)->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, 999);

    manager_->DoCleanup();
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return manager_->recovery_complete_.load(std::memory_order_acquire); },
                          std::chrono::seconds(3)));
    EXPECT_EQ(31u, Indexer(kDefaultInstance)->GetStorageUsage());

    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"committed", "incomplete"});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(2u, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_FALSE(values[1].found);
}

TEST_F(KvMetaManagerTest, RejectsMalformedRequestsWithoutMutation) {
    EXPECT_EQ(EC_BADARGS, manager_->StartWrite(&context_, kDefaultInstance, {}, {}, 30).first);
    EXPECT_EQ(EC_DUPLICATE_ENTITY, manager_->StartWrite(&context_, kDefaultInstance, {"dup", "dup"}, {1, 1}, 30).first);
    EXPECT_EQ(EC_BADARGS, manager_->StartWrite(&context_, kDefaultInstance, {"key"}, {}, 30).first);
    EXPECT_EQ(EC_OUT_OF_LIMIT, manager_->StartWrite(&context_, kDefaultInstance, {"key"}, {0}, 30).first);
    EXPECT_EQ(EC_BADARGS, manager_->StartWrite(&context_, kDefaultInstance, {"key"}, {1}, 0).first);
    EXPECT_EQ(0u, Indexer(kDefaultInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, RejectsAReclaimThresholdWithoutHeadroom) {
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(&context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(group && group->cache_config() && group->cache_config()->reclaim_strategy());

    auto cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(cache_config->FromJsonString(group->cache_config()->ToJsonString()));
    auto strategy = std::make_shared<CacheReclaimStrategy>(*cache_config->reclaim_strategy());
    TriggerStrategy trigger = strategy->trigger_strategy();
    trigger.set_used_percentage(1.0);
    strategy->set_trigger_strategy(trigger);
    cache_config->set_reclaim_strategy(strategy);
    InstanceGroup updated(*group);
    updated.set_cache_config(cache_config);
    updated.set_version(group->version() + 1);
    ASSERT_EQ(EC_OK, registry_manager_->UpdateInstanceGroup(&context_, updated, group->version()));

    EXPECT_EQ(EC_CONFIG_ERROR, manager_->StartWrite(&context_, kDefaultInstance, {"no-headroom"}, {1}, 30).first);
}

} // namespace kv_cache_manager
