#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/common/timestamp_util.h"
#ifdef KVCM_REAL_REDIS_TEST
#include "kv_cache_manager/meta/meta_redis_backend.h"
#endif
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
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
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

class FailingSyncBackend : public MetaLocalBackend {
public:
    bool Sync(const KeyTypeVec &) noexcept override { return false; }
};

class RetryDeleteBackend : public MetaLocalBackend {
public:
    std::vector<ErrorCode> Delete(RequestContext *context, const KeyTypeVec &keys) noexcept override {
        if (delete_attempts.fetch_add(1) == 0) {
            return std::vector<ErrorCode>(keys.size(), EC_ERROR);
        }
        return MetaLocalBackend::Delete(context, keys);
    }

    std::atomic<int> delete_attempts{0};
};

class PartialCommitBackend : public MetaLocalBackend {
public:
    explicit PartialCommitBackend(KeyType failing_key) : failing_key_(failing_key) {}

    std::vector<ErrorCode> Upsert(RequestContext *context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override {
        if (keys.size() != locations.size() || keys.size() != properties.size()) {
            return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
        }
        std::vector<ErrorCode> results;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            bool serving = false;
            for (const auto &[_, location] : locations[i]) {
                serving = serving || (location && location->status() == CLS_SERVING);
            }
            if (serving && keys[i] == failing_key_) {
                if (before_failed_commit) {
                    before_failed_commit();
                }
                results.push_back(EC_IO_ERROR);
                continue;
            }
            const auto result = MetaLocalBackend::Upsert(context, {keys[i]}, {locations[i]}, {properties[i]});
            results.push_back(result.size() == 1 ? result[0] : EC_MISMATCH);
        }
        return results;
    }

    std::function<void()> before_failed_commit;

private:
    KeyType failing_key_;
};

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
        ASSERT_EQ(EC_OK, registry_manager_->DoRecover());

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
    EXPECT_EQ(2u, pace_backend_->delete_count());
}

TEST_F(KvMetaManagerTest, PartialFinishRollbackHonorsReadersOfCommittedKeys) {
    constexpr const char *kGroup = "partial-finish-delay";
    constexpr const char *kInstance = "partial-finish-delay-instance";
    CreateGroup(kGroup, kInstance, 100, 0.8, 100);
    ASSERT_EQ(nullptr, Indexer(kInstance)->backend_manager_->cache_backend_);
    auto indexer = Indexer(kInstance);
    indexer->batch_key_size_ = 1;
    std::string readable = "readable";
    std::string failed = "failed";
    ASSERT_NE(indexer->GetMutexShardIndex(KvMetaManager::InternalKey(readable)),
              indexer->GetMutexShardIndex(KvMetaManager::InternalKey(failed)));
    if (indexer->GetMutexShardIndex(KvMetaManager::InternalKey(readable)) >
        indexer->GetMutexShardIndex(KvMetaManager::InternalKey(failed))) {
        std::swap(readable, failed);
    }
    // MakeBatches commits the lower shard first. Inject the second failure
    // only after releasing the readable key's shard, so public Get can do
    // its generation-fenced LRU update without a re-entrant lock.
    auto backend = std::make_unique<PartialCommitBackend>(KvMetaManager::InternalKey(failed));
    ASSERT_EQ(EC_OK, backend->Init(kInstance, std::make_shared<MetaStorageBackendConfig>()));
    ASSERT_EQ(EC_OK, backend->Open());
    auto *partial_backend = backend.get();
    Indexer(kInstance)->backend_manager_->persistent_backend_ = std::move(backend);
    auto start = Start(kInstance, {readable, failed}, {17, 29});
    ASSERT_EQ(2u, start.locations.size());

    bool reader_observed_serving = false;
    std::chrono::steady_clock::time_point read_time;
    partial_backend->before_failed_commit = [&]() {
        RequestContext reader_context("partial_finish_reader");
        auto [ec, values] = manager_->Get(&reader_context, kInstance, {readable});
        EXPECT_EQ(EC_OK, ec);
        if (ec == EC_OK && values.size() == 1 && values[0].found) {
            reader_observed_serving = true;
            read_time = std::chrono::steady_clock::now();
            EXPECT_EQ(start.locations[0].specs, values[0].location.specs);
            EXPECT_EQ(0u, pace_backend_->delete_count());
        }
    };

    EXPECT_NE(EC_OK, manager_->FinishWrite(&context_, kInstance, start.write_session_id, {true, true}));
    ASSERT_TRUE(reader_observed_serving);
    EXPECT_GE(std::chrono::steady_clock::now() - read_time, std::chrono::milliseconds(80));
    EXPECT_EQ(2u, pace_backend_->delete_count());
    EXPECT_EQ(0u, Indexer(kInstance)->GetStorageUsage());
    auto [ec, values] = manager_->Get(&context_, kInstance, {readable, failed});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(2u, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
}

TEST_F(KvMetaManagerTest, CancelledPartialFinishRollbackRetainsReadersMemory) {
    constexpr const char *kGroup = "partial-finish-cancel";
    constexpr const char *kInstance = "partial-finish-cancel-instance";
    CreateGroup(kGroup, kInstance, 100, 0.8, 30000);
    ASSERT_EQ(nullptr, Indexer(kInstance)->backend_manager_->cache_backend_);
    auto indexer = Indexer(kInstance);
    indexer->batch_key_size_ = 1;
    std::string readable = "readable";
    std::string failed = "failed";
    ASSERT_NE(indexer->GetMutexShardIndex(KvMetaManager::InternalKey(readable)),
              indexer->GetMutexShardIndex(KvMetaManager::InternalKey(failed)));
    if (indexer->GetMutexShardIndex(KvMetaManager::InternalKey(readable)) >
        indexer->GetMutexShardIndex(KvMetaManager::InternalKey(failed))) {
        std::swap(readable, failed);
    }
    // MakeBatches commits the lower shard first. Inject the second failure
    // only after releasing the readable key's shard, so public Get can do
    // its generation-fenced LRU update without a re-entrant lock.
    auto backend = std::make_unique<PartialCommitBackend>(KvMetaManager::InternalKey(failed));
    ASSERT_EQ(EC_OK, backend->Init(kInstance, std::make_shared<MetaStorageBackendConfig>()));
    ASSERT_EQ(EC_OK, backend->Open());
    auto *partial_backend = backend.get();
    Indexer(kInstance)->backend_manager_->persistent_backend_ = std::move(backend);
    auto start = Start(kInstance, {readable, failed}, {17, 29});
    ASSERT_EQ(2u, start.locations.size());

    bool reader_observed_serving = false;
    partial_backend->before_failed_commit = [&]() {
        RequestContext reader_context("cancelled_partial_finish_reader");
        auto [ec, values] = manager_->Get(&reader_context, kInstance, {readable});
        EXPECT_EQ(EC_OK, ec);
        reader_observed_serving = ec == EC_OK && values.size() == 1 && values[0].found;
        manager_->CancelMaintenance();
    };

    const auto begin = std::chrono::steady_clock::now();
    EXPECT_NE(EC_OK, manager_->FinishWrite(&context_, kInstance, start.write_session_id, {true, true}));
    EXPECT_TRUE(reader_observed_serving);
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(2));
    EXPECT_EQ(0u, pace_backend_->delete_count());
    EXPECT_EQ(0u, Indexer(kInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, MissingIndexerDoesNotAuthorizePhysicalDeletion) {
    auto start = Start(kDefaultInstance, {"object"}, {31});
    cache_manager_->meta_indexer_manager()->DoCleanup();
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST,
              manager_->FinishWrite(&context_, kDefaultInstance, start.write_session_id, {true}));
    EXPECT_EQ(0u, pace_backend_->delete_count());
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

TEST_F(KvMetaManagerTest, StoragePreferenceMustMatchTheSinglePaceTier) {
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(&context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(group && group->cache_config());

    auto cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(cache_config->FromJsonString(group->cache_config()->ToJsonString()));
    cache_config->set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_3FS);
    InstanceGroup updated(*group);
    updated.set_cache_config(cache_config);
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

TEST_F(KvMetaManagerTest, ExpiredSessionRetriesCleanupWithoutReleasingAllocation) {
    auto backend = std::make_unique<RetryDeleteBackend>();
    ASSERT_EQ(EC_OK, backend->Init("expiry-retry", std::make_shared<MetaStorageBackendConfig>()));
    ASSERT_EQ(EC_OK, backend->Open());
    auto *retry_backend = backend.get();
    Indexer(kDefaultInstance)->backend_manager_->persistent_backend_ = std::move(backend);
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
        std::chrono::seconds(5)));
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&context_, kDefaultInstance, start.write_session_id, {true}));
    EXPECT_GE(retry_backend->delete_attempts.load(), 2);
    EXPECT_EQ(0u, Indexer(kDefaultInstance)->GetStorageUsage());
    // The SDK timeout is advisory; the old writer may still hold its address.
    EXPECT_EQ(0u, pace_backend_->delete_count());
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
    EXPECT_EQ(0u, pace_backend_->delete_count());

    auto replacement = Start(kDefaultInstance, {"incomplete"}, {47});
    ASSERT_EQ(1u, replacement.locations.size());
    EXPECT_NE(incomplete.locations[0].specs, replacement.locations[0].specs);
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&context_, kDefaultInstance, incomplete.write_session_id, {false}));
    EXPECT_EQ(0u, pace_backend_->delete_count());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&context_, kDefaultInstance, replacement.write_session_id, {true}));
}

TEST_F(KvMetaManagerTest, RecoveryWaitsForTheCompleteRegistry) {
    Commit(kDefaultInstance, "committed", 31);
    Indexer(kDefaultInstance)->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, 999);
    registry_manager_->recover_complete_.store(false);

    EXPECT_EQ(EC_CONFIG_ERROR, manager_->DoRecover());
    EXPECT_EQ(999u, Indexer(kDefaultInstance)->GetStorageUsage());
    EXPECT_EQ(EC_CONFIG_ERROR, manager_->Get(&context_, kDefaultInstance, {"committed"}).first);

    registry_manager_->recover_complete_.store(true);
    EXPECT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(31u, Indexer(kDefaultInstance)->GetStorageUsage());
    EXPECT_EQ(EC_OK, manager_->Get(&context_, kDefaultInstance, {"committed"}).first);
}

TEST_F(KvMetaManagerTest, FailedDeleteConfirmationNeverReleasesTheAllocation) {
    auto backend = std::make_unique<FailingSyncBackend>();
    ASSERT_EQ(EC_OK, backend->Init("delete-failure", std::make_shared<MetaStorageBackendConfig>()));
    ASSERT_EQ(EC_OK, backend->Open());
    Indexer(kDefaultInstance)->backend_manager_->persistent_backend_ = std::move(backend);
    Commit(kDefaultInstance, "object", 31);

    EXPECT_EQ(EC_ERROR, manager_->Remove(&context_, kDefaultInstance, {"object"}));
    EXPECT_EQ(0u, pace_backend_->delete_count());
    EXPECT_EQ(0u, Indexer(kDefaultInstance)->GetStorageUsage());
    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"object"});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, CancelInterruptsRemoveGraceWithoutReleasingReadersMemory) {
    CreateGroup("cancel-remove", "cancel-remove-instance", 100, 0.8, 30000);
    Commit("cancel-remove-instance", "object", 31);
    auto removed = std::async(std::launch::async, [&]() {
        RequestContext context("cancel_remove");
        return manager_->Remove(&context, "cancel-remove-instance", {"object"});
    });
    ASSERT_TRUE(WaitUntil([&]() { return Indexer("cancel-remove-instance")->GetStorageUsage() == 0; },
                          std::chrono::seconds(2)));

    manager_->CancelMaintenance();
    ASSERT_EQ(std::future_status::ready, removed.wait_for(std::chrono::seconds(2)));
    EXPECT_EQ(EC_ERROR, removed.get());
    EXPECT_EQ(0u, pace_backend_->delete_count());
}

TEST_F(KvMetaManagerTest, CancelInterruptsReclaimerGrace) {
    CreateGroup("cancel-reclaim", "cancel-reclaim-instance", 100, 0.8, 30000);
    Commit("cancel-reclaim-instance", "object", 90);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&context_, 10);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return Indexer("cancel-reclaim-instance")->GetStorageUsage() == 0; },
                          std::chrono::seconds(2)));

    auto stopped = std::async(std::launch::async, [&]() { manager_->DoCleanup(); });
    ASSERT_EQ(std::future_status::ready, stopped.wait_for(std::chrono::seconds(2)));
    stopped.get();
    EXPECT_EQ(0u, pace_backend_->delete_count());
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

#ifdef KVCM_REAL_REDIS_TEST
// Dedicated Redis endpoint; unique per-test namespaces, never FLUSHDB.
class HookRedisBackend : public MetaRedisBackend {
public:
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *context,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &ids,
                                                     LocationsPerKey &out) noexcept override {
        auto result = MetaRedisBackend::GetLocations(context, keys, ids, out);
        auto hook = std::move(after_read);
        after_read = nullptr;
        if (hook) {
            hook();
        }
        return result;
    }
    std::vector<ErrorCode> Upsert(RequestContext *context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override {
        if (fail_upsert) {
            return std::vector<ErrorCode>(keys.size(), EC_IO_ERROR);
        }
        return MetaRedisBackend::Upsert(context, keys, locations, properties);
    }
    ErrorCode RandomSample(RequestContext *context, int64_t count, KeyTypeVec &out) noexcept override {
        if (!fixed_sample.empty()) {
            // The ordering regression controls candidate coverage, while all
            // location/property reads and writes still use the real Redis.
            out = fixed_sample;
            return EC_OK;
        }
        return MetaRedisBackend::RandomSample(context, count, out);
    }
    std::function<void()> after_read;
    KeyTypeVec fixed_sample;
    bool fail_upsert = false;
};

class KvMetaManagerRedisTest : public KvMetaManagerTest {
protected:
    void SetUp() override {
        KvMetaManagerTest::SetUp();
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        const char *uri = std::getenv("KVCM_TEST_KVMETA_REDIS_URI");
        ASSERT_NE(nullptr, uri) << "Use a dedicated test Redis, never application Redis";
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
        config->SetStorageUri(uri);
        auto backend = std::make_unique<HookRedisBackend>();
        const std::string name = "kvmeta-lru-test-" + std::to_string(TimestampUtil::GetCurrentTimeUs()) + "-" +
                                 ::testing::UnitTest::GetInstance()->current_test_info()->name();
        ASSERT_EQ(EC_OK, backend->Init(name, config));
        ASSERT_EQ(EC_OK, backend->Open());
        redis_ = backend.get();
        ASSERT_EQ(nullptr, Indexer(kDefaultInstance)->backend_manager_->cache_backend_);
        Indexer(kDefaultInstance)->backend_manager_->persistent_backend_ = std::move(backend);
    }
    std::int64_t Lru(const std::string &key) {
        PropertyMapVector properties;
        const auto ecs = redis_->GetProperties(
            &context_, {KvMetaManager::InternalKey(key)}, {PROPERTY_LRU_TIME}, properties);
        EXPECT_EQ(1u, ecs.size());
        EXPECT_EQ(1u, properties.size());
        if (properties.size() != 1 || properties[0].count(PROPERTY_LRU_TIME) == 0) {
            return -1;
        }
        return std::stoll(properties[0].at(PROPERTY_LRU_TIME));
    }
    void SetLru(const std::string &key, std::int64_t value) {
        ASSERT_EQ(std::vector<ErrorCode>{EC_OK},
                  redis_->Upsert(&context_, {KvMetaManager::InternalKey(key)}, {{}},
                                 {{{PROPERTY_LRU_TIME, std::to_string(value)}}}));
    }
    void ExpectAbsent(const std::string &key) {
        std::vector<bool> exists;
        ASSERT_EQ(std::vector<ErrorCode>{EC_OK},
                  redis_->Exists(&context_, {KvMetaManager::InternalKey(key)}, exists));
        EXPECT_EQ(std::vector<bool>{false}, exists);
    }
    HookRedisBackend *redis_ = nullptr;
};

TEST_F(KvMetaManagerRedisTest, CommitInitializesPersistentLruAndHitAdvancesIt) {
    const auto before_commit = TimestampUtil::GetCurrentTimeUs();
    Commit(kDefaultInstance, "object", 17);
    const auto committed = Lru("object");
    EXPECT_GE(committed, before_commit);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"object"});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_GT(Lru("object"), committed);
    EXPECT_EQ(17u, Indexer(kDefaultInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerRedisTest, MissWritingAndMaintenanceDoNotTouch) {
    auto writing = Start(kDefaultInstance, {"writing"}, {17});
    Commit(kDefaultInstance, "serving", 23);
    SetLru("serving", 123);
    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"absent", "writing"});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(2u, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    ExpectAbsent("absent");
    EXPECT_EQ(-1, Lru("writing"));
    LocationsPerKey locations;
    const auto read = Indexer(kDefaultInstance)->backend_manager_->GetLocationsForMaintenance(
        &context_, {KvMetaManager::InternalKey("serving")}, {{KvMetaManager::StableLocationId("serving")}}, locations);
    ASSERT_EQ((std::vector<std::vector<ErrorCode>>{{EC_OK}}), read);
    redis_->fixed_sample = {KvMetaManager::InternalKey("serving"), KvMetaManager::InternalKey("writing")};
    ReclaimCandidateVector candidates;
    ASSERT_EQ(EC_OK, Indexer(kDefaultInstance)->SamplePersistentReclaimCandidates(&context_, 32, candidates, true));
    ASSERT_FALSE(candidates.empty());
    EXPECT_EQ(123, Lru("serving"));
}

TEST_F(KvMetaManagerRedisTest, FailedBatchDoesNotTouchEarlierValidHit) {
    Commit(kDefaultInstance, "valid", 17);
    Commit(kDefaultInstance, "corrupt", 23);
    SetLru("valid", 123);
    LocationsPerKey locations;
    ASSERT_EQ((std::vector<std::vector<ErrorCode>>{{EC_OK}}),
              redis_->GetLocations(&context_, {KvMetaManager::InternalKey("corrupt")},
                                   {{KvMetaManager::StableLocationId("corrupt")}}, locations));
    auto corrupt = std::make_shared<CacheLocation>(*locations[0][0]);
    corrupt->set_spec_size(2);
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK},
              redis_->Upsert(&context_, {KvMetaManager::InternalKey("corrupt")},
                             {{{KvMetaManager::StableLocationId("corrupt"), corrupt}}}, {{}}));
    EXPECT_NE(EC_OK, manager_->Get(&context_, kDefaultInstance, {"valid", "corrupt"}).first);
    EXPECT_EQ(123, Lru("valid"));
}

TEST_F(KvMetaManagerRedisTest, DelayedHitCannotResurrectDeletedMetadata) {
    Commit(kDefaultInstance, "object", 17);
    std::promise<void> read_done, release;
    auto released = release.get_future();
    redis_->after_read = [&]() { read_done.set_value(); released.wait(); };
    auto reader = std::async(std::launch::async, [&]() {
        RequestContext context("redis_delayed_reader");
        return manager_->Get(&context, kDefaultInstance, {"object"});
    });
    read_done.get_future().wait();
    const auto remove_ec = manager_->Remove(&context_, kDefaultInstance, {"object"});
    release.set_value();
    const auto [ec, values] = reader.get();
    ASSERT_EQ(EC_OK, remove_ec);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, values.size());
    EXPECT_TRUE(values[0].found);
    ExpectAbsent("object");
    EXPECT_EQ(0u, Indexer(kDefaultInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerRedisTest, DelayedHitCannotTouchReplacementGeneration) {
    Commit(kDefaultInstance, "object", 17);
    std::promise<void> read_done, release;
    auto released = release.get_future();
    redis_->after_read = [&]() { read_done.set_value(); released.wait(); };
    auto reader = std::async(std::launch::async, [&]() {
        RequestContext context("redis_replaced_reader");
        return manager_->Get(&context, kDefaultInstance, {"object"});
    });
    read_done.get_future().wait();
    const auto remove_ec = manager_->Remove(&context_, kDefaultInstance, {"object"});
    Commit(kDefaultInstance, "object", 29);
    SetLru("object", 123);
    release.set_value();
    const auto [ec, values] = reader.get();
    ASSERT_EQ(EC_OK, remove_ec);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_EQ(17u, values[0].location.value_size);
    EXPECT_EQ(123, Lru("object"));
    EXPECT_EQ(29u, Indexer(kDefaultInstance)->GetStorageUsage());
}

TEST_F(KvMetaManagerRedisTest, AuxiliaryLruWriteFailurePreservesHit) {
    Commit(kDefaultInstance, "object", 17);
    SetLru("object", 123);
    redis_->fail_upsert = true;
    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"object"});
    redis_->fail_upsert = false;
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_EQ(123, Lru("object"));
}

TEST_F(KvMetaManagerRedisTest, RealRedisReclaimerEvictsOldestUntouchedObject) {
    Commit(kDefaultInstance, "a", 30);
    Commit(kDefaultInstance, "b", 30);
    Commit(kDefaultInstance, "c", 30);
    SetLru("a", 100);
    SetLru("b", 200);
    SetLru("c", 300);
    redis_->fixed_sample = {KvMetaManager::InternalKey("a"), KvMetaManager::InternalKey("b"),
                            KvMetaManager::InternalKey("c")};
    ASSERT_EQ(EC_OK, manager_->Get(&context_, kDefaultInstance, {"a"}).first);
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(&context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    InstanceGroup updated(*group);
    auto config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(config->FromJsonString(group->cache_config()->ToJsonString()));
    auto strategy = std::make_shared<CacheReclaimStrategy>(*config->reclaim_strategy());
    TriggerStrategy trigger = strategy->trigger_strategy();
    trigger.set_used_percentage(0.8);
    strategy->set_trigger_strategy(trigger);
    strategy->set_reclaim_policy(ReclaimPolicy::POLICY_LRU);
    strategy->set_delay_before_delete_ms(0);
    config->set_reclaim_strategy(strategy);
    updated.set_cache_config(config);
    updated.set_quota(InstanceGroupQuota(80, {QuotaConfig(80, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL)}));
    updated.set_version(group->version() + 1);
    ASSERT_EQ(EC_OK, registry_manager_->UpdateInstanceGroup(&context_, updated, group->version()));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&context_, 10);
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&context_, 32));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&context_, 8));
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return Indexer(kDefaultInstance)->GetStorageUsage() == 60; }, std::chrono::seconds(3)));
    auto [ec, values] = manager_->Get(&context_, kDefaultInstance, {"a", "b", "c"});
    manager_->CancelMaintenance();
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(3u, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_FALSE(values[1].found);
    EXPECT_TRUE(values[2].found);
    EXPECT_EQ(1u, pace_backend_->delete_count());
}
#endif

} // namespace kv_cache_manager
