#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/config/quota_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/kv_meta_instance.h"
#include "kv_cache_manager/manager/kv_meta_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

class OverlongCreateNfsBackend : public NfsBackend {
public:
    explicit OverlongCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        std::vector<std::pair<ErrorCode, DataStorageUri>> result;
        for (const char *path : {"/malformed/first", "/malformed/second"}) {
            DataStorageUri uri;
            uri.SetProtocol("file");
            uri.SetPath(path);
            uri.SetParam("size", std::to_string(size_per_key));
            result.emplace_back(EC_OK, std::move(uri));
        }
        DataStorageUri foreign_uri;
        foreign_uri.SetProtocol("dummy");
        foreign_uri.SetPath("/must-not-delete-through-nfs");
        foreign_uri.SetParam("size", std::to_string(size_per_key));
        result.emplace_back(EC_OK, std::move(foreign_uri));
        if (cb) {
            cb();
        }
        return result;
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        deleted_uris.insert(deleted_uris.end(), storage_uris.begin(), storage_uris.end());
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<DataStorageUri> deleted_uris;
};

class NonSingletonCreateNfsBackend : public NfsBackend {
public:
    explicit NonSingletonCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        DataStorageUri uri;
        uri.SetProtocol("file");
        uri.SetPath("/malformed/shared-file");
        uri.SetParam("size", std::to_string(size_per_key));
        uri.SetParam("blkid", "1");
        if (cb) {
            cb();
        }
        return {{EC_OK, std::move(uri)}};
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        delete_calls += storage_uris.size();
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::size_t delete_calls{0};
};

class DuplicateSingletonCreateNfsBackend : public NfsBackend {
public:
    explicit DuplicateSingletonCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        DataStorageUri uri;
        uri.SetProtocol("file");
        uri.SetPath("/malformed/reused-singleton");
        uri.SetParam("blkid", "0");
        uri.SetParam("size", std::to_string(size_per_key));
        if (cb) {
            cb();
        }
        return {{EC_OK, std::move(uri)}};
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        delete_calls += storage_uris.size();
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::size_t delete_calls{0};
};

class BlockingDeleteNfsBackend : public NfsBackend {
public:
    explicit BlockingDeleteNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        std::unique_lock<std::mutex> lock(mutex_);
        delete_entered_ = true;
        condition_.notify_all();
        condition_.wait(lock, [&]() { return release_delete_; });
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    bool WaitForDelete(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return delete_entered_; });
    }

    void ReleaseDelete() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_delete_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool delete_entered_{false};
    bool release_delete_{false};
};

class BlockingThenFailDeleteNfsBackend : public NfsBackend {
public:
    explicit BlockingThenFailDeleteNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        bool fail = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ++delete_attempts_;
            fail = delete_attempts_ > 1;
            if (!fail) {
                first_delete_entered_ = true;
                condition_.notify_all();
                condition_.wait(lock, [&]() { return release_first_delete_; });
            }
        }
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), fail ? EC_IO_ERROR : EC_OK);
    }

    bool WaitForFirstDelete(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return first_delete_entered_; });
    }

    void ReleaseFirstDelete() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_first_delete_ = true;
        condition_.notify_all();
    }

    std::size_t DeleteAttempts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return delete_attempts_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool first_delete_entered_{false};
    bool release_first_delete_{false};
    std::size_t delete_attempts_{0};
};

class FaultingDeleteNfsBackend : public NfsBackend {
public:
    enum class Mode {
        kError,
        kShortResult,
        kStandardException,
        kUnknownException,
    };

    FaultingDeleteNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry, Mode mode)
        : NfsBackend(std::move(metrics_registry)), mode_(mode) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++delete_attempts_;
            condition_.notify_all();
        }
        if (cb) {
            cb();
        }
        switch (mode_) {
        case Mode::kError:
            return std::vector<ErrorCode>(storage_uris.size(), EC_IO_ERROR);
        case Mode::kShortResult:
            return std::vector<ErrorCode>(storage_uris.empty() ? 0 : storage_uris.size() - 1, EC_OK);
        case Mode::kStandardException:
            throw std::runtime_error("injected secret provider detail");
        case Mode::kUnknownException:
            throw 17;
        }
        throw std::runtime_error("unreachable fault mode");
    }

    bool WaitForDeleteAttempts(std::size_t expected, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return delete_attempts_ >= expected; });
    }

    std::size_t DeleteAttempts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return delete_attempts_;
    }

private:
    Mode mode_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t delete_attempts_{0};
};

class ThrowingSecondCreateNfsBackend : public NfsBackend {
public:
    explicit ThrowingSecondCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        ++create_attempts_;
        if (create_attempts_ == 2) {
            throw std::runtime_error("injected secret create detail");
        }
        return NfsBackend::Create(keys, size_per_key, trace_id, std::move(cb));
    }

    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override {
        delete_items_ += storage_uris.size();
        return NfsBackend::Delete(storage_uris, trace_id, std::move(cb));
    }

    std::size_t CreateAttempts() const { return create_attempts_; }
    std::size_t DeleteItems() const { return delete_items_; }

private:
    std::size_t create_attempts_{0};
    std::size_t delete_items_{0};
};

class FailNextMaintenanceDeleteSyncBackend : public MetaLocalBackend {
public:
    void FailNextMaintenanceDeleteSync() {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_next_delete_sync_ = true;
    }

    std::vector<ErrorCode> DeleteLocationsForMaintenance(RequestContext *request_context,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdsPerKey &location_ids) noexcept override {
        auto result = MetaLocalBackend::DeleteLocationsForMaintenance(request_context, keys, location_ids);
        ArmFailureAfterSuccessfulDelete(result);
        return result;
    }

    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept override {
        auto result = MetaLocalBackend::Delete(request_context, keys);
        ArmFailureAfterSuccessfulDelete(result);
        return result;
    }

    bool Sync(const KeyTypeVec &keys) noexcept override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (fail_next_sync_) {
                fail_next_sync_ = false;
                sync_failed_ = true;
                condition_.notify_all();
                return false;
            }
        }
        return MetaLocalBackend::Sync(keys);
    }

    bool WaitForSyncFailure(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return sync_failed_; });
    }

private:
    void ArmFailureAfterSuccessfulDelete(const std::vector<ErrorCode> &result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fail_next_delete_sync_ &&
            std::any_of(result.begin(), result.end(), [](const ErrorCode ec) { return ec == EC_OK; })) {
            fail_next_delete_sync_ = false;
            fail_next_sync_ = true;
        }
    }
    std::mutex mutex_;
    std::condition_variable condition_;
    bool fail_next_delete_sync_{false};
    bool fail_next_sync_{false};
    bool sync_failed_{false};
};

class OversamplingMetaLocalBackend : public MetaLocalBackend {
public:
    ErrorCode SampleReclaimCandidates(RequestContext *request_context,
                                      int64_t count,
                                      ReclaimCandidateVector &out_candidates,
                                      bool require_read_success) noexcept override {
        const auto ec =
            MetaLocalBackend::SampleReclaimCandidates(request_context, count, out_candidates, require_read_success);
        if (ec == EC_OK && !out_candidates.empty()) {
            out_candidates.push_back(out_candidates.front());
        }
        return ec;
    }
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

    void CreateReclaimGroup(const std::string &group_name,
                            const std::string &instance_id,
                            std::int64_t capacity,
                            double threshold,
                            std::int32_t delay_before_delete_ms,
                            std::size_t max_key_count = MetaIndexerConfig::kDefaultMaxKeyCount,
                            std::optional<std::int64_t> storage_type_capacity = std::nullopt,
                            ReclaimPolicy reclaim_policy = ReclaimPolicy::POLICY_LRU,
                            ErrorCode expected_registration_ec = EC_OK) {
        const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
        ASSERT_EQ(EC_OK, group_ec);
        ASSERT_TRUE(default_group);
        ASSERT_TRUE(default_group->cache_config());
        ASSERT_TRUE(default_group->cache_config()->reclaim_strategy());

        auto cache_config = std::make_shared<CacheConfig>();
        ASSERT_TRUE(cache_config->FromJsonString(default_group->cache_config()->ToJsonString()));
        auto reclaim_strategy = std::make_shared<CacheReclaimStrategy>(*cache_config->reclaim_strategy());
        TriggerStrategy trigger = reclaim_strategy->trigger_strategy();
        trigger.set_used_percentage(threshold);
        reclaim_strategy->set_trigger_strategy(trigger);
        reclaim_strategy->set_delay_before_delete_ms(delay_before_delete_ms);
        reclaim_strategy->set_reclaim_policy(reclaim_policy);
        cache_config->set_reclaim_strategy(reclaim_strategy);
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>(*cache_config->meta_indexer_config());
        meta_indexer_config->SetMaxKeyCount(max_key_count);
        if (max_key_count != MetaIndexerConfig::kDefaultMaxKeyCount) {
            std::size_t mutex_shard_num = 1;
            while (mutex_shard_num <= max_key_count / 2) {
                mutex_shard_num *= 2;
            }
            meta_indexer_config->SetMutexShardNum(mutex_shard_num);
        }
        cache_config->set_meta_indexer_config(meta_indexer_config);

        InstanceGroup object_group(*default_group);
        object_group.set_name(group_name);
        object_group.set_global_quota_group_name(group_name + "-quota");
        object_group.set_version(1);
        object_group.set_cache_config(cache_config);
        object_group.set_quota(InstanceGroupQuota(
            capacity, {QuotaConfig(storage_type_capacity.value_or(capacity), DataStorageType::DATA_STORAGE_TYPE_NFS)}));
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));
        ASSERT_EQ(expected_registration_ec,
                  manager_->RegisterInstance(&request_context_, group_name, instance_id, "reclaim-test").first);
    }

    FailNextMaintenanceDeleteSyncBackend *InstallFailingSyncBackend(const std::string &instance_id) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        if (!indexer || !indexer->backend_manager_) {
            return nullptr;
        }
        auto config = std::make_shared<MetaStorageBackendConfig>();
        auto backend = std::make_unique<FailNextMaintenanceDeleteSyncBackend>();
        if (backend->Init(KvMetaManager::InternalInstanceId(instance_id), config) != EC_OK ||
            backend->Open() != EC_OK) {
            return nullptr;
        }
        auto *backend_raw = backend.get();
        if (indexer->backend_manager_->persistent_backend_) {
            indexer->backend_manager_->persistent_backend_->Close();
        }
        indexer->backend_manager_->persistent_backend_ = std::move(backend);
        indexer->backend_manager_->cache_backend_.reset();
        return backend_raw;
    }

    OversamplingMetaLocalBackend *InstallOversamplingBackend(const std::string &instance_id) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        if (!indexer || !indexer->backend_manager_) {
            return nullptr;
        }
        auto config = std::make_shared<MetaStorageBackendConfig>();
        auto backend = std::make_unique<OversamplingMetaLocalBackend>();
        if (backend->Init(KvMetaManager::InternalInstanceId(instance_id), config) != EC_OK ||
            backend->Open() != EC_OK) {
            return nullptr;
        }
        auto *backend_raw = backend.get();
        if (indexer->backend_manager_->persistent_backend_) {
            indexer->backend_manager_->persistent_backend_->Close();
        }
        indexer->backend_manager_->persistent_backend_ = std::move(backend);
        indexer->backend_manager_->cache_backend_.reset();
        return backend_raw;
    }

    void CommitObject(const std::string &instance_id, const std::string &key, std::uint64_t size) {
        auto [start_ec, start] = manager_->StartWrite(&request_context_, instance_id, {key}, {size}, 30);
        ASSERT_EQ(EC_OK, start_ec);
        ASSERT_EQ(1, start.locations.size());
        ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, instance_id, start.write_session_id, {true}));
    }

    void MutateObject(const std::string &instance_id,
                      const std::string &key,
                      const std::function<void(CacheLocation &)> &mutation) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        ASSERT_TRUE(indexer);
        const auto internal_key = KvMetaManager::InternalKey(key);
        const auto location_id = KvMetaManager::StableLocationId(key);
        auto modifier = [&mutation](const std::vector<ErrorCode> &get_ecs,
                                    const LocationIdVector &,
                                    std::size_t,
                                    CacheLocationVector &locations,
                                    PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || get_ecs[0] != EC_OK || locations.size() != 1 || !locations[0]) {
                return {MA_FAIL, {EC_CORRUPTION}};
            }
            auto replacement = std::make_shared<CacheLocation>(*locations[0]);
            mutation(*replacement);
            locations[0] = std::move(replacement);
            return {MA_OK, {EC_OK}};
        };
        const auto result =
            indexer->ReadModifyWriteTargetLocations(&request_context_, {internal_key}, {{location_id}}, modifier);
        ASSERT_EQ(EC_OK, result.ec);
        ASSERT_EQ(1, result.per_location_error_codes.size());
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), result.per_location_error_codes.front());
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
    EXPECT_EQ(2, start.session_item_count);
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

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));
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
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
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

TEST_F(KvMetaManagerTest, MalformedCreateResponseReleasesOnlyOwnedAllocations) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<OverlongCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] = manager_->StartWrite(&request_context_, kInstanceId, {"malformed-create"}, {17}, 30);

    EXPECT_EQ(EC_MISMATCH, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    ASSERT_EQ(2, malformed->deleted_uris.size());
    EXPECT_NE(malformed->deleted_uris[0].ToUriString(), malformed->deleted_uris[1].ToUriString());

    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"malformed-create"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, CreateProviderExceptionFailsClosedAndReleasesEarlierCandidates) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<ThrowingSecondCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }

    const auto [ec, result] =
        manager_->StartWrite(&request_context_, kInstanceId, {"create-before-throw", "create-throws"}, {11, 13}, 30);

    EXPECT_EQ(EC_IO_ERROR, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_EQ(2, throwing->CreateAttempts());
    EXPECT_EQ(1, throwing->DeleteItems());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"create-before-throw", "create-throws"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RejectsPackedFileMemberWithoutDeletingItsSharedAllocation) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<NonSingletonCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] = manager_->StartWrite(&request_context_, kInstanceId, {"packed-member"}, {17}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    // A non-zero block belongs to a shared file by definition. The malformed
    // backend result is retained rather than risking deletion of other data.
    EXPECT_EQ(0, malformed->delete_calls);
    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"packed-member"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, RejectsAStorageBackendThatReusesOneSingletonForTwoKeys) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<DuplicateSingletonCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] =
        manager_->StartWrite(&request_context_, kInstanceId, {"duplicate-uri-a", "duplicate-uri-b"}, {17, 17}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_EQ(1, malformed->delete_calls);
    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"duplicate-uri-a", "duplicate-uri-b"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
}

TEST_F(KvMetaManagerTest, AtomicFinishFailureRollsBackEveryValue) {
    const std::vector<std::string> keys{"atomic-a", "atomic-b"};
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, keys, {64, 128}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(2, start.locations.size());

    // One failed item aborts the complete generic-object transaction.
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, false}));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);

    // Exact metadata was removed, so the same keys can be admitted again.
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, keys, {7, 9}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ((std::vector<bool>{false, false}), retry.key_mask);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false, false}));
}

TEST_F(KvMetaManagerTest, RecoveryRebuildsExactDynamicByteUsage) {
    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {"recover-a", "recover-b"}, {17, 33}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(2, start.locations.size());
    ASSERT_EQ(start.locations[0].type, start.locations[1].type);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
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

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(kObjectCount, indexer->GetStorageUsage());

    // A demotion cancels an unbounded namespace walk before the server waits
    // for KVMeta RPCs. Cancellation is sticky until the next successful
    // leader recovery explicitly resumes maintenance.
    manager_->CancelMaintenance();
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->TrimAll(&request_context_, kInstanceId, false));
    auto [before_resume_ec, before_resume] = manager_->Get(&request_context_, kInstanceId, {"trim-0", "trim-256"});
    ASSERT_EQ(EC_OK, before_resume_ec);
    ASSERT_EQ(2, before_resume.size());
    EXPECT_TRUE(before_resume[0].found);
    EXPECT_TRUE(before_resume[1].found);

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_EQ(0, indexer->GetStorageUsage());

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"trim-0", "trim-128", "trim-256"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(3, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    EXPECT_FALSE(values[2].found);
}

TEST_F(KvMetaManagerTest, TrimDoesNotReplayFailedPhysicalDelete) {
    constexpr const char *kKey = "trim-delete-fails";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {31}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    EXPECT_EQ(EC_IO_ERROR, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_EQ(1, failing->DeleteAttempts());
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    // The retry sees durable metadata absence and cannot rediscover or replay
    // the old reusable-address URI.
    EXPECT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_EQ(1, failing->DeleteAttempts());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ExistingKeysAreMaskedAndInflightKeysAreRetryable) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {21}, 30);
    ASSERT_EQ(EC_OK, first_ec);
    ASSERT_EQ((std::vector<bool>{false}), first.key_mask);

    auto [wrong_active_ec, wrong_active] =
        manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {999}, 30);
    EXPECT_EQ(EC_MISMATCH, wrong_active_ec);
    EXPECT_TRUE(wrong_active.key_mask.empty());
    EXPECT_TRUE(wrong_active.locations.empty());

    auto [second_ec, second] = manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {21}, 30);
    EXPECT_EQ(EC_EXIST, second_ec);
    EXPECT_TRUE(second.key_mask.empty());
    EXPECT_TRUE(second.locations.empty());
    EXPECT_TRUE(second.write_session_id.empty());

    auto [mixed_ec, mixed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"same-key", "must-not-allocate"}, {21, 7}, 30);
    EXPECT_EQ(EC_EXIST, mixed_ec);
    EXPECT_TRUE(mixed.key_mask.empty());
    EXPECT_TRUE(mixed.locations.empty());
    EXPECT_TRUE(mixed.write_session_id.empty());
    auto [mixed_get_ec, mixed_values] =
        manager_->Get(&request_context_, kInstanceId, {"same-key", "must-not-allocate"});
    ASSERT_EQ(EC_OK, mixed_get_ec);
    ASSERT_EQ(2, mixed_values.size());
    EXPECT_FALSE(mixed_values[0].found);
    EXPECT_FALSE(mixed_values[1].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(21, indexer->GetStorageUsage());

    // A malformed finish request must not consume the valid session.
    EXPECT_EQ(EC_BADARGS, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {}));
    EXPECT_EQ(EC_MISMATCH, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {true, true}));
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

TEST_F(KvMetaManagerTest, TrimRejectsActiveSessionsWithoutDeletingCommittedValues) {
    auto [committed_ec, committed] = manager_->StartWrite(&request_context_, kInstanceId, {"trim-committed"}, {13}, 30);
    ASSERT_EQ(EC_OK, committed_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, committed.write_session_id, {true}));

    auto [active_ec, active] = manager_->StartWrite(&request_context_, kInstanceId, {"trim-active"}, {17}, 30);
    ASSERT_EQ(EC_OK, active_ec);
    ASSERT_FALSE(active.write_session_id.empty());

    EXPECT_EQ(EC_EXIST, manager_->TrimAll(&request_context_, kInstanceId, false));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"trim-committed", "trim-active"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_FALSE(values[1].found);

    // Rejection does not consume the writer's session. Once the caller ends
    // it, the same explicit Trim can safely remove the whole namespace.
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, active.write_session_id, {false}));
    ASSERT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
    auto [after_ec, after] = manager_->Get(&request_context_, kInstanceId, {"trim-committed"});
    ASSERT_EQ(EC_OK, after_ec);
    ASSERT_EQ(1, after.size());
    EXPECT_FALSE(after[0].found);
}

TEST_F(KvMetaManagerTest, TrimWaitBarrierIncludesSessionFinalization) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"finalizing-trim"}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto finish = std::async(std::launch::async, [&]() {
        RequestContext context("finish-while-trim");
        return manager_->FinishWrite(&context, kInstanceId, start.write_session_id, {false});
    });
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    auto trim = std::async(std::launch::async, [&]() {
        RequestContext context("trim-while-finalizing");
        return manager_->TrimAll(&context, kInstanceId, false);
    });
    const auto trim_status = trim.wait_for(std::chrono::seconds(1));
    blocking->ReleaseDelete();

    EXPECT_EQ(std::future_status::ready, trim_status);
    EXPECT_EQ(EC_EXIST, trim.get());
    EXPECT_EQ(EC_OK, finish.get());
    EXPECT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
}

TEST_F(KvMetaManagerTest, TrimFenceDoesNotHoldTheGroupShardAcrossStorageIo) {
    constexpr const char *kPeerInstance = "trim-peer-instance";
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, "default", kPeerInstance, "peer").first);
    CommitObject(kInstanceId, "trim-slow-delete", 17);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto trim = std::async(std::launch::async, [&]() {
        RequestContext context("slow-trim");
        return manager_->TrimAll(&context, kInstanceId, false);
    });
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    auto same_instance_write = std::async(std::launch::async, [&]() {
        RequestContext context("write-during-same-instance-trim");
        return manager_->StartWrite(&context, kInstanceId, {"must-be-fenced"}, {19}, 30).first;
    });
    auto duplicate_trim = std::async(std::launch::async, [&]() {
        RequestContext context("duplicate-trim");
        return manager_->TrimAll(&context, kInstanceId, false);
    });
    auto same_instance_remove = std::async(std::launch::async, [&]() {
        RequestContext context("remove-during-same-instance-trim");
        return manager_->Remove(&context, kInstanceId, {"must-be-fenced"});
    });
    auto peer_write = std::async(std::launch::async, [&]() {
        RequestContext context("peer-write-during-trim");
        return manager_->StartWrite(&context, kPeerInstance, {"peer-value"}, {23}, 30);
    });

    const auto same_status = same_instance_write.wait_for(std::chrono::seconds(1));
    const auto duplicate_status = duplicate_trim.wait_for(std::chrono::seconds(1));
    const auto remove_status = same_instance_remove.wait_for(std::chrono::seconds(1));
    const auto peer_status = peer_write.wait_for(std::chrono::seconds(1));
    blocking->ReleaseDelete();

    ASSERT_EQ(std::future_status::ready, same_status);
    EXPECT_EQ(EC_EXIST, same_instance_write.get());
    ASSERT_EQ(std::future_status::ready, duplicate_status);
    EXPECT_EQ(EC_EXIST, duplicate_trim.get());
    ASSERT_EQ(std::future_status::ready, remove_status);
    EXPECT_EQ(EC_EXIST, same_instance_remove.get());
    ASSERT_EQ(std::future_status::ready, peer_status);
    auto [peer_ec, peer_start] = peer_write.get();
    ASSERT_EQ(EC_OK, peer_ec);
    ASSERT_FALSE(peer_start.write_session_id.empty());

    EXPECT_EQ(EC_OK, trim.get());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kPeerInstance, peer_start.write_session_id, {false}));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RemoveDoesNotInvalidateAnActiveWriteSession) {
    auto [committed_start_ec, committed_start] =
        manager_->StartWrite(&request_context_, kInstanceId, {"committed-remove-guard"}, {13}, 30);
    ASSERT_EQ(EC_OK, committed_start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, committed_start.write_session_id, {true}));

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"active-remove"}, {21}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    EXPECT_EQ(EC_EXIST, manager_->Remove(&request_context_, kInstanceId, {"committed-remove-guard", "active-remove"}));

    // Batch validation finishes before DeleteItems, so the committed key is
    // preserved when a later key in the same request is still active.
    auto [before_finish_ec, before_finish] =
        manager_->Get(&request_context_, kInstanceId, {"committed-remove-guard", "active-remove"});
    ASSERT_EQ(EC_OK, before_finish_ec);
    ASSERT_EQ(2, before_finish.size());
    EXPECT_TRUE(before_finish[0].found);
    EXPECT_FALSE(before_finish[1].found);

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"active-remove"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {"committed-remove-guard", "active-remove"}));
}

TEST_F(KvMetaManagerTest, RemoveFinishesPhysicalDeleteBeforeReadmittingTheKey) {
    constexpr const char *key = "remove-recreate";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {key}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto remove = std::async(std::launch::async, [&]() {
        RequestContext context("remove-before-recreate");
        return manager_->Remove(&context, kInstanceId, {key});
    });
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    auto recreate = std::async(std::launch::async, [&]() {
        RequestContext context("recreate-after-remove");
        return manager_->StartWrite(&context, kInstanceId, {key}, {19}, 30);
    });
    // Remove still owns the KVMeta group shard while the old allocation is
    // being deleted, so a new generation cannot reach backend allocation yet.
    EXPECT_EQ(std::future_status::timeout, recreate.wait_for(std::chrono::milliseconds(100)));

    blocking->ReleaseDelete();
    EXPECT_EQ(EC_OK, remove.get());
    auto [recreate_ec, recreated] = recreate.get();
    ASSERT_EQ(EC_OK, recreate_ec);
    ASSERT_EQ((std::vector<bool>{false}), recreated.key_mask);
    ASSERT_EQ(1, recreated.locations.size());
    ASSERT_FALSE(recreated.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, recreated.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RemoveContainsPhysicalDeleteExceptionWithoutReplay) {
    constexpr const char *kKey = "remove-delete-throws";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_,
                                                               FaultingDeleteNfsBackend::Mode::kStandardException);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }

    EXPECT_EQ(EC_IO_ERROR, manager_->Remove(&request_context_, kInstanceId, {kKey}));
    EXPECT_EQ(1, throwing->DeleteAttempts());
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    // An API retry is idempotent at the metadata layer and must not replay the
    // uncertain physical delete after the old URI is no longer discoverable.
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {kKey}));
    EXPECT_EQ(1, throwing->DeleteAttempts());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }

    auto [recreate_ec, recreated] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 30);
    ASSERT_EQ(EC_OK, recreate_ec);
    ASSERT_FALSE(recreated.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, recreated.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RollbackFinishesPhysicalDeleteBeforeReadmittingTheKey) {
    constexpr const char *key = "rollback-recreate";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {key}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto rollback = std::async(std::launch::async, [&]() {
        RequestContext context("rollback-before-recreate");
        return manager_->FinishWrite(&context, kInstanceId, start.write_session_id, {false});
    });
    // Metadata is already durably absent when the physical delete blocks, but
    // the group shard still prevents a new generation from being allocated
    // until this one-shot delete attempt has returned.
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    auto recreate = std::async(std::launch::async, [&]() {
        RequestContext context("recreate-after-rollback");
        return manager_->StartWrite(&context, kInstanceId, {key}, {19}, 30);
    });
    EXPECT_EQ(std::future_status::timeout, recreate.wait_for(std::chrono::milliseconds(100)));

    blocking->ReleaseDelete();
    EXPECT_EQ(EC_OK, rollback.get());
    auto [recreate_ec, recreated] = recreate.get();
    ASSERT_EQ(EC_OK, recreate_ec);
    ASSERT_EQ((std::vector<bool>{false}), recreated.key_mask);
    ASSERT_EQ(1, recreated.locations.size());
    ASSERT_FALSE(recreated.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, recreated.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RollbackContainsPhysicalDeleteExceptionWithoutReplay) {
    constexpr const char *kKey = "rollback-delete-throws";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_,
                                                               FaultingDeleteNfsBackend::Mode::kStandardException);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }

    // The metadata-first rollback converts the provider exception into a
    // deterministic error. It must not replay an uncertain delete because a
    // reusable backend address may already belong to a successor object.
    EXPECT_EQ(EC_IO_ERROR, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    EXPECT_EQ(1, throwing->DeleteAttempts());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }

    auto [recreate_ec, recreated] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 30);
    ASSERT_EQ(EC_OK, recreate_ec);
    ASSERT_EQ((std::vector<bool>{false}), recreated.key_mask);
    ASSERT_FALSE(recreated.write_session_id.empty());
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, recreated.write_session_id, {false}));
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, RollbackRejectsMalformedPhysicalDeleteResult) {
    constexpr const char *kKey = "rollback-short-delete-result";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto malformed =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kShortResult);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    EXPECT_EQ(EC_MISMATCH, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    EXPECT_EQ(1, malformed->DeleteAttempts());
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, PartialCommitWithIncompleteRollbackReturnsUnknownOutcome) {
    constexpr const char *kFirstKey = "partial-commit-first";
    constexpr const char *kSecondKey = "partial-commit-second";
    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {kFirstKey, kSecondKey}, {17, 23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto second_key = KvMetaManager::InternalKey(kSecondKey);
    const auto delete_result = indexer->Delete(&request_context_, {second_key});
    ASSERT_EQ(EC_OK, delete_result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_result.error_codes);
    ASSERT_TRUE(indexer->Sync({second_key}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));
    EXPECT_GE(failing->DeleteAttempts(), 1);

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kFirstKey, kSecondKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ExpiredFinishPreservesTimeoutWhenCleanupFails) {
    auto [blocker_ec, blocker] =
        manager_->StartWrite(&request_context_, kInstanceId, {"expired-cleanup-blocker"}, {11}, 1);
    ASSERT_EQ(EC_OK, blocker_ec);
    auto [target_ec, target] =
        manager_->StartWrite(&request_context_, kInstanceId, {"expired-cleanup-failure"}, {19}, 2);
    ASSERT_EQ(EC_OK, target_ec);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingThenFailDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }
    ASSERT_TRUE(blocking->WaitForFirstDelete(std::chrono::seconds(3)));

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    auto finish = std::async(std::launch::async, [&]() {
        RequestContext context("expired-finish-with-cleanup-failure");
        return manager_->FinishWrite(&context, kInstanceId, target.write_session_id, {true});
    });
    EXPECT_EQ(std::future_status::timeout, finish.wait_for(std::chrono::milliseconds(100)));
    blocking->ReleaseFirstDelete();
    EXPECT_EQ(EC_TIMEOUT, finish.get());
    EXPECT_EQ(2, blocking->DeleteAttempts());
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&request_context_, kInstanceId, blocker.write_session_id, {true}));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ExpiredSessionIsCleanedBeforeTheKeyCanBeWrittenAgain) {
    constexpr const char *kKey = "expires-and-retries";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    ErrorCode retry_ec = EC_UNKNOWN;
    KvMetaManager::StartWriteResult retry;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        auto attempt = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 30);
        retry_ec = attempt.first;
        retry = std::move(attempt.second);
        if (retry_ec == EC_OK && !retry.write_session_id.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while (std::chrono::steady_clock::now() < deadline);

    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, ExpiryDoesNotReplayFailedPhysicalDelete) {
    constexpr const char *kKey = "expiry-delete-fails";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    ASSERT_TRUE(failing->WaitForDeleteAttempts(1, std::chrono::seconds(3)));
    // A replay implementation would make a second call after its first short
    // backoff. Leave enough time to detect that regression without coupling
    // the production implementation to any retry interval.
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    EXPECT_EQ(1, failing->DeleteAttempts());
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {31}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ExpiryWorkerSurvivesUnknownPhysicalDeleteException) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"expiry-throws-a"}, {37}, 1);
    ASSERT_EQ(EC_OK, first_ec);
    ASSERT_FALSE(first.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_,
                                                               FaultingDeleteNfsBackend::Mode::kUnknownException);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }

    ASSERT_TRUE(throwing->WaitForDeleteAttempts(1, std::chrono::seconds(3)));
    auto [second_ec, second] = manager_->StartWrite(&request_context_, kInstanceId, {"expiry-throws-b"}, {41}, 1);
    ASSERT_EQ(EC_OK, second_ec);
    ASSERT_FALSE(second.write_session_id.empty());
    // Processing a second expiry proves the worker caught the first provider
    // exception instead of letting it terminate the process or its thread.
    ASSERT_TRUE(throwing->WaitForDeleteAttempts(2, std::chrono::seconds(3)));
    EXPECT_EQ(2, throwing->DeleteAttempts());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"expiry-throws-a", "expiry-throws-b"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, FinishCannotCommitAfterItsLeaseDeadlineWhileExpiryIsBusy) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"expiry-blocker"}, {11}, 1);
    ASSERT_EQ(EC_OK, first_ec);
    auto [late_ec, late] = manager_->StartWrite(&request_context_, kInstanceId, {"late-finish"}, {13}, 1);
    ASSERT_EQ(EC_OK, late_ec);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    // The expiry worker is now blocked cleaning the first session, so the
    // second session remains addressable even after its own deadline. A late
    // Finish must take it as expired and clean it, never commit it.
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(3)));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    auto finish = std::async(std::launch::async, [&]() {
        RequestContext context("late-finish-after-deadline");
        return manager_->FinishWrite(&context, kInstanceId, late.write_session_id, {true});
    });
    EXPECT_EQ(std::future_status::timeout, finish.wait_for(std::chrono::milliseconds(100)));
    blocking->ReleaseDelete();
    EXPECT_EQ(EC_TIMEOUT, finish.get());

    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {true}));
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto usage_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (indexer->GetStorageUsage() != 0 && std::chrono::steady_clock::now() < usage_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, OversizedSessionIdIsRejectedWithoutConsumingTheSession) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"bounded-session-id"}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    EXPECT_EQ(EC_BADARGS,
              manager_->FinishWrite(&request_context_,
                                    kInstanceId,
                                    std::string(manager_->limits().max_write_session_id_bytes + 1, 'x'),
                                    {true}));
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
}

TEST_F(KvMetaManagerTest, FinishRechecksLeaseAfterWaitingForTheGroupShard) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"finish-lock-wait"}, {17}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto [instance_ec, instance_info] = manager_->GetValidatedInstanceInfo(&request_context_, kInstanceId);
    ASSERT_EQ(EC_OK, instance_ec);
    ASSERT_TRUE(instance_info);
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % manager_->quota_admission_mutexes_.size();

    std::promise<void> shard_locked;
    auto release_shard = shard_locked.get_future();
    std::thread blocker([&]() {
        std::unique_lock<std::mutex> lock(manager_->quota_admission_mutexes_[quota_shard]);
        shard_locked.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    });
    release_shard.wait();

    RequestContext finish_context("finish-after-group-lock-wait");
    EXPECT_EQ(EC_TIMEOUT, manager_->FinishWrite(&finish_context, kInstanceId, start.write_session_id, {true}));
    blocker.join();

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"finish-lock-wait"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
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

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
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
    const auto rmw =
        indexer->ReadModifyWriteTargetLocations(&request_context_, {internal_key}, {{location_id}}, corrupt_scheme);
    ASSERT_EQ(EC_OK, rmw.ec);
    ASSERT_EQ(1, rmw.per_location_error_codes.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), rmw.per_location_error_codes[0]);

    EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstanceId, {key}).first);
    EXPECT_EQ(EC_CORRUPTION, manager_->Remove(&request_context_, kInstanceId, {key}));
}

TEST_F(KvMetaManagerTest, PersistedObjectUriAndAccountingMustRemainUnambiguous) {
    const std::vector<std::string> keys{
        "duplicate-size", "fragment", "accounting-mismatch", "event-report", "oversized-uri"};
    for (const auto &key : keys) {
        CommitObject(kInstanceId, key, 31);
    }

    MutateObject(kInstanceId, "duplicate-size", [](CacheLocation &location) {
        auto &spec = location.mutable_location_specs().front();
        spec.set_uri(spec.uri() + "&size=31");
    });
    MutateObject(kInstanceId, "fragment", [](CacheLocation &location) {
        auto &spec = location.mutable_location_specs().front();
        spec.set_uri(spec.uri() + "#ignored-by-some-parsers");
    });
    MutateObject(
        kInstanceId, "accounting-mismatch", [](CacheLocation &location) { location.set_validated_total_size(30); });
    MutateObject(kInstanceId, "event-report", [](CacheLocation &location) {
        location.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    });
    MutateObject(kInstanceId, "oversized-uri", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs/" + std::string(kMaxKvMetaLocationUriBytes, 'x') +
                                                          "?size=31");
    });

    for (const auto &key : keys) {
        SCOPED_TRACE(key);
        EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstanceId, {key}).first);
        EXPECT_EQ(EC_CORRUPTION, manager_->Remove(&request_context_, kInstanceId, {key}));
    }
}

TEST_F(KvMetaManagerTest, RejectsAmbiguousOrUnboundedRequestsBeforeAllocation) {
    auto [duplicate_ec, duplicate] = manager_->StartWrite(&request_context_, kInstanceId, {"dup", "dup"}, {1, 2}, 30);
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

TEST_F(KvMetaManagerTest, RejectsWriteTimeoutLimitOutsideTheProtocolRange) {
    KvMetaManager::Limits limits;
    limits.max_write_timeout_seconds = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
    KvMetaManager invalid_manager(cache_manager_, registry_manager_, limits);

    EXPECT_FALSE(invalid_manager.Init());
}

TEST_F(KvMetaManagerTest, RejectsLocationUriLimitOutsideTheClientContract) {
    KvMetaManager::Limits limits;
    limits.max_location_uri_bytes = 0;
    KvMetaManager zero_limit_manager(cache_manager_, registry_manager_, limits);
    EXPECT_FALSE(zero_limit_manager.Init());

    limits.max_location_uri_bytes = kMaxKvMetaLocationUriBytes + 1;
    KvMetaManager oversized_limit_manager(cache_manager_, registry_manager_, limits);
    EXPECT_FALSE(oversized_limit_manager.Init());
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
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"demoted-active"}, {19}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    LocationsPerKey active_location;
    const auto active_result = indexer->GetLocations(&request_context_,
                                                     {KvMetaManager::InternalKey("demoted-active")},
                                                     {{KvMetaManager::StableLocationId("demoted-active")}},
                                                     active_location);
    ASSERT_EQ(1, active_result.per_location_error_codes.size());
    ASSERT_EQ(1, active_location.size());
    ASSERT_EQ(1, active_location[0].size());
    ASSERT_TRUE(active_location[0][0]);
    // Tagged positive values persist the write-lease deadline. This is well
    // above any ordinary Unix timestamp in microseconds.
    EXPECT_GT(active_location[0][0]->create_time(), std::int64_t{1} << 62);

    manager_->DoCleanup();
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
    auto [hidden_ec, hidden] = manager_->Get(&request_context_, kInstanceId, {"demoted-active"});
    ASSERT_EQ(EC_OK, hidden_ec);
    ASSERT_EQ(1, hidden.size());
    EXPECT_FALSE(hidden[0].found);

    const auto recovery_start = std::chrono::steady_clock::now();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    const auto recovery_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - recovery_start);
    // Recovery must not immediately delete a lease that an old leader has
    // already handed to a client. Keep the lower bound loose for slow ASAN
    // hosts while still distinguishing it from the old eager deletion.
    EXPECT_GE(recovery_elapsed.count(), 100);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"demoted-active"}, {19}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ((std::vector<bool>{false}), retry.key_mask);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RecoveryProtectsUntaggedActiveMarkerDuringRollingUpgrade) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"legacy-active"}, {19}, 30);
    ASSERT_EQ(EC_OK, start_ec);

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto internal_key = KvMetaManager::InternalKey("legacy-active");
    const auto location_id = KvMetaManager::StableLocationId("legacy-active");
    auto replace_with_legacy_marker = [](const std::vector<ErrorCode> &get_ecs,
                                         const LocationIdVector &,
                                         std::size_t,
                                         CacheLocationVector &locations,
                                         PropertyMap &) -> LocationModifierResult {
        if (get_ecs.size() != 1 || get_ecs[0] != EC_OK || locations.size() != 1 || !locations[0]) {
            return {MA_FAIL, {EC_CORRUPTION}};
        }
        auto legacy = std::make_shared<CacheLocation>(*locations[0]);
        legacy->set_create_time(std::max<std::int64_t>(1, TimestampUtil::GetCurrentTimeUs()));
        locations[0] = std::move(legacy);
        return {MA_OK, {EC_OK}};
    };
    const auto rmw = indexer->ReadModifyWriteTargetLocations(
        &request_context_, {internal_key}, {{location_id}}, replace_with_legacy_marker);
    ASSERT_EQ(EC_OK, rmw.ec);
    ASSERT_TRUE(indexer->Sync({internal_key}));

    manager_->DoCleanup();
    const auto recovery_start = std::chrono::steady_clock::now();
    const auto abort_after_poll = [&]() {
        return std::chrono::steady_clock::now() - recovery_start >= std::chrono::milliseconds(150);
    };
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->DoRecover(abort_after_poll));

    LocationsPerKey still_active;
    const auto active_result = indexer->GetLocations(&request_context_, {internal_key}, {{location_id}}, still_active);
    ASSERT_EQ(1, active_result.per_location_error_codes.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), active_result.per_location_error_codes[0]);
    ASSERT_EQ(1, still_active.size());
    ASSERT_EQ(1, still_active[0].size());
    ASSERT_TRUE(still_active[0][0]);
    EXPECT_GT(still_active[0][0]->create_time(), 0);
    EXPECT_LT(still_active[0][0]->create_time(), std::int64_t{1} << 62);
}

TEST_F(KvMetaManagerTest, RecoveryContainsPhysicalDeleteExceptionWithoutReplay) {
    constexpr const char *kKey = "recovery-delete-throws";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {41}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_,
                                                               FaultingDeleteNfsBackend::Mode::kStandardException);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));

    manager_->DoCleanup();
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }
    EXPECT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(1, throwing->DeleteAttempts());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    LocationsPerKey active_location;
    const auto active_result = indexer->GetLocations(&request_context_,
                                                     {KvMetaManager::InternalKey(kKey)},
                                                     {{KvMetaManager::StableLocationId(kKey)}},
                                                     active_location);
    ASSERT_EQ(1, active_result.per_location_error_codes.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}), active_result.per_location_error_codes[0]);
    ASSERT_EQ(1, active_location.size());
    ASSERT_EQ(1, active_location[0].size());
    EXPECT_FALSE(active_location[0][0]);
    EXPECT_EQ(0, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
    // Recovery sees no stale metadata on the next pass and therefore never
    // replays the uncertain physical delete against a reusable URI.
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(1, throwing->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {43}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, CancellationClosesSessionAdmissionBeforeWorkerJoin) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"before-cancel"}, {19}, 30);
    ASSERT_EQ(EC_OK, first_ec);

    manager_->CancelMaintenance();
    auto [cancelled_ec, cancelled] = manager_->StartWrite(&request_context_, kInstanceId, {"after-cancel"}, {23}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, cancelled_ec);
    EXPECT_TRUE(cancelled.locations.empty());
    EXPECT_FALSE(manager_->ResumeMaintenance());

    // An already admitted Finish may still drain cleanly. No new session can
    // be published after cancellation, and DoCleanup performs the final join.
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {false}));
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"after-cancel"}, {23}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ActiveSessionCountIsBoundedBeforeAllocation) {
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_active_write_sessions = 1;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());

    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"session-a"}, {7}, 30);
    ASSERT_EQ(EC_OK, first_ec);
    auto [second_ec, second] = manager_->StartWrite(&request_context_, kInstanceId, {"session-b"}, {9}, 30);
    EXPECT_EQ(EC_NOSPC, second_ec);
    EXPECT_TRUE(second.key_mask.empty());
    EXPECT_TRUE(second.locations.empty());

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {false}));
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"session-b"}, {9}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
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
    object_group.set_quota(InstanceGroupQuota(20, {QuotaConfig(20, DataStorageType::DATA_STORAGE_TYPE_NFS)}));
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));
    ASSERT_EQ(EC_OK,
              manager_->RegisterInstance(&request_context_, "small-object-group", "small-object-instance", "").first);

    auto [oversized_ec, oversized] =
        manager_->StartWrite(&request_context_, "small-object-instance", {"a", "b"}, {17, 4}, 30);
    EXPECT_EQ(EC_NOSPC, oversized_ec);
    EXPECT_TRUE(oversized.locations.empty());

    auto [start_ec, start] = manager_->StartWrite(&request_context_, "small-object-instance", {"a"}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, "small-object-instance", start.write_session_id, {true}));
    auto [remaining_ec, remaining] = manager_->StartWrite(&request_context_, "small-object-instance", {"b"}, {4}, 30);
    EXPECT_EQ(EC_NOSPC, remaining_ec);
    EXPECT_TRUE(remaining.locations.empty());

    auto [fill_ec, fill] = manager_->StartWrite(&request_context_, "small-object-instance", {"b"}, {3}, 30);
    ASSERT_EQ(EC_OK, fill_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, "small-object-instance", fill.write_session_id, {true}));
    auto [full_ec, full] = manager_->StartWrite(&request_context_, "small-object-instance", {"c"}, {1}, 30);
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
    object_group.set_quota(InstanceGroupQuota(20, {QuotaConfig(20, DataStorageType::DATA_STORAGE_TYPE_NFS)}));
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));
    ASSERT_EQ(EC_OK,
              manager_->RegisterInstance(&request_context_, "concurrent-object-group", "concurrent-object-instance", "")
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
            auto [ec, result] =
                manager_->StartWrite(&context, "concurrent-object-instance", {"key-" + std::to_string(i)}, {15}, 30);
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

    const std::size_t success_count = static_cast<std::size_t>(std::count(errors.begin(), errors.end(), EC_OK));
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
                      manager_->FinishWrite(
                          &request_context_, "concurrent-object-instance", results[i].write_session_id, {false}));
        } else {
            EXPECT_TRUE(results[i].key_mask.empty());
            EXPECT_TRUE(results[i].locations.empty());
        }
    }
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, ReclaimerEvictsTheLeastRecentlyUsedCommittedObjectAtTheByteWatermark) {
    constexpr const char *kGroup = "reclaim-lru-group";
    constexpr const char *kInstance = "reclaim-lru-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);

    CommitObject(kInstance, "old", 30);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "middle", 30);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "hot", 30);
    auto [touch_ec, touch] = manager_->Get(&request_context_, kInstance, {"hot"});
    ASSERT_EQ(EC_OK, touch_ec);
    ASSERT_EQ(1, touch.size());
    ASSERT_TRUE(touch[0].found);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(90, indexer->GetStorageUsage());
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 60; }, std::chrono::seconds(2)));
    ASSERT_TRUE(
        WaitUntil([&]() { return metrics_registry_->GetGauge("kv_meta_reclaimer.pending_object_count").Get() == 0; },
                  std::chrono::seconds(2)));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"old", "middle", "hot"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(3, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_TRUE(values[1].found);
    EXPECT_TRUE(values[2].found);
    EXPECT_GE(metrics_registry_->GetCounter("kv_meta_reclaimer.round_count").Get(), 1);
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.retired_object_count").Get());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.reclaimed_object_count").Get());
    EXPECT_EQ(30, metrics_registry_->GetCounter("kv_meta_reclaimer.reclaimed_bytes").Get());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_attempted_object_count").Get());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_object_count").Get());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_bytes").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.pending_bytes").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.blocked_group_count").Get());
}

TEST_F(KvMetaManagerTest, ReclaimerRetiresMetadataBeforeWaitingForTheReadGracePeriod) {
    constexpr const char *kGroup = "reclaim-grace-group";
    constexpr const char *kInstance = "reclaim-grace-instance";
    constexpr const char *kKey = "grace-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 250);
    CommitObject(kInstance, kKey, 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto internal_key = KvMetaManager::InternalKey(kKey);
    const auto location_id = KvMetaManager::StableLocationId(kKey);
    const auto read_status = [&]() {
        CacheLocationMapVector maps;
        const auto result = indexer->GetLocationMapsForMaintenance(&request_context_, {internal_key}, maps);
        if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
            return CLS_NOT_FOUND;
        }
        const auto it = maps[0].find(location_id);
        return it == maps[0].end() || !it->second ? CLS_NOT_FOUND : it->second->status();
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return read_status() == CLS_DELETING; }, std::chrono::seconds(2)));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_EQ(90, indexer->GetStorageUsage());
    EXPECT_EQ(EC_EXIST, manager_->TrimAll(&request_context_, kInstance, false));

    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    EXPECT_EQ(CLS_NOT_FOUND, read_status());
}

TEST_F(KvMetaManagerTest, ReclaimerNeverEvictsAnActiveWriteAndReclaimsAfterCommit) {
    constexpr const char *kGroup = "reclaim-active-group";
    constexpr const char *kInstance = "reclaim-active-instance";
    constexpr const char *kKey = "active-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstance, {kKey}, {90}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    LocationsPerKey locations;
    auto exact = indexer->GetLocations(
        &request_context_, {KvMetaManager::InternalKey(kKey)}, {{KvMetaManager::StableLocationId(kKey)}}, locations);
    ASSERT_EQ(EC_OK, exact.ec);
    ASSERT_EQ(1, locations.size());
    ASSERT_EQ(1, locations[0].size());
    ASSERT_TRUE(locations[0][0]);
    EXPECT_EQ(CLS_NEW, locations[0][0]->status());
    EXPECT_GT(locations[0][0]->create_time(), 0);
    EXPECT_EQ(90, indexer->GetStorageUsage());

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, start.write_session_id, {true}));
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerDoesNotReplayAnUncertainPhysicalDelete) {
    constexpr const char *kGroup = "reclaim-delete-failure-group";
    constexpr const char *kInstance = "reclaim-delete-failure-instance";
    constexpr const char *kKey = "delete-failure-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, kKey, 90);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    std::shared_ptr<DataStorageBackend> original;
    auto faulting =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        original = storage_manager->storage_map_.at("nfs_01");
        storage_manager->storage_map_["nfs_01"] = faulting;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(faulting->WaitForDeleteAttempts(1, std::chrono::seconds(2)));
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(1, faulting->DeleteAttempts());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_attempted_object_count").Get());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_object_count").Get());
    EXPECT_EQ(90, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_bytes").Get());

    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ReclaimerPendingCreditPreventsOverEvictionDuringTheGracePeriod) {
    constexpr const char *kGroup = "reclaim-credit-group";
    constexpr const char *kInstance = "reclaim-credit-instance";
    const std::vector<std::string> keys{"credit-old", "credit-middle", "credit-new"};
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 1000);
    for (const auto &key : keys) {
        CommitObject(kInstance, key, 30);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto retired_count = [&]() {
        std::size_t count = 0;
        for (const auto &key : keys) {
            CacheLocationMapVector maps;
            const auto result =
                indexer->GetLocationMapsForMaintenance(&request_context_, {KvMetaManager::InternalKey(key)}, maps);
            if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
                continue;
            }
            const auto it = maps[0].find(KvMetaManager::StableLocationId(key));
            if (it != maps[0].end() && it->second && it->second->status() == CLS_DELETING) {
                ++count;
            }
        }
        return count;
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return retired_count() == 1; }, std::chrono::seconds(2)));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(1, retired_count());
    EXPECT_EQ(90, indexer->GetStorageUsage());

    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 60; }, std::chrono::seconds(2)));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, keys);
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(3, values.size());
    EXPECT_EQ(2, std::count_if(values.begin(), values.end(), [](const auto &value) { return value.found; }));
}

TEST_F(KvMetaManagerTest, ReclaimerGracePeriodForOneGroupDoesNotBlockAnotherGroup) {
    constexpr const char *kSlowGroup = "reclaim-a-slow-group";
    constexpr const char *kSlowInstance = "reclaim-a-slow-instance";
    constexpr const char *kFastGroup = "reclaim-z-fast-group";
    constexpr const char *kFastInstance = "reclaim-z-fast-instance";
    constexpr const char *kSlowKey = "slow-object";
    CreateReclaimGroup(kSlowGroup, kSlowInstance, 100, 0.8, 750);
    CreateReclaimGroup(kFastGroup, kFastInstance, 100, 0.8, 0);
    CommitObject(kSlowInstance, kSlowKey, 90);
    CommitObject(kFastInstance, "fast-object", 90);

    auto slow_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kSlowInstance));
    auto fast_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kFastInstance));
    ASSERT_TRUE(slow_indexer);
    ASSERT_TRUE(fast_indexer);
    const auto slow_is_retired = [&]() {
        CacheLocationMapVector maps;
        const auto result = slow_indexer->GetLocationMapsForMaintenance(
            &request_context_, {KvMetaManager::InternalKey(kSlowKey)}, maps);
        if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
            return false;
        }
        const auto it = maps[0].find(KvMetaManager::StableLocationId(kSlowKey));
        return it != maps[0].end() && it->second && it->second->status() == CLS_DELETING;
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(slow_is_retired, std::chrono::seconds(2)));
    ASSERT_TRUE(WaitUntil([&]() { return fast_indexer->GetStorageUsage() == 0; }, std::chrono::milliseconds(250)));
    EXPECT_EQ(90, slow_indexer->GetStorageUsage());
    ASSERT_TRUE(WaitUntil([&]() { return slow_indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerRetriesMetadataSyncBeforePhysicalDeleteAndBlocksUnsafeAdmission) {
    constexpr const char *kGroup = "reclaim-sync-failure-group";
    constexpr const char *kInstance = "reclaim-sync-failure-instance";
    constexpr const char *kKey = "sync-failure-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    auto *meta_backend = InstallFailingSyncBackend(kInstance);
    ASSERT_NE(nullptr, meta_backend);
    CommitObject(kInstance, kKey, 90);
    meta_backend->FailNextMaintenanceDeleteSync();

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    std::shared_ptr<DataStorageBackend> original;
    auto faulting =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        original = storage_manager->storage_map_.at("nfs_01");
        storage_manager->storage_map_["nfs_01"] = faulting;
    }

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 200);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(meta_backend->WaitForSyncFailure(std::chrono::seconds(2)));
    ASSERT_TRUE(
        WaitUntil([&]() { return metrics_registry_->GetGauge("kv_meta_reclaimer.blocked_group_count").Get() == 1; },
                  std::chrono::seconds(2)));

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"new-object"}, {10}, 30);
    EXPECT_EQ(EC_EXIST, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    EXPECT_EQ(0, faulting->DeleteAttempts());

    ASSERT_TRUE(faulting->WaitForDeleteAttempts(1, std::chrono::seconds(2)));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    EXPECT_EQ(1, faulting->DeleteAttempts());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstance, {"new-object"}, {10}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(1, start.locations.size());
    EXPECT_GE(metrics_registry_->GetCounter("kv_meta_reclaimer.retry_count").Get(), 1);
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.blocked_group_count").Get());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, start.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerHonorsTheIndependentKeyCountWatermark) {
    constexpr const char *kGroup = "reclaim-key-count-group";
    constexpr const char *kInstance = "reclaim-key-count-instance";
    CreateReclaimGroup(kGroup, kInstance, 10'000, 0.8, 0, 4);
    CommitObject(kInstance, "key-old", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "key-middle", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "key-new", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "key-newest", 10);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(4, indexer->GetKeyCount());
    ASSERT_EQ(40, indexer->GetStorageUsage());
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetKeyCount() == 3; }, std::chrono::seconds(2)));
    EXPECT_EQ(30, indexer->GetStorageUsage());

    auto [get_ec, values] =
        manager_->Get(&request_context_, kInstance, {"key-old", "key-middle", "key-new", "key-newest"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(4, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_TRUE(values[1].found);
    EXPECT_TRUE(values[2].found);
    EXPECT_TRUE(values[3].found);
}

TEST_F(KvMetaManagerTest, ReclaimerHonorsTheStorageTypeByteWatermark) {
    constexpr const char *kGroup = "reclaim-storage-type-group";
    constexpr const char *kInstance = "reclaim-storage-type-instance";
    // Group usage is only 9%, but NFS usage is 90% of its independent quota.
    CreateReclaimGroup(kGroup, kInstance, 1000, 0.8, 0, MetaIndexerConfig::kDefaultMaxKeyCount, 100);
    CommitObject(kInstance, "type-pressure", 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(90, indexer->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerCreatesHeadroomForARejectedRequestBelowTheWatermark) {
    constexpr const char *kGroup = "reclaim-admission-demand-group";
    constexpr const char *kInstance = "reclaim-admission-demand-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 1.0, 0);
    CommitObject(kInstance, "old-object", 70);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_EQ(70, indexer->GetStorageUsage());

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"large-new-object"}, {40}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());

    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstance, {"large-new-object"}, {40}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(1, retry.locations.size());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerCreatesStorageTypeHeadroomBelowTheGroupWatermark) {
    constexpr const char *kGroup = "reclaim-type-admission-demand-group";
    constexpr const char *kInstance = "reclaim-type-admission-demand-instance";
    CreateReclaimGroup(kGroup, kInstance, 1000, 1.0, 0, MetaIndexerConfig::kDefaultMaxKeyCount, 100);
    CommitObject(kInstance, "old-type-object", 70);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_EQ(70, indexer->GetStorageUsage());

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"new-type-object"}, {40}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));

    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstance, {"new-type-object"}, {40}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerTargetsKeyAdmissionPressureToTheFullInstance) {
    constexpr const char *kGroup = "reclaim-key-admission-demand-group";
    constexpr const char *kFullInstance = "z-reclaim-key-admission-full-instance";
    constexpr const char *kPeerInstance = "a-reclaim-key-admission-peer-instance";
    CreateReclaimGroup(kGroup, kFullInstance, 10'000, 1.0, 0, 2);
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, kGroup, kPeerInstance, "reclaim-test").first);

    // Make the peer object globally oldest. Targeted pressure must still free
    // a primary metadata key from the full instance, not evict the peer.
    CommitObject(kPeerInstance, "peer-oldest", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kFullInstance, "full-old", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kFullInstance, "full-new", 10);

    auto full_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kFullInstance));
    auto peer_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kPeerInstance));
    ASSERT_TRUE(full_indexer);
    ASSERT_TRUE(peer_indexer);
    ASSERT_EQ(2, full_indexer->GetKeyCount());
    ASSERT_EQ(1, peer_indexer->GetKeyCount());
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 1));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&request_context_, 1));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 1000);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kFullInstance, {"third-key"}, {10}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    // The admission wake must sample the targeted (lexically last) instance
    // immediately rather than waiting one full idle interval for rotation.
    ASSERT_TRUE(WaitUntil([&]() { return full_indexer->GetKeyCount() == 1; }, std::chrono::milliseconds(500)));
    EXPECT_EQ(1, peer_indexer->GetKeyCount());
    auto [peer_get_ec, peer_values] = manager_->Get(&request_context_, kPeerInstance, {"peer-oldest"});
    ASSERT_EQ(EC_OK, peer_get_ec);
    ASSERT_EQ(1, peer_values.size());
    EXPECT_TRUE(peer_values.front().found);

    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kFullInstance, {"third-key"}, {10}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kFullInstance, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerUsesTargetedKeyEvictionToAlsoSatisfyGroupAndTypePressure) {
    constexpr const char *kGroup = "reclaim-overlapping-pressure-group";
    constexpr const char *kFullInstance = "reclaim-overlapping-pressure-full-instance";
    constexpr const char *kPeerInstance = "reclaim-overlapping-pressure-peer-instance";
    CreateReclaimGroup(kGroup, kFullInstance, 100, 0.9, 0, 2);
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, kGroup, kPeerInstance, "reclaim-test").first);

    // The peer is globally oldest, but retiring it would satisfy only byte
    // pressure. Retiring one object from the full instance satisfies the
    // targeted key demand and the group/type byte pressure at the same time.
    CommitObject(kPeerInstance, "peer-oldest", 35);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kFullInstance, "full-old", 30);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kFullInstance, "full-new", 30);

    auto full_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kFullInstance));
    auto peer_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kPeerInstance));
    ASSERT_TRUE(full_indexer);
    ASSERT_TRUE(peer_indexer);
    cache_manager_->PauseReclaimer();
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kFullInstance, {"third-key"}, {1}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());

    cache_manager_->ResumeReclaimer();
    ASSERT_TRUE(WaitUntil([&]() { return full_indexer->GetKeyCount() == 1; }, std::chrono::seconds(2)));
    EXPECT_EQ(1, peer_indexer->GetKeyCount());
    EXPECT_EQ(35, peer_indexer->GetStorageUsage());
    auto [peer_get_ec, peer_values] = manager_->Get(&request_context_, kPeerInstance, {"peer-oldest"});
    ASSERT_EQ(EC_OK, peer_get_ec);
    ASSERT_EQ(1, peer_values.size());
    EXPECT_TRUE(peer_values.front().found);
}

TEST_F(KvMetaManagerTest, ImpossibleAdmissionDemandDoesNotEvictUsefulCacheEntries) {
    constexpr const char *kGroup = "reclaim-impossible-admission-group";
    constexpr const char *kInstance = "reclaim-impossible-admission-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 1.0, 0);
    CommitObject(kInstance, "must-stay", 70);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"cannot-fit"}, {101}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(70, indexer->GetStorageUsage());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"must-stay"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, ImpossibleStorageTypeAdmissionDoesNotEvictUsefulCacheEntries) {
    constexpr const char *kGroup = "reclaim-impossible-type-admission-group";
    constexpr const char *kInstance = "reclaim-impossible-type-admission-instance";
    CreateReclaimGroup(kGroup, kInstance, 1000, 1.0, 0, MetaIndexerConfig::kDefaultMaxKeyCount, 100);
    CommitObject(kInstance, "must-stay", 70);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"cannot-fit-type"}, {101}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(70, indexer->GetStorageUsage());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"must-stay"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, ImpossibleKeyAdmissionDemandDoesNotEvictUsefulCacheEntries) {
    constexpr const char *kGroup = "reclaim-impossible-key-admission-group";
    constexpr const char *kInstance = "reclaim-impossible-key-admission-instance";
    CreateReclaimGroup(kGroup, kInstance, 1000, 1.0, 0, 2);
    CommitObject(kInstance, "must-stay", 10);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [blocked_ec, blocked] =
        manager_->StartWrite(&request_context_, kInstance, {"new-a", "new-b", "new-c"}, {1, 1, 1}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(1, indexer->GetKeyCount());
    EXPECT_EQ(10, indexer->GetStorageUsage());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"must-stay"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, ReclaimerSupportsAZeroWatermark) {
    constexpr const char *kGroup = "reclaim-zero-watermark-group";
    constexpr const char *kInstance = "reclaim-zero-watermark-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.0, 0);
    CommitObject(kInstance, "remove-at-zero", 10);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerSharesTheOperationalPauseSwitch) {
    constexpr const char *kGroup = "reclaim-pause-group";
    constexpr const char *kInstance = "reclaim-pause-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, "paused-object", 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    cache_manager_->PauseReclaimer();
    ASSERT_TRUE(manager_->ResumeMaintenance());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(90, indexer->GetStorageUsage());

    cache_manager_->ResumeReclaimer();
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerRotatesAOneKeySampleBudgetAcrossInstances) {
    constexpr const char *kGroup = "reclaim-sample-rotation-group";
    constexpr const char *kActiveInstance = "a-reclaim-active-instance";
    constexpr const char *kCommittedInstance = "z-reclaim-committed-instance";
    CreateReclaimGroup(kGroup, kActiveInstance, 100, 0.8, 0);
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, kGroup, kCommittedInstance, "reclaim-test").first);

    auto [active_ec, active] = manager_->StartWrite(&request_context_, kActiveInstance, {"active"}, {45}, 30);
    ASSERT_EQ(EC_OK, active_ec);
    CommitObject(kCommittedInstance, "committed", 45);

    auto active_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kActiveInstance));
    auto committed_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kCommittedInstance));
    ASSERT_TRUE(active_indexer);
    ASSERT_TRUE(committed_indexer);
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 1));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&request_context_, 1));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    ASSERT_TRUE(WaitUntil([&]() { return committed_indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    EXPECT_EQ(45, active_indexer->GetStorageUsage());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kActiveInstance, active.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerRejectsAnOverlongBackendSampleWithoutDeleting) {
    constexpr const char *kGroup = "reclaim-overlong-sample-group";
    constexpr const char *kInstance = "reclaim-overlong-sample-instance";
    constexpr const char *kKey = "must-not-be-deleted";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    ASSERT_NE(nullptr, InstallOversamplingBackend(kInstance));
    CommitObject(kInstance, kKey, 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 1));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return metrics_registry_->GetCounter("kv_meta_reclaimer.error_count").Get() != 0; },
                          std::chrono::seconds(2)));
    cache_manager_->PauseReclaimer();

    EXPECT_EQ(90, indexer->GetStorageUsage());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.retired_object_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, ReclaimerIsolatesACorruptCandidateAndStillMakesProgress) {
    constexpr const char *kGroup = "reclaim-corrupt-candidate-group";
    constexpr const char *kInstance = "reclaim-corrupt-candidate-instance";
    constexpr const char *kCorruptKey = "corrupt-object";
    constexpr const char *kValidKey = "valid-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.5, 0);
    CommitObject(kInstance, kCorruptKey, 30);
    CommitObject(kInstance, kValidKey, 30);
    MutateObject(kInstance, kCorruptKey, [](CacheLocation &location) {
        auto &spec = location.mutable_location_specs().front();
        spec.set_uri(spec.uri() + "&size=30");
    });

    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 10));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&request_context_, 10));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(
        [&]() {
            auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kValidKey});
            return get_ec == EC_OK && values.size() == 1 && !values.front().found;
        },
        std::chrono::seconds(2)));
    cache_manager_->PauseReclaimer();

    EXPECT_NE(0, metrics_registry_->GetCounter("kv_meta_reclaimer.error_count").Get());
    EXPECT_NE(0, metrics_registry_->GetCounter("kv_meta_reclaimer.reclaimed_object_count").Get());
    EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstance, {kCorruptKey}).first);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsAnUnsupportedReclaimPolicy) {
    constexpr const char *kGroup = "reclaim-unsupported-policy-group";
    constexpr const char *kInstance = "reclaim-unsupported-policy-instance";
    CreateReclaimGroup(kGroup,
                       kInstance,
                       100,
                       0.8,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_TTL,
                       EC_CONFIG_ERROR);
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, manager_->GetInstanceInfo(&request_context_, kInstance).first);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsInvalidReclaimBounds) {
    CreateReclaimGroup("reclaim-watermark-negative-group",
                       "reclaim-watermark-negative-instance",
                       100,
                       -0.01,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-watermark-high-group",
                       "reclaim-watermark-high-instance",
                       100,
                       1.01,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-watermark-infinity-group",
                       "reclaim-watermark-infinity-instance",
                       100,
                       std::numeric_limits<double>::infinity(),
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-watermark-nan-group",
                       "reclaim-watermark-nan-instance",
                       100,
                       std::numeric_limits<double>::quiet_NaN(),
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-negative-delay-group",
                       "reclaim-negative-delay-instance",
                       100,
                       0.8,
                       -1,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-excessive-delay-group",
                       "reclaim-excessive-delay-instance",
                       100,
                       0.8,
                       1'800'001,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsStorageWithoutExactObjectOwnership) {
    constexpr const char *kStorage = "external-event-report";
    auto event_spec = std::make_shared<EventReportStorageSpec>();
    StorageConfig event_config(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, kStorage, event_spec);
    ASSERT_EQ(EC_OK,
              registry_manager_->data_storage_manager()->RegisterStorage(&request_context_, kStorage, event_config));

    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup object_group(*default_group);
    object_group.set_name("event-report-object-group");
    object_group.set_global_quota_group_name("event-report-object-quota");
    object_group.set_storage_candidates({kStorage});
    object_group.set_quota(
        InstanceGroupQuota(100, {QuotaConfig(100, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5)}));
    object_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));

    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_
            ->RegisterInstance(&request_context_, object_group.name(), "event-report-object-instance", "reclaim-test")
            .first);

    constexpr const char *kHotGroup = "hot-storage-object-group";
    constexpr const char *kHotInstance = "hot-storage-object-instance";
    CreateReclaimGroup(kHotGroup, kHotInstance, 100, 0.8, 0);
    CommitObject(kHotInstance, "existing", 17);
    const auto [hot_group_ec, hot_group] = registry_manager_->GetInstanceGroup(&request_context_, kHotGroup);
    ASSERT_EQ(EC_OK, hot_group_ec);
    ASSERT_TRUE(hot_group);
    InstanceGroup updated_hot_group(*hot_group);
    updated_hot_group.set_storage_candidates({kStorage});
    updated_hot_group.set_version(hot_group->version() + 1);
    ASSERT_EQ(EC_OK,
              registry_manager_->UpdateInstanceGroup(&request_context_, updated_hot_group, hot_group->version()));

    auto [hit_ec, hit] = manager_->StartWrite(&request_context_, kHotInstance, {"existing"}, {17}, 30);
    ASSERT_EQ(EC_OK, hit_ec);
    EXPECT_EQ((std::vector<bool>{true}), hit.key_mask);
    EXPECT_EQ(EC_CONFIG_ERROR, manager_->StartWrite(&request_context_, kHotInstance, {"new-object"}, {17}, 30).first);
    EXPECT_EQ(EC_OK, registry_manager_->data_storage_manager()->UnRegisterStorage(kStorage));
}

TEST_F(KvMetaManagerTest, RegistrationRejectsDuplicateStorageCandidates) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    ASSERT_FALSE(default_group->storage_candidates().empty());
    InstanceGroup duplicate_group(*default_group);
    duplicate_group.set_name("duplicate-storage-object-group");
    duplicate_group.set_global_quota_group_name("duplicate-storage-object-quota");
    duplicate_group.set_storage_candidates(
        {default_group->storage_candidates().front(), default_group->storage_candidates().front()});
    duplicate_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, duplicate_group));
    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_->RegisterInstance(&request_context_, duplicate_group.name(), "duplicate-storage-object-instance", "")
            .first);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsMissingStorageCandidates) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);

    const auto create_and_register = [&](const std::string &group_name,
                                         const std::string &instance_id,
                                         std::vector<std::string> storage_candidates) {
        InstanceGroup group(*default_group);
        group.set_name(group_name);
        group.set_global_quota_group_name(group_name + "-quota");
        group.set_storage_candidates(std::move(storage_candidates));
        group.set_version(1);
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, group));
        EXPECT_EQ(EC_CONFIG_ERROR,
                  manager_->RegisterInstance(&request_context_, group_name, instance_id, "reclaim-test").first);
    };

    create_and_register("empty-storage-object-group", "empty-storage-object-instance", {});
    create_and_register(
        "missing-storage-object-group", "missing-storage-object-instance", {"unregistered-kvmeta-storage"});
}

TEST_F(KvMetaManagerTest, ZeroReclaimerTuningBlocksOnlyNewAllocation) {
    CommitObject(kInstanceId, "existing", 17);
    const auto cache_reclaimer = cache_manager_->cache_reclaimer();
    ASSERT_NE(nullptr, cache_reclaimer);
    const auto original_sampling = cache_reclaimer->GetSamplingSize(&request_context_);
    const auto original_batching = cache_reclaimer->GetBatchingSize(&request_context_);

    ASSERT_EQ(EC_OK, cache_reclaimer->SetSamplingSize(&request_context_, 0));
    auto [hit_ec, hit] = manager_->StartWrite(&request_context_, kInstanceId, {"existing"}, {17}, 30);
    ASSERT_EQ(EC_OK, hit_ec);
    EXPECT_EQ((std::vector<bool>{true}), hit.key_mask);
    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->StartWrite(&request_context_, kInstanceId, {"sampling-disabled"}, {17}, 30).first);
    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->RegisterInstance(&request_context_, "default", "sampling-disabled-instance", "").first);

    ASSERT_EQ(EC_OK, cache_reclaimer->SetSamplingSize(&request_context_, original_sampling));
    ASSERT_EQ(EC_OK, cache_reclaimer->SetBatchingSize(&request_context_, 0));
    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->StartWrite(&request_context_, kInstanceId, {"batching-disabled"}, {17}, 30).first);

    // A broken GC knob closes admission, but never closes the operator's
    // cleanup path for objects already present in the cache.
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {"existing"}));
    ASSERT_EQ(EC_OK, cache_reclaimer->SetBatchingSize(&request_context_, original_batching));
}

TEST_F(KvMetaManagerTest, InvalidHotUpdatedReclaimPolicyBlocksOnlyNewAllocation) {
    constexpr const char *kGroup = "reclaim-hot-update-group";
    constexpr const char *kInstance = "reclaim-hot-update-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, "existing", 17);

    const auto [group_ec, current_group] = registry_manager_->GetInstanceGroup(&request_context_, kGroup);
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(current_group);
    ASSERT_TRUE(current_group->cache_config());
    ASSERT_TRUE(current_group->cache_config()->reclaim_strategy());
    auto updated_cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(updated_cache_config->FromJsonString(current_group->cache_config()->ToJsonString()));
    auto unsupported_strategy = std::make_shared<CacheReclaimStrategy>(*updated_cache_config->reclaim_strategy());
    unsupported_strategy->set_reclaim_policy(ReclaimPolicy::POLICY_TTL);
    updated_cache_config->set_reclaim_strategy(unsupported_strategy);
    InstanceGroup updated_group(*current_group);
    updated_group.set_cache_config(updated_cache_config);
    updated_group.set_version(current_group->version() + 1);
    ASSERT_EQ(EC_OK,
              registry_manager_->UpdateInstanceGroup(&request_context_, updated_group, current_group->version()));

    auto [hit_ec, hit] = manager_->StartWrite(&request_context_, kInstance, {"existing"}, {17}, 30);
    ASSERT_EQ(EC_OK, hit_ec);
    EXPECT_EQ((std::vector<bool>{true}), hit.key_mask);
    EXPECT_TRUE(hit.locations.empty());
    EXPECT_TRUE(hit.write_session_id.empty());

    auto [miss_ec, miss] = manager_->StartWrite(&request_context_, kInstance, {"new"}, {17}, 30);
    EXPECT_EQ(EC_CONFIG_ERROR, miss_ec);
    EXPECT_TRUE(miss.locations.empty());
    EXPECT_TRUE(miss.write_session_id.empty());
    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->RegisterInstance(&request_context_, kGroup, "another-instance", "reclaim-test").first);

    // Configuration failure closes admission, not cleanup: operators must
    // still be able to drain already committed cache objects safely.
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstance, {"existing"}));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"existing"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);
}

TEST_F(KvMetaManagerTest, RecoveryCompletesAReclaimerRetirementLeftByDemotion) {
    constexpr const char *kGroup = "reclaim-recovery-group";
    constexpr const char *kInstance = "reclaim-recovery-instance";
    constexpr const char *kKey = "retired-before-demotion";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 250);
    CommitObject(kInstance, kKey, 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto internal_key = KvMetaManager::InternalKey(kKey);
    const auto location_id = KvMetaManager::StableLocationId(kKey);
    const auto is_retired = [&]() {
        CacheLocationMapVector maps;
        const auto result = indexer->GetLocationMapsForMaintenance(&request_context_, {internal_key}, maps);
        if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
            return false;
        }
        const auto it = maps[0].find(location_id);
        return it != maps[0].end() && it->second && it->second->status() == CLS_DELETING;
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(is_retired, std::chrono::seconds(2)));
    manager_->CancelMaintenance();
    manager_->DoCleanup();

    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    ASSERT_TRUE(manager_->ResumeMaintenance());
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
