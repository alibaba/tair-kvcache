#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/test/redis_test_environment.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/meta_async_redis_backend.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_redis_backend.h"

namespace kv_cache_manager {
namespace {

// These barriers only control scheduling. Reads, queue admission, pipeline
// execution and Sync below all delegate to the production Redis code.
class OneShotBarrier {
public:
    void Arm() {
        std::lock_guard<std::mutex> lock(mutex_);
        armed_ = true;
    }

    void PauseIfArmed() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!armed_) {
            return;
        }
        armed_ = false;
        entered_ = true;
        cv_.notify_all();
        if (!cv_.wait_for(lock, std::chrono::seconds(10), [this] { return released_; })) {
            timed_out_ = true;
        }
    }

    bool WaitUntilEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [this] { return entered_; });
    }

    void Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        armed_ = false;
        released_ = true;
        cv_.notify_all();
    }

    bool TimedOut() {
        std::lock_guard<std::mutex> lock(mutex_);
        return timed_out_;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool armed_ = false;
    bool entered_ = false;
    bool released_ = false;
    bool timed_out_ = false;
};

struct RedisSchedule {
    OneShotBarrier snapshot;
    OneShotBarrier pipeline;
    std::string paused_command = "DEL";
    std::mutex sync_mutex;
    int sync_calls = 0;

    void ObserveSync() {
        std::lock_guard<std::mutex> lock(sync_mutex);
        ++sync_calls;
    }

    void ReleaseAll() {
        pipeline.Release();
        snapshot.Release();
    }
};

class ScheduledRedisClient : public RedisClient {
public:
    ScheduledRedisClient(const StandardUri &uri, std::shared_ptr<RedisSchedule> schedule)
        : RedisClient(uri), schedule_(std::move(schedule)) {}

    std::vector<ReplyUPtr> TryExecPipeline(const std::vector<CmdArgs> &commands) override {
        if (std::any_of(commands.begin(), commands.end(), [this](const CmdArgs &command) {
                return !command.empty() && command.front() == schedule_->paused_command;
            })) {
            schedule_->pipeline.PauseIfArmed();
        }
        return RedisClient::TryExecPipeline(commands);
    }

private:
    std::shared_ptr<RedisSchedule> schedule_;
};

template <typename Backend> class SnapshotRedisBackend : public Backend {
public:
    explicit SnapshotRedisBackend(std::shared_ptr<RedisSchedule> schedule) : schedule_(std::move(schedule)) {}

    std::vector<ErrorCode> Get(RequestContext *context,
                               const KeyTypeVec &keys,
                               CacheLocationMapVector &locations,
                               PropertyMapVector &properties) noexcept override {
        auto results = Backend::Get(context, keys, locations, properties);
        schedule_->snapshot.PauseIfArmed();
        return results;
    }

protected:
    std::shared_ptr<RedisSchedule> schedule_;
};

class ScheduledAsyncRedisBackend : public SnapshotRedisBackend<MetaAsyncRedisBackend> {
public:
    using SnapshotRedisBackend::SnapshotRedisBackend;

    bool Sync(const KeyTypeVec &keys) noexcept override {
        schedule_->ObserveSync();
        return MetaAsyncRedisBackend::Sync(keys);
    }

private:
    std::shared_ptr<RedisClient> CreateRedisClient() const override {
        return std::make_shared<ScheduledRedisClient>(storage_uri_, schedule_);
    }
};

} // namespace

class GcRecoverRedisFixture : public TESTBASE {
public:
    void SetUp() override {
        schedule_ = std::make_shared<RedisSchedule>();
        context_ = std::make_shared<RequestContext>("gc_recover_real_redis");
        instance_ = "gc_recover_" + GetCurrentTestName() + "_" + std::to_string(getpid()) + "_" +
                    std::to_string(TimestampUtil::GetCurrentTimeUs());
    }

    void TearDown() override {
        schedule_->ReleaseAll();
        // Release before destroying the manager: Close joins recovery and
        // consumer threads even when an ASSERT above aborted the test body.
        indexer_.reset();
        EXPECT_FALSE(schedule_->snapshot.TimedOut());
        EXPECT_FALSE(schedule_->pipeline.TimedOut());
        if (redis_) {
            const auto results = redis_->Delete(nullptr, cleanup_keys_);
            for (const auto result : results) {
                EXPECT_TRUE(result == EC_OK || result == EC_NOENT);
            }
            const auto result = redis_->DeleteMetaData();
            EXPECT_TRUE(result == EC_OK || result == EC_NOENT);
            EXPECT_EQ(EC_OK, redis_->Close());
        }
    }

protected:
    static CacheLocationConstPtr Location(const std::string &id, const std::string &uri = "old_uri") {
        auto location = std::make_shared<CacheLocation>();
        location->set_id(id);
        location->set_status(CLS_SERVING);
        location->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        location->set_spec_size(1);
        location->set_location_specs({LocationSpec("default", uri)});
        return location;
    }

    std::shared_ptr<MetaStorageBackendConfig> Config(const std::string &type,
                                                    bool memory_primary = false,
                                                    int queue_capacity = 100) {
        auto config = std::make_shared<MetaStorageBackendConfig>(type);
        config->SetMemoryPrimary(memory_primary);
        config->SetStorageUri(redis_test::Uri(
            "test_redis_user:test_redis_password",
            redis_test::kMetaStorageBackendManagerDb,
            "client_max_pool_size=8&timeout_ms=500&retry_count=0&persistent_type=async_redis&cache_type=local"
            "&async_queue_count=1&async_max_batch=32&async_wait_us=1000&async_drain_ms=2000"
            "&async_enqueue_timeout_ms=50&async_max_size=" +
                std::to_string(queue_capacity) + "&async_sync_timeout_ms=2000"));
        return config;
    }

    void Start(bool memory_primary,
               const KeyVector &keys,
               const CacheLocationMapVector &locations,
               bool pause_recovery = true,
               int queue_capacity = 100,
               bool async = true) {
        cleanup_keys_ = keys;
        // Reserve this fixture's auxiliary keys for exact cleanup as well.
        cleanup_keys_.insert(cleanup_keys_.end(), {43, 44, 45});
        redis_ = std::make_unique<MetaRedisBackend>();
        ASSERT_EQ(EC_OK, redis_->Init(instance_, Config(META_REDIS_BACKEND_TYPE_STR)));
        ASSERT_EQ(EC_OK, redis_->Open());
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK),
                  redis_->Put(nullptr, keys, locations, PropertyMapVector(keys.size())));

        // Init the real indexer's locks/config normally, then install the
        // schedule-controlled Redis subclass before opening its real manager.
        indexer_ = std::make_unique<MetaIndexer>();
        auto local_config = std::make_shared<MetaStorageBackendConfig>(META_LOCAL_BACKEND_TYPE_STR);
        ASSERT_EQ(EC_OK, indexer_->Init(instance_, std::make_shared<MetaIndexerConfig>(100, 1, 100, local_config)));
        auto manager = std::make_unique<MetaStorageBackendManager>();
        auto config = Config(META_CACHED_BACKEND_TYPE_STR, memory_primary, queue_capacity);
        if (!async) {
            config->SetStorageUri(redis_test::Uri("test_redis_user:test_redis_password",
                                                  redis_test::kMetaStorageBackendManagerDb,
                                                  "client_max_pool_size=8&persistent_type=redis&cache_type=local"));
        }
        ASSERT_EQ(EC_OK, manager->Init(instance_, config));
        std::unique_ptr<MetaStorageBackend> persistent;
        if (async) {
            auto backend = std::make_unique<ScheduledAsyncRedisBackend>(schedule_);
            async_ = backend.get();
            persistent = std::move(backend);
        } else {
            persistent = std::make_unique<SnapshotRedisBackend<MetaRedisBackend>>(schedule_);
        }
        ASSERT_EQ(EC_OK, persistent->Init(instance_, config));
        manager->persistent_backend_ = std::move(persistent);
        indexer_->backend_manager_ = std::move(manager);
        if (pause_recovery) {
            schedule_->snapshot.Arm();
        }
        ASSERT_EQ(EC_OK, Manager().Open());
        indexer_->key_count_.store(keys.size());
        if (pause_recovery) {
            ASSERT_TRUE(schedule_->snapshot.WaitUntilEntered());
            ASSERT_EQ(MetaStorageBackendManager::RecoverState::kRecover, Manager().GetRecoverState());
        } else {
            FinishRecovery();
        }
    }

    MetaStorageBackendManager &Manager() { return *indexer_->backend_manager_; }

    void HotPut(const KeyVector &keys, const CacheLocationMapVector &locations) {
        ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK),
                  Manager().cache_backend_->Put(nullptr, keys, locations, PropertyMapVector(keys.size())));
    }

    void FinishRecovery() {
        schedule_->snapshot.Release();
        if (Manager().recover_thread_.joinable()) {
            Manager().recover_thread_.join();
        }
        EXPECT_EQ(MetaStorageBackendManager::RecoverState::kRunning, Manager().GetRecoverState());
        EXPECT_TRUE(Manager().deleted_keys_.empty());
    }

    void ExpectLocations(MetaStorageBackend &backend, KeyType key, const CacheLocationMap &expected) {
        CacheLocationMapVector actual;
        ASSERT_EQ(std::vector<ErrorCode>{expected.empty() ? EC_NOENT : EC_OK},
                  backend.GetLocations(nullptr, {key}, actual));
        ASSERT_EQ(1u, actual.size());
        ASSERT_EQ(expected.size(), actual[0].size());
        for (const auto &[id, location] : expected) {
            ASSERT_EQ(1u, actual[0].count(id));
            EXPECT_EQ(location->ToJsonString(), actual[0].at(id)->ToJsonString());
        }
    }

    MetaIndexer::LocationResult Remove(const KeyVector &keys, const LocationIdsPerKey &ids) {
        return indexer_->ReadModifyWriteLocationsForMaintenance(
            context_.get(), keys, ids, [](const std::vector<ErrorCode> &ecs,
                                         const LocationIdVector &,
                                         size_t,
                                         CacheLocationVector &,
                                         PropertyMap &) -> LocationModifierResult { return {MA_DELETE, ecs}; });
    }

    void RunPausedDeleteRecreation(bool memory_primary, bool block_rmw) {
        const auto old = Location("L1");
        const auto added = Location("L2", "new_uri");
        ASSERT_NO_FATAL_FAILURE(Start(memory_primary, {42}, {{{"L1", old}}}));
        ASSERT_NO_FATAL_FAILURE(HotPut({42}, {{{"L1", old}}}));
        schedule_->pipeline.Arm();
        ASSERT_EQ(EC_OK, Remove({42}, {{"L1"}}).ec);
        ASSERT_TRUE(schedule_->pipeline.WaitUntilEntered());
        EXPECT_EQ(0u, indexer_->GetKeyCount());
        EXPECT_EQ(1u, Manager().deleted_keys_.count(42));
        ExpectLocations(*redis_, 42, {{"L1", old}}); // DEL has not reached Redis.
        ExpectLocations(*Manager().cache_backend_, 42, {});

        CacheLocationMapVector reads;
        EXPECT_EQ(std::vector<ErrorCode>{EC_NOENT}, Manager().GetLocations(nullptr, {42}, reads));
        std::vector<bool> exists;
        EXPECT_EQ(std::vector<ErrorCode>{EC_OK}, Manager().Exists(nullptr, {42}, exists));
        EXPECT_EQ(std::vector<bool>{false}, exists);
        LocationIdsPerKey ids;
        EXPECT_EQ(std::vector<ErrorCode>{EC_NOENT}, Manager().GetLocationIds(nullptr, {42}, ids));

        if (block_rmw) {
            const auto result = indexer_->ReadModifyWriteBlock(
                context_.get(), {42}, [added](const LocationIdVector &existing,
                                              ErrorCode ec,
                                              size_t,
                                              PropertyMap &,
                                              CacheLocationMap &locations) -> ModifierResult {
                    EXPECT_EQ(EC_NOENT, ec);
                    EXPECT_TRUE(existing.empty());
                    locations.emplace("L2", added);
                    return {MA_OK, EC_OK};
                });
            ASSERT_EQ(EC_OK, result.ec);
        } else {
            const auto result = indexer_->ReadModifyWriteTargetLocations(
                context_.get(), {42}, {{"L2"}}, [added](const std::vector<ErrorCode> &ecs,
                                                       const LocationIdVector &,
                                                       size_t,
                                                       CacheLocationVector &locations,
                                                       PropertyMap &) -> LocationModifierResult {
                    EXPECT_EQ(std::vector<ErrorCode>{EC_NOENT}, ecs);
                    EXPECT_EQ(nullptr, locations[0]);
                    locations[0] = added;
                    return {MA_OK, {EC_OK}};
                });
            ASSERT_EQ(EC_OK, result.ec);
        }
        EXPECT_EQ(1u, indexer_->GetKeyCount());
        ExpectLocations(*Manager().cache_backend_, 42, {{"L2", added}});
        ExpectLocations(*redis_, 42, {{"L1", old}});
        ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, Manager().GetLocations(nullptr, {42}, reads));
        EXPECT_EQ(0u, reads[0].count("L1"));
        EXPECT_EQ(1u, reads[0].count("L2"));
        LocationsPerKey targeted;
        EXPECT_EQ((std::vector<std::vector<ErrorCode>>{{EC_NOENT, EC_OK}}),
                  Manager().GetLocations(nullptr, {42}, {{"L1", "L2"}}, targeted));
        EXPECT_EQ(std::vector<ErrorCode>{EC_OK}, Manager().EnsureCacheForGc(nullptr, {42}));
        {
            std::lock_guard<std::mutex> lock(schedule_->sync_mutex);
            EXPECT_EQ(0, schedule_->sync_calls); // Neither maintenance nor business adds a barrier.
        }

        // Normal cold-key hydration must still work: only the deleted key is
        // suppressed. This independent Redis key has no recovery tombstone.
        ASSERT_EQ(std::vector<ErrorCode>{EC_OK},
                  redis_->Put(nullptr, {43}, {{{"L1", old}}}, PropertyMapVector(1)));
        indexer_->key_count_.store(2); // Include the separately seeded existing key.
        const auto cold_result = indexer_->ReadModifyWriteTargetLocations(
            context_.get(), {43}, {{"L2"}}, [added](const std::vector<ErrorCode> &ecs,
                                                   const LocationIdVector &,
                                                   size_t,
                                                   CacheLocationVector &locations,
                                                   PropertyMap &) -> LocationModifierResult {
                EXPECT_EQ(std::vector<ErrorCode>{EC_NOENT}, ecs);
                locations[0] = added;
                return {MA_OK, {EC_OK}};
            });
        ASSERT_EQ(EC_OK, cold_result.ec);
        EXPECT_EQ(2u, indexer_->GetKeyCount());
        EXPECT_EQ(0u, Manager().deleted_keys_.count(43));
        ExpectLocations(*Manager().cache_backend_, 43, {{"L1", old}, {"L2", added}});
        schedule_->pipeline.Release();
        ASSERT_TRUE(Manager().Sync({42, 43}));
        ExpectLocations(*redis_, 42, {{"L2", added}});
        ExpectLocations(*redis_, 43, {{"L1", old}, {"L2", added}});
        FinishRecovery(); // The real background task now receives the old L1 snapshot.
        ExpectLocations(*Manager().cache_backend_, 42, {{"L2", added}});
        ExpectLocations(*Manager().cache_backend_, 43, {{"L1", old}, {"L2", added}});
        EXPECT_EQ(2u, indexer_->GetKeyCount());
    }

    std::shared_ptr<RedisSchedule> schedule_;
    std::shared_ptr<RequestContext> context_;
    std::string instance_;
    KeyVector cleanup_keys_;
    std::unique_ptr<MetaRedisBackend> redis_;
    std::unique_ptr<MetaIndexer> indexer_;
    ScheduledAsyncRedisBackend *async_ = nullptr;
};

class GcRecoverRealRedisTest : public GcRecoverRedisFixture, public ::testing::WithParamInterface<bool> {};

TEST_P(GcRecoverRealRedisTest, NormalOpenRecoversAndRunningWritesReachRedis) {
    const auto old = Location("L1");
    const auto added = Location("L2", "new_uri");
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42}, {{{"L1", old}}}, false));
    ExpectLocations(*Manager().cache_backend_, 42, {{"L1", old}});
    BatchMetaData batch;
    batch.batch_keys = {42};
    batch.batch_locations = {{{"L2", added}}};
    batch.batch_properties.resize(1);
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, Manager().Upsert(context_.get(), batch));
    ASSERT_TRUE(Manager().Sync({42}));
    ExpectLocations(*Manager().cache_backend_, 42, {{"L1", old}, {"L2", added}});
    ExpectLocations(*redis_, 42, {{"L1", old}, {"L2", added}});

    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, redis_->Put(nullptr, {43}, {{{"L1", old}}}, PropertyMapVector(1)));
    CacheLocationMapVector locations;
    EXPECT_EQ(std::vector<ErrorCode>{EC_NOENT}, Manager().GetLocations(nullptr, {43}, locations));
}

TEST_P(GcRecoverRealRedisTest, PausedDeleteAllowsTargetedRmwWithoutRevivingOldLocation) {
    RunPausedDeleteRecreation(GetParam(), false);
}

TEST_P(GcRecoverRealRedisTest, PausedDeleteAllowsBlockRmwWithoutRevivingOldLocation) {
    RunPausedDeleteRecreation(GetParam(), true);
}

TEST_P(GcRecoverRealRedisTest, WholeKeyDeletionBlocksDelayedRealRecoverySnapshot) {
    const auto old = Location("L1");
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42}, {{{"L1", old}}}));
    ASSERT_NO_FATAL_FAILURE(HotPut({42}, {{{"L1", old}}}));
    ASSERT_EQ(EC_OK, Remove({42}, {{"L1"}}).ec);
    EXPECT_EQ(0u, indexer_->GetKeyCount());
    EXPECT_EQ(1u, Manager().deleted_keys_.count(42));
    ASSERT_TRUE(Manager().Sync({42}));
    ExpectLocations(*redis_, 42, {});
    FinishRecovery();
    ExpectLocations(*Manager().cache_backend_, 42, {});
    EXPECT_EQ(0u, indexer_->GetKeyCount());
}

TEST_P(GcRecoverRealRedisTest, PartialDeletePreservesSiblingAndSkipsOnlyColdKey) {
    const auto target = Location("L1");
    const auto sibling = Location("L2", "keep_uri");
    const CacheLocationMap both{{"L1", target}, {"L2", sibling}};
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42, 43, 44}, {both, both, {{"L1", target}}}));
    ASSERT_NO_FATAL_FAILURE(HotPut({43, 44}, {both, {{"L1", target}}}));
    std::vector<int64_t> before;
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK},
              Manager().cache_backend_->GetLastAccessTimesForMaintenance(nullptr, {43}, before));
    const auto result = Remove({42, 43, 44}, {{"L1"}, {"L1"}, {"L1"}});
    EXPECT_EQ(EC_OK, result.ec);
    EXPECT_EQ((std::vector<std::vector<ErrorCode>>{{EC_NOENT}, {EC_OK}, {EC_OK}}),
              result.per_location_error_codes);
    EXPECT_EQ(2u, indexer_->GetKeyCount());
    EXPECT_EQ(0u, Manager().deleted_keys_.count(42));
    EXPECT_EQ(0u, Manager().deleted_keys_.count(43));
    EXPECT_EQ(1u, Manager().deleted_keys_.count(44));
    std::vector<int64_t> after;
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK},
              Manager().cache_backend_->GetLastAccessTimesForMaintenance(nullptr, {43}, after));
    EXPECT_EQ(before, after);
    ASSERT_TRUE(Manager().Sync({42, 43, 44}));
    ExpectLocations(*redis_, 42, both);
    ExpectLocations(*redis_, 43, {{"L2", sibling}});
    ExpectLocations(*redis_, 44, {});
    ExpectLocations(*Manager().cache_backend_, 42, {});
    ExpectLocations(*Manager().cache_backend_, 43, {{"L2", sibling}});
    FinishRecovery();
    ExpectLocations(*Manager().cache_backend_, 42, both);
    ExpectLocations(*Manager().cache_backend_, 43, {{"L2", sibling}});
    ExpectLocations(*Manager().cache_backend_, 44, {});
    ASSERT_EQ(EC_OK, Remove({42}, {{"L1"}}).ec); // Deferred key is eligible next round.
    ASSERT_TRUE(Manager().Sync({42}));
    ExpectLocations(*redis_, 42, {{"L2", sibling}});
    EXPECT_EQ(2u, indexer_->GetKeyCount());
}

TEST_P(GcRecoverRealRedisTest, HotUpdateRejectsOldExpectedValueWhileRedisConsumptionIsPaused) {
    const auto old = Location("L1");
    const auto replacement = Location("L1", "new_uri");
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42}, {{{"L1", old}}}));
    ASSERT_NO_FATAL_FAILURE(HotPut({42}, {{{"L1", old}}}));
    schedule_->paused_command = "HSET";
    schedule_->pipeline.Arm();
    const auto updated = indexer_->ReadModifyWriteTargetLocations(
        context_.get(), {42}, {{"L1"}}, [replacement](const std::vector<ErrorCode> &ecs,
                                                     const LocationIdVector &,
                                                     size_t,
                                                     CacheLocationVector &locations,
                                                     PropertyMap &) -> LocationModifierResult {
            locations[0] = replacement;
            return {MA_OK, ecs};
        });
    ASSERT_EQ(EC_OK, updated.ec);
    ASSERT_TRUE(schedule_->pipeline.WaitUntilEntered());
    auto result = std::async(std::launch::async, [this, old] {
        return indexer_->ReadModifyWriteLocationsForMaintenance(
            context_.get(), {42}, {{"L1"}}, [old](const std::vector<ErrorCode> &ecs,
                                                  const LocationIdVector &,
                                                  size_t,
                                                  CacheLocationVector &locations,
                                                  PropertyMap &) -> LocationModifierResult {
                EXPECT_EQ(std::vector<ErrorCode>{EC_OK}, ecs);
                return {MA_DELETE, {locations[0]->ToJsonString() == old->ToJsonString() ? EC_OK : EC_MISMATCH}};
            });
    });
    const auto ready = result.wait_for(std::chrono::seconds(1));
    EXPECT_EQ(std::future_status::ready, ready);
    // Always release before joining, including a regression that blocks Sync.
    schedule_->pipeline.Release();
    const auto removed = result.get();
    EXPECT_EQ(EC_OK, removed.ec);
    EXPECT_EQ((std::vector<std::vector<ErrorCode>>{{EC_MISMATCH}}), removed.per_location_error_codes);
    EXPECT_TRUE(Manager().deleted_keys_.empty());
    EXPECT_EQ(1u, indexer_->GetKeyCount());
    {
        std::lock_guard<std::mutex> lock(schedule_->sync_mutex);
        EXPECT_EQ(0, schedule_->sync_calls);
    }
    ASSERT_TRUE(Manager().Sync({42}));
    ExpectLocations(*redis_, 42, {{"L1", replacement}});
    ExpectLocations(*Manager().cache_backend_, 42, {{"L1", replacement}});
}

TEST_P(GcRecoverRealRedisTest, LastHotLocationReclaimsKeyWhileRedisStillHasQueuedSibling) {
    const auto first = Location("L1");
    const auto second = Location("L2", "second_uri");
    const CacheLocationMap both{{"L1", first}, {"L2", second}};
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42}, {both}));
    ASSERT_NO_FATAL_FAILURE(HotPut({42}, {both}));
    schedule_->paused_command = "HDEL";
    schedule_->pipeline.Arm();
    ASSERT_EQ(EC_OK, Remove({42}, {{"L2"}}).ec);
    ASSERT_TRUE(schedule_->pipeline.WaitUntilEntered());
    ExpectLocations(*Manager().cache_backend_, 42, {{"L1", first}});
    ExpectLocations(*redis_, 42, both);

    // The stale Redis sibling must not turn the final deletion into HDEL and
    // leave an empty hot key. No queue barrier is needed to decide from hot.
    const auto removed = Remove({42}, {{"L1"}});
    EXPECT_EQ(EC_OK, removed.ec);
    EXPECT_EQ((std::vector<std::vector<ErrorCode>>{{EC_OK}}), removed.per_location_error_codes);
    EXPECT_EQ(0u, indexer_->GetKeyCount());
    EXPECT_EQ(1u, Manager().deleted_keys_.count(42));
    ExpectLocations(*Manager().cache_backend_, 42, {});
    {
        std::lock_guard<std::mutex> lock(schedule_->sync_mutex);
        EXPECT_EQ(0, schedule_->sync_calls);
    }
    schedule_->pipeline.Release();
    ASSERT_TRUE(Manager().Sync({42}));
    ExpectLocations(*redis_, 42, {});
    FinishRecovery();
    ExpectLocations(*Manager().cache_backend_, 42, {});
}

TEST_P(GcRecoverRealRedisTest, OrdinaryGcDoesNotRestoreEventReportDeletedSiblingAcrossRecovery) {
    const auto first = Location("L1");
    const auto second = Location("L2", "second_uri");
    const CacheLocationMap both{{"L1", first}, {"L2", second}};
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42}, {both}));
    ASSERT_NO_FATAL_FAILURE(HotPut({42}, {both}));
    schedule_->paused_command = "HDEL";
    schedule_->pipeline.Arm();
    ASSERT_EQ(EC_OK, Remove({42}, {{"L1"}}).ec);
    ASSERT_TRUE(schedule_->pipeline.WaitUntilEntered());
    ExpectLocations(*Manager().cache_backend_, 42, {{"L2", second}});
    ExpectLocations(*redis_, 42, both);

    auto deleting = std::make_shared<CacheLocation>(*second);
    deleting->set_status(CLS_DELETING);
    const auto marked = indexer_->ReadModifyWriteLocation(
        context_.get(), {42}, {{"L2"}}, [second, deleting](const std::vector<ErrorCode> &ecs,
                                                         const LocationIdVector &,
                                                         size_t,
                                                         CacheLocationVector &locations,
                                                         PropertyMap &) -> LocationModifierResult {
            EXPECT_EQ(std::vector<ErrorCode>{EC_OK}, ecs);
            EXPECT_EQ(second->ToJsonString(), locations[0]->ToJsonString());
            locations[0] = deleting;
            return {MA_OK, {EC_OK}};
        }, true, true);
    EXPECT_EQ(EC_OK, marked.ec);
    EXPECT_EQ((std::vector<std::vector<ErrorCode>>{{EC_OK}}), marked.per_location_error_codes);
    ExpectLocations(*Manager().cache_backend_, 42, {{"L2", deleting}});
    ExpectLocations(*redis_, 42, both);
    EXPECT_TRUE(Manager().deleted_keys_.empty()); // Partial deletion does not need a key tombstone.
    {
        std::lock_guard<std::mutex> lock(schedule_->sync_mutex);
        EXPECT_EQ(0, schedule_->sync_calls);
    }

    FinishRecovery(); // Publishing Running does not drain pending HDEL/HSET.
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, Manager().EnsureCacheForGc(nullptr, {42}));
    ExpectLocations(*Manager().cache_backend_, 42, {{"L2", deleting}});
    schedule_->pipeline.Release();
    ASSERT_TRUE(Manager().Sync({42})); // The ordinary GC's existing physical-delete barrier.
    ExpectLocations(*redis_, 42, {{"L2", deleting}});
    EXPECT_EQ(1u, indexer_->GetKeyCount());
}

TEST_P(GcRecoverRealRedisTest, OrdinaryGcHydratesCompleteColdRecoverKeyBeforeCheckingTarget) {
    const auto first = Location("L1");
    const auto second = Location("L2", "second_uri");
    const CacheLocationMap both{{"L1", first}, {"L2", second}};
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42}, {both}));
    ExpectLocations(*Manager().cache_backend_, 42, {});
    auto deleting = std::make_shared<CacheLocation>(*second);
    deleting->set_status(CLS_DELETING);
    const auto marked = indexer_->ReadModifyWriteLocation(
        context_.get(), {42}, {{"L2"}}, [second, deleting](const std::vector<ErrorCode> &ecs,
                                                         const LocationIdVector &,
                                                         size_t,
                                                         CacheLocationVector &locations,
                                                         PropertyMap &) -> LocationModifierResult {
            EXPECT_EQ(std::vector<ErrorCode>{EC_OK}, ecs);
            EXPECT_EQ(second->ToJsonString(), locations[0]->ToJsonString());
            locations[0] = deleting;
            return {MA_OK, {EC_OK}};
        }, true, true);
    EXPECT_EQ(EC_OK, marked.ec);
    EXPECT_EQ((std::vector<std::vector<ErrorCode>>{{EC_OK}}), marked.per_location_error_codes);
    ExpectLocations(*Manager().cache_backend_, 42, {{"L1", first}, {"L2", deleting}});
    ASSERT_TRUE(Manager().Sync({42}));
    ExpectLocations(*redis_, 42, {{"L1", first}, {"L2", deleting}});
    FinishRecovery();
    ExpectLocations(*Manager().cache_backend_, 42, {{"L1", first}, {"L2", deleting}});
}

TEST_P(GcRecoverRealRedisTest, PersistentQueueAdmissionFailureKeepsHotCopyAndNoTombstone) {
    const auto old = Location("L1");
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42}, {{{"L1", old}}}, true, 1));
    ASSERT_NO_FATAL_FAILURE(HotPut({42}, {{{"L1", old}}}));
    schedule_->paused_command = "HSET";
    schedule_->pipeline.Arm();
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK},
              async_->Upsert(nullptr, {43}, {{{"L1", old}}}, PropertyMapVector(1)));
    ASSERT_TRUE(schedule_->pipeline.WaitUntilEntered());
    // The first write is in flight; the second occupies the sole queue slot.
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK},
              async_->Upsert(nullptr, {44}, {{{"L1", old}}}, PropertyMapVector(1)));
    int32_t reclaimed = -1;
    EXPECT_EQ(std::vector<ErrorCode>{EC_TIMEOUT},
              Manager().DeleteLocationsForMaintenance(
                  context_.get(), {42}, {{"L1"}}, reclaimed, Manager().UseLocalPrimaryForMaintenance()));
    EXPECT_EQ(0, reclaimed);
    EXPECT_TRUE(Manager().deleted_keys_.empty());
    ExpectLocations(*redis_, 42, {{"L1", old}});
    ExpectLocations(*Manager().cache_backend_, 42, {{"L1", old}});
    schedule_->pipeline.Release();
    ASSERT_TRUE(Manager().SyncAll());
}

TEST_P(GcRecoverRealRedisTest, MissingHotTargetIsSkippedEvenWhileRedisStillContainsIt) {
    const auto first = Location("L1");
    const auto second = Location("L2", "second_uri");
    const CacheLocationMap both{{"L1", first}, {"L2", second}};
    ASSERT_NO_FATAL_FAILURE(Start(GetParam(), {42}, {both}));
    ASSERT_NO_FATAL_FAILURE(HotPut({42}, {both}));
    schedule_->paused_command = "HDEL";
    schedule_->pipeline.Arm();
    ASSERT_EQ(EC_OK, Remove({42}, {{"L1"}}).ec);
    ASSERT_TRUE(schedule_->pipeline.WaitUntilEntered());
    const auto repeated = Remove({42}, {{"L1"}});
    EXPECT_EQ(EC_OK, repeated.ec);
    EXPECT_EQ((std::vector<std::vector<ErrorCode>>{{EC_NOENT}}), repeated.per_location_error_codes);
    ExpectLocations(*Manager().cache_backend_, 42, {{"L2", second}});
    ExpectLocations(*redis_, 42, both);
    EXPECT_TRUE(Manager().deleted_keys_.empty());
    EXPECT_EQ(1u, indexer_->GetKeyCount());
    schedule_->pipeline.Release();
    ASSERT_TRUE(Manager().Sync({42}));
    ExpectLocations(*redis_, 42, {{"L2", second}});
}

INSTANTIATE_TEST_SUITE_P(MemoryPrimaryModes, GcRecoverRealRedisTest, ::testing::Bool());

TEST_F(GcRecoverRedisFixture, SynchronousRedisNoentStillRemovesHotCopyAndMarksRecovery) {
    const auto old = Location("L1");
    ASSERT_NO_FATAL_FAILURE(Start(false, {42}, {{{"L1", old}}}, true, 100, false));
    ASSERT_NO_FATAL_FAILURE(HotPut({42}, {{{"L1", old}}}));
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, redis_->Delete(nullptr, {42}));
    EXPECT_EQ(EC_OK, Remove({42}, {{"L1"}}).ec);
    EXPECT_EQ(0u, indexer_->GetKeyCount());
    EXPECT_EQ(1u, Manager().deleted_keys_.count(42));
    ExpectLocations(*redis_, 42, {});
    ExpectLocations(*Manager().cache_backend_, 42, {});
    FinishRecovery();
    ExpectLocations(*Manager().cache_backend_, 42, {});
}

} // namespace kv_cache_manager
