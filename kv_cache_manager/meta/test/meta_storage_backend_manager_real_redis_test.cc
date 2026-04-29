#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

// Dual-backend end-to-end test against a real redis service. This target is
// tagged "manual" + "redis" in BUILD so it is excluded from default
// `bazel test //...` runs; opt in with `--test_tag_filters=redis`.
class MetaStorageBackendManagerRealRedisTest : public TESTBASE {
public:
    void SetUp() override { request_context_ = std::make_shared<RequestContext>("test_trace_id"); }

    // URI with persistent=redis + cache=local forces dual-backend mode.
    static std::shared_ptr<MetaStorageBackendConfig> MakeDualConfig() {
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
        config->SetStorageUri(
            "redis://test_redis_user:test_redis_password@localhost:6379/"
            "?client_max_pool_size=4&persistent_type=redis&cache_type=local");
        return config;
    }

    // URI without persistent_type/cache_type params -> single-backend mode
    // routed straight to the real redis instance.
    static std::shared_ptr<MetaStorageBackendConfig> MakeSingleConfig() {
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
        config->SetStorageUri(
            "redis://test_redis_user:test_redis_password@localhost:6379/?client_max_pool_size=4");
        return config;
    }

    static CacheLocation MakeLocation(const std::string &id, const std::string &uri) {
        CacheLocation loc;
        loc.set_id(id);
        loc.set_status(CacheLocationStatus::CLS_SERVING);
        loc.set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        loc.set_spec_size(1);
        std::vector<LocationSpec> specs;
        specs.emplace_back("default", uri);
        loc.set_location_specs(std::move(specs));
        return loc;
    }

    static BatchMetaData MakeBatch(const KeyVector &keys) {
        BatchMetaData batch;
        batch.batch_keys = keys;
        batch.batch_indexs.reserve(keys.size());
        batch.batch_locations.resize(keys.size());
        batch.batch_properties.resize(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            batch.batch_indexs.emplace_back(static_cast<int32_t>(i));
            const std::string loc_id = "loc_" + std::to_string(keys[i]);
            batch.batch_locations[i].emplace(loc_id, MakeLocation(loc_id, "uri_" + std::to_string(keys[i])));
            batch.batch_properties[i]["p0"] = "p0_" + std::to_string(keys[i]);
        }
        return batch;
    }

    // Mirror MetaStorageBackendManager::BuildEffectiveFieldMaps for tests that
    // bypass the manager and write straight to persistent_backend_. Without
    // this merge the persistent side would only carry block-level properties
    // and later GetLocations fallbacks would find no location fields -> the
    // caller's map::at / out-of-bounds access on the returned locations would
    // throw or read garbage.
    static void SerializeLocationsIntoProperties(BatchMetaData &batch) {
        if (batch.batch_locations.empty()) {
            return;
        }
        if (batch.batch_properties.empty()) {
            batch.batch_properties.resize(batch.batch_keys.size());
        }
        for (size_t i = 0; i < batch.batch_keys.size(); ++i) {
            for (const auto &loc_kv : batch.batch_locations[i]) {
                const CacheLocation &location = loc_kv.second;
                batch.batch_properties[i][LOCATION_PREFIX + location.id()] = location.ToJsonString();
            }
        }
    }

    static void WaitRunning(MetaStorageBackendManager &mgr) {
        for (int i = 0; i < 200; ++i) {
            if (mgr.GetRecoverState() == MetaStorageBackendManager::RecoverState::kRunning) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        FAIL() << "recover did not finish in time";
    }

    // Best-effort Delete; swallow NOENT so repeated runs against the same
    // redis instance stay idempotent even when a previous run aborted midway.
    static void Cleanup(MetaStorageBackendManager &mgr, const KeyVector &keys) {
        mgr.Delete(keys);
    }

protected:
    std::shared_ptr<RequestContext> request_context_;
};

// --- Dual-backend: Put + GetLocations round-trip against real redis ----------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestDualBackendPutAndGet) {
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_redis_dual", MakeDualConfig()));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    KeyVector keys = {10001, 10002};
    Cleanup(mgr, keys); // defensive: wipe any leftover from a prior aborted run
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    LocationMapVector out_locations;
    auto get_ecs = mgr.GetLocations(request_context_.get(), keys, out_locations);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), get_ecs);
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string loc_id = "loc_" + std::to_string(keys[i]);
        ASSERT_EQ(1u, out_locations[i].size());
        ASSERT_EQ("uri_" + std::to_string(keys[i]),
                  out_locations[i].at(loc_id).location_specs().front().uri());
    }

    Cleanup(mgr, keys);
    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Init: invalid URI types rejected against a live redis URI ---------------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestInitInvalidBackendTypesRejected) {
    // Unknown persistent_type -> EC_ERROR (factory cannot construct bogus).
    {
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
        config->SetStorageUri(
            "redis://test_redis_user:test_redis_password@localhost:6379/"
            "?client_max_pool_size=4&persistent_type=bogus&cache_type=local");
        MetaStorageBackendManager mgr;
        ASSERT_EQ(EC_ERROR, mgr.Init("inst_redis_bad_persistent", config));
    }
    // Unknown cache_type -> EC_ERROR.
    {
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
        config->SetStorageUri(
            "redis://test_redis_user:test_redis_password@localhost:6379/"
            "?client_max_pool_size=4&persistent_type=redis&cache_type=bogus");
        MetaStorageBackendManager mgr;
        ASSERT_EQ(EC_ERROR, mgr.Init("inst_redis_bad_cache", config));
    }
}

// --- Single-backend: end-to-end CRUD straight against redis ------------------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestSingleBackendCrud) {
    // No persistent_type/cache_type params -> single-backend mode, every op
    // goes straight to redis. Exercises the no-local branches inside Get /
    // Exists / GetLocations / Delete(location_ids)+reclaim.
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_redis_single", MakeSingleConfig()));
    ASSERT_EQ(EC_OK, mgr.Open());
    ASSERT_EQ(MetaStorageBackendManager::RecoverState::kRunning, mgr.GetRecoverState());
    ASSERT_FALSE(mgr.local_backend_);

    KeyVector keys = {20001, 20002};
    Cleanup(mgr, keys);
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    FieldMapVec field_maps;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Get(keys, {"p0"}, field_maps));
    ASSERT_EQ("p0_20001", field_maps[0].at("p0"));

    std::vector<bool> exists_vec;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Exists(keys, exists_vec));
    ASSERT_EQ((std::vector<bool>{true, true}), exists_vec);

    LocationMapVector out_locs;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              mgr.GetLocations(request_context_.get(), keys, out_locs));
    ASSERT_EQ("uri_20001", out_locs[0].at("loc_20001").location_specs().front().uri());

    // Delete the only location of each key -> MaybeReclaimEmptyKeys walks
    // through persistent (redis) to discover keys with no remaining locations.
    LocationIdsPerKey loc_ids = {{"loc_20001"}, {"loc_20002"}};
    int32_t reclaimed = 0;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Delete(keys, loc_ids, reclaimed));
    ASSERT_EQ(2, reclaimed);

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Exists(keys, exists_vec));
    ASSERT_EQ((std::vector<bool>{false, false}), exists_vec);

    Cleanup(mgr, keys);
    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Recover phase: local miss falls back to redis ---------------------------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestRecoverReadFallbackToPersistent) {
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_redis_recover_read", MakeDualConfig()));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Key 30001 dual-written via manager (seen by local + redis).
    KeyVector seeded = {30001};
    Cleanup(mgr, seeded);
    auto seeded_batch = MakeBatch(seeded);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), seeded_batch));

    // Key 30002 only in redis (bypass manager): simulates a pre-restart key
    // that has not yet been back-filled into local. Must serialize locations
    // into properties manually because we are bypassing the manager's
    // BuildEffectiveFieldMaps; otherwise the later GetLocations fallback
    // would observe no location fields.
    KeyVector extra = {30002};
    auto extra_batch = MakeBatch(extra);
    SerializeLocationsIntoProperties(extra_batch);
    {
        std::vector<ErrorCode> ecs =
            mgr.persistent_backend_->Put(extra_batch.batch_keys, extra_batch.batch_properties);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), ecs);
    }

    // Flip back to Recover so local-miss triggers the persistent fallback.
    mgr.recover_state_.store(MetaStorageBackendManager::RecoverState::kRecover,
                             std::memory_order_release);

    KeyVector keys = {30001, 30002, 30003};
    FieldMapVec fms;
    auto ecs = mgr.Get(keys, {"p0"}, fms);
    ASSERT_EQ(EC_OK, ecs[0]);
    ASSERT_EQ(EC_OK, ecs[1]);
    ASSERT_EQ("p0_30001", fms[0].at("p0"));
    ASSERT_EQ("p0_30002", fms[1].at("p0"));

    std::vector<bool> exists_vec;
    auto exists_ecs = mgr.Exists(keys, exists_vec);
    ASSERT_EQ(EC_OK, exists_ecs[0]);
    ASSERT_EQ(EC_OK, exists_ecs[1]);
    ASSERT_EQ((std::vector<bool>{true, true, false}), exists_vec);

    LocationIdsPerKey ids = {{"loc_30001"}, {"loc_30002"}};
    LocationsPerKey per_key_locs;
    auto per_ecs = mgr.GetLocations(request_context_.get(), KeyVector{30001, 30002}, ids, per_key_locs);
    ASSERT_EQ(EC_OK, per_ecs[0][0]);
    ASSERT_EQ(EC_OK, per_ecs[1][0]);
    ASSERT_EQ("uri_30002", per_key_locs[1][0].location_specs().front().uri());

    Cleanup(mgr, KeyVector{30001, 30002});
    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Recover phase: dual-write + Delete tombstone blocks backfill ------------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestRecoverWriteDualWriteAndDeleteTombstone) {
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_redis_recover_write", MakeDualConfig()));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Force Recover so Delete writes a tombstone and UpdateFields runs
    // EnsureKeyInLocal before the conditional write.
    mgr.recover_state_.store(MetaStorageBackendManager::RecoverState::kRecover,
                             std::memory_order_release);

    KeyVector keys = {40042};
    Cleanup(mgr, keys);

    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), batch));

    // Delete in Recover -> tombstone recorded.
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Delete(keys));
    {
        std::lock_guard<std::mutex> lock(mgr.deleted_keys_mutex_);
        ASSERT_EQ(1u, mgr.deleted_keys_.count(40042));
    }

    // Simulate a late backfill after Delete: BackfillKeysToLocal must see the
    // tombstone and refuse to reinsert the key into local.
    FieldMapVec stale_fms(1);
    stale_fms[0]["p0"] = "stale";
    ASSERT_EQ(0, mgr.BackfillKeysToLocal(keys, stale_fms, {EC_OK}));
    std::vector<bool> exists_vec;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.local_backend_->Exists(keys, exists_vec));
    ASSERT_FALSE(exists_vec[0]);

    // Key 40007 lives only in redis: UpdateFields under Recover must hydrate
    // it into local via EnsureKeyInLocal before the conditional write, so the
    // subsequent Get observes the update.
    KeyVector k7 = {40007};
    Cleanup(mgr, k7);
    auto batch7 = MakeBatch(k7);
    SerializeLocationsIntoProperties(batch7);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              mgr.persistent_backend_->Put(batch7.batch_keys, batch7.batch_properties));

    BatchMetaData update_batch;
    update_batch.batch_keys = k7;
    update_batch.batch_properties.resize(1);
    update_batch.batch_properties[0]["p0"] = "p0_40007_updated";
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.UpdateFields(request_context_.get(), update_batch));

    FieldMapVec fms;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Get(k7, {"p0"}, fms));
    ASSERT_EQ("p0_40007_updated", fms[0].at("p0"));

    Cleanup(mgr, k7);
    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Running phase: reads stay local-only, no redis fallback -----------------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestRunningReadLocalOnly) {
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_redis_running", MakeDualConfig()));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    KeyVector k1 = {50001};
    KeyVector k2 = {50002};
    Cleanup(mgr, KeyVector{50001, 50002});

    auto b1 = MakeBatch(k1);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), b1));

    // Bypass manager and write k2 only into redis; in Running state reads
    // must not see it (local-only path).
    auto b2 = MakeBatch(k2);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              mgr.persistent_backend_->Put(b2.batch_keys, b2.batch_properties));

    std::vector<bool> exists_vec;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              mgr.Exists(KeyVector{50001, 50002}, exists_vec));
    ASSERT_EQ((std::vector<bool>{true, false}), exists_vec);

    FieldMapVec fms;
    auto ecs = mgr.Get(KeyVector{50002}, {"p0"}, fms);
    ASSERT_TRUE(ecs[0] == EC_NOENT || (ecs[0] == EC_OK && fms[0].empty()));

    Cleanup(mgr, KeyVector{50001, 50002});
    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- GetLocations(keys, location_ids): per-id EC semantics -------------------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestGetLocationsPerLocationIdSemantics) {
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_redis_get_loc_ids", MakeDualConfig()));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    KeyVector keys = {60005, 60006};
    Cleanup(mgr, keys);
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    // Two ids per key: one existing, one non-existent -> {EC_OK, EC_NOENT}.
    LocationIdsPerKey ids = {{"loc_60005", "missing_loc"}, {"loc_60006", "missing_loc"}};
    LocationsPerKey out_locs;
    auto ecs = mgr.GetLocations(request_context_.get(), keys, ids, out_locs);
    ASSERT_EQ(2u, ecs.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}), ecs[0]);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}), ecs[1]);
    ASSERT_EQ("uri_60005", out_locs[0][0].location_specs().front().uri());
    ASSERT_EQ("uri_60006", out_locs[1][0].location_specs().front().uri());

    Cleanup(mgr, keys);
    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Cross-batch APIs: ListKeys + RandomSample -------------------------------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestListKeysAndRandomSample) {
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_redis_list", MakeDualConfig()));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    KeyVector keys = {70100, 70200, 70300};
    Cleanup(mgr, keys);
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    // ListKeys must eventually surface every inserted key. Other tenants may
    // share the redis instance, so we only assert inclusion (not equality).
    std::set<KeyType> seen;
    std::string cursor = SCAN_BASE_CURSOR;
    for (int i = 0; i < 50 && seen.size() < keys.size(); ++i) {
        std::string next;
        KeyTypeVec out;
        ASSERT_EQ(EC_OK, mgr.ListKeys(cursor, /*limit*/ 100, next, out));
        for (auto k : out) {
            seen.insert(k);
        }
        cursor = next;
        if (cursor == SCAN_BASE_CURSOR) {
            break;
        }
    }
    for (auto k : keys) {
        ASSERT_TRUE(seen.count(k) > 0) << "missing key=" << k;
    }

    KeyTypeVec sampled;
    ASSERT_EQ(EC_OK, mgr.RandomSample(/*count*/ 1, sampled));
    ASSERT_LE(sampled.size(), 1u);

    Cleanup(mgr, keys);
    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- PutMetaData / GetMetaData always routed to redis ------------------------

TEST_F(MetaStorageBackendManagerRealRedisTest, TestPutGetMetaData) {
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_redis_metadata", MakeDualConfig()));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    FieldMap input = {{"k1", "v1"}, {"k2", "v2"}};
    ASSERT_EQ(EC_OK, mgr.PutMetaData(input));
    FieldMap output;
    ASSERT_EQ(EC_OK, mgr.GetMetaData(output));
    ASSERT_EQ("v1", output["k1"]);
    ASSERT_EQ("v2", output["k2"]);

    ASSERT_EQ(EC_OK, mgr.Close());
}

} // namespace kv_cache_manager
