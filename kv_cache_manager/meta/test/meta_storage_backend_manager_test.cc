#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_dummy_backend.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

class MetaStorageBackendManagerTest : public TESTBASE {
public:
    void SetUp() override { request_context_ = std::make_shared<RequestContext>("test_trace_id"); }

    // Build a dual-backend config URI with persistent=dummy (file-backed) so
    // the test does not depend on a running redis service. The dummy backend
    // persists to `path` which is cleaned up between test cases via
    // GetPrivateTestRuntimeDataPath().
    std::shared_ptr<MetaStorageBackendConfig> MakeDualConfig(const std::string &path) {
        auto config = std::make_shared<MetaStorageBackendConfig>();
        // storage_type must be "cached" so that Init enters the dual-backend
        // code path which parses persistent_type/cache_type from the URI.
        config->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
        config->SetStorageUri("file://" + path + "?persistent_type=dummy&cache_type=local");
        return config;
    }

    std::shared_ptr<MetaStorageBackendConfig> MakeSingleConfig(const std::string &path) {
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageType(META_DUMMY_BACKEND_TYPE_STR);
        config->SetStorageUri("file://" + path);
        return config;
    }

    // Construct a single-location CacheLocation with id/uri wired up so the
    // round-trip through JSON can be asserted.
    static CacheLocationConstPtr MakeLocation(const std::string &id, const std::string &uri) {
        auto loc = std::make_shared<CacheLocation>();
        loc->set_id(id);
        loc->set_status(CacheLocationStatus::CLS_SERVING);
        loc->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        loc->set_spec_size(1);
        std::vector<LocationSpec> specs;
        specs.emplace_back("default", uri);
        loc->set_location_specs(std::move(specs));
        return loc;
    }

    // Build a BatchMetaData for every key in `keys` with one location each.
    // Populates both batch_locations (to exercise BuildEffectiveFieldMaps) and
    // a block-level property so the Put path touches both code branches.
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
    // caller's map::at on the returned CacheLocationMapVector would throw.
    static void SerializeLocationsIntoProperties(BatchMetaData &batch) {
        if (batch.batch_locations.empty()) {
            return;
        }
        if (batch.batch_properties.empty()) {
            batch.batch_properties.resize(batch.batch_keys.size());
        }
        for (size_t i = 0; i < batch.batch_keys.size(); ++i) {
            for (const auto &[loc_id, loc_ptr] : batch.batch_locations[i]) {
                if (!loc_ptr)
                    continue;
                batch.batch_properties[i][LOCATION_PREFIX + loc_ptr->id()] = loc_ptr->ToJsonString();
            }
        }
    }

    // Spin until recover finishes (dual-backend only). The background thread
    // scans the (tiny) persistent store and flips to kRunning; a 1 s budget is
    // plenty under the dummy backend and keeps the test snappy.
    static void WaitRunning(MetaStorageBackendManager &mgr) {
        for (int i = 0; i < 100; ++i) {
            if (mgr.GetRecoverState() == MetaStorageBackendManager::RecoverState::kRunning) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        FAIL() << "recover did not finish in time";
    }

protected:
    std::shared_ptr<RequestContext> request_context_;
};

// --- Init / lifecycle ---------------------------------------------------------

TEST_F(MetaStorageBackendManagerTest, TestInitBadArgs) {
    MetaStorageBackendManager mgr;
    // empty instance_id and null config both rejected.
    ASSERT_EQ(EC_BADARGS, mgr.Init(/*instance_id*/ "", std::make_shared<MetaStorageBackendConfig>()));
    ASSERT_EQ(EC_BADARGS, mgr.Init(/*instance_id*/ "inst", nullptr));
}

TEST_F(MetaStorageBackendManagerTest, TestInitSingleBackend) {
    // No persistent_type/cache_type params in URI -> single-backend mode
    // (cache_backend_ stays null, recover_state goes straight to kRunning).
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_single";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_single", MakeSingleConfig(path)));
    ASSERT_TRUE(mgr.persistent_backend_);
    ASSERT_FALSE(mgr.cache_backend_);
    ASSERT_EQ(EC_OK, mgr.Open());
    ASSERT_EQ(MetaStorageBackendManager::RecoverState::kRunning, mgr.GetRecoverState());
    ASSERT_EQ(EC_OK, mgr.Close());
}

TEST_F(MetaStorageBackendManagerTest, TestInitDualBackend) {
    // URI params present -> dual-backend mode (persistent=dummy + local cache).
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_dual_init";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_dual", MakeDualConfig(path)));
    ASSERT_TRUE(mgr.persistent_backend_);
    ASSERT_TRUE(mgr.cache_backend_);
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);
    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Put/Get: CacheLocation serialization round-trip --------------------------

TEST_F(MetaStorageBackendManagerTest, TestPutAndGetLocationsRoundTrip) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_put_get";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_put", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    KeyVector keys = {1, 2, 3};
    auto batch = MakeBatch(keys);
    auto put_ecs = mgr.Put(request_context_.get(), batch);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), put_ecs);

    // New API stores locations separately from properties — verify locations
    // are populated in batch_locations (already set by MakeBatch) and that the
    // Put call did not corrupt them.
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string loc_id = "loc_" + std::to_string(keys[i]);
        ASSERT_EQ(1u, batch.batch_locations[i].size());
        ASSERT_TRUE(batch.batch_locations[i].count(loc_id) > 0)
            << "location missing in batch_locations for key=" << keys[i];
    }

    // GetLocations must deserialize back into the same (id, uri) pairs.
    CacheLocationMapVector out_locations;
    auto get_ecs = mgr.GetLocations(request_context_.get(), keys, out_locations);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), get_ecs);
    ASSERT_EQ(keys.size(), out_locations.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string loc_id = "loc_" + std::to_string(keys[i]);
        ASSERT_EQ(1u, out_locations[i].size());
        auto it = out_locations[i].find(loc_id);
        ASSERT_TRUE(it != out_locations[i].end());
        ASSERT_EQ("uri_" + std::to_string(keys[i]), it->second->location_specs().front().uri());
    }

    // Block-level properties should be preserved alongside the location fields.
    PropertyMapVector field_maps;
    auto field_ecs = mgr.GetProperties(nullptr, keys, {"p0"}, field_maps);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), field_ecs);
    for (size_t i = 0; i < keys.size(); ++i) {
        ASSERT_EQ("p0_" + std::to_string(keys[i]), field_maps[i].at("p0"));
    }

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Location-field Delete + empty-key reclamation ----------------------------

TEST_F(MetaStorageBackendManagerTest, TestDeleteLocationFieldsReclaimsEmptyKeys) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_delete_fields";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_del", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    KeyVector keys = {10, 20};
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    // Delete the sole location of each key -> keys become empty and should be
    // reclaimed by MaybeReclaimEmptyKeys.
    LocationIdsPerKey location_ids = {{"loc_10"}, {"loc_20"}};
    int32_t reclaimed = 0;
    auto del_ecs = mgr.Delete(nullptr, keys, location_ids, reclaimed);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), del_ecs);
    ASSERT_EQ(2, reclaimed);

    // After reclaim both keys must be gone.
    std::vector<bool> exists_vec;
    auto exists_ecs = mgr.Exists(nullptr, keys, exists_vec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), exists_ecs);
    ASSERT_EQ((std::vector<bool>{false, false}), exists_vec);

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Cross-batch APIs ---------------------------------------------------------

TEST_F(MetaStorageBackendManagerTest, TestListKeysAndRandomSample) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_listkeys";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_list", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    KeyVector keys = {100, 200, 300};
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    // ListKeys eventually surfaces every key.
    std::set<KeyType> seen;
    std::string cursor = SCAN_BASE_CURSOR;
    for (int i = 0; i < 20 && seen.size() < keys.size(); ++i) {
        std::string next;
        KeyTypeVec out;
        ASSERT_EQ(EC_OK, mgr.ListKeys(nullptr, cursor, /*limit*/ 50, next, out));
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

    // RandomSample should return at most `count` keys from the set above.
    KeyTypeVec sampled;
    ASSERT_EQ(EC_OK, mgr.RandomSample(nullptr, /*count*/ 1, sampled));
    ASSERT_LE(sampled.size(), 1u);

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- PutMetaData / GetMetaData always routed to persistent --------------------

TEST_F(MetaStorageBackendManagerTest, TestPutGetMetaData) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_metadata";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_meta", MakeDualConfig(path)));
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

// --- Init: invalid URI types rejected ----------------------------------------

TEST_F(MetaStorageBackendManagerTest, TestInitInvalidBackendTypesRejected) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_invalid";
    std::filesystem::remove(path);
    // Unknown persistent_type -> factory returns nullptr -> EC_ERROR.
    {
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
        config->SetStorageUri("file://" + path + "?persistent_type=bogus&cache_type=local");
        MetaStorageBackendManager mgr;
        ASSERT_EQ(EC_ERROR, mgr.Init("inst_bad_persistent", config));
    }
    // Unknown cache_type -> EC_ERROR (persistent constructed, local fails).
    {
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
        config->SetStorageUri("file://" + path + "?persistent_type=dummy&cache_type=bogus");
        MetaStorageBackendManager mgr;
        ASSERT_EQ(EC_ERROR, mgr.Init("inst_bad_cache", config));
    }
}

// --- Single-backend: end-to-end CRUD -----------------------------------------

TEST_F(MetaStorageBackendManagerTest, TestSingleBackendCrud) {
    // Single-backend mode has no local cache / recover; every op goes straight
    // to the persistent backend. Exercises Put -> Get -> GetLocations ->
    // Delete(location_ids) -> reclaim to cover the no-local branches inside
    // each API (esp. MaybeReclaimEmptyKeys falling back to persistent).
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_single_crud";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_single_crud", MakeSingleConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    ASSERT_EQ(MetaStorageBackendManager::RecoverState::kRunning, mgr.GetRecoverState());
    ASSERT_FALSE(mgr.cache_backend_);

    KeyVector keys = {1, 2};
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    PropertyMapVector field_maps;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.GetProperties(nullptr, keys, {"p0"}, field_maps));
    ASSERT_EQ("p0_1", field_maps[0].at("p0"));

    std::vector<bool> exists_vec;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Exists(nullptr, keys, exists_vec));
    ASSERT_EQ((std::vector<bool>{true, true}), exists_vec);

    CacheLocationMapVector out_locs;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.GetLocations(request_context_.get(), keys, out_locs));
    ASSERT_EQ("uri_1", out_locs[0].at("loc_1")->location_specs().front().uri());

    // Delete location field -> reclaim path resolves emptiness via persistent.
    LocationIdsPerKey loc_ids = {{"loc_1"}, {"loc_2"}};
    int32_t reclaimed = 0;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Delete(nullptr, keys, loc_ids, reclaimed));
    ASSERT_EQ(2, reclaimed);

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Exists(nullptr, keys, exists_vec));
    ASSERT_EQ((std::vector<bool>{false, false}), exists_vec);

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Recover phase: reads fall back to persistent when local misses ----------

TEST_F(MetaStorageBackendManagerTest, TestRecoverReadFallbackToPersistent) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_recover_read";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_recover_read", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Key 1 dual-write via manager (present in both local + persistent).
    KeyVector seeded = {1};
    auto batch = MakeBatch(seeded);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), batch));

    // Key 2 only in persistent: write via persistent_backend_ directly so
    // local never sees it, simulating a pre-restart key awaiting back-fill.
    // Must serialize locations into properties manually because we are
    // bypassing the manager's BuildEffectiveFieldMaps; otherwise the later
    // GetLocations fallback would observe no location fields and throw.
    KeyVector extra = {2};
    auto extra_batch = MakeBatch(extra);
    SerializeLocationsIntoProperties(extra_batch);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              mgr.persistent_backend_->Put(
                  nullptr, extra_batch.batch_keys, extra_batch.batch_locations, extra_batch.batch_properties));

    // Flip back to Recover to force the local-miss -> persistent-fallback path.
    mgr.recover_state_.store(MetaStorageBackendManager::RecoverState::kRecover, std::memory_order_release);

    KeyVector keys = {1, 2, 3};
    PropertyMapVector fms;
    auto ecs = mgr.GetProperties(nullptr, keys, {"p0"}, fms);
    ASSERT_EQ(EC_OK, ecs[0]);
    ASSERT_EQ(EC_OK, ecs[1]);
    ASSERT_EQ("p0_1", fms[0].at("p0"));
    ASSERT_EQ("p0_2", fms[1].at("p0"));

    std::vector<bool> exists_vec;
    auto exists_ecs = mgr.Exists(nullptr, keys, exists_vec);
    ASSERT_EQ(EC_OK, exists_ecs[0]);
    ASSERT_EQ(EC_OK, exists_ecs[1]);
    ASSERT_EQ((std::vector<bool>{true, true, false}), exists_vec);

    CacheLocationMapVector locs;
    auto loc_ecs = mgr.GetLocations(request_context_.get(), KeyVector{1, 2}, locs);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), loc_ecs);
    ASSERT_EQ("uri_2", locs[1].at("loc_2")->location_specs().front().uri());

    // Targeted GetLocations(keys, location_ids) also falls back on miss.
    LocationIdsPerKey ids = {{"loc_1"}, {"loc_2"}};
    LocationsPerKey per_key_locs;
    auto per_ecs = mgr.GetLocations(request_context_.get(), KeyVector{1, 2}, ids, per_key_locs);
    ASSERT_EQ(EC_OK, per_ecs[0][0]);
    ASSERT_EQ(EC_OK, per_ecs[1][0]);
    ASSERT_EQ("uri_2", per_key_locs[1][0]->location_specs().front().uri());

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Recover phase: writes dual-write; Delete records tombstone --------------

TEST_F(MetaStorageBackendManagerTest, TestRecoverWriteDualWriteAndDeleteTombstone) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_recover_write";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_recover_write", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Force Recover so Upsert hits EnsureKeyInLocal and Delete
    // inserts into deleted_keys_.
    mgr.recover_state_.store(MetaStorageBackendManager::RecoverState::kRecover, std::memory_order_release);

    // Put in Recover -> dual-write; then Delete in Recover -> tombstone set.
    KeyVector keys = {42};
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), batch));

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Delete(nullptr, keys));
    {
        std::lock_guard<std::mutex> lock(mgr.deleted_keys_mutex_);
        ASSERT_EQ(1u, mgr.deleted_keys_.count(42));
    }

    // Simulate a late backfill racing after Delete: BackfillKeysToCache must
    // see the tombstone and refuse to reinsert the key into local.
    CacheLocationMapVector stale_locs(1);
    PropertyMapVector stale_props(1);
    stale_props[0]["p0"] = "stale";
    ASSERT_EQ(0, mgr.BackfillKeysToCache(keys, stale_locs, stale_props, {EC_OK}));
    std::vector<bool> exists_vec;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.cache_backend_->Exists(nullptr, keys, exists_vec));
    ASSERT_FALSE(exists_vec[0]);

    // Upsert under Recover hydrates missing keys via EnsureKeyInLocal
    // before the conditional write. Seed key 7 into persistent only and
    // verify the upsert is observable afterwards.
    KeyVector k7 = {7};
    auto batch7 = MakeBatch(k7);
    SerializeLocationsIntoProperties(batch7);
    ASSERT_EQ(
        (std::vector<ErrorCode>{EC_OK}),
        mgr.persistent_backend_->Put(nullptr, batch7.batch_keys, batch7.batch_locations, batch7.batch_properties));

    BatchMetaData upsert_batch;
    upsert_batch.batch_keys = k7;
    upsert_batch.batch_properties.resize(1);
    upsert_batch.batch_properties[0]["p0"] = "p0_7_updated";
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Upsert(request_context_.get(), upsert_batch));

    PropertyMapVector fms;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.GetProperties(nullptr, k7, {"p0"}, fms));
    ASSERT_EQ("p0_7_updated", fms[0].at("p0"));

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Running phase: reads stay local-only, no persistent fallback ------------

TEST_F(MetaStorageBackendManagerTest, TestRunningReadLocalOnly) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_running_read";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_running", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Dual-write key 1 (visible from both).
    KeyVector k1 = {1};
    auto b1 = MakeBatch(k1);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), b1));

    // Bypass manager and write key 2 directly into persistent so local does
    // not know about it; in Running state reads must not see it.
    KeyVector k2 = {2};
    auto b2 = MakeBatch(k2);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              mgr.persistent_backend_->Put(nullptr, b2.batch_keys, b2.batch_locations, b2.batch_properties));

    std::vector<bool> exists_vec;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Exists(nullptr, KeyVector{1, 2}, exists_vec));
    ASSERT_EQ((std::vector<bool>{true, false}), exists_vec);

    PropertyMapVector fms;
    auto ecs = mgr.GetProperties(nullptr, KeyVector{2}, {"p0"}, fms);
    // Local miss: EC_OK with empty map OR EC_NOENT, both must not leak the
    // persistent-only entry.
    ASSERT_TRUE(ecs[0] == EC_NOENT || (ecs[0] == EC_OK && fms[0].empty()));

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- GetLocations(keys, location_ids): per-field EC semantics ----------------

TEST_F(MetaStorageBackendManagerTest, TestGetLocationsPerLocationIdSemantics) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_get_loc_ids";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_get_loc_ids", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Two keys, each with one real location. Request two ids per key: one
    // existing, one non-existent -> per-id EC must be {EC_OK, EC_NOENT}.
    KeyVector keys = {5, 6};
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    LocationIdsPerKey ids = {{"loc_5", "missing_loc"}, {"loc_6", "missing_loc"}};
    LocationsPerKey out_locs;
    auto ecs = mgr.GetLocations(request_context_.get(), keys, ids, out_locs);
    ASSERT_EQ(2u, ecs.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}), ecs[0]);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}), ecs[1]);
    ASSERT_EQ("uri_5", out_locs[0][0]->location_specs().front().uri());
    ASSERT_EQ("uri_6", out_locs[1][0]->location_specs().front().uri());

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Empty inputs on key-level APIs ------------------------------------------

TEST_F(MetaStorageBackendManagerTest, TestEmptyInputsAreNoOp) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_empty";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_empty", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    int32_t reclaimed = -1;
    auto del_ecs = mgr.Delete(nullptr, /*keys*/ {}, /*location_ids*/ {}, reclaimed);
    ASSERT_TRUE(del_ecs.empty());
    ASSERT_EQ(0, reclaimed);

    BatchMetaData empty_batch;
    auto put_ecs = mgr.Put(request_context_.get(), empty_batch);
    ASSERT_TRUE(put_ecs.empty());

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Tombstone: GetLocationIds ignores empty-value fields --------------------

TEST_F(MetaStorageBackendManagerTest, TestGetLocationIdsIgnoresTombstone) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_tombstone";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_tomb", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Put key 10 with a real location.
    KeyVector keys = {10};
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), batch));

    // Verify GetLocationIds returns the real location.
    LocationIdsPerKey loc_ids;
    auto ecs = mgr.GetLocationIds(nullptr, keys, loc_ids);
    ASSERT_EQ(EC_OK, ecs[0]);
    ASSERT_EQ(1u, loc_ids[0].size());
    EXPECT_EQ("loc_10", loc_ids[0][0]);

    // Now delete the location via DeleteLocations on both backends directly.
    LocationIdsPerKey del_loc_ids = {{"loc_10"}};
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.persistent_backend_->DeleteLocations(nullptr, keys, del_loc_ids));
    if (mgr.cache_backend_) {
        mgr.cache_backend_->DeleteLocations(nullptr, keys, del_loc_ids);
    }

    // GetLocationIds should now return EC_OK with empty location ids
    // (key still exists but has no valid non-tombstone locations).
    loc_ids.clear();
    ecs = mgr.GetLocationIds(nullptr, keys, loc_ids);
    ASSERT_EQ(EC_OK, ecs[0]);
    EXPECT_TRUE(loc_ids[0].empty());

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Delete with empty location_ids per key is a no-op -----------------------

TEST_F(MetaStorageBackendManagerTest, TestDeleteEmptyLocationIdsIsNoOp) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_del_empty_lids";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_del_empty", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Put key 20 with a real location.
    KeyVector keys = {20};
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), batch));

    // Delete with empty location_ids should be a no-op → EC_OK, 0 reclaimed.
    int32_t reclaimed = -1;
    LocationIdsPerKey empty_ids = {{}};
    auto del_ecs = mgr.Delete(nullptr, keys, empty_ids, reclaimed);
    ASSERT_EQ(EC_OK, del_ecs[0]);
    ASSERT_EQ(0, reclaimed);

    // The original location should still exist.
    LocationIdsPerKey loc_ids;
    auto get_ecs = mgr.GetLocationIds(nullptr, keys, loc_ids);
    ASSERT_EQ(EC_OK, get_ecs[0]);
    ASSERT_EQ(1u, loc_ids[0].size());
    EXPECT_EQ("loc_20", loc_ids[0][0]);

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- MaybeReclaimEmptyKeys after deleting last location ----------------------

TEST_F(MetaStorageBackendManagerTest, TestMaybeReclaimEmptyKeysAfterLastLocationDeleted) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_reclaim_empty";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_reclaim", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Put key 30 with a single location.
    KeyVector keys = {30};
    auto batch = MakeBatch(keys);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), mgr.Put(request_context_.get(), batch));

    // Delete that location → the key should be auto-reclaimed.
    int32_t reclaimed = 0;
    LocationIdsPerKey ids = {{"loc_30"}};
    auto del_ecs = mgr.Delete(nullptr, keys, ids, reclaimed);
    ASSERT_EQ(EC_OK, del_ecs[0]);
    EXPECT_EQ(1, reclaimed);

    // The key should no longer exist.
    std::vector<bool> exists_vec;
    auto exists_ecs = mgr.Exists(nullptr, keys, exists_vec);
    ASSERT_EQ(EC_OK, exists_ecs[0]);
    EXPECT_FALSE(exists_vec[0]);

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Multi-key, multi-location: gradual deletion until key reclaimed ----------

TEST_F(MetaStorageBackendManagerTest, TestMultiKeyMultiLocationGradualDeletion) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_multi_loc_gradual";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_multi_loc", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // Put 3 keys, each with 3 locations in a single batch (Put is overwrite,
    // so all locations must be in the same batch).
    KeyVector keys = {40, 50, 60};
    BatchMetaData batch;
    batch.batch_keys = keys;
    batch.batch_indexs = {0, 1, 2};
    batch.batch_locations.resize(keys.size());
    batch.batch_properties.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string key_str = std::to_string(keys[i]);
        const std::string loc_a = "loc_" + key_str;
        const std::string loc_b = "loc_" + key_str + "_b";
        const std::string loc_c = "loc_" + key_str + "_c";
        batch.batch_locations[i].emplace(loc_a, MakeLocation(loc_a, "uri_a_" + key_str));
        batch.batch_locations[i].emplace(loc_b, MakeLocation(loc_b, "uri_b_" + key_str));
        batch.batch_locations[i].emplace(loc_c, MakeLocation(loc_c, "uri_c_" + key_str));
        batch.batch_properties[i]["p0"] = "p0_" + key_str;
    }
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    // Verify each key now has 3 locations.
    {
        CacheLocationMapVector out_locations;
        auto get_ecs = mgr.GetLocations(request_context_.get(), keys, out_locations);
        for (size_t i = 0; i < keys.size(); ++i) {
            ASSERT_EQ(EC_OK, get_ecs[i]) << "key=" << keys[i];
            ASSERT_EQ(3u, out_locations[i].size()) << "key=" << keys[i] << " should have 3 locations";
        }
    }

    // --- Round 1: delete 1 location from each key → keys still alive ---------
    {
        LocationIdsPerKey del_ids = {
            {"loc_40"},
            {"loc_50"},
            {"loc_60"},
        };
        int32_t reclaimed = -1;
        auto del_ecs = mgr.Delete(nullptr, keys, del_ids, reclaimed);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), del_ecs);
        EXPECT_EQ(0, reclaimed) << "keys still have 2 locations each, none should be reclaimed";

        // All keys should still exist.
        std::vector<bool> exists_vec;
        auto exists_ecs = mgr.Exists(nullptr, keys, exists_vec);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), exists_ecs);
        ASSERT_EQ((std::vector<bool>{true, true, true}), exists_vec);

        // Each key should have exactly 2 remaining locations.
        CacheLocationMapVector out_locations;
        auto get_ecs = mgr.GetLocations(request_context_.get(), keys, out_locations);
        for (size_t i = 0; i < keys.size(); ++i) {
            ASSERT_EQ(EC_OK, get_ecs[i]);
            EXPECT_EQ(2u, out_locations[i].size()) << "key=" << keys[i] << " should have 2 locations left";
        }
    }

    // --- Round 2: delete another location from each key → keys still alive ---
    {
        LocationIdsPerKey del_ids = {
            {"loc_40_b"},
            {"loc_50_b"},
            {"loc_60_b"},
        };
        int32_t reclaimed = -1;
        auto del_ecs = mgr.Delete(nullptr, keys, del_ids, reclaimed);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), del_ecs);
        EXPECT_EQ(0, reclaimed) << "keys still have 1 location each, none should be reclaimed";

        // All keys should still exist.
        std::vector<bool> exists_vec;
        auto exists_ecs = mgr.Exists(nullptr, keys, exists_vec);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), exists_ecs);
        ASSERT_EQ((std::vector<bool>{true, true, true}), exists_vec);

        // Each key should have exactly 1 remaining location.
        CacheLocationMapVector out_locations;
        auto get_ecs = mgr.GetLocations(request_context_.get(), keys, out_locations);
        for (size_t i = 0; i < keys.size(); ++i) {
            ASSERT_EQ(EC_OK, get_ecs[i]);
            EXPECT_EQ(1u, out_locations[i].size()) << "key=" << keys[i] << " should have 1 location left";
            // Verify the remaining location is the "_c" one.
            const std::string expected_id = "loc_" + std::to_string(keys[i]) + "_c";
            EXPECT_TRUE(out_locations[i].count(expected_id) > 0) << "remaining location should be " << expected_id;
        }
    }

    // --- Round 3: delete the last location → all keys should be reclaimed ----
    {
        LocationIdsPerKey del_ids = {
            {"loc_40_c"},
            {"loc_50_c"},
            {"loc_60_c"},
        };
        int32_t reclaimed = -1;
        auto del_ecs = mgr.Delete(nullptr, keys, del_ids, reclaimed);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), del_ecs);
        EXPECT_EQ(3, reclaimed) << "all 3 keys should be reclaimed after last location deleted";

        // All keys should be gone.
        std::vector<bool> exists_vec;
        auto exists_ecs = mgr.Exists(nullptr, keys, exists_vec);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), exists_ecs);
        ASSERT_EQ((std::vector<bool>{false, false, false}), exists_vec);
    }

    ASSERT_EQ(EC_OK, mgr.Close());
}

// --- Mixed scenario: some keys reclaimed, some survive -----------------------

TEST_F(MetaStorageBackendManagerTest, TestMultiKeyPartialReclamation) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "mgr_partial_reclaim";
    std::filesystem::remove(path);
    MetaStorageBackendManager mgr;
    ASSERT_EQ(EC_OK, mgr.Init("inst_partial", MakeDualConfig(path)));
    ASSERT_EQ(EC_OK, mgr.Open());
    WaitRunning(mgr);

    // key 70: 2 locations (loc_70, loc_70_b)
    // key 80: 3 locations (loc_80, loc_80_b, loc_80_c)
    // key 90: 1 location  (loc_90)
    // All locations must be in a single Put per key (Put is overwrite).
    KeyVector keys = {70, 80, 90};
    BatchMetaData batch;
    batch.batch_keys = keys;
    batch.batch_indexs = {0, 1, 2};
    batch.batch_locations.resize(3);
    batch.batch_properties.resize(3);
    // key 70: 2 locations
    batch.batch_locations[0].emplace("loc_70", MakeLocation("loc_70", "uri_70"));
    batch.batch_locations[0].emplace("loc_70_b", MakeLocation("loc_70_b", "uri_70_b"));
    batch.batch_properties[0]["p0"] = "p0_70";
    // key 80: 3 locations
    batch.batch_locations[1].emplace("loc_80", MakeLocation("loc_80", "uri_80"));
    batch.batch_locations[1].emplace("loc_80_b", MakeLocation("loc_80_b", "uri_80_b"));
    batch.batch_locations[1].emplace("loc_80_c", MakeLocation("loc_80_c", "uri_80_c"));
    batch.batch_properties[1]["p0"] = "p0_80";
    // key 90: 1 location
    batch.batch_locations[2].emplace("loc_90", MakeLocation("loc_90", "uri_90"));
    batch.batch_properties[2]["p0"] = "p0_90";
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), mgr.Put(request_context_.get(), batch));

    // Verify initial state: key70=2, key80=3, key90=1
    {
        CacheLocationMapVector out_locations;
        auto get_ecs = mgr.GetLocations(request_context_.get(), keys, out_locations);
        ASSERT_EQ(EC_OK, get_ecs[0]);
        ASSERT_EQ(EC_OK, get_ecs[1]);
        ASSERT_EQ(EC_OK, get_ecs[2]);
        EXPECT_EQ(2u, out_locations[0].size());
        EXPECT_EQ(3u, out_locations[1].size());
        EXPECT_EQ(1u, out_locations[2].size());
    }

    // Delete: key70 loses 1 of 2, key80 loses all 3, key90 loses its only 1.
    // Expected: key70 survives, key80 and key90 are reclaimed.
    {
        LocationIdsPerKey del_ids = {
            {"loc_70"},                         // key70: 1 of 2 deleted → survives
            {"loc_80", "loc_80_b", "loc_80_c"}, // key80: all 3 deleted → reclaimed
            {"loc_90"},                         // key90: only 1 deleted → reclaimed
        };
        int32_t reclaimed = -1;
        auto del_ecs = mgr.Delete(nullptr, keys, del_ids, reclaimed);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), del_ecs);
        EXPECT_EQ(2, reclaimed) << "key80 and key90 should be reclaimed";
    }

    // Verify: key70 still exists with 1 location, key80 and key90 are gone.
    {
        std::vector<bool> exists_vec;
        auto exists_ecs = mgr.Exists(nullptr, keys, exists_vec);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), exists_ecs);
        EXPECT_TRUE(exists_vec[0]) << "key70 should still exist";
        EXPECT_FALSE(exists_vec[1]) << "key80 should be reclaimed";
        EXPECT_FALSE(exists_vec[2]) << "key90 should be reclaimed";
    }

    // key70 should have exactly 1 remaining location.
    {
        CacheLocationMapVector out_locations;
        auto get_ecs = mgr.GetLocations(request_context_.get(), {70}, out_locations);
        ASSERT_EQ(EC_OK, get_ecs[0]);
        ASSERT_EQ(1u, out_locations[0].size());
        EXPECT_TRUE(out_locations[0].count("loc_70_b") > 0);
    }

    ASSERT_EQ(EC_OK, mgr.Close());
}

} // namespace kv_cache_manager
