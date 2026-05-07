#include "kv_cache_manager/meta/test/meta_indexer_test_base.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_set>

#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_search_cache.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

CacheLocation MetaIndexerTestBase::MakeLocation(const std::string &id, const std::string &uri) {
    // Build a minimal one-spec CacheLocation; uri field carries the payload the
    // tests compare against after a round-trip through the backend.
    CacheLocation location;
    location.set_id(id);
    location.set_status(CacheLocationStatus::CLS_SERVING);
    location.set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    location.set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back(/*name*/ "default", uri);
    location.set_location_specs(std::move(specs));
    return location;
}

void MetaIndexerTestBase::MakeKVData(const int64_t start, const int64_t end, KVData &data) const {
    data.keys.clear();
    data.location_maps.clear();
    data.properties.clear();
    for (int64_t i = start; i < end; ++i) {
        data.keys.push_back(i);
        const std::string loc_id = "loc_" + std::to_string(i);
        LocationMap loc_map;
        loc_map.emplace(loc_id, MakeLocation(loc_id, "uri_" + std::to_string(i)));
        data.location_maps.push_back(std::move(loc_map));

        PropertyMap map;
        map["p0"] = "p0_" + std::to_string(i);
        map["p1"] = "p1_" + std::to_string(i);
        data.properties.push_back(std::move(map));
    }
}

void MetaIndexerTestBase::MakeRandomKVData(const int64_t count,
                                           const int64_t min,
                                           const int64_t max,
                                           KVData &data) const {
    data.keys.clear();
    data.location_maps.clear();
    data.properties.clear();
    thread_local std::mt19937 rng(std::hash<std::thread::id>()(std::this_thread::get_id()) + std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(min, max);
    std::unordered_set<int64_t> seen;
    for (int64_t i = 0; i < count; ++i) {
        int64_t key = dist(rng);
        if (!seen.insert(key).second) {
            continue;
        }
        data.keys.push_back(key);
        const std::string loc_id = "loc_" + std::to_string(key);
        LocationMap loc_map;
        loc_map.emplace(loc_id, MakeLocation(loc_id, "uri_" + std::to_string(key)));
        data.location_maps.push_back(std::move(loc_map));

        PropertyMap map;
        map["p0"] = "p0_" + std::to_string(key);
        map["p1"] = "p1_" + std::to_string(key);
        data.properties.push_back(std::move(map));
    }
}

void MetaIndexerTestBase::AssertGet(const KeyVector &keys,
                                    const LocationMapVector &expect_location_maps,
                                    const Result &expect_result) {
    LocationMapVector out_locations;
    PropertyMapVector out_properties;
    auto result = meta_indexer_->Get(request_context_.get(), keys, out_locations, out_properties);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    ASSERT_EQ(keys.size(), out_locations.size());
    ASSERT_EQ(keys.size(), expect_location_maps.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto &expect_map = expect_location_maps[i];
        const auto &actual_map = out_locations[i];
        ASSERT_EQ(expect_map.size(), actual_map.size()) << "key=" << keys[i];
        for (const auto &kv : expect_map) {
            auto it = actual_map.find(kv.first);
            ASSERT_TRUE(it != actual_map.end()) << "key=" << keys[i] << " location_id=" << kv.first;
            // Compare on the round-trippable projection (id/uri) instead of the
            // full object to avoid coupling to every CacheLocation field.
            ASSERT_EQ(kv.second.id(), it->second.id());
            ASSERT_EQ(kv.second.location_specs().size(), it->second.location_specs().size());
            if (!kv.second.location_specs().empty()) {
                ASSERT_EQ(kv.second.location_specs().front().uri(), it->second.location_specs().front().uri());
            }
        }
    }
}

void MetaIndexerTestBase::AssertGet(const KeyVector &keys,
                                    const LocationMapVector &expect_location_maps,
                                    const PropertyMapVector &expect_properties,
                                    const Result &expect_result) {
    LocationMapVector out_locations;
    PropertyMapVector out_properties;
    auto result = meta_indexer_->Get(request_context_.get(), keys, out_locations, out_properties);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    ASSERT_EQ(keys.size(), out_locations.size());
    ASSERT_EQ(keys.size(), out_properties.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto &expect_loc = expect_location_maps[i];
        const auto &actual_loc = out_locations[i];
        ASSERT_EQ(expect_loc.size(), actual_loc.size()) << "key=" << keys[i];
        for (const auto &kv : expect_loc) {
            auto it = actual_loc.find(kv.first);
            ASSERT_TRUE(it != actual_loc.end()) << "key=" << keys[i] << " location_id=" << kv.first;
            ASSERT_EQ(kv.second.id(), it->second.id());
        }
        for (const auto &prop : expect_properties[i]) {
            ASSERT_EQ(prop.second, out_properties[i][prop.first]) << "key=" << keys[i] << " prop=" << prop.first;
        }
    }
}

void MetaIndexerTestBase::AssertGetProperties(const KeyVector &keys,
                                              const std::vector<std::string> &property_names,
                                              const PropertyMapVector &expect_properties,
                                              const Result &expect_result) {
    PropertyMapVector properties;
    auto result = meta_indexer_->GetProperties(request_context_.get(), keys, property_names, properties);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    ASSERT_EQ(keys.size(), properties.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        for (const auto &name : property_names) {
            ASSERT_EQ(expect_properties[i].count(name) ? expect_properties[i].at(name) : std::string(),
                      properties[i].count(name) ? properties[i].at(name) : std::string())
                << "key=" << keys[i] << " prop=" << name;
        }
    }
}

void MetaIndexerTestBase::DoPutTest() {
    KVData data;
    const int32_t key_count = 3;
    MakeKVData(/*start*/ 0, /*end*/ 3, data);

    // 1. Put fresh keys and verify round-tripped location/property payloads.
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
    auto expect_result = Result(key_count);
    // Snapshot expected payload before the Put call moves the inputs.
    LocationMapVector expect_locations;
    expect_locations.reserve(key_count);
    for (const auto &m : data.location_maps) {
        expect_locations.emplace_back(m);
    }
    PropertyMapVector expect_properties = data.properties;
    auto result = meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());

    AssertGet(data.keys, expect_locations, expect_result);
    AssertGet(data.keys, expect_locations, expect_properties, expect_result);
    PropertyMapVector expect_p0 = {{{"p0", "p0_0"}}, {{"p0", "p0_1"}}, {{"p0", "p0_2"}}};
    AssertGetProperties(data.keys, {"p0"}, expect_p0, expect_result);
    PropertyMapVector expect_p1 = {{{"p1", "p1_0"}}, {{"p1", "p1_1"}}, {{"p1", "p1_2"}}};
    AssertGetProperties(data.keys, {"p1"}, expect_p1, expect_result);

    // 2. Cleanup so downstream sub-tests start from an empty store.
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoUpdateTest() {
    KVData data;
    const int32_t key_count = 3;
    MakeKVData(/*start*/ 0, /*end*/ 3, data);

    // 1. Update against NOENT keys -> every key returns EC_NOENT.
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
    KVData update_data;
    MakeKVData(0, 3, update_data);
    for (auto &m : update_data.properties) {
        m["p0"] = m["p0"] + "_new";
    }
    PropertyMapVector keep_properties = update_data.properties;
    LocationMapVector keep_locations;
    for (const auto &m : update_data.location_maps) {
        keep_locations.emplace_back(m);
    }

    Result expect_result(key_count);
    expect_result.ec = EC_ERROR;
    expect_result.error_codes = {EC_NOENT, EC_NOENT, EC_NOENT};
    auto result = meta_indexer_->Update(
        request_context_.get(), update_data.keys, update_data.location_maps, update_data.properties);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);

    // 2. Put then Update: new locations + new properties take effect.
    result = meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());

    // Re-materialise update payload since the previous Update may have moved some of it.
    update_data = {};
    MakeKVData(0, 3, update_data);
    for (auto &m : update_data.properties) {
        m["p0"] = m["p0"] + "_new";
    }
    keep_properties = update_data.properties;
    keep_locations.clear();
    for (const auto &m : update_data.location_maps) {
        keep_locations.emplace_back(m);
    }
    expect_result = Result(key_count);
    result = meta_indexer_->Update(
        request_context_.get(), update_data.keys, update_data.location_maps, update_data.properties);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);

    AssertGet(data.keys, keep_locations, expect_result);
    AssertGet(data.keys, keep_locations, keep_properties, expect_result);
    PropertyMapVector expect_p0 = {{{"p0", "p0_0_new"}}, {{"p0", "p0_1_new"}}, {{"p0", "p0_2_new"}}};
    AssertGetProperties(data.keys, {"p0"}, expect_p0, expect_result);

    // 3. Cleanup
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoDeleteAndExistTest() {
    KVData data;
    const int32_t key_count = 3;
    MakeKVData(0, 3, data);

    // 1. Exist/Delete on NOENT keys.
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
    std::vector<bool> is_exists;
    auto result = meta_indexer_->Exist(request_context_.get(), data.keys, is_exists);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ((std::vector<bool>{false, false, false}), is_exists);
    result = meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(EC_ERROR, result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT, EC_NOENT, EC_NOENT}), result.error_codes);

    // 2. Put then delete a subset; Exist must reflect the partial state.
    auto expect_result = Result(key_count);
    result = meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());

    result = meta_indexer_->Exist(request_context_.get(), data.keys, is_exists);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ((std::vector<bool>{true, true, true}), is_exists);

    KeyVector delete_keys = {0, 1};
    result = meta_indexer_->Delete(request_context_.get(), delete_keys);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(key_count - static_cast<int32_t>(delete_keys.size()), meta_indexer_->GetKeyCount());

    result = meta_indexer_->Exist(request_context_.get(), data.keys, is_exists);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ((std::vector<bool>{false, false, true}), is_exists);

    LocationMapVector expect_locations(key_count);
    expect_locations[2].emplace("loc_2", MakeLocation("loc_2", "uri_2"));
    PropertyMapVector expect_properties(key_count);
    expect_properties[2] = {{"p0", "p0_2"}, {"p1", "p1_2"}};
    Result expect_get_result(key_count);
    expect_get_result.ec = EC_PARTIAL_OK;
    expect_get_result.error_codes = {EC_NOENT, EC_NOENT, EC_OK};
    AssertGet(data.keys, expect_locations, expect_get_result);
    AssertGet(data.keys, expect_locations, expect_properties, expect_get_result);

    // 3. Cleanup
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoScanAndSampleReclaimKeysTest() {
    KVData data;
    const int32_t key_count = 3;
    MakeKVData(0, 3, data);
    auto result = meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    ASSERT_EQ(EC_OK, result.ec);

    // 1. Scan must eventually surface every key.
    std::string cursor = SCAN_BASE_CURSOR;
    KeyVector keys;
    int64_t try_count = 100;
    int64_t scan_count = key_count;
    while (try_count-- && scan_count > 0) {
        std::string next_cursor;
        KeyVector out_keys;
        ASSERT_EQ(EC_OK, meta_indexer_->Scan(cursor, /*limit*/ 50, next_cursor, out_keys));
        cursor = next_cursor;
        scan_count -= out_keys.size();
        keys.insert(keys.end(), out_keys.begin(), out_keys.end());
    }
    ASSERT_GT(try_count, 0);
    ASSERT_EQ(static_cast<size_t>(key_count), keys.size());
    std::sort(keys.begin(), keys.end());
    ASSERT_EQ((KeyVector{0, 1, 2}), keys);

    // 2. SampleReclaimKeys must converge to the full key set after enough tries.
    keys.clear();
    try_count = 100;
    while (try_count-- && keys.size() < static_cast<size_t>(key_count)) {
        KeyVector out_keys;
        ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(request_context_.get(), key_count, out_keys));
        for (const auto key : out_keys) {
            if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
                keys.push_back(key);
            }
        }
    }
    ASSERT_GT(try_count, 0);
    std::sort(keys.begin(), keys.end());
    ASSERT_EQ((KeyVector{0, 1, 2}), keys);

    // 3. Cleanup
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoReadModifyWriteBlockTest() {
    // upsert_modifier: creates a new location when the key is absent, or adds
    // a second location (id="loc_new_<i>") when the key already exists.
    auto upsert_modifier = [](const LocationMap &existing_locations,
                              ErrorCode get_ec,
                              size_t /*key_index*/,
                              PropertyMap &upsert_property_map,
                              LocationMap &out_new_locations) -> ModifierResult {
        if (get_ec != EC_OK && get_ec != EC_NOENT) {
            return {MA_FAIL, get_ec};
        }
        const std::string suffix = std::to_string(existing_locations.size());
        const std::string loc_id = "rmw_loc_" + suffix;
        out_new_locations.emplace(loc_id, MetaIndexerTestBase::MakeLocation(loc_id, "rmw_uri_" + suffix));
        upsert_property_map["rmw_prop"] = "rmw_val_" + suffix;
        return {MA_OK, EC_OK};
    };
    auto delete_modifier_lenient = [](const LocationMap & /*existing_locations*/,
                                      ErrorCode get_ec,
                                      size_t,
                                      PropertyMap &,
                                      LocationMap &) -> ModifierResult {
        if (get_ec == EC_OK) {
            return {MA_DELETE, EC_OK};
        }
        if (get_ec == EC_NOENT) {
            return {MA_SKIP, EC_OK};
        }
        return {MA_FAIL, get_ec};
    };
    auto delete_modifier_strict = [](const LocationMap & /*existing_locations*/,
                                     ErrorCode get_ec,
                                     size_t,
                                     PropertyMap &,
                                     LocationMap &) -> ModifierResult {
        if (get_ec == EC_OK) {
            return {MA_DELETE, EC_OK};
        }
        return {MA_FAIL, get_ec};
    };

    // 1. RMW on NOENT keys -> upsert inserts every key.
    KeyVector keys = {0, 1, 2};
    Result expect(keys.size());
    auto result = meta_indexer_->ReadModifyWriteBlock(request_context_.get(), keys, upsert_modifier);
    ASSERT_EQ(expect.ec, result.ec);
    ASSERT_EQ(expect.error_codes, result.error_codes);
    ASSERT_EQ(3, meta_indexer_->GetKeyCount());
    // Verify rmw_prop was stored.
    PropertyMapVector expect_props(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        expect_props[i] = {{"rmw_prop", "rmw_val_0"}};
    }
    AssertGetProperties(keys, {"rmw_prop"}, expect_props, expect);

    // 2. Delete via RMW with lenient modifier: skip NOENT, delete existing.
    KeyVector rmw_keys = {1, 2, 3};
    Result expect2(rmw_keys.size()); // all MA_SKIP/MA_DELETE(EC_OK) -> EC_OK
    result = meta_indexer_->ReadModifyWriteBlock(request_context_.get(), rmw_keys, delete_modifier_lenient);
    ASSERT_EQ(expect2.ec, result.ec);
    ASSERT_EQ(expect2.error_codes, result.error_codes);
    ASSERT_EQ(1, meta_indexer_->GetKeyCount());

    // 3. Strict delete on NOENT surfaces EC_NOENT.
    // key 0 still exists (lenient step above targeted {1,2,3} so key 0 was
    // untouched); keys 1/2 were already deleted there. Strict modifier
    // therefore deletes key 0 and fails keys 1/2 -> partial batch.
    KeyVector strict_keys = {0, 1, 2};
    result = meta_indexer_->ReadModifyWriteBlock(request_context_.get(), strict_keys, delete_modifier_strict);
    ASSERT_EQ(EC_PARTIAL_OK, result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT, EC_NOENT}), result.error_codes);

    // 4. Cleanup
    meta_indexer_->Delete(request_context_.get(), {0, 1, 2, 3});
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoReadModifyWriteLocationTest() {
    // Seed two keys, each with two locations (loc_a/loc_b).
    KeyVector keys = {0, 1};
    LocationMapVector seed_locations(keys.size());
    PropertyMapVector seed_properties(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        seed_locations[i].emplace("loc_a", MakeLocation("loc_a", "uri_a_" + std::to_string(keys[i])));
        seed_locations[i].emplace("loc_b", MakeLocation("loc_b", "uri_b_" + std::to_string(keys[i])));
        seed_properties[i] = {{"p0", "p0_" + std::to_string(keys[i])}};
    }
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), keys, seed_locations, seed_properties).ec);

    // 1. Upsert: rewrite loc_a, add loc_new; loc_b stays untouched.
    auto upsert_modifier = [](CacheLocationVector &loc,
                              std::vector<ErrorCode> get_ec,
                              size_t /*key_index*/,
                              const LocationIdVector &loc_id,
                              PropertyMap &upsert_property_map) -> LocationModifierResult {
        std::vector<ErrorCode> ecs(loc_id.size(), EC_OK);
        for (size_t k = 0; k < loc_id.size(); ++k) {
            if (get_ec[k] != EC_OK && get_ec[k] != EC_NOENT) {
                ecs[k] = get_ec[k];
                continue;
            }
            loc[k] = MetaIndexerTestBase::MakeLocation(loc_id[k], "rmw_" + loc_id[k]);
        }
        upsert_property_map["rmw_prop"] = "v";
        return {MA_OK, std::move(ecs)};
    };
    LocationIdsPerKey upsert_ids = {{"loc_a", "loc_new"}, {"loc_a", "loc_new"}};
    auto loc_result = meta_indexer_->ReadModifyWriteLocation(request_context_.get(), keys, upsert_ids, upsert_modifier);
    ASSERT_EQ(EC_OK, loc_result.ec);
    for (const auto &per_key : loc_result.per_location_error_codes) {
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), per_key);
    }
    LocationMapVector expect_locations(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        expect_locations[i].emplace("loc_a", MakeLocation("loc_a", "rmw_loc_a"));
        expect_locations[i].emplace("loc_b", MakeLocation("loc_b", "uri_b_" + std::to_string(keys[i])));
        expect_locations[i].emplace("loc_new", MakeLocation("loc_new", "rmw_loc_new"));
    }
    AssertGet(keys, expect_locations, Result(keys.size()));
    PropertyMapVector expect_props(keys.size(), {{"rmw_prop", "v"}});
    AssertGetProperties(keys, {"rmw_prop"}, expect_props, Result(keys.size()));

    // 2. Delete: drop loc_b on key 0; key 1 untouched.
    auto delete_modifier = [](CacheLocationVector & /*loc*/,
                              std::vector<ErrorCode> get_ec,
                              size_t /*key_index*/,
                              const LocationIdVector &loc_id,
                              PropertyMap & /*upsert_property_map*/) -> LocationModifierResult {
        std::vector<ErrorCode> ecs(loc_id.size(), EC_OK);
        for (size_t k = 0; k < loc_id.size(); ++k) {
            if (get_ec[k] != EC_OK) {
                ecs[k] = get_ec[k];
            }
        }
        return {MA_DELETE, std::move(ecs)};
    };
    LocationIdsPerKey delete_ids = {{"loc_b"}, {}};
    loc_result = meta_indexer_->ReadModifyWriteLocation(request_context_.get(), keys, delete_ids, delete_modifier);
    ASSERT_EQ(EC_OK, loc_result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), loc_result.per_location_error_codes[0]);
    ASSERT_TRUE(loc_result.per_location_error_codes[1].empty());
    expect_locations[0].erase("loc_b");
    AssertGet(keys, expect_locations, Result(keys.size()));

    // 3. NOENT key surfaces per-location EC_NOENT and leaves the store unchanged.
    auto noent_modifier = [](CacheLocationVector & /*loc*/,
                             std::vector<ErrorCode> get_ec,
                             size_t /*key_index*/,
                             const LocationIdVector &loc_id,
                             PropertyMap & /*upsert_property_map*/) -> LocationModifierResult {
        std::vector<ErrorCode> ecs(loc_id.size(), EC_OK);
        for (size_t k = 0; k < loc_id.size(); ++k) {
            ecs[k] = get_ec[k];
        }
        return {MA_SKIP, std::move(ecs)};
    };
    KeyVector miss_keys = {99};
    LocationIdsPerKey miss_ids = {{"loc_a"}};
    loc_result = meta_indexer_->ReadModifyWriteLocation(request_context_.get(), miss_keys, miss_ids, noent_modifier);
    ASSERT_EQ(EC_OK, loc_result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}), loc_result.per_location_error_codes[0]);

    // 4. Bad args: keys/location_ids size mismatch.
    LocationIdsPerKey bad_ids = {{"loc_a"}};
    auto bad_result = meta_indexer_->ReadModifyWriteLocation(request_context_.get(), keys, bad_ids, upsert_modifier);
    ASSERT_EQ(EC_BADARGS, bad_result.ec);

    // 5. Cleanup
    meta_indexer_->Delete(request_context_.get(), keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoSimpleTest() {
    DoPutTest();
    DoUpdateTest();
    DoDeleteAndExistTest();
    DoScanAndSampleReclaimKeysTest();
    DoReadModifyWriteBlockTest();
    DoReadModifyWriteLocationTest();
}

void MetaIndexerTestBase::DoMultiThreadTest() {
    // Concurrent RMW (upsert) + Delete stress. The upsert modifier always
    // inserts/overwrites a single location so the outcome is deterministic per
    // key across threads.
    auto upsert_modifier = [](const LocationMap & /*existing_locations*/,
                              ErrorCode get_ec,
                              size_t /*key_index*/,
                              PropertyMap &upsert_property_map,
                              LocationMap &out_new_locations) -> ModifierResult {
        if (get_ec != EC_OK && get_ec != EC_NOENT) {
            return {MA_FAIL, get_ec};
        }
        out_new_locations.emplace("mt_loc", MetaIndexerTestBase::MakeLocation("mt_loc", "mt_uri"));
        upsert_property_map["mt_prop"] = "mt_val";
        return {MA_OK, EC_OK};
    };

    auto add_func = [&](const int32_t count, const int32_t min, const int32_t max) {
        KVData data;
        MakeRandomKVData(count, min, max, data);
        auto result = meta_indexer_->ReadModifyWriteBlock(request_context_.get(), data.keys, upsert_modifier);
        ASSERT_GE(static_cast<size_t>(count), result.error_codes.size());
        ASSERT_EQ(EC_OK, result.ec);
    };
    auto delete_func = [&](const int32_t count, const int32_t min, const int32_t max) {
        KVData data;
        MakeRandomKVData(count, min, max, data);
        auto result = meta_indexer_->Delete(request_context_.get(), data.keys);
        for (const auto &ec : result.error_codes) {
            ASSERT_TRUE(ec == EC_OK || ec == EC_NOENT);
        }
    };

    const int32_t epoch = 20;
    const int32_t thread_num = 8;
    const int32_t count = 64;
    const int32_t min = 0;
    const int32_t max = 128;
    for (int32_t i = 0; i < epoch; ++i) {
        std::vector<std::thread> threads;
        for (int j = 0; j < thread_num; ++j) {
            threads.emplace_back(add_func, count, min, max);
            if (j > thread_num / 2) {
                threads.emplace_back(delete_func, count, min, max);
            }
        }
        for (auto &t : threads) {
            t.join();
        }
    }
    KVData data;
    MakeKVData(min, max + 1, data);
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

} // namespace kv_cache_manager
