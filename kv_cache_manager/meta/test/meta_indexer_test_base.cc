#include "kv_cache_manager/meta/test/meta_indexer_test_base.h"

#include <random>
#include <thread>
#include <unordered_set>

#include "kv_cache_manager/meta/meta_search_cache.h"

namespace kv_cache_manager {
void MetaIndexerTestBase::MakeKVData(const int64_t start, const int64_t end, KVData &data) const {
    data.keys.clear();
    data.uris.clear();
    data.properties.clear();
    for (int32_t i = start; i < end; i++) {
        data.keys.push_back(i);
        data.uris.push_back("uri_" + std::to_string(i));
        MetaIndexer::PropertyMap map;
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
    data.uris.clear();
    data.properties.clear();
    std::random_device rd;
    thread_local std::mt19937 rng(std::hash<std::thread::id>()(std::this_thread::get_id()) + std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(min, max);
    std::unordered_set<int64_t> seen;
    for (int32_t i = 0; i < count; i++) {
        int32_t key = dist(rng);
        // filter duplicate keys
        if (seen.insert(key).second) {
            data.keys.push_back(key);
            data.uris.push_back("uri_" + std::to_string(key));
            MetaIndexer::PropertyMap map;
            map["p0"] = "p0_" + std::to_string(key);
            map["p1"] = "p1_" + std::to_string(key);
            data.properties.push_back(std::move(map));
        }
    }
}

void MetaIndexerTestBase::AssertGet(const KeyVector &keys, const UriVector &expect_uris, const Result &expect_result) {
    UriVector uris;
    auto result = meta_indexer_->Get(request_context_.get(), keys, uris);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    ASSERT_EQ(keys.size(), uris.size());
    for (int32_t i = 0; i < keys.size(); ++i) {
        ASSERT_EQ(expect_uris[i], uris[i]);
    }
}

void MetaIndexerTestBase::AssertSearchCacheGet(const KeyVector &keys,
                                               const UriVector &expect_uris,
                                               const std::vector<ErrorCode> &expect_error_codes) {
    if (!meta_indexer_->cache_) {
        // if no search cache, just return
        return;
    }
    for (int32_t i = 0; i < keys.size(); i++) {
        std::string out_uri;
        auto ec = meta_indexer_->cache_->Get(keys[i], &out_uri);
        ASSERT_EQ(expect_error_codes[i], ec);
        ASSERT_EQ(expect_uris[i], out_uri);
    }
}

void MetaIndexerTestBase::AssertGet(const KeyVector &keys,
                                    const UriVector &expect_uris,
                                    const PropertyMapVector &expect_properties,
                                    const Result &expect_result) {
    UriVector uris;
    PropertyMapVector properties;
    auto result = meta_indexer_->Get(request_context_.get(), keys, uris, properties);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    ASSERT_EQ(keys.size(), uris.size());
    ASSERT_EQ(keys.size(), properties.size());
    for (int32_t i = 0; i < keys.size(); ++i) {
        ASSERT_EQ(expect_uris[i], uris[i]);
        const auto &expect_property = expect_properties[i];
        for (const auto &iter : expect_property) {
            ASSERT_EQ(iter.second, properties[i][iter.first]);
        }
    }
}

void MetaIndexerTestBase::AssertGetProperties(const KeyVector &keys,
                                              const std::vector<std::string> &property_names,
                                              PropertyMapVector &expect_properties,
                                              const Result &expect_result) {
    PropertyMapVector properties;
    auto result = meta_indexer_->GetProperties(request_context_.get(), keys, property_names, properties);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    ASSERT_EQ(keys.size(), properties.size());
    for (int32_t i = 0; i < keys.size(); ++i) {
        for (const auto &name : property_names) {
            ASSERT_EQ(expect_properties[i][name], properties[i][name]);
        }
    }
}

void MetaIndexerTestBase::AssertReadModifyWrite(const KeyVector &keys,
                                                const MetaIndexer::ModifierFunc &modifier,
                                                const Result &expect_result) {
    auto result = meta_indexer_->ReadModifyWrite(request_context_.get(), keys, modifier);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
}

void MetaIndexerTestBase::DoPutTest() {
    KVData data;
    int32_t key_count = 3;
    MakeKVData(/*start*/ 0, /*end*/ 3, data);
    // 1. test put keys
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
    auto expect_result = Result(key_count);
    auto result = meta_indexer_->Put(request_context_.get(), data.keys, data.uris, data.properties);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    // data uris and properties will be moved to batch property maps
    for (int32_t i = 0; i < key_count; i++) {
        ASSERT_TRUE(data.uris[i].empty());
        ASSERT_TRUE(data.properties[i].empty());
    }
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    UriVector expect_uris = {"uri_0", "uri_1", "uri_2"};
    AssertSearchCacheGet(data.keys, {"", "", ""}, {EC_NOENT, EC_NOENT, EC_NOENT});
    AssertGet(data.keys, expect_uris, expect_result);
    AssertSearchCacheGet(data.keys, expect_uris, {EC_OK, EC_OK, EC_OK});
    PropertyMapVector expect_maps = {
        {{"p0", "p0_0"}, {"p1", "p1_0"}}, {{"p0", "p0_1"}, {"p1", "p1_1"}}, {{"p0", "p0_2"}, {"p1", "p1_2"}}};
    AssertGet(data.keys, expect_uris, expect_maps, expect_result);
    PropertyMapVector expect_p0_maps = {{{"p0", "p0_0"}}, {{"p0", "p0_1"}}, {{"p0", "p0_2"}}};
    AssertGetProperties(data.keys, {"p0"}, expect_p0_maps, expect_result);
    PropertyMapVector expect_p1_maps = {{{"p1", "p1_0"}}, {{"p1", "p1_1"}}, {{"p1", "p1_2"}}};
    AssertGetProperties(data.keys, {"p1"}, expect_p1_maps, expect_result);

    // 2. delete all keys to avoid affecting other test cases
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoUpdateTest() {
    KVData data;
    int32_t key_count = 3;
    MakeKVData(/*start*/ 0, /*end*/ 3, data);
    // 1. test update noent keys
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
    UriVector update_uris = {"uri_0_new", "uri_1_new", "uri_2_new"};
    PropertyMapVector update_p0_maps = {{{"p0", "p0_0_new"}}, {{"p0", "p0_1_new"}}, {{"p0", "p0_2_new"}}};
    auto expect_result = Result(key_count);
    expect_result.ec = EC_ERROR;
    expect_result.error_codes = {EC_NOENT, EC_NOENT, EC_NOENT};
    auto result = meta_indexer_->Update(request_context_.get(), data.keys, update_uris, update_p0_maps);
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);

    // 2. test update keys
    result = meta_indexer_->Put(request_context_.get(), data.keys, data.uris, data.properties);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    expect_result = Result(key_count);
    // update_uris and update_p0_maps will be moved to batch property maps
    update_uris = {"uri_0_new", "uri_1_new", "uri_2_new"};
    update_p0_maps = {{{"p0", "p0_0_new"}}, {{"p0", "p0_1_new"}}, {{"p0", "p0_2_new"}}};
    UriVector expect_uris = update_uris;
    PropertyMapVector expect_p0_maps = update_p0_maps;
    result = meta_indexer_->Update(request_context_.get(), data.keys, update_uris, update_p0_maps);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    for (int32_t i = 0; i < update_uris.size(); i++) {
        ASSERT_TRUE(update_uris[i].empty());
        ASSERT_TRUE(update_p0_maps[i].empty());
    }
    AssertSearchCacheGet(data.keys, {"", "", ""}, {EC_NOENT, EC_NOENT, EC_NOENT});
    AssertGet(data.keys, expect_uris, expect_result);
    AssertSearchCacheGet(data.keys, expect_uris, {EC_OK, EC_OK, EC_OK});
    AssertGetProperties(data.keys, {"p0"}, expect_p0_maps, expect_result);

    PropertyMapVector update_p1_maps = {{{"p1", "p1_0_new"}}, {{"p1", "p1_1_new"}}, {{"p1", "p1_2_new"}}};
    PropertyMapVector expect_p1_maps = update_p1_maps;
    result = meta_indexer_->Update(request_context_.get(), data.keys, update_p1_maps);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    for (int32_t i = 0; i < update_uris.size(); i++) {
        ASSERT_TRUE(update_p1_maps[i].empty());
    }
    ASSERT_EQ(expect_result.ec, result.ec);
    ASSERT_EQ(expect_result.error_codes, result.error_codes);
    AssertGet(data.keys, expect_uris, expect_result);
    AssertGetProperties(data.keys, {"p1"}, expect_p1_maps, expect_result);
    PropertyMapVector expect_maps = {{{"p0", "p0_0_new"}, {"p1", "p1_0_new"}},
                                     {{"p0", "p0_1_new"}, {"p1", "p1_1_new"}},
                                     {{"p0", "p0_2_new"}, {"p1", "p1_2_new"}}};
    AssertGet(data.keys, expect_uris, expect_maps, expect_result);

    // 3. delete all keys to avoid affecting other test cases
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoDeleteAndExistTest() {
    KVData data;
    int32_t key_count = 3;
    MakeKVData(/*start*/ 0, /*end*/ 3, data);
    // 1. test delete and exist noent keys
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
    std::vector<bool> is_exists;
    auto result = meta_indexer_->Exist(request_context_.get(), data.keys, is_exists);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(is_exists, std::vector<bool>({false, false, false}));
    result = meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(EC_ERROR, result.ec);
    ASSERT_EQ(std::vector<ErrorCode>({EC_NOENT, EC_NOENT, EC_NOENT}), result.error_codes);

    // 2. test delete some keys and exist
    auto expect_result = Result(key_count);
    result = meta_indexer_->Put(request_context_.get(), data.keys, data.uris, data.properties);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    result = meta_indexer_->Exist(request_context_.get(), data.keys, is_exists);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(is_exists, std::vector<bool>({true, true, true}));
    KeyVector delete_keys = {0, 1};
    auto expect_delete_result = Result(delete_keys.size());
    result = meta_indexer_->Delete(request_context_.get(), delete_keys);
    ASSERT_EQ(key_count - delete_keys.size(), meta_indexer_->GetKeyCount());
    ASSERT_EQ(expect_delete_result.ec, result.ec);
    ASSERT_EQ(expect_delete_result.error_codes, result.error_codes);
    result = meta_indexer_->Exist(request_context_.get(), data.keys, is_exists);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(is_exists, std::vector<bool>({false, false, true}));
    UriVector expect_uris = {"", "", "uri_2"};
    PropertyMapVector expect_maps = {{}, {}, {{"p0", "p0_2"}, {"p1", "p1_2"}}};
    PropertyMapVector expect_p0_maps = {{}, {}, {{"p0", "p0_2"}}};
    PropertyMapVector expect_p1_maps = {{}, {}, {{"p1", "p1_2"}}};
    auto expect_get_result = Result(key_count);
    expect_get_result.ec = EC_PARTIAL_OK;
    expect_get_result.error_codes = {EC_NOENT, EC_NOENT, EC_OK};
    AssertSearchCacheGet(data.keys, {"", "", ""}, {EC_NOENT, EC_NOENT, EC_NOENT});
    AssertGet(data.keys, expect_uris, expect_get_result);
    AssertSearchCacheGet(data.keys, expect_uris, expect_get_result.error_codes);
    AssertGet(data.keys, expect_uris, expect_maps, expect_get_result);
    auto expect_get_properties_result = Result(key_count);
    AssertGetProperties(data.keys, {"p0"}, expect_p0_maps, expect_get_properties_result);
    AssertGetProperties(data.keys, {"p1"}, expect_p1_maps, expect_get_properties_result);

    // 3. delete all keys to avoid affecting other test cases
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoScanAndRandomSampleTest() {
    KVData data;
    int32_t key_count = 3;
    MakeKVData(/*start*/ 0, /*end*/ 3, data);
    auto result = meta_indexer_->Put(request_context_.get(), data.keys, data.uris, data.properties);
    ASSERT_EQ(key_count, meta_indexer_->GetKeyCount());
    ASSERT_EQ(EC_OK, result.ec);
    // 1. test scan
    std::string cursor = SCAN_BASE_CURSOR;
    KeyVector keys;
    int64_t try_count = 100;
    int64_t scan_count = 3;
    while (try_count-- && scan_count > 0) {
        std::string next_cursor;
        KeyVector out_keys;
        ASSERT_EQ(EC_OK, meta_indexer_->Scan(cursor, /*limit*/ 50, next_cursor, out_keys));
        cursor = next_cursor;
        scan_count -= out_keys.size();
        keys.insert(keys.end(), out_keys.begin(), out_keys.end());
    }
    ASSERT_GT(try_count, 0);
    ASSERT_EQ(3, keys.size());
    std::sort(keys.begin(), keys.end());
    KeyVector expect_keys = {0, 1, 2};
    ASSERT_EQ(expect_keys, keys);

    // 2. test random sample
    keys.clear();
    try_count = 100;
    while (try_count-- && keys.size() < key_count) {
        KeyVector out_keys;
        ASSERT_EQ(EC_OK, meta_indexer_->RandomSample(key_count, out_keys));
        for (const auto key : out_keys) {
            if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
                keys.push_back(key);
            }
        }
    }
    ASSERT_GT(try_count, 0);
    ASSERT_EQ(key_count, keys.size());
    std::sort(keys.begin(), keys.end());
    expect_keys = {0, 1, 2};
    ASSERT_EQ(expect_keys, keys);

    // 3. delete all keys to avoid affecting other test cases
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoSimpleTest() {
    DoPutTest();
    DoUpdateTest();
    DoDeleteAndExistTest();
    DoScanAndRandomSampleTest();
    DoReadModifyWriteTest();
}

void MetaIndexerTestBase::DoMultiThreadTest() {
    auto modifier =
        [&](std::string &uri, ErrorCode get_ec, size_t index, PropertyMap &map) -> MetaIndexer::ModifierResult {
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            return {MetaIndexer::MA_FAIL, get_ec};
        }
        if (get_ec == ErrorCode::EC_OK) {
            uri = "update_" + uri;
        } else {
            uri = "new_" + uri;
        }
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };

    // test add location
    auto add_uri_func = [&](const int32_t count, const int32_t min, const int32_t max) {
        KVData data;
        MakeRandomKVData(count, min, max, data);
        auto result = meta_indexer_->ReadModifyWrite(request_context_.get(), data.keys, modifier);
        ASSERT_GE(count, result.error_codes.size());
        ASSERT_EQ(EC_OK, result.ec);
    };

    auto delete_uri_func = [&](const int32_t count, const int32_t min, const int32_t max) {
        KVData data;
        MakeRandomKVData(count, min, max, data);
        // delete all keys
        auto result = meta_indexer_->Delete(request_context_.get(), data.keys);
        for (const auto &ec : result.error_codes) {
            ASSERT_TRUE(ec == EC_OK || ec == EC_NOENT);
        }
    };

    int32_t epoch = 20;
    int32_t thread_num = 8;
    int32_t count = 64;
    int32_t min = 0;
    int32_t max = 128;
    for (int32_t i = 0; i < epoch; i++) {
        std::vector<std::thread> threads;
        for (int j = 0; j < thread_num; j++) {
            threads.push_back(std::thread(add_uri_func, count, min, max));
            if (j > thread_num / 2) {
                threads.push_back(std::thread(delete_uri_func, count, min, max));
            }
        }
        for (auto &thread : threads) {
            thread.join();
        }
    }
    // delete all keys to avoid affecting other test cases
    KVData data;
    MakeKVData(min, max + 1, data);
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

void MetaIndexerTestBase::DoReadModifyWriteTest() {
    KVData data;
    auto upsert_modifier =
        [&data](std::string &uri, ErrorCode get_ec, size_t index, PropertyMap &map) -> MetaIndexer::ModifierResult {
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            return {MetaIndexer::MA_FAIL, get_ec};
        }
        if (get_ec == ErrorCode::EC_OK) {
            uri = "update_" + data.uris[index];
            map["test_property_key"] = "test_property_value_updated_" + data.uris[index];
        } else {
            uri = "new_" + data.uris[index];
            map["test_property_key"] = "test_property_value_" + data.uris[index];
        }
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };
    auto delete_modifier1 =
        [](std::string &uri, ErrorCode get_ec, size_t index, PropertyMap &map) -> MetaIndexer::ModifierResult {
        if (get_ec == ErrorCode::EC_OK) {
            return {MetaIndexer::MA_DELETE, ErrorCode::EC_OK};
        } else if (get_ec == ErrorCode::EC_NOENT) {
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_OK};
        } else {
            return {MetaIndexer::MA_FAIL, get_ec};
        }
    };
    auto delete_modifier2 =
        [](std::string &uri, ErrorCode get_ec, size_t index, PropertyMap &map) -> MetaIndexer::ModifierResult {
        if (get_ec == ErrorCode::EC_OK) {
            return {MetaIndexer::MA_DELETE, ErrorCode::EC_OK};
        } else {
            return {MetaIndexer::MA_FAIL, get_ec};
        }
    };

    // 1. test all put keys in ReadModifyWrite
    {
        MakeKVData(0, 3, data);
        auto expect_result = Result(data.keys.size());
        AssertReadModifyWrite(data.keys, upsert_modifier, expect_result);
        ASSERT_EQ(3, meta_indexer_->GetKeyCount());
        UriVector expect_uris = {"new_uri_0", "new_uri_1", "new_uri_2"};
        AssertSearchCacheGet(data.keys, {"", "", ""}, {EC_NOENT, EC_NOENT, EC_NOENT});
        AssertGet(data.keys, expect_uris, expect_result);
        AssertSearchCacheGet(data.keys, expect_uris, expect_result.error_codes);
        const std::vector<std::string> property_names = {"test_property_key"};
        PropertyMapVector expect_properties = {{{"test_property_key", "test_property_value_uri_0"}},
                                               {{"test_property_key", "test_property_value_uri_1"}},
                                               {{"test_property_key", "test_property_value_uri_2"}}};
        AssertGetProperties(data.keys, property_names, expect_properties, expect_result);
    }

    // 2. test update and put keys in ReadModifyWrite
    {
        MakeKVData(1, 4, data);
        auto expect_result = Result(data.keys.size());
        AssertReadModifyWrite(data.keys, upsert_modifier, expect_result);
        ASSERT_EQ(4, meta_indexer_->GetKeyCount());
        KeyVector get_keys = {0, 1, 2, 3};
        AssertSearchCacheGet(get_keys, {"new_uri_0", "", "", ""}, {EC_OK, EC_NOENT, EC_NOENT, EC_NOENT});
        UriVector expect_uris = {"new_uri_0", "update_uri_1", "update_uri_2", "new_uri_3"};
        expect_result = Result(get_keys.size());
        AssertGet(get_keys, expect_uris, expect_result);
        AssertSearchCacheGet(get_keys, expect_uris, expect_result.error_codes);
        const std::vector<std::string> property_names = {"test_property_key"};
        PropertyMapVector expect_properties = {{{"test_property_key", "test_property_value_uri_0"}},
                                               {{"test_property_key", "test_property_value_updated_uri_1"}},
                                               {{"test_property_key", "test_property_value_updated_uri_2"}},
                                               {{"test_property_key", "test_property_value_uri_3"}}};
        AssertGetProperties(get_keys, property_names, expect_properties, expect_result);
    }

    // 3. test delete keys in ReadModifyWrite
    {
        // test delete_modifier1
        MakeKVData(3, 5, data);
        auto expect_result = Result(data.keys.size());
        AssertReadModifyWrite(data.keys, delete_modifier1, expect_result);
        ASSERT_EQ(3, meta_indexer_->GetKeyCount());
        KeyVector get_keys = {0, 1, 2, 3, 4};
        expect_result = Result(get_keys.size());
        expect_result.ec = EC_PARTIAL_OK;
        expect_result.error_codes = {EC_OK, EC_OK, EC_OK, EC_NOENT, EC_NOENT};
        UriVector expect_uris = {"new_uri_0", "update_uri_1", "update_uri_2", "", ""};
        AssertSearchCacheGet(get_keys, expect_uris, expect_result.error_codes);
        AssertGet(get_keys, expect_uris, expect_result);
        AssertSearchCacheGet(get_keys, expect_uris, expect_result.error_codes);
        // test delete_modifier2
        MakeKVData(2, 4, data);
        expect_result = Result(data.keys.size());
        expect_result.ec = EC_PARTIAL_OK;
        expect_result.error_codes = {EC_OK, EC_NOENT};
        AssertReadModifyWrite(data.keys, delete_modifier2, expect_result);
        ASSERT_EQ(2, meta_indexer_->GetKeyCount());
        expect_result = Result(get_keys.size());
        expect_result.ec = EC_PARTIAL_OK;
        expect_result.error_codes = {EC_OK, EC_OK, EC_NOENT, EC_NOENT, EC_NOENT};
        expect_uris = {"new_uri_0", "update_uri_1", "", "", ""};
        AssertSearchCacheGet(get_keys, expect_uris, expect_result.error_codes);
        AssertGet(get_keys, expect_uris, expect_result);
        AssertSearchCacheGet(get_keys, expect_uris, expect_result.error_codes);
    }

    // 4. delete all keys to avoid affecting other test cases
    MakeKVData(0, 5, data);
    meta_indexer_->Delete(request_context_.get(), data.keys);
    ASSERT_EQ(0, meta_indexer_->GetKeyCount());
}

} // namespace kv_cache_manager
