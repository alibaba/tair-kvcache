#include <filesystem>
#include <set>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/test/meta_storage_backend_test_base.h"

namespace kv_cache_manager {
class MetaLocalBackendTest : public MetaStorageBackendTestBase, public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

    void ConstructMetaStorageBackend();
    void ConstructMetaStorageBackendConfig();
    std::string ExpectedStorageType() const;

    MetaLocalBackend *GetLocalBackend() const { return static_cast<MetaLocalBackend *>(meta_storage_backend_.get()); }

protected:
    std::shared_ptr<MetaStorageBackend> meta_storage_backend_;
    std::shared_ptr<MetaStorageBackendConfig> meta_storage_backend_config_;
};

void MetaLocalBackendTest::SetUp() {
    ConstructMetaStorageBackend();
    ConstructMetaStorageBackendConfig();
}

void MetaLocalBackendTest::ConstructMetaStorageBackend() {
    meta_storage_backend_ = std::make_shared<MetaLocalBackend>();
}

void MetaLocalBackendTest::ConstructMetaStorageBackendConfig() {
    meta_storage_backend_config_ = std::make_shared<MetaStorageBackendConfig>();

    std::string local_path = GetPrivateTestRuntimeDataPath() + "_meta_local_backend_file1";
    std::error_code ec;
    bool exists = std::filesystem::exists(local_path, ec);
    ASSERT_FALSE(ec) << local_path; // false means correct
    if (exists) {
        std::remove(local_path.c_str());
    }
    meta_storage_backend_config_->SetStorageUri("file://" + local_path);
}

std::string MetaLocalBackendTest::ExpectedStorageType() const { return META_LOCAL_BACKEND_TYPE_STR; }

TEST_F(MetaLocalBackendTest, TestSimple) {
    ASSERT_EQ(ExpectedStorageType(), meta_storage_backend_->GetStorageType());

    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    // Put two entries with uri field
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{PROPERTY_URI, "uri1"}}, {{PROPERTY_URI, "uri2"}}}));
    // UpdateFields to add lru_time
    ASSERT_EQ(
        (std::vector<ErrorCode>{EC_OK, EC_OK}),
        meta_storage_backend_->UpdateFields({1, 2}, {{{PROPERTY_LRU_TIME, "100"}}, {{PROPERTY_LRU_TIME, "200"}}}));

    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {true, true, false});
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_LRU_TIME},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}});
    AssertListKeys(meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 3, EC_OK, SCAN_BASE_CURSOR, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 1, EC_OK, {1, 2});

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              meta_storage_backend_->Put({3}, {{{PROPERTY_URI, "uri3"}, {PROPERTY_LRU_TIME, "300"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              meta_storage_backend_->UpdateFields({2}, {{{PROPERTY_URI, "uri2-updated"}}}));

    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {false, true, true});
    AssertGet(meta_storage_backend_.get(),
              {1, 2, 3},
              {PROPERTY_URI, PROPERTY_LRU_TIME},
              {EC_NOENT, EC_OK, EC_OK},
              {{},
               {{PROPERTY_URI, "uri2-updated"}, {PROPERTY_LRU_TIME, "200"}},
               {{PROPERTY_URI, "uri3"}, {PROPERTY_LRU_TIME, "300"}}});
    AssertListKeys(meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 3, EC_OK, SCAN_BASE_CURSOR, {2, 3});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 1, EC_OK, {2, 3});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestInit) {
    // invalid config
    ASSERT_EQ(EC_BADARGS, meta_storage_backend_->Init("test_instance_0", /*config*/ nullptr));
    ASSERT_EQ(EC_BADARGS, meta_storage_backend_->Init(/*instance_id*/ "", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
}

TEST_F(MetaLocalBackendTest, TestPut) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}}));
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_LRU_TIME},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}});

    // Put again to overwrite
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({1}, {{{PROPERTY_URI, "uri1-new"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {PROPERTY_URI},
              {EC_OK, EC_OK},
              {{{PROPERTY_URI, "uri1-new"}}, {{PROPERTY_URI, "uri2"}}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestUpdateFields) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}}));

    // Update uri only, lru_time should be preserved
    ASSERT_EQ(
        (std::vector<ErrorCode>{EC_OK, EC_OK}),
        meta_storage_backend_->UpdateFields({1, 2}, {{{PROPERTY_URI, "uri1-updated"}}, {{PROPERTY_LRU_TIME, "250"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {PROPERTY_URI, PROPERTY_LRU_TIME},
              {EC_OK, EC_OK},
              {{{PROPERTY_URI, "uri1-updated"}, {PROPERTY_LRU_TIME, "100"}},
               {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "250"}}});

    // Cannot update key that does not exist
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}), meta_storage_backend_->UpdateFields({3}, {{{PROPERTY_URI, "uri3"}}}));

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestUpsert) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}}));

    // Upsert: update existing, insert new
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}),
              meta_storage_backend_->Upsert(
                  {1, 2, 3},
                  {{{PROPERTY_URI, "uri1-upserted"}}, {{PROPERTY_LRU_TIME, "250"}}, {{PROPERTY_URI, "uri3-new"}}}));
    // Verify existing keys preserve unmodified fields
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {PROPERTY_URI, PROPERTY_LRU_TIME},
              {EC_OK, EC_OK},
              {{{PROPERTY_URI, "uri1-upserted"}, {PROPERTY_LRU_TIME, "100"}},
               {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "250"}}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestIncrFields) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}}));

    // Increment lru_time
    ASSERT_EQ((std::vector<ErrorCode>{EC_UNIMPLEMENTED, EC_UNIMPLEMENTED}),
              meta_storage_backend_->IncrFields({1, 2}, {{PROPERTY_LRU_TIME, 50}}));

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestDelete) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}),
              meta_storage_backend_->Put(
                  {1, 2, 3}, {{{PROPERTY_URI, "uri1"}}, {{PROPERTY_URI, "uri2"}}, {{PROPERTY_URI, "uri3"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), meta_storage_backend_->Delete({1, 3}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT, EC_NOENT}), meta_storage_backend_->Delete({1, 3}));
    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {false, true, false});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestGet) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}}));

    // Get single field
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {PROPERTY_URI},
              {EC_OK, EC_OK},
              {{{PROPERTY_URI, "uri1"}}, {{PROPERTY_URI, "uri2"}}});

    // Get all supported fields
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_LRU_TIME},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}});

    // Get no fields
    AssertGet(meta_storage_backend_.get(), {1, 2}, {}, {EC_OK, EC_OK}, FieldMapVec(2));

    // Get non-existent key returns empty values
    AssertGet(meta_storage_backend_.get(), {3}, {PROPERTY_URI, PROPERTY_LRU_TIME}, {EC_NOENT}, {{}});

    // Get unsupported field returns empty string
    AssertGet(meta_storage_backend_.get(), {1}, {"unsupported_field"}, {EC_OK}, {{}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestGetAll) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}}));
    AssertGetAllFields(
        meta_storage_backend_.get(),
        {1, 2},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}});
    AssertGetAllFields(meta_storage_backend_.get(), {3}, {EC_NOENT}, FieldMapVec(1)); // no entry

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestExists) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{PROPERTY_URI, "uri1"}}, {{PROPERTY_URI, "uri2"}}}));
    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {true, true, false});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestListKeys) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({1}, {{{PROPERTY_URI, "uri1"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({2}, {{{PROPERTY_URI, "uri2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({3}, {{{PROPERTY_URI, "uri3"}}}));

    // List all keys
    AssertListKeys(meta_storage_backend_.get(),
                   SCAN_BASE_CURSOR,
                   /*limit*/ std::numeric_limits<int64_t>::max(),
                   EC_OK,
                   SCAN_BASE_CURSOR,
                   {1, 2, 3});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestSampleReclaimKeys) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({1}, {{{PROPERTY_URI, "uri1"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({2}, {{{PROPERTY_URI, "uri2"}}}));
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 0, EC_OK, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 1, EC_OK, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 2, EC_OK, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 3, EC_OK, {1, 2});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestMetaMemCacheItemFieldMap) {
    // Verify MetaMemCacheItem stores all fields in a single FieldMap
    MetaMemCacheItem::FieldMap fields = {{PROPERTY_URI, "test://some/long/uri/path/for/testing"},
                                         {PROPERTY_LRU_TIME, "1234567890"}};

    MetaMemCacheItem *item = MetaMemCacheItem::Create(fields);
    ASSERT_NE(nullptr, item);
    ASSERT_EQ("test://some/long/uri/path/for/testing", item->GetFields().at(PROPERTY_URI));
    ASSERT_EQ("1234567890", item->GetFields().at(PROPERTY_LRU_TIME));
    ASSERT_EQ(2u, item->GetFields().size());
    // Size should be at least sizeof(MetaMemCacheItem) plus string heap overhead
    ASSERT_GE(item->Size(), sizeof(MetaMemCacheItem));

    MetaMemCacheItem::Deleter(item, nullptr);

    // Test with empty fields
    MetaMemCacheItem *empty_item = MetaMemCacheItem::Create({});
    ASSERT_NE(nullptr, empty_item);
    ASSERT_TRUE(empty_item->GetFields().empty());
    ASSERT_EQ(sizeof(MetaMemCacheItem), empty_item->Size());

    MetaMemCacheItem::Deleter(empty_item, nullptr);

    // Test with extra custom fields
    MetaMemCacheItem::FieldMap custom_fields = {
        {PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}, {"custom_key", "custom_value"}};
    MetaMemCacheItem *custom_item = MetaMemCacheItem::Create(custom_fields);
    ASSERT_NE(nullptr, custom_item);
    ASSERT_EQ(3u, custom_item->GetFields().size());
    ASSERT_EQ("custom_value", custom_item->GetFields().at("custom_key"));

    MetaMemCacheItem::Deleter(custom_item, nullptr);
}

TEST_F(MetaLocalBackendTest, TestGetOldestKeysFromRandomShard) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    // No entries inserted, should return empty
    MetaLocalBackend *local_backend = GetLocalBackend();
    std::vector<MetaStorageBackend::KeyType> oldest_keys;
    ErrorCode ret = local_backend->GetOldestKeysFromRandomShard(5, oldest_keys);
    ASSERT_EQ(EC_OK, ret);
    ASSERT_TRUE(oldest_keys.empty());

    // count = 0 should return EC_BADARGS
    ret = local_backend->GetOldestKeysFromRandomShard(0, oldest_keys);
    ASSERT_EQ(EC_BADARGS, ret);

    // Insert multiple entries
    for (int64_t i = 1; i <= 20; ++i) {
        std::string uri = "uri" + std::to_string(i);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({i}, {{{PROPERTY_URI, uri}}}));
    }

    // GetOldestKeysFromRandomShard should return some keys
    ret = local_backend->GetOldestKeysFromRandomShard(5, oldest_keys);
    ASSERT_EQ(EC_OK, ret);
    // The returned keys should be a subset of inserted keys
    std::set<int64_t> all_keys;
    for (int64_t i = 1; i <= 20; ++i) {
        all_keys.insert(i);
    }
    for (const auto &key : oldest_keys) {
        ASSERT_TRUE(all_keys.count(key) > 0) << "Unexpected key: " << key;
    }

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestGetOldestKeysAfterDelete) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    // Insert entries
    for (int64_t i = 1; i <= 10; ++i) {
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
                  meta_storage_backend_->Put({i}, {{{PROPERTY_URI, "uri" + std::to_string(i)}}}));
    }

    // Delete some entries
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), meta_storage_backend_->Delete({1, 3, 5}));

    // GetOldestKeysFromRandomShard should not return deleted keys
    MetaLocalBackend *local_backend = GetLocalBackend();
    std::set<int64_t> deleted_keys = {1, 3, 5};
    std::set<int64_t> remaining_keys = {2, 4, 6, 7, 8, 9, 10};

    // Run multiple times to cover different random shards
    std::vector<MetaStorageBackend::KeyType> oldest_keys;
    ErrorCode ret = local_backend->GetOldestKeysFromRandomShard(10, oldest_keys);
    ASSERT_EQ(EC_OK, ret);
    for (const auto &key : oldest_keys) {
        ASSERT_TRUE(remaining_keys.count(key) > 0) << "Deleted key " << key << " should not appear in oldest keys";
    }

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestUpsertMergesFields) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    // Put initial entry
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              meta_storage_backend_->Put({1}, {{{PROPERTY_URI, "original_uri"}, {PROPERTY_LRU_TIME, "100"}}}));

    // Upsert with only uri change - lru_time should be preserved
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Upsert({1}, {{{PROPERTY_URI, "updated_uri"}}}));

    FieldMapVec out_field_maps;
    auto results = meta_storage_backend_->Get({1}, {PROPERTY_URI, PROPERTY_LRU_TIME}, out_field_maps);
    ASSERT_EQ(EC_OK, results[0]);
    ASSERT_EQ("updated_uri", out_field_maps[0][PROPERTY_URI]);
    ASSERT_EQ("100", out_field_maps[0][PROPERTY_LRU_TIME]);

    // Upsert with only lru_time change - uri should be preserved
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Upsert({1}, {{{PROPERTY_LRU_TIME, "999"}}}));

    FieldMapVec out_field_maps2;
    results = meta_storage_backend_->Get({1}, {PROPERTY_URI, PROPERTY_LRU_TIME}, out_field_maps2);
    ASSERT_EQ(EC_OK, results[0]);
    ASSERT_EQ("updated_uri", out_field_maps2[0][PROPERTY_URI]);
    ASSERT_EQ("999", out_field_maps2[0][PROPERTY_LRU_TIME]);

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestPutMetaDataAndGetMetaData) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    FieldMap field_maps;
    field_maps["key1"] = "value1";
    ASSERT_EQ(EC_OK, meta_storage_backend_->PutMetaData(field_maps));

    FieldMap out_field_maps;
    ASSERT_EQ(EC_NOENT, meta_storage_backend_->GetMetaData(out_field_maps));

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestPutIfAbsent) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    MetaLocalBackend *local_backend = GetLocalBackend();

    // Insert new keys should succeed
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              local_backend->PutIfAbsent({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}},
                                         {EC_OK, EC_OK}));

    // Verify inserted data
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_LRU_TIME},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}});

    // Insert again with same keys should return EC_EXIST and not overwrite
    ASSERT_EQ((std::vector<ErrorCode>{EC_EXIST, EC_EXIST}),
              local_backend->PutIfAbsent({1, 2},
                                         {{{PROPERTY_URI, "uri1-new"}, {PROPERTY_LRU_TIME, "999"}},
                                          {{PROPERTY_URI, "uri2-new"}, {PROPERTY_LRU_TIME, "888"}}},
                                         {EC_OK, EC_OK}));

    // Verify original data is unchanged
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_LRU_TIME},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_LRU_TIME, "200"}}});

    // Mixed: one existing key and one new key
    ASSERT_EQ((std::vector<ErrorCode>{EC_EXIST, EC_OK}),
              local_backend->PutIfAbsent(
                  {1, 3},
                  {{{PROPERTY_URI, "uri1-again"}}, {{PROPERTY_URI, "uri3"}, {PROPERTY_LRU_TIME, "300"}}},
                  {EC_OK, EC_OK}));

    // Verify key 1 unchanged, key 3 inserted
    AssertGet(
        meta_storage_backend_.get(),
        {1, 3},
        {PROPERTY_URI, PROPERTY_LRU_TIME},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}}, {{PROPERTY_URI, "uri3"}, {PROPERTY_LRU_TIME, "300"}}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestPutIfAbsentThenDelete) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    MetaLocalBackend *local_backend = GetLocalBackend();

    // Insert a key
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              local_backend->PutIfAbsent({1}, {{{PROPERTY_URI, "uri1"}, {PROPERTY_LRU_TIME, "100"}}}, {EC_OK, EC_OK}));

    // Delete the key
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Delete({1}));

    // PutIfAbsent should succeed again after deletion
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              local_backend->PutIfAbsent(
                  {1}, {{{PROPERTY_URI, "uri1-reinserted"}, {PROPERTY_LRU_TIME, "200"}}}, {EC_OK, EC_OK}));

    // Verify the reinserted data
    AssertGet(meta_storage_backend_.get(),
              {1},
              {PROPERTY_URI, PROPERTY_LRU_TIME},
              {EC_OK},
              {{{PROPERTY_URI, "uri1-reinserted"}, {PROPERTY_LRU_TIME, "200"}}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

// ==================== storage_uri parsing tests ====================

TEST_F(MetaLocalBackendTest, TestParseStorageUri) {
    // URI with all parameters specified
    auto config = std::make_shared<MetaStorageBackendConfig>();
    config->SetStorageUri("local://?capacity=1024&num_shard_bits=4&sample_times=50");

    auto backend = std::make_shared<MetaLocalBackend>();
    ASSERT_EQ(EC_OK, backend->Init("test_instance_uri_all", config));
    ASSERT_EQ(EC_OK, backend->Open());
    ASSERT_EQ(1024 * 1024 * 1024, backend->cache_->GetCapacity());
    uint32_t expected_shard_mask = (1 << 4) - 1;
    ASSERT_EQ(expected_shard_mask, backend->shard_mask_);
    ASSERT_EQ(50, backend->sample_times_);

    // Verify the backend is functional with custom parameters
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend->Put({1}, {{{PROPERTY_URI, "uri1"}}}));
    AssertGet(backend.get(), {1}, {PROPERTY_URI}, {EC_OK}, {{{PROPERTY_URI, "uri1"}}});

    ASSERT_EQ(EC_OK, backend->Close());
}

TEST_F(MetaLocalBackendTest, TestParseStorageUriEmpty) {
    // Empty URI should use all default values
    auto config = std::make_shared<MetaStorageBackendConfig>();
    config->SetStorageUri("");

    auto backend = std::make_shared<MetaLocalBackend>();
    ASSERT_EQ(EC_OK, backend->Init("test_instance_uri_empty", config));
    ASSERT_EQ(EC_OK, backend->Open());
    ASSERT_EQ(META_LOCAL_BACKEND_DEFAULT_CAPACITY * 1024 * 1024, backend->cache_->GetCapacity());
    uint32_t expected_shard_mask = (1 << META_LOCAL_BACKEND_DEFAULT_NUM_SHARD_BITS) - 1;
    ASSERT_EQ(expected_shard_mask, backend->shard_mask_);
    ASSERT_EQ(META_LOCAL_BACKEND_DEFAULT_SAMPLE_TIMES, backend->sample_times_);

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend->Put({1}, {{{PROPERTY_URI, "uri1"}}}));
    AssertGet(backend.get(), {1}, {PROPERTY_URI}, {EC_OK}, {{{PROPERTY_URI, "uri1"}}});

    ASSERT_EQ(EC_OK, backend->Close());
}

TEST_F(MetaLocalBackendTest, TestParseStorageUriInvalidFormat) {
    // Invalid URI format (no protocol) should fall back to defaults
    auto config = std::make_shared<MetaStorageBackendConfig>();
    config->SetStorageUri("not_a_valid_uri");

    auto backend = std::make_shared<MetaLocalBackend>();
    ASSERT_EQ(EC_BADARGS, backend->Init("test_instance_uri_invalid", config));
}

} // namespace kv_cache_manager
