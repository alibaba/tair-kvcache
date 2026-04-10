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
}

std::string MetaLocalBackendTest::ExpectedStorageType() const { return META_LOCAL_BACKEND_TYPE_STR; }

TEST_F(MetaLocalBackendTest, TestSimple) {
    ASSERT_EQ(ExpectedStorageType(), meta_storage_backend_->GetStorageType());

    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    // Put two entries with uri field
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{PROPERTY_URI, "uri1"}}, {{PROPERTY_URI, "uri2"}}}));
    // UpdateFields to add hit_count
    ASSERT_EQ(
        (std::vector<ErrorCode>{EC_OK, EC_OK}),
        meta_storage_backend_->UpdateFields({1, 2}, {{{PROPERTY_HIT_COUNT, "100"}}, {{PROPERTY_HIT_COUNT, "200"}}}));

    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {true, true, false});
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_HIT_COUNT},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}});
    AssertListKeys(meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 3, EC_OK, SCAN_BASE_CURSOR, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 1, EC_OK, {1, 2});

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              meta_storage_backend_->Put({3}, {{{PROPERTY_URI, "uri3"}, {PROPERTY_HIT_COUNT, "300"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              meta_storage_backend_->UpdateFields({2}, {{{PROPERTY_URI, "uri2-updated"}}}));

    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {false, true, true});
    AssertGet(meta_storage_backend_.get(),
              {1, 2, 3},
              {PROPERTY_URI, PROPERTY_HIT_COUNT},
              {EC_NOENT, EC_OK, EC_OK},
              {{},
               {{PROPERTY_URI, "uri2-updated"}, {PROPERTY_HIT_COUNT, "200"}},
               {{PROPERTY_URI, "uri3"}, {PROPERTY_HIT_COUNT, "300"}}});
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

TEST_F(MetaLocalBackendTest, TestPut) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}}));
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_HIT_COUNT},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}});

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
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}}));

    // Update uri only, lru_time should be preserved
    ASSERT_EQ(
        (std::vector<ErrorCode>{EC_OK, EC_OK}),
        meta_storage_backend_->UpdateFields({1, 2}, {{{PROPERTY_URI, "uri1-updated"}}, {{PROPERTY_HIT_COUNT, "250"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {PROPERTY_URI, PROPERTY_HIT_COUNT},
              {EC_OK, EC_OK},
              {{{PROPERTY_URI, "uri1-updated"}, {PROPERTY_HIT_COUNT, "100"}},
               {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "250"}}});

    // Cannot update key that does not exist
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}), meta_storage_backend_->UpdateFields({3}, {{{PROPERTY_URI, "uri3"}}}));

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestUpsert) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}}));

    // Upsert: update existing, insert new
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}),
              meta_storage_backend_->Upsert(
                  {1, 2, 3},
                  {{{PROPERTY_URI, "uri1-upserted"}}, {{PROPERTY_HIT_COUNT, "250"}}, {{PROPERTY_URI, "uri3-new"}}}));
    // Verify existing keys preserve unmodified fields
    AssertGet(meta_storage_backend_.get(),
              {1, 2, 3},
              {PROPERTY_URI, PROPERTY_HIT_COUNT},
              {EC_OK, EC_OK, EC_OK},
              {{{PROPERTY_URI, "uri1-upserted"}, {PROPERTY_HIT_COUNT, "100"}},
               {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "250"}},
               {{PROPERTY_URI, "uri3-new"}}});

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
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}}));

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
        {PROPERTY_URI, PROPERTY_HIT_COUNT},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}});

    // Get no fields
    AssertGet(meta_storage_backend_.get(), {1, 2}, {}, {EC_OK, EC_OK}, FieldMapVec(2));

    // Get non-existent key returns empty values
    AssertGet(meta_storage_backend_.get(), {3}, {PROPERTY_URI, PROPERTY_HIT_COUNT}, {EC_NOENT}, {{}});

    // Get unsupported field returns empty string
    AssertGet(meta_storage_backend_.get(), {1}, {"unsupported_field"}, {EC_OK}, {{}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestGetAll) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2},
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}}));
    // GetAllFields returns all stored fields plus a dynamically generated PROPERTY_LRU_TIME.
    // AssertGetAllFields skips PROPERTY_LRU_TIME value comparison (only checks existence).
    AssertGetAllFields(meta_storage_backend_.get(),
                       {1, 2},
                       {EC_OK, EC_OK},
                       {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}, {PROPERTY_LRU_TIME, ""}},
                        {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}, {PROPERTY_LRU_TIME, ""}}});
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
                                         {PROPERTY_HIT_COUNT, "1234567890"}};

    MetaMemCacheItem *item = MetaMemCacheItem::Create(fields);
    ASSERT_NE(nullptr, item);
    ASSERT_EQ("test://some/long/uri/path/for/testing", item->GetFields().at(PROPERTY_URI));
    ASSERT_EQ("1234567890", item->GetFields().at(PROPERTY_HIT_COUNT));
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
        {PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}, {"custom_key", "custom_value"}};
    MetaMemCacheItem *custom_item = MetaMemCacheItem::Create(custom_fields);
    ASSERT_NE(nullptr, custom_item);
    ASSERT_EQ(3u, custom_item->GetFields().size());
    ASSERT_EQ("custom_value", custom_item->GetFields().at("custom_key"));

    MetaMemCacheItem::Deleter(custom_item, nullptr);
}

TEST_F(MetaLocalBackendTest, TestRandomSample) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    // No entries inserted, should return empty
    std::vector<MetaStorageBackend::KeyType> oldest_keys;
    ErrorCode ret = meta_storage_backend_->RandomSample(5, oldest_keys);
    ASSERT_EQ(EC_OK, ret);
    ASSERT_TRUE(oldest_keys.empty());

    // count = 0 should return OK with no keys
    ret = meta_storage_backend_->RandomSample(0, oldest_keys);
    ASSERT_EQ(EC_OK, ret);
    ASSERT_TRUE(oldest_keys.empty());

    // Insert multiple entries
    for (int64_t i = 1; i <= 20; ++i) {
        std::string uri = "uri" + std::to_string(i);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({i}, {{{PROPERTY_URI, uri}}}));
    }

    // RandomSample should return some keys
    ret = meta_storage_backend_->RandomSample(5, oldest_keys);
    ASSERT_EQ(EC_OK, ret);
    // The returned keys should be a subset of inserted keys
    for (const auto &key : oldest_keys) {
        ASSERT_TRUE(1 <= key && key <= 20) << "Unexpected key: " << key;
    }

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestRandomSampleAfterDelete) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    // Insert entries
    for (int64_t i = 1; i <= 10; ++i) {
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
                  meta_storage_backend_->Put({i}, {{{PROPERTY_URI, "uri" + std::to_string(i)}}}));
    }

    // Delete some entries
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), meta_storage_backend_->Delete({1, 3, 5}));

    // RandomSample should not return deleted keys
    std::set<int64_t> remaining_keys = {2, 4, 6, 7, 8, 9, 10};

    std::vector<MetaStorageBackend::KeyType> oldest_keys;
    ErrorCode ret = meta_storage_backend_->RandomSample(10, oldest_keys);
    ASSERT_EQ(EC_OK, ret);
    for (const auto &key : oldest_keys) {
        ASSERT_TRUE(remaining_keys.count(key) > 0) << "Deleted key " << key << " should not appear in oldest keys";
    }

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestPutMetaDataAndGetMetaData) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    FieldMap field_maps;
    field_maps["key1"] = "value1";
    ASSERT_EQ(EC_UNIMPLEMENTED, meta_storage_backend_->PutMetaData(field_maps));

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
                                         {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}},
                                          {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}},
                                         {EC_OK, EC_OK}));

    // Verify inserted data
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_HIT_COUNT},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}});

    // Insert again with same keys should return EC_EXIST and not overwrite
    ASSERT_EQ((std::vector<ErrorCode>{EC_EXIST, EC_EXIST}),
              local_backend->PutIfAbsent({1, 2},
                                         {{{PROPERTY_URI, "uri1-new"}, {PROPERTY_HIT_COUNT, "999"}},
                                          {{PROPERTY_URI, "uri2-new"}, {PROPERTY_HIT_COUNT, "888"}}},
                                         {EC_OK, EC_OK}));

    // Verify original data is unchanged
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2},
        {PROPERTY_URI, PROPERTY_HIT_COUNT},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}}, {{PROPERTY_URI, "uri2"}, {PROPERTY_HIT_COUNT, "200"}}});

    // Mixed: one existing key and one new key
    ASSERT_EQ((std::vector<ErrorCode>{EC_EXIST, EC_OK}),
              local_backend->PutIfAbsent(
                  {1, 3},
                  {{{PROPERTY_URI, "uri1-again"}}, {{PROPERTY_URI, "uri3"}, {PROPERTY_HIT_COUNT, "300"}}},
                  {EC_OK, EC_OK}));

    // Verify key 1 unchanged, key 3 inserted
    AssertGet(
        meta_storage_backend_.get(),
        {1, 3},
        {PROPERTY_URI, PROPERTY_HIT_COUNT},
        {EC_OK, EC_OK},
        {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}}, {{PROPERTY_URI, "uri3"}, {PROPERTY_HIT_COUNT, "300"}}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestPutIfAbsentThenDelete) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    MetaLocalBackend *local_backend = GetLocalBackend();

    // Insert a key
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              local_backend->PutIfAbsent({1}, {{{PROPERTY_URI, "uri1"}, {PROPERTY_HIT_COUNT, "100"}}}, {EC_OK, EC_OK}));

    // Delete the key
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Delete({1}));

    // PutIfAbsent should succeed again after deletion
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              local_backend->PutIfAbsent(
                  {1}, {{{PROPERTY_URI, "uri1-reinserted"}, {PROPERTY_HIT_COUNT, "200"}}}, {EC_OK, EC_OK}));

    // Verify the reinserted data
    AssertGet(meta_storage_backend_.get(),
              {1},
              {PROPERTY_URI, PROPERTY_HIT_COUNT},
              {EC_OK},
              {{{PROPERTY_URI, "uri1-reinserted"}, {PROPERTY_HIT_COUNT, "200"}}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestLruTimeUpdatedByReadWriteOps) {
    // Verify that PROPERTY_LRU_TIME (backed by last_access_time_) is updated
    // by all read/write operations: Get, GetAllFields, Exists, UpdateFields,
    // Upsert, and PutIfAbsent.
    meta_storage_backend_config_->SetStorageUri("local://?capacity=64&num_shard_bits=0&sample_times=1");
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_lru_time_ops", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    MetaLocalBackend *local_backend = GetLocalBackend();

    // Helper lambda: get LRU time for a key via Get.
    auto getLruTime = [&](int64_t key) -> int64_t {
        FieldMapVec out;
        auto ec = meta_storage_backend_->Get({key}, {PROPERTY_LRU_TIME}, out);
        EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), ec);
        return std::stoll(out[0][PROPERTY_LRU_TIME]);
    };

    // --- Put ---
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({1}, {{{PROPERTY_URI, "uri1"}}}));
    int64_t lru_after_put = getLruTime(1);
    ASSERT_GT(lru_after_put, 0);

    // Each operation sleeps 1ms (1000us) before running, so the LRU time
    // difference between consecutive operations should be at least 1000us.
    constexpr int64_t kMinTimeDiffUs = 1000;

    // --- Get updates LRU time ---
    usleep(1000);
    FieldMapVec all_out;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Get({1}, {PROPERTY_URI}, all_out));
    ASSERT_EQ("uri1", all_out[0][PROPERTY_URI]);
    int64_t lru_after_get = getLruTime(1);
    ASSERT_GE(lru_after_get - lru_after_put, kMinTimeDiffUs) << "Get should update LRU time by >= 1000us";

    // --- GetAllFields updates LRU time ---
    usleep(1000);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->GetAllFields({1}, all_out));
    ASSERT_EQ("uri1", all_out[0][PROPERTY_URI]);
    int64_t lru_after_getall = getLruTime(1);
    ASSERT_GE(lru_after_getall - lru_after_get, kMinTimeDiffUs) << "GetAllFields should update LRU time by >= 1000us";

    // --- Exists updates LRU time ---
    usleep(1000);
    std::vector<bool> exist_vec;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Exists({1}, exist_vec));
    ASSERT_TRUE(exist_vec[0]);
    int64_t lru_after_exists = getLruTime(1);
    ASSERT_GE(lru_after_exists - lru_after_getall, kMinTimeDiffUs) << "Exists should update LRU time by >= 1000us";

    // --- UpdateFields updates LRU time ---
    usleep(1000);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              meta_storage_backend_->UpdateFields({1}, {{{PROPERTY_URI, "uri1-updated"}}}));
    int64_t lru_after_update = getLruTime(1);
    ASSERT_GE(lru_after_update - lru_after_exists, kMinTimeDiffUs)
        << "UpdateFields should update LRU time by >= 1000us";

    // --- Upsert updates LRU time ---
    usleep(1000);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Upsert({1}, {{{PROPERTY_URI, "uri1-upserted"}}}));
    int64_t lru_after_upsert = getLruTime(1);
    ASSERT_GE(lru_after_upsert - lru_after_update, kMinTimeDiffUs) << "Upsert should update LRU time by >= 1000us";

    // --- PutIfAbsent on new key ---
    usleep(1000);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), local_backend->PutIfAbsent({2}, {{{PROPERTY_URI, "uri2"}}}, {EC_OK}));
    int64_t lru_key2 = getLruTime(2);
    ASSERT_GT(lru_key2, 0) << "PutIfAbsent should set LRU time on new key";

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestShardOldestAccessTimeEmptyToNonEmpty) {
    // Use 2 shards (num_shard_bits=1) to test 0→1 boundary.
    meta_storage_backend_config_->SetStorageUri("local://?capacity=64&num_shard_bits=1&sample_times=2");
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_0to1", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    MetaLocalBackend *local_backend = GetLocalBackend();

    // Initially both shards should have INT64_MAX (empty).
    ASSERT_EQ(INT64_MAX, local_backend->shard_oldest_access_time_[0].load(std::memory_order_relaxed));
    ASSERT_EQ(INT64_MAX, local_backend->shard_oldest_access_time_[1].load(std::memory_order_relaxed));

    // Insert a key.
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({100}, {{{PROPERTY_URI, "uri100"}}}));

    // At least one shard should now have a non-INT64_MAX timestamp.
    bool any_updated = (local_backend->shard_oldest_access_time_[0].load(std::memory_order_relaxed) < INT64_MAX) ||
                       (local_backend->shard_oldest_access_time_[1].load(std::memory_order_relaxed) < INT64_MAX);
    ASSERT_TRUE(any_updated) << "Inserting into an empty shard should trigger tail-change callback";

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestShardOldestAccessTimeNonEmptyToEmpty) {
    // Use 1 shard (num_shard_bits=0) to test 1→0 boundary.
    meta_storage_backend_config_->SetStorageUri("local://?capacity=64&num_shard_bits=0&sample_times=1");
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_1to0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    MetaLocalBackend *local_backend = GetLocalBackend();

    // Insert one key.
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({1}, {{{PROPERTY_URI, "uri1"}}}));
    ASSERT_LT(local_backend->shard_oldest_access_time_[0].load(std::memory_order_relaxed), INT64_MAX);

    // Delete the key — shard becomes empty, timestamp should go back to INT64_MAX.
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ(INT64_MAX, local_backend->shard_oldest_access_time_[0].load(std::memory_order_relaxed));

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestSampleReclaimKeysSelectsOldestShard) {
    // Use 2 shards (num_shard_bits=1), sample_times=2 so we can observe shard selection.
    meta_storage_backend_config_->SetStorageUri("local://?capacity=64&num_shard_bits=1&sample_times=2");
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_oldest_shard", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    MetaLocalBackend *local_backend = GetLocalBackend();

    // Insert keys into both shards.
    for (int64_t i = 1; i <= 100; ++i) {
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
                  meta_storage_backend_->Put({i}, {{{PROPERTY_URI, "uri" + std::to_string(i)}}}));
    }

    // Both shards should have entries now.
    int64_t ts0 = local_backend->shard_oldest_access_time_[0].load(std::memory_order_relaxed);
    int64_t ts1 = local_backend->shard_oldest_access_time_[1].load(std::memory_order_relaxed);
    ASSERT_LT(ts0, INT64_MAX) << "Shard 0 should have entries";
    ASSERT_LT(ts1, INT64_MAX) << "Shard 1 should have entries";

    // Sleep and then access all keys via Get to make them "newer".
    for (int64_t i = 1; i <= 50; ++i) {
        FieldMapVec out;
        meta_storage_backend_->Get({i}, {PROPERTY_URI}, out);
    }
    usleep(1000);
    for (int64_t i = 51; i <= 100; ++i) {
        FieldMapVec out;
        meta_storage_backend_->Get({i}, {PROPERTY_URI}, out);
    }

    // Now SampleReclaimKeys should return keys.
    std::vector<MetaStorageBackend::KeyType> reclaim_keys;
    ASSERT_EQ(EC_OK, meta_storage_backend_->SampleReclaimKeys(10, reclaim_keys));
    ASSERT_FALSE(reclaim_keys.empty()) << "SampleReclaimKeys should return some keys";

    // All returned keys should be valid (subset of inserted keys).
    for (const auto &key : reclaim_keys) {
        ASSERT_TRUE(1 <= key && key <= 50) << "Unexpected key: " << key;
    }

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestSampleReclaimKeysEmptyCache) {
    meta_storage_backend_config_->SetStorageUri("local://?capacity=64&num_shard_bits=2&sample_times=4");
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_empty_sample", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    // SampleReclaimKeys on empty cache should return OK with no keys.
    std::vector<MetaStorageBackend::KeyType> reclaim_keys;
    ASSERT_EQ(EC_OK, meta_storage_backend_->SampleReclaimKeys(10, reclaim_keys));
    ASSERT_TRUE(reclaim_keys.empty());

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestSampleReclaimKeysAfterDeleteUpdatesTimestamp) {
    // Use 1 shard for simplicity.
    meta_storage_backend_config_->SetStorageUri("local://?capacity=64&num_shard_bits=0&sample_times=1");
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_delete_ts", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    MetaLocalBackend *local_backend = GetLocalBackend();

    // Insert two keys with a time gap.
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({1}, {{{PROPERTY_URI, "uri1"}}}));
    usleep(1000);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({2}, {{{PROPERTY_URI, "uri2"}}}));

    int64_t ts_before_delete = local_backend->shard_oldest_access_time_[0].load(std::memory_order_relaxed);

    // Sample should return key 1 (oldest in LRU).
    std::vector<MetaStorageBackend::KeyType> reclaim_keys;
    ASSERT_EQ(EC_OK, meta_storage_backend_->SampleReclaimKeys(1, reclaim_keys));
    ASSERT_EQ(1u, reclaim_keys.size());
    ASSERT_EQ(1, reclaim_keys[0]);

    // Delete the sampled key.
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Delete(reclaim_keys));

    // After deletion, the shard's oldest timestamp should have changed.
    int64_t ts_after_delete = local_backend->shard_oldest_access_time_[0].load(std::memory_order_relaxed);
    ASSERT_GE(ts_after_delete, ts_before_delete) << "After deleting the oldest key, shard timestamp should advance";
    ASSERT_EQ(EC_OK, meta_storage_backend_->SampleReclaimKeys(1, reclaim_keys));
    ASSERT_EQ(1u, reclaim_keys.size());
    ASSERT_EQ(2, reclaim_keys[0]);

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

} // namespace kv_cache_manager
