#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_dummy_backend.h"
#include "kv_cache_manager/meta/test/meta_storage_backend_test_base.h"

using namespace kv_cache_manager;

class MetaDummyBackendTest : public MetaStorageBackendTestBase, public TESTBASE {
public:
    void SetUp() override;
    void TearDown() override;
    void ConstructMetaStorageBackend();
    void ConstructMetaStorageBackendConfig();

private:
    std::shared_ptr<MetaStorageBackend> meta_storage_backend_;
    std::shared_ptr<MetaStorageBackendConfig> meta_storage_backend_config_;
};

void MetaDummyBackendTest::SetUp() {
    ConstructMetaStorageBackend();
    ConstructMetaStorageBackendConfig();
}

void MetaDummyBackendTest::TearDown() {}

void MetaDummyBackendTest::ConstructMetaStorageBackend() {
    meta_storage_backend_ = std::make_shared<MetaDummyBackend>();
}

void MetaDummyBackendTest::ConstructMetaStorageBackendConfig() {
    meta_storage_backend_config_ = std::make_shared<MetaStorageBackendConfig>();

    const std::string local_path = GetPrivateTestRuntimeDataPath() + "_meta_dummy_backend_file1";
    std::error_code ec;
    const bool exists = std::filesystem::exists(local_path, ec);
    ASSERT_FALSE(ec) << local_path; // false means correct
    if (exists) {
        std::remove(local_path.c_str());
    }
    meta_storage_backend_config_->SetStorageUri("file://" + local_path);
}

TEST_F(MetaDummyBackendTest, TestSimple) {
    ASSERT_EQ(META_DUMMY_BACKEND_TYPE_STR, meta_storage_backend_->GetStorageType());

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}}, {{"f1", "v2-1"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->UpdateFields({1, 2}, {{{"f2", "v1-2"}}, {{"f2", "v2-2"}}}));

    AssertExists(meta_storage_backend_.get(),
                 {1, 2, 3},
                 {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK},
                 /*is_exist*/ {true, true, false});
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    AssertListKeys(
        meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 3, ErrorCode::EC_OK, SCAN_BASE_CURSOR, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 1, ErrorCode::EC_OK, {1, 2});

    ConstructMetaStorageBackend(); // recover
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());
    AssertExists(meta_storage_backend_.get(),
                 {1, 2, 3},
                 (std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK}),
                 /*is_exist*/ {true, true, false});

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_NOENT}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              meta_storage_backend_->Put({3}, {{{"f1", "v3-1"}, {"f2", "v3-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              meta_storage_backend_->UpdateFields({2}, {{{"f1", "v2-1-1"}}}));

    AssertExists(meta_storage_backend_.get(),
                 {1, 2, 3},
                 {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK},
                 /*is_exist*/ {false, true, true});
    AssertGet(meta_storage_backend_.get(),
              {1, 2, 3},
              {"f1", "f2"},
              {ErrorCode::EC_NOENT, ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{}, {{"f1", "v2-1-1"}, {"f2", "v2-2"}}, {{"f1", "v3-1"}, {"f2", "v3-2"}}});
    AssertListKeys(
        meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 3, ErrorCode::EC_OK, SCAN_BASE_CURSOR, {2, 3});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 1, ErrorCode::EC_OK, {2, 3});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestInit) {
    // invalid config
    ASSERT_NE(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", /*config*/ nullptr));
    ASSERT_NE(ErrorCode::EC_OK, meta_storage_backend_->Init(/*instance_id*/ "", meta_storage_backend_config_));
}

TEST_F(MetaDummyBackendTest, TestPut) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1}, {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}})); // cover old value
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2", "f3"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestUpdateFields) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->UpdateFields(
                  {1, 2}, {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}, {{"f2", "v2-2-1"}}})); // merge old value
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2", "f3"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f1", "v1-1-1"}, {"f2", "v1-2"}, {"f3", "v1-3"}}, {{"f1", "v2-1"}, {"f2", "v2-2-1"}}});

    // can not update key that dont exist
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_NOENT}),
              meta_storage_backend_->UpdateFields({3}, {{{"f1", "v3-1"}}}));

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestUpsert) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    // update or insert
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Upsert(
                  {1, 2, 3}, {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}, {{"f2", "v2-2-1"}}, {{"f3", "v3-1"}}}));
    AssertGet(
        meta_storage_backend_.get(),
        {1, 2, 3},
        {"f1", "f2", "f3"},
        {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK},
        {{{"f1", "v1-1-1"}, {"f2", "v1-2"}, {"f3", "v1-3"}}, {{"f1", "v2-1"}, {"f2", "v2-2-1"}}, {{"f3", "v3-1"}}});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestDelete) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ(
        (std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK}),
        meta_storage_backend_->Put(
            {1, 2, 3},
            {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}, {{"f1", "v3-1"}, {"f2", "v3-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}), meta_storage_backend_->Delete({1, 3}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_NOENT, ErrorCode::EC_NOENT}),
              meta_storage_backend_->Delete({1, 3}));
    AssertExists(meta_storage_backend_.get(),
                 {1, 2, 3},
                 {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK},
                 /*is_exist*/ {false, true, false});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestGet) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f1", "v1-1"}}, {{"f1", "v2-1"}}}); // part fields
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}); // all fields
    AssertGet(
        meta_storage_backend_.get(), {1, 2}, {}, {ErrorCode::EC_OK, ErrorCode::EC_OK}, FieldMapVec(2)); // no fields
    AssertGet(meta_storage_backend_.get(), {3}, {"f1", "f2"}, {ErrorCode::EC_NOENT}, {{}});             // key not exist

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestGetAll) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGetAllFields(meta_storage_backend_.get(),
                       {1, 2},
                       {ErrorCode::EC_OK, ErrorCode::EC_OK},
                       {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    AssertGetAllFields(meta_storage_backend_.get(), {3}, {ErrorCode::EC_NOENT}, FieldMapVec(1)); // no entry

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestExists) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertExists(meta_storage_backend_.get(),
                 {1, 2, 3},
                 {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK},
                 /*is_exist*/ {true, true, false});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestListKeys) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              meta_storage_backend_->Put({2}, {{{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              meta_storage_backend_->Put({3}, {{{"f1", "v3-1"}, {"f2", "v3-2"}}}));

    // list keys by step
    std::string current_cursor = SCAN_BASE_CURSOR;
    for (std::string next_cursor; current_cursor != SCAN_BASE_CURSOR; current_cursor = next_cursor) {
        AssertListKeysByStep(
            meta_storage_backend_.get(), current_cursor, /*limit*/ 1, ErrorCode::EC_OK, {1, 2, 3}, next_cursor);
    }

    // list all keys
    AssertListKeys(meta_storage_backend_.get(),
                   SCAN_BASE_CURSOR,
                   /*limit*/ std::numeric_limits<std::int64_t>::max(),
                   ErrorCode::EC_OK,
                   SCAN_BASE_CURSOR,
                   {1, 2, 3});

    // invalid cursor
    AssertListKeys(meta_storage_backend_.get(), "invalid_cursor", /*limit*/ 1, ErrorCode::EC_BADARGS, "", {1, 2, 3});

    // list no key, limit = 0
    std::string next_cursor;
    AssertListKeysByStep(
        meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 0, ErrorCode::EC_OK, {1, 2, 3}, next_cursor);

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestRandomSample) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              meta_storage_backend_->Put({2}, {{{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 0, ErrorCode::EC_OK, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 1, ErrorCode::EC_OK, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 2, ErrorCode::EC_OK, {1, 2});
    AssertSampleReclaimKeys(meta_storage_backend_.get(), /*count*/ 3, ErrorCode::EC_OK, {1, 2});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestRecover) {
    for (std::int32_t i = 0; i != 10; ++i) {
        ConstructMetaStorageBackend(); // new meta storage backend
        ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
        ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

        {
            std::string keyStr = std::to_string(i);
            ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
                      meta_storage_backend_->Put({i}, {{{"f" + keyStr, "v" + keyStr}}}));
        }

        for (std::int32_t j = 0; j <= i; ++j) {
            std::string keyStr = std::to_string(j);
            AssertGet(
                meta_storage_backend_.get(), {j}, {"f" + keyStr}, {ErrorCode::EC_OK}, {{{"f" + keyStr, "v" + keyStr}}});
        }

        ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
    }
}

TEST_F(MetaDummyBackendTest, TestRecoverBinarySafe) {
    for (std::int32_t i = 0; i != 10; ++i) {
        ConstructMetaStorageBackend(); // new meta storage backend
        ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test instance 0", meta_storage_backend_config_));
        ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

        {
            std::string keyStr = std::to_string(i);
            ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
                      meta_storage_backend_->Put({i}, {{{"f " + keyStr, "v " + keyStr}}}));
        }

        for (std::int32_t j = 0; j <= i; ++j) {
            std::string keyStr = std::to_string(j);
            AssertGet(meta_storage_backend_.get(),
                      {j},
                      {"f " + keyStr},
                      {ErrorCode::EC_OK},
                      {{{"f " + keyStr, "v " + keyStr}}});
        }

        ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
    }
}

TEST_F(MetaDummyBackendTest, TestDeleteFields) {
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_storage_backend_->Init("test_instance_delete_fields", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put(
                  {1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}, {"f3", "v1-3"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));

    // key 1: delete one of three fields; key 2: delete all listed fields;
    // key 3: not exist -> EC_NOENT.
    AssertDeleteFields(meta_storage_backend_.get(),
                       {1, 2, 3},
                       {{"f1"}, {"f1", "f2"}, {"f1"}},
                       {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_NOENT});

    // Surviving fields are intact; deleted fields are absent.
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2", "f3"},
              {ErrorCode::EC_OK, ErrorCode::EC_OK},
              {{{"f2", "v1-2"}, {"f3", "v1-3"}}, {}});

    // Deleting a non-existent field on an existing key still returns EC_OK.
    AssertDeleteFields(meta_storage_backend_.get(), {1}, {{"not_exist_field"}}, {ErrorCode::EC_OK});

    // Empty field list on an existing key is a no-op (EC_OK), original fields kept.
    AssertDeleteFields(meta_storage_backend_.get(), {1}, {{}}, {ErrorCode::EC_OK});
    AssertGet(meta_storage_backend_.get(), {1}, {"f2", "f3"}, {ErrorCode::EC_OK}, {{{"f2", "v1-2"}, {"f3", "v1-3"}}});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestExistsFieldWithPrefix) {
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_storage_backend_->Init("test_instance_exists_prefix", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    // key 1: has LOCATION_PREFIX field; key 2: has only normal fields; key 3: not exist.
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{LOCATION_PREFIX + "a", "la"}, {"p0", "v0"}}, {{"p0", "v0"}}}));

    AssertExistsFieldWithPrefix(meta_storage_backend_.get(),
                                {1, 2, 3},
                                LOCATION_PREFIX,
                                {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_NOENT},
                                {true, false, false});

    // After removing the only LOCATION_PREFIX field from key 1, prefix check is false.
    AssertDeleteFields(meta_storage_backend_.get(), {1}, {{LOCATION_PREFIX + "a"}}, {ErrorCode::EC_OK});
    AssertExistsFieldWithPrefix(meta_storage_backend_.get(), {1}, LOCATION_PREFIX, {ErrorCode::EC_OK}, {false});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestGetPerKeyFields) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Put(
                  {1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}, {"f3", "v1-3"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));

    // Each key queries a different subset; key 3 does not exist; key 2 asks for
    // a missing field "f3" which is silently dropped from the returned map.
    FieldMapVec field_maps;
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_NOENT}),
              meta_storage_backend_->Get(
                  {1, 2, 3}, std::vector<std::vector<std::string>>{{"f1", "f3"}, {"f2", "f3"}, {"f1"}}, field_maps));
    ASSERT_EQ((FieldMapVec{{{"f1", "v1-1"}, {"f3", "v1-3"}}, {{"f2", "v2-2"}}, {}}), field_maps);

    // Size mismatch between keys and field_names_vec -> EC_BADARGS per key.
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_BADARGS, ErrorCode::EC_BADARGS}),
              meta_storage_backend_->Get({1, 2}, std::vector<std::vector<std::string>>{{"f1"}}, field_maps));

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}
