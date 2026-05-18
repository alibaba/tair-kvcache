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
              PutWithFieldMaps(meta_storage_backend_.get(), {1, 2}, {{{"f1", "v1-1"}}, {{"f1", "v2-1"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              UpsertWithFieldMaps(meta_storage_backend_.get(), {1, 2}, {{{"f2", "v1-2"}}, {{"f2", "v2-2"}}}));

    AssertExists(meta_storage_backend_.get(),
                 {1, 2, 3},
                 {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK},
                 /*is_exist*/ {true, true, false});
    AssertGetProperties(meta_storage_backend_.get(),
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

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}), meta_storage_backend_->Delete(nullptr, {1}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_NOENT}), meta_storage_backend_->Delete(nullptr, {1}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              PutWithFieldMaps(meta_storage_backend_.get(), {3}, {{{"f1", "v3-1"}, {"f2", "v3-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              UpsertWithFieldMaps(meta_storage_backend_.get(), {2}, {{{"f1", "v2-1-1"}}}));

    AssertExists(meta_storage_backend_.get(),
                 {1, 2, 3},
                 {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK},
                 /*is_exist*/ {false, true, true});
    AssertGetProperties(meta_storage_backend_.get(),
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
              PutWithFieldMaps(meta_storage_backend_.get(),
                               {1, 2},
                               {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGetProperties(meta_storage_backend_.get(),
                        {1, 2},
                        {"f1", "f2"},
                        {ErrorCode::EC_OK, ErrorCode::EC_OK},
                        {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});

    ASSERT_EQ(
        (std::vector<ErrorCode>{ErrorCode::EC_OK}),
        PutWithFieldMaps(meta_storage_backend_.get(), {1}, {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}})); // cover old value
    AssertGetProperties(meta_storage_backend_.get(),
                        {1, 2},
                        {"f1", "f2", "f3"},
                        {ErrorCode::EC_OK, ErrorCode::EC_OK},
                        {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestUpsert) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              PutWithFieldMaps(meta_storage_backend_.get(),
                               {1, 2},
                               {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGetProperties(meta_storage_backend_.get(),
                        {1, 2},
                        {"f1", "f2"},
                        {ErrorCode::EC_OK, ErrorCode::EC_OK},
                        {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    // update or insert
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_OK}),
              UpsertWithFieldMaps(meta_storage_backend_.get(),
                                  {1, 2, 3},
                                  {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}, {{"f2", "v2-2-1"}}, {{"f3", "v3-1"}}}));
    AssertGetProperties(
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
        PutWithFieldMaps(
            meta_storage_backend_.get(),
            {1, 2, 3},
            {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}, {{"f1", "v3-1"}, {"f2", "v3-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              meta_storage_backend_->Delete(nullptr, {1, 3}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_NOENT, ErrorCode::EC_NOENT}),
              meta_storage_backend_->Delete(nullptr, {1, 3}));
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
              PutWithFieldMaps(meta_storage_backend_.get(),
                               {1, 2},
                               {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGetProperties(meta_storage_backend_.get(),
                        {1, 2},
                        {"f1"},
                        {ErrorCode::EC_OK, ErrorCode::EC_OK},
                        {{{"f1", "v1-1"}}, {{"f1", "v2-1"}}}); // part fields
    AssertGetProperties(meta_storage_backend_.get(),
                        {1, 2},
                        {"f1", "f2"},
                        {ErrorCode::EC_OK, ErrorCode::EC_OK},
                        {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}); // all fields
    AssertGetProperties(
        meta_storage_backend_.get(), {1, 2}, {}, {ErrorCode::EC_OK, ErrorCode::EC_OK}, FieldMapVec(2)); // no fields
    AssertGetProperties(meta_storage_backend_.get(), {3}, {"f1", "f2"}, {ErrorCode::EC_NOENT}, {{}});   // key not exist

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestGetAll) {
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              PutWithFieldMaps(meta_storage_backend_.get(),
                               {1, 2},
                               {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
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
              PutWithFieldMaps(meta_storage_backend_.get(),
                               {1, 2},
                               {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
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
              PutWithFieldMaps(meta_storage_backend_.get(), {1}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              PutWithFieldMaps(meta_storage_backend_.get(), {2}, {{{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              PutWithFieldMaps(meta_storage_backend_.get(), {3}, {{{"f1", "v3-1"}, {"f2", "v3-2"}}}));

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
              PutWithFieldMaps(meta_storage_backend_.get(), {1}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK}),
              PutWithFieldMaps(meta_storage_backend_.get(), {2}, {{{"f1", "v2-1"}, {"f2", "v2-2"}}}));
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
                      PutWithFieldMaps(meta_storage_backend_.get(), {i}, {{{"f" + keyStr, "v" + keyStr}}}));
        }

        for (std::int32_t j = 0; j <= i; ++j) {
            std::string keyStr = std::to_string(j);
            AssertGetProperties(
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
                      PutWithFieldMaps(meta_storage_backend_.get(), {i}, {{{"f " + keyStr, "v " + keyStr}}}));
        }

        for (std::int32_t j = 0; j <= i; ++j) {
            std::string keyStr = std::to_string(j);
            AssertGetProperties(meta_storage_backend_.get(),
                                {j},
                                {"f " + keyStr},
                                {ErrorCode::EC_OK},
                                {{{"f " + keyStr, "v " + keyStr}}});
        }

        ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
    }
}

TEST_F(MetaDummyBackendTest, TestDeleteLocations) {
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_storage_backend_->Init("test_instance_delete_locations", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    // Seed: key 1 has two locations + one property; key 2 has one location + one property.
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              PutWithFieldMaps(meta_storage_backend_.get(),
                               {1, 2},
                               {{{LOCATION_PREFIX + "a", "la"}, {LOCATION_PREFIX + "b", "lb"}, {"p0", "v0"}},
                                {{LOCATION_PREFIX + "c", "lc"}, {"p0", "v0"}}}));

    // key 1: delete one of two locations; key 2: delete its only location;
    // key 3: does not exist -> EC_NOENT.
    AssertDeleteLocations(meta_storage_backend_.get(),
                          {1, 2, 3},
                          {{"a"}, {"c"}, {"anything"}},
                          {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_NOENT});

    // Deleting a non-existent location on an existing key still returns EC_OK.
    AssertDeleteLocations(meta_storage_backend_.get(), {1}, {{"not_exist_loc"}}, {ErrorCode::EC_OK});

    // Empty location id list on an existing key is a no-op (EC_OK).
    AssertDeleteLocations(meta_storage_backend_.get(), {2}, {{}}, {ErrorCode::EC_OK});

    // Properties survive location deletion.
    AssertGetProperties(meta_storage_backend_.get(),
                        {1, 2},
                        {"p0"},
                        {ErrorCode::EC_OK, ErrorCode::EC_OK},
                        {{{"p0", "v0"}}, {{"p0", "v0"}}});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaDummyBackendTest, TestExistsLocation) {
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_storage_backend_->Init("test_instance_exists_location", meta_storage_backend_config_));
    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Open());

    // key 1: has location; key 2: has only properties; key 3: not exist.
    ASSERT_EQ((std::vector<ErrorCode>{ErrorCode::EC_OK, ErrorCode::EC_OK}),
              PutWithFieldMaps(meta_storage_backend_.get(),
                               {1, 2},
                               {{{LOCATION_PREFIX + "a", "la"}, {"p0", "v0"}}, {{"p0", "v0"}}}));

    AssertExistsLocation(meta_storage_backend_.get(),
                         {1, 2, 3},
                         {ErrorCode::EC_OK, ErrorCode::EC_OK, ErrorCode::EC_NOENT},
                         {true, false, false});

    // After removing the only location from key 1, ExistsLocation is false.
    AssertDeleteLocations(meta_storage_backend_.get(), {1}, {{"a"}}, {ErrorCode::EC_OK});
    AssertExistsLocation(meta_storage_backend_.get(), {1}, {ErrorCode::EC_OK}, {false});

    ASSERT_EQ(ErrorCode::EC_OK, meta_storage_backend_->Close());
}
