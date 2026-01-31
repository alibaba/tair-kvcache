#include <filesystem>

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

private:
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

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}}, {{"f1", "v2-1"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->UpdateFields({1, 2}, {{{"f2", "v1-2"}}, {{"f2", "v2-2"}}}));

    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {true, true, false});
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    AssertListKeys(meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 3, EC_OK, SCAN_BASE_CURSOR, {1, 2});
    AssertRandomSample(meta_storage_backend_.get(), /*count*/ 1, EC_OK, {1, 2});

    ConstructMetaStorageBackend(); // recover
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());
    AssertExists(meta_storage_backend_.get(),
                 {1, 2, 3},
                 (std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}),
                 /*is_exist*/ {true, true, false});

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}), meta_storage_backend_->Delete({1}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({3}, {{{"f1", "v3-1"}, {"f2", "v3-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->UpdateFields({2}, {{{"f1", "v2-1-1"}}}));

    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {false, true, true});
    AssertGet(meta_storage_backend_.get(),
              {1, 2, 3},
              {"f1", "f2"},
              {EC_OK, EC_OK, EC_OK},
              {{{"f1", ""}, {"f2", ""}}, {{"f1", "v2-1-1"}, {"f2", "v2-2"}}, {{"f1", "v3-1"}, {"f2", "v3-2"}}});
    AssertListKeys(meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 3, EC_OK, SCAN_BASE_CURSOR, {2, 3});
    AssertRandomSample(meta_storage_backend_.get(), /*count*/ 1, EC_OK, {2, 3});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestInit) {
    // invalid config
    ASSERT_NE(EC_OK, meta_storage_backend_->Init("test_instance_0", /*config*/ nullptr));
    ASSERT_NE(EC_OK, meta_storage_backend_->Init(/*instance_id*/ "", meta_storage_backend_config_));
}

TEST_F(MetaLocalBackendTest, TestPut) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              meta_storage_backend_->Put({1}, {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}})); // cover old value
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2", "f3"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1-1"}, {"f2", ""}, {"f3", "v1-3"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}, {"f3", ""}}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestUpdateFields) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->UpdateFields(
                  {1, 2}, {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}, {{"f2", "v2-2-1"}}})); // merge old value
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2", "f3"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1-1"}, {"f2", "v1-2"}, {"f3", "v1-3"}}, {{"f1", "v2-1"}, {"f2", "v2-2-1"}, {"f3", ""}}});

    // can not update key that dont exist
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}), meta_storage_backend_->UpdateFields({3}, {{{"f1", "v3-1"}}}));

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestUpsert) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    // update or insert
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}),
              meta_storage_backend_->Upsert(
                  {1, 2, 3}, {{{"f1", "v1-1-1"}, {"f3", "v1-3"}}, {{"f2", "v2-2-1"}}, {{"f3", "v3-1"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2, 3},
              {"f1", "f2", "f3"},
              {EC_OK, EC_OK, EC_OK},
              {{{"f1", "v1-1-1"}, {"f2", "v1-2"}, {"f3", "v1-3"}},
               {{"f1", "v2-1"}, {"f2", "v2-2-1"}, {"f3", ""}},
               {{"f1", ""}, {"f2", ""}, {"f3", "v3-1"}}});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestIncrFields) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put(
                  {1, 2}, {{{"hit_count", "10"}, {"weight", "100"}}, {{"hit_count", "20"}, {"weight", "200"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"hit_count", "weight"},
              {EC_OK, EC_OK},
              {{{"hit_count", "10"}, {"weight", "100"}}, {{"hit_count", "20"}, {"weight", "200"}}});

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->IncrFields({1, 2}, {{"hit_count", /*amount*/ 2}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"hit_count", "weight"},
              {EC_OK, EC_OK},
              {{{"hit_count", "12"}, {"weight", "100"}}, {{"hit_count", "22"}, {"weight", "200"}}});
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->IncrFields({1, 2}, {{"weight", /*amount*/ 3}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"hit_count", "weight"},
              {EC_OK, EC_OK},
              {{{"hit_count", "12"}, {"weight", "103"}}, {{"hit_count", "22"}, {"weight", "203"}}});

    // can not update key that dont exist
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT}),
              meta_storage_backend_->IncrFields({3}, {{"hit_count", /*amount*/ 2}}));

    // can not update field that is not num
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({4}, {{{"f1", "v1"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_BADARGS}), meta_storage_backend_->IncrFields({4}, {{"f1", /*amount*/ 2}}));
    // can not update field that is not exist
    ASSERT_EQ((std::vector<ErrorCode>{EC_BADARGS}), meta_storage_backend_->IncrFields({4}, {{"f2", /*amount*/ 2}}));

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestDelete) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ(
        (std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}),
        meta_storage_backend_->Put(
            {1, 2, 3},
            {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}, {{"f1", "v3-1"}, {"f2", "v3-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), meta_storage_backend_->Delete({1, 3}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_NOENT, EC_NOENT}), meta_storage_backend_->Delete({1, 3}));
    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {false, true, false});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestGet) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1"}}, {{"f1", "v2-1"}}}); // part fields
    AssertGet(meta_storage_backend_.get(),
              {1, 2},
              {"f1", "f2"},
              {EC_OK, EC_OK},
              {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});                // all fields
    AssertGet(meta_storage_backend_.get(), {1, 2}, {}, {EC_OK, EC_OK}, FieldMapVec(2));             // no fields
    AssertGet(meta_storage_backend_.get(), {3}, {"f1", "f2"}, {EC_OK}, {{{"f1", ""}, {"f2", ""}}}); // key not exist

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestGetAll) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertGetAllFields(meta_storage_backend_.get(),
                       {1, 2},
                       {EC_OK, EC_OK},
                       {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}});
    AssertGetAllFields(meta_storage_backend_.get(), {3}, {EC_NOENT}, FieldMapVec(1)); // no entry

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestExists) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}),
              meta_storage_backend_->Put({1, 2}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}, {{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertExists(meta_storage_backend_.get(), {1, 2, 3}, {EC_OK, EC_OK, EC_OK}, /*is_exist*/ {true, true, false});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestListKeys) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({1}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({2}, {{{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({3}, {{{"f1", "v3-1"}, {"f2", "v3-2"}}}));

    // list keys by step
    std::string current_cursor = SCAN_BASE_CURSOR;
    for (std::string next_cursor; current_cursor != SCAN_BASE_CURSOR; current_cursor = next_cursor) {
        AssertListKeysByStep(meta_storage_backend_.get(), current_cursor, /*limit*/ 1, EC_OK, {1, 2, 3}, next_cursor);
    }

    // list all keys
    AssertListKeys(meta_storage_backend_.get(),
                   SCAN_BASE_CURSOR,
                   /*limit*/ std::numeric_limits<int64_t>::max(),
                   EC_OK,
                   SCAN_BASE_CURSOR,
                   {1, 2, 3});

    // invalid cursor
    AssertListKeys(meta_storage_backend_.get(), "invalid_cursor", /*limit*/ 1, EC_BADARGS, "", {1, 2, 3});

    // list no key, limit = 0
    std::string next_cursor;
    AssertListKeysByStep(meta_storage_backend_.get(), SCAN_BASE_CURSOR, /*limit*/ 0, EC_OK, {1, 2, 3}, next_cursor);

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestRandomSample) {
    ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
    ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({1}, {{{"f1", "v1-1"}, {"f2", "v1-2"}}}));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({2}, {{{"f1", "v2-1"}, {"f2", "v2-2"}}}));
    AssertRandomSample(meta_storage_backend_.get(), /*count*/ 0, EC_OK, {1, 2});
    AssertRandomSample(meta_storage_backend_.get(), /*count*/ 1, EC_OK, {1, 2});
    AssertRandomSample(meta_storage_backend_.get(), /*count*/ 2, EC_OK, {1, 2});
    AssertRandomSample(meta_storage_backend_.get(), /*count*/ 3, EC_OK, {1, 2});

    ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
}

TEST_F(MetaLocalBackendTest, TestRecover) {
    for (int32_t i = 0; i < 10; ++i) {
        ConstructMetaStorageBackend(); // new meta storage backend
        ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test_instance_0", meta_storage_backend_config_));
        ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

        std::string keyStr = std::to_string(i);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({i}, {{{"f" + keyStr, "v" + keyStr}}}));
        for (int j = 0; j <= i; ++j) {
            std::string keyStr = std::to_string(j);
            AssertGet(meta_storage_backend_.get(), {j}, {"f" + keyStr}, {EC_OK}, {{{"f" + keyStr, "v" + keyStr}}});
        }

        ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
    }
}

TEST_F(MetaLocalBackendTest, TestRecoverBinarySafe) {
    for (int32_t i = 0; i < 10; ++i) {
        ConstructMetaStorageBackend(); // new meta storage backend
        ASSERT_EQ(EC_OK, meta_storage_backend_->Init("test instance 0", meta_storage_backend_config_));
        ASSERT_EQ(EC_OK, meta_storage_backend_->Open());

        std::string keyStr = std::to_string(i);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), meta_storage_backend_->Put({i}, {{{"f " + keyStr, "v " + keyStr}}}));
        for (int j = 0; j <= i; ++j) {
            std::string keyStr = std::to_string(j);
            AssertGet(meta_storage_backend_.get(), {j}, {"f " + keyStr}, {EC_OK}, {{{"f " + keyStr, "v " + keyStr}}});
        }

        ASSERT_EQ(EC_OK, meta_storage_backend_->Close());
    }
}

} // namespace kv_cache_manager
