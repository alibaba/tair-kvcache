#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/dummy_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

class DummyBackendTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        test_root_ = GetPrivateTestRuntimeDataPath() + "dummy_root/";
    }
    void TearDown() override {}

    // helper: build a StorageConfig with a DummyStorageSpec
    StorageConfig MakeConfig(const std::string &root_path, int32_t key_count_per_file = 1) {
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path(root_path);
        spec->set_key_count_per_file(key_count_per_file);
        return StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "test_dummy", spec);
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::string test_root_;
};

TEST_F(DummyBackendTest, TestGetType) {
    DummyBackend backend(metrics_registry_);
    ASSERT_EQ(backend.GetType(), DataStorageType::DATA_STORAGE_TYPE_DUMMY);
}

TEST_F(DummyBackendTest, TestAvailableBeforeAndAfterOpen) {
    DummyBackend backend(metrics_registry_);
    ASSERT_FALSE(backend.Available());

    auto config = MakeConfig(test_root_);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));
    ASSERT_TRUE(backend.Available());

    ASSERT_EQ(EC_OK, backend.Close());
    ASSERT_FALSE(backend.Available());
}

TEST_F(DummyBackendTest, TestOpenCreatesRootDirectory) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));
    ASSERT_TRUE(std::filesystem::is_directory(test_root_));
}

TEST_F(DummyBackendTest, TestOpenEmptyRootPathFails) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig("");
    ASSERT_NE(EC_OK, backend.Open(config, "trace_1"));
    ASSERT_FALSE(backend.Available());
}

TEST_F(DummyBackendTest, TestOpenWrongSpecTypeFails) {
    DummyBackend backend(metrics_registry_);
    // use an NfsStorageSpec instead of DummyStorageSpec
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_root_path(test_root_);
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "test_dummy", spec);
    ASSERT_NE(EC_OK, backend.Open(config, "trace_1"));
    ASSERT_FALSE(backend.Available());
}

TEST_F(DummyBackendTest, TestGetStorageUsageRatio) {
    DummyBackend backend(metrics_registry_);
    ASSERT_DOUBLE_EQ(0.0, backend.GetStorageUsageRatio("trace_1"));
}

TEST_F(DummyBackendTest, TestCreateSingleKeyPerFile) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    std::vector<std::string> keys = {"key1", "key2", "key3"};
    auto results = backend.Create(keys, 128, "trace_2", []() {});

    ASSERT_EQ(results.size(), keys.size());
    for (auto &[ec, uri] : results) {
        ASSERT_EQ(ec, EC_OK);
        EXPECT_EQ(uri.GetProtocol(), "dummy");
    }
    // each key maps to its own file path under base_path_
    EXPECT_NE(std::string::npos, results[0].second.ToUriString().find("key1"));
    EXPECT_NE(std::string::npos, results[1].second.ToUriString().find("key2"));
    EXPECT_NE(std::string::npos, results[2].second.ToUriString().find("key3"));
}

TEST_F(DummyBackendTest, TestCreateMultipleKeysPerFile) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 2);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    std::vector<std::string> keys = {"a", "b", "c"};
    auto results = backend.Create(keys, 64, "trace_2", []() {});

    ASSERT_EQ(results.size(), keys.size());
    for (auto &[ec, uri] : results) {
        ASSERT_EQ(ec, EC_OK);
    }
    // first two keys share a batch (should have blkid param)
    EXPECT_NE(std::string::npos, results[0].second.ToUriString().find("blkid=0"));
    EXPECT_NE(std::string::npos, results[1].second.ToUriString().find("blkid=1"));
    // third key is a single-element batch
    EXPECT_NE(std::string::npos, results[2].second.ToUriString().find("blkid=0"));
}

TEST_F(DummyBackendTest, TestCreateEmptyKeys) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    bool cb_called = false;
    auto results = backend.Create({}, 128, "trace_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_TRUE(results.empty());
}

TEST_F(DummyBackendTest, TestCreateCallbackInvoked) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    bool cb_called = false;
    backend.Create({"k"}, 64, "trace_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
}

TEST_F(DummyBackendTest, TestCreateWithZeroBatchSize) {
    DummyBackend backend(metrics_registry_);
    // key_count_per_file = 0 should be treated as 1
    auto config = MakeConfig(test_root_, 0);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    std::vector<std::string> keys = {"x", "y"};
    auto results = backend.Create(keys, 32, "trace_2", []() {});
    ASSERT_EQ(results.size(), keys.size());
    // with batch_size clamped to 1, no blkid param expected
    EXPECT_EQ(std::string::npos, results[0].second.ToUriString().find("blkid"));
    EXPECT_EQ(std::string::npos, results[1].second.ToUriString().find("blkid"));
}

TEST_F(DummyBackendTest, TestExistAndMightExistOnNonExistentFiles) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    // create URIs pointing to non-existent files
    DataStorageUri uri1;
    uri1.SetProtocol("dummy");
    uri1.SetPath(test_root_ + "no_such_file_1");
    DataStorageUri uri2;
    uri2.SetProtocol("dummy");
    uri2.SetPath(test_root_ + "no_such_file_2");
    std::vector<DataStorageUri> uris = {uri1, uri2};

    auto exist_res = backend.Exist(uris);
    ASSERT_EQ(exist_res.size(), 2u);
    EXPECT_FALSE(exist_res[0]);
    EXPECT_FALSE(exist_res[1]);

    // MightExist delegates to Exist, so same results
    auto might_res = backend.MightExist(uris);
    ASSERT_EQ(might_res.size(), 2u);
    EXPECT_FALSE(might_res[0]);
    EXPECT_FALSE(might_res[1]);
}

TEST_F(DummyBackendTest, TestExistAndMightExistOnExistingFiles) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    // create a file on disk
    std::string file_path = test_root_ + "existing_file";
    std::ofstream ofs(file_path);
    ofs.close();
    ASSERT_TRUE(std::filesystem::exists(file_path));

    DataStorageUri uri;
    uri.SetProtocol("dummy");
    uri.SetPath(file_path);
    std::vector<DataStorageUri> uris = {uri};

    auto exist_res = backend.Exist(uris);
    ASSERT_EQ(exist_res.size(), 1u);
    EXPECT_TRUE(exist_res[0]);

    auto might_res = backend.MightExist(uris);
    ASSERT_EQ(might_res.size(), 1u);
    EXPECT_TRUE(might_res[0]);
}

TEST_F(DummyBackendTest, TestMightExistDelegatesToExist) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    // create one file, leave another missing
    std::string present = test_root_ + "present";
    std::ofstream(present).close();
    std::string absent = test_root_ + "absent";

    DataStorageUri uri_present, uri_absent;
    uri_present.SetProtocol("dummy");
    uri_present.SetPath(present);
    uri_absent.SetProtocol("dummy");
    uri_absent.SetPath(absent);

    std::vector<DataStorageUri> uris = {uri_present, uri_absent};

    auto exist_res = backend.Exist(uris);
    auto might_res = backend.MightExist(uris);

    // MightExist must return exactly the same results as Exist
    ASSERT_EQ(exist_res.size(), might_res.size());
    for (std::size_t i = 0; i < exist_res.size(); ++i) {
        EXPECT_EQ(exist_res[i], might_res[i]) << "mismatch at index " << i;
    }
}

TEST_F(DummyBackendTest, TestDeleteExistingFiles) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    // create files first
    std::string f1 = test_root_ + "del_file1";
    std::string f2 = test_root_ + "del_file2";
    std::ofstream(f1).close();
    std::ofstream(f2).close();
    ASSERT_TRUE(std::filesystem::exists(f1));
    ASSERT_TRUE(std::filesystem::exists(f2));

    DataStorageUri uri1, uri2;
    uri1.SetProtocol("dummy");
    uri1.SetPath(f1);
    uri2.SetProtocol("dummy");
    uri2.SetPath(f2);

    bool cb_called = false;
    auto res = backend.Delete({uri1, uri2}, "trace_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(res.size(), 2u);
    EXPECT_EQ(res[0], EC_OK);
    EXPECT_EQ(res[1], EC_OK);

    // files should be gone
    EXPECT_FALSE(std::filesystem::exists(f1));
    EXPECT_FALSE(std::filesystem::exists(f2));
}

TEST_F(DummyBackendTest, TestDeleteNonExistentFileReturnsOk) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    DataStorageUri uri;
    uri.SetProtocol("dummy");
    uri.SetPath(test_root_ + "nonexistent");

    auto res = backend.Delete({uri}, "trace_2", []() {});
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0], EC_OK);
}

TEST_F(DummyBackendTest, TestLockAndUnLockReturnOk) {
    DummyBackend backend(metrics_registry_);

    DataStorageUri uri1, uri2, uri3;
    uri1.SetProtocol("dummy");
    uri1.SetPath("/any/path1");
    uri2.SetProtocol("dummy");
    uri2.SetPath("/any/path2");
    uri3.SetProtocol("dummy");
    uri3.SetPath("/any/path3");
    std::vector<DataStorageUri> uris = {uri1, uri2, uri3};

    auto lock_res = backend.Lock(uris);
    auto unlock_res = backend.UnLock(uris);

    ASSERT_EQ(lock_res.size(), uris.size());
    ASSERT_EQ(unlock_res.size(), uris.size());
    for (std::size_t i = 0; i < uris.size(); ++i) {
        EXPECT_EQ(lock_res[i], EC_OK);
        EXPECT_EQ(unlock_res[i], EC_OK);
    }
}

TEST_F(DummyBackendTest, TestCreateThenExistRoundTrip) {
    // end-to-end: Create produces URIs, then Exist and MightExist
    // confirm the files were NOT actually written (Create does not
    // touch disk in the current implementation -- it only generates
    // URIs). If the implementation changes to touch disk, update this.
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig(test_root_, 1);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_1"));

    std::vector<std::string> keys = {"round_trip_key"};
    auto create_res = backend.Create(keys, 256, "trace_2", []() {});
    ASSERT_EQ(create_res.size(), 1u);
    ASSERT_EQ(create_res[0].first, EC_OK);

    std::vector<DataStorageUri> uris = {create_res[0].second};

    // Create does not write to disk, so the file should not exist
    auto exist_res = backend.Exist(uris);
    ASSERT_EQ(exist_res.size(), 1u);
    EXPECT_FALSE(exist_res[0]);

    auto might_res = backend.MightExist(uris);
    ASSERT_EQ(might_res.size(), 1u);
    EXPECT_FALSE(might_res[0]);
}
