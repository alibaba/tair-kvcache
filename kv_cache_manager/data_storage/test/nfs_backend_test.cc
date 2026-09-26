#include <algorithm>
#include <atomic>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

namespace {

class FaultInjectingNfsBackend final : public NfsBackend {
public:
    enum class Mode {
        kError,
        kShortResult
    };

    FaultInjectingNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry, Mode mode)
        : NfsBackend(std::move(metrics_registry)), mode_(mode) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        if (cb) {
            cb();
        }
        if (mode_ == Mode::kShortResult) {
            return std::vector<ErrorCode>(storage_uris.empty() ? 0 : storage_uris.size() - 1, EC_OK);
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_IO_ERROR);
    }

private:
    Mode mode_;
};

class DirectorySyncFailingNfsBackend final : public NfsBackend {
public:
    explicit DirectorySyncFailingNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::size_t sync_attempts() const noexcept { return sync_attempts_.load(std::memory_order_relaxed); }

protected:
    bool SyncKvMetaDeleteDirectory(const std::string &) const noexcept override {
        sync_attempts_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

private:
    mutable std::atomic<std::size_t> sync_attempts_{0};
};

class AdmissionRootSyncFailingNfsBackend final : public NfsBackend {
public:
    explicit AdmissionRootSyncFailingNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::size_t sync_attempts() const noexcept { return sync_attempts_.load(std::memory_order_relaxed); }

protected:
    bool PersistKvMetaAdmissionRoot() const noexcept override {
        sync_attempts_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

private:
    mutable std::atomic<std::size_t> sync_attempts_{0};
};

class AdmissionRootSyncCountingNfsBackend final : public NfsBackend {
public:
    explicit AdmissionRootSyncCountingNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::size_t sync_attempts() const noexcept { return sync_attempts_; }

protected:
    bool PersistKvMetaAdmissionRoot() const noexcept override {
        ++sync_attempts_;
        return true;
    }

private:
    mutable std::size_t sync_attempts_{0};
};

class CapacityProbeNfsBackend final : public NfsBackend {
public:
    CapacityProbeNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry, ErrorCode result)
        : NfsBackend(std::move(metrics_registry)), result_(result) {}

    std::size_t capacity_attempts() const noexcept { return capacity_attempts_; }

protected:
    CreatePreflightResult
    CheckKvMetaAdmissionCapacity(const std::vector<CreatePreflightItem> &items) const noexcept override {
        ++capacity_attempts_;
        observed_sizes_.clear();
        observed_keys_.clear();
        std::uint64_t requested_bytes = 0;
        for (const CreatePreflightItem &item : items) {
            observed_sizes_.push_back(item.value_size);
            observed_keys_.push_back(item.allocation_key);
            requested_bytes += item.value_size;
        }
        return {result_, result_ == EC_NOSPC ? requested_bytes : 0, result_ == EC_NOSPC ? items.size() : 0};
    }

public:
    std::uint64_t observed_bytes() const noexcept {
        return std::accumulate(observed_sizes_.begin(), observed_sizes_.end(), std::uint64_t{0});
    }
    std::size_t observed_objects() const noexcept { return observed_sizes_.size(); }
    const std::vector<std::uint64_t> &observed_sizes() const noexcept { return observed_sizes_; }
    const std::vector<std::string> &observed_keys() const noexcept { return observed_keys_; }

private:
    ErrorCode result_;
    mutable std::size_t capacity_attempts_{0};
    mutable std::vector<std::uint64_t> observed_sizes_;
    mutable std::vector<std::string> observed_keys_;
};

class CapacityPolicyNfsBackend : public NfsBackend {
public:
    using NfsBackend::ComputeKvMetaCapacityPressure;
    using NfsBackend::CountMissingKvMetaNamespaceDirectories;
};

} // namespace

class NfsBackendTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        test_root_ = GetPrivateTestRuntimeDataPath() + "nfs_root/";
        outside_root_ = GetPrivateTestRuntimeDataPath() + "nfs_outside/";
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(test_root_, ec);
        ec.clear();
        std::filesystem::remove_all(outside_root_, ec);
    }
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::string test_root_;
    std::string outside_root_;
};

// TestSimple参考版，使用Open(StorageConfig)初始化，其他测试用例都调整如下：
TEST_F(NfsBackendTest, TestSimple) {
    // 一个key是一个文件的形式
    {
        NfsBackend backend(metrics_registry_);
        std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
        spec->set_key_count_per_file(1);
        spec->set_root_path("/data/");
        StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
        ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
        std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
        auto results = backend.Create(keys, 128, "fake_trace_id_2", []() {});
        ASSERT_EQ(results.size(), keys.size());
        ASSERT_EQ("file:///data/key1?size=128", results[0].second.ToUriString());
        ASSERT_EQ("file:///data/key2?size=128", results[1].second.ToUriString());
        ASSERT_EQ("file:///data/key3?size=128", results[2].second.ToUriString());
        ASSERT_EQ("file:///data/key4?size=128", results[3].second.ToUriString());
        ASSERT_EQ("file:///data/key5?size=128", results[4].second.ToUriString());
    }
    // 多个key一个文件的形式
    {
        NfsBackend backend(metrics_registry_);
        std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
        spec->set_key_count_per_file(2);
        spec->set_root_path("/data/");
        StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
        ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_3"));
        std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
        auto results = backend.Create(keys, 128, "fake_trace_id_4", []() {});
        ASSERT_EQ(results.size(), keys.size());
        EXPECT_EQ("file:///data/key1_5a560a3d977cc6f2?blkid=0&size=128", results[0].second.ToUriString());
        EXPECT_EQ("file:///data/key1_5a560a3d977cc6f2?blkid=1&size=128", results[1].second.ToUriString());
        EXPECT_EQ("file:///data/key3_1184f2d3fc112241?blkid=0&size=128", results[2].second.ToUriString());
        EXPECT_EQ("file:///data/key3_1184f2d3fc112241?blkid=1&size=128", results[3].second.ToUriString());
        EXPECT_EQ("file:///data/key5?blkid=0&size=128", results[4].second.ToUriString());
    }
}

TEST_F(NfsBackendTest, TestGetTypeAndAvailableStatus) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(2);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_FALSE(backend.Available());
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    ASSERT_TRUE(backend.Available());
    ASSERT_EQ(backend.Close(), EC_OK);
    ASSERT_FALSE(backend.Available());
}

TEST_F(NfsBackendTest, TestCreateWithBatchingAndCallbackInvocation) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(2);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
    bool callback_called = false;
    auto callback = [&callback_called]() { callback_called = true; };
    auto results = backend.Create(keys, 100, "fake_trace_id_2", callback);
    ASSERT_TRUE(callback_called);
    ASSERT_EQ(results.size(), keys.size());
    EXPECT_EQ(results[0].second.ToUriString(), "file:///data/key1_5a560a3d977cc6f2?blkid=0&size=100");
    EXPECT_EQ(results[1].second.ToUriString(), "file:///data/key1_5a560a3d977cc6f2?blkid=1&size=100");
    EXPECT_EQ(results[2].second.ToUriString(), "file:///data/key3_1184f2d3fc112241?blkid=0&size=100");
    EXPECT_EQ(results[3].second.ToUriString(), "file:///data/key3_1184f2d3fc112241?blkid=1&size=100");
    EXPECT_EQ(results[4].second.ToUriString(), "file:///data/key5?blkid=0&size=100");
    for (size_t i = 0; i < results.size(); ++i) {
        ASSERT_EQ(results[i].first, EC_OK);
    }
}

TEST_F(NfsBackendTest, TestCreateWithBatchSizeOneAndEmptyKeys) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    bool callback_called = false;
    auto cb = [&callback_called]() { callback_called = true; };
    auto results_empty = backend.Create({}, 100, "fake_trace_id_2", cb);
    ASSERT_TRUE(callback_called);
    ASSERT_TRUE(results_empty.empty());

    std::vector<std::string> keys = {"a", "b"};
    callback_called = false;
    auto results = backend.Create(keys, 100, "fake_trace_id_3", [&callback_called]() { callback_called = true; });
    ASSERT_TRUE(callback_called);
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_EQ(results[0].second.ToUriString(), "file:///data/a?size=100");
    ASSERT_EQ(results[1].second.ToUriString(), "file:///data/b?size=100");
}

TEST_F(NfsBackendTest, TestDeleteReturnsOkAndSameSize) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<DataStorageUri> uris(3);
    auto res = backend.Delete(uris, "fake_trace_id_2", []() {});
    ASSERT_EQ(res.size(), uris.size());
    for (auto code : res) {
        ASSERT_EQ(code, EC_OK);
    }
}

TEST_F(NfsBackendTest, TestKvMetaCreateRequiresPreExistingDurableRootBeforeReturningUri) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'a');

    NfsBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    EXPECT_TRUE(backend.HasDedicatedKvMetaCreate());
    ASSERT_FALSE(std::filesystem::exists(test_root_));
    bool callback_called = false;
    const auto missing = backend.CreateForKvMeta({object_key}, 17, "missing_root", [&]() { callback_called = true; });
    ASSERT_EQ(1u, missing.size());
    EXPECT_EQ(EC_IO_ERROR, missing.front().first);
    EXPECT_FALSE(missing.front().second.Valid());
    EXPECT_FALSE(callback_called);
    EXPECT_FALSE(std::filesystem::exists(test_root_));

    ASSERT_TRUE(std::filesystem::create_directories(test_root_));
    const auto created = backend.CreateForKvMeta({object_key}, 17, "durable_root", [&]() { callback_called = true; });
    ASSERT_EQ(1u, created.size());
    EXPECT_EQ(EC_OK, created.front().first);
    EXPECT_TRUE(callback_called);
    EXPECT_EQ((std::filesystem::path(test_root_) / object_key).string(), created.front().second.GetPath());
    EXPECT_EQ("17", created.front().second.GetParam("size"));
}

TEST_F(NfsBackendTest, TestKvMetaCreateRejectsAmbiguousOrOverbroadConfiguredRoots) {
    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'a');
    for (const std::string &root :
         {std::string("relative/"), std::string("/"), std::string("/tmp/../cache/"), std::string("/tmp//cache/")}) {
        auto spec = std::make_shared<NfsStorageSpec>();
        spec->set_key_count_per_file(1);
        spec->set_root_path(root);
        const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
        EXPECT_FALSE(HasSafeConfiguredKvMetaNamespace(storage_config));

        NfsBackend backend(metrics_registry_);
        ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
        const auto result = backend.CreateForKvMeta({object_key}, 17, "ambiguous_root", nullptr);
        ASSERT_EQ(1u, result.size());
        EXPECT_EQ(EC_BADARGS, result.front().first) << root;
        EXPECT_FALSE(result.front().second.Valid()) << root;
    }
    EXPECT_TRUE(HasCanonicalKvMetaNfsRootPath("/tmp/cache/"));
}

TEST_F(NfsBackendTest, TestKvMetaCreateReturnsNoAuthorityWhenRootSyncFails) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    AdmissionRootSyncFailingNfsBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 's');
    bool callback_called = false;
    const auto result = backend.CreateForKvMeta({object_key}, 17, "sync_failure", [&]() { callback_called = true; });
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(EC_IO_ERROR, result.front().first);
    EXPECT_FALSE(result.front().second.Valid());
    EXPECT_EQ(1u, backend.sync_attempts());
    EXPECT_FALSE(callback_called);
    const auto retry = backend.CreateForKvMeta({object_key}, 17, "sync_failure_retry", nullptr);
    ASSERT_EQ(1u, retry.size());
    EXPECT_EQ(EC_IO_ERROR, retry.front().first);
    EXPECT_EQ(2u, backend.sync_attempts()) << "a failed durability barrier must never be cached";
}

TEST_F(NfsBackendTest, TestKvMetaCreateReportsAuthoritativeCapacityFailureBeforeReturningUri) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    CapacityProbeNfsBackend backend(metrics_registry_, EC_NOSPC);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'c');
    bool callback_called = false;
    const auto result = backend.CreateForKvMeta({object_key}, 123, "capacity", [&]() { callback_called = true; });

    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(EC_NOSPC, result.front().first);
    EXPECT_FALSE(result.front().second.Valid());
    EXPECT_FALSE(callback_called);
    EXPECT_EQ(1u, backend.capacity_attempts());
    EXPECT_EQ(123u, backend.observed_bytes());
    EXPECT_EQ(1u, backend.observed_objects());
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(test_root_) / object_key));
}

TEST_F(NfsBackendTest, TestKvMetaPreflightChecksTheCompleteBatchBytesAndObjectCount) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    CapacityProbeNfsBackend backend(metrics_registry_, EC_NOSPC);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    const std::vector<std::uint64_t> value_sizes{1, 2, 3, 5, 8, 13, 268};
    std::vector<KvMetaDataStorageBackendExtension::CreatePreflightItem> items;
    for (std::size_t i = 0; i < value_sizes.size(); ++i) {
        items.push_back(
            {"kvmeta/1/" + std::to_string(i + 1) + "/" + std::string(kKvMetaObjectNonceBytes, 'a'), value_sizes[i]});
    }
    const auto result = backend.PreflightKvMetaCreate(items);

    EXPECT_EQ(EC_NOSPC, result.ec);
    EXPECT_EQ(300u, result.reclaim_bytes);
    EXPECT_EQ(7u, result.reclaim_objects);
    EXPECT_EQ(1u, backend.capacity_attempts());
    EXPECT_EQ(300u, backend.observed_bytes());
    EXPECT_EQ(7u, backend.observed_objects());
    EXPECT_EQ(value_sizes, backend.observed_sizes());
    ASSERT_EQ(items.size(), backend.observed_keys().size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        EXPECT_EQ(items[i].allocation_key, backend.observed_keys()[i]);
    }
}

TEST_F(NfsBackendTest, TestKvMetaCapacityPolicyPreservesVariableSizeBlockDistribution) {
    constexpr std::uint64_t block_size = 4096;
    constexpr std::uint64_t available_blocks = 2;

    const auto fits = CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure(
        {4095, 4095}, 0, block_size, available_blocks, false, 0);
    EXPECT_EQ(EC_OK, fits.ec);
    EXPECT_EQ(0u, fits.reclaim_bytes);
    EXPECT_EQ(0u, fits.reclaim_objects);

    // The same object count and a smaller aggregate logical size can require
    // three blocks when its exact per-object distribution is retained.
    const auto full =
        CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure({4097, 1}, 0, block_size, available_blocks, false, 0);
    EXPECT_EQ(EC_NOSPC, full.ec);
    EXPECT_EQ(4096u, full.reclaim_bytes);
    EXPECT_EQ(0u, full.reclaim_objects);
}

TEST_F(NfsBackendTest, TestKvMetaCapacityPolicySeparatesInodePressureFromBytePressure) {
    constexpr std::uint64_t block_size = 4096;
    const auto inode_full = CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure({1, 1}, 4, block_size, 6, true, 5);
    EXPECT_EQ(EC_NOSPC, inode_full.ec);
    EXPECT_EQ(0u, inode_full.reclaim_bytes);
    EXPECT_EQ(1u, inode_full.reclaim_objects);

    const auto inode_accounting_unknown =
        CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure({1, 1}, 4, block_size, 6, false, 0);
    EXPECT_EQ(EC_OK, inode_accounting_unknown.ec);
}

TEST_F(NfsBackendTest, TestKvMetaCapacityPolicyAccountsForMissingNamespaceBlocks) {
    constexpr std::uint64_t block_size = 4096;
    const auto directory_full =
        CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure({1}, 3, block_size, 3, false, 0);
    EXPECT_EQ(EC_NOSPC, directory_full.ec);
    EXPECT_EQ(1u, directory_full.reclaim_bytes);
    EXPECT_EQ(0u, directory_full.reclaim_objects);

    const auto fits = CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure({1}, 3, block_size, 4, false, 0);
    EXPECT_EQ(EC_OK, fits.ec);
}

TEST_F(NfsBackendTest, TestKvMetaNamespaceProbeDeduplicatesSharedAncestorsAndRejectsSymlinks) {
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));
    const int root_fd = ::open(test_root_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    ASSERT_GE(root_fd, 0);

    const std::string nonce(kKvMetaObjectNonceBytes, 'a');
    std::vector<KvMetaDataStorageBackendExtension::CreatePreflightItem> items{
        {"kvmeta/1/a/" + nonce, 1}, {"kvmeta/1/b/" + nonce, 2}, {"kvmeta/1/a/" + nonce, 3}};
    std::uint64_t missing_directories = 99;
    EXPECT_TRUE(CapacityPolicyNfsBackend::CountMissingKvMetaNamespaceDirectories(root_fd, items, missing_directories));
    EXPECT_EQ(4u, missing_directories) << "kvmeta and instance ancestors are shared across the batch";

    ASSERT_TRUE(std::filesystem::create_directories(std::filesystem::path(test_root_) / "kvmeta/1/a"));
    EXPECT_TRUE(CapacityPolicyNfsBackend::CountMissingKvMetaNamespaceDirectories(root_fd, items, missing_directories));
    EXPECT_EQ(1u, missing_directories) << "only the second key namespace is absent";

    ASSERT_TRUE(std::filesystem::create_directories(outside_root_));
    const auto symlink_path = std::filesystem::path(test_root_) / "kvmeta/1/c";
    std::filesystem::create_directory_symlink(outside_root_, symlink_path);
    ASSERT_TRUE(std::filesystem::is_symlink(symlink_path));
    items = {{"kvmeta/1/c/" + nonce, 1}};
    EXPECT_FALSE(CapacityPolicyNfsBackend::CountMissingKvMetaNamespaceDirectories(root_fd, items, missing_directories));
    EXPECT_EQ(0u, missing_directories);

    EXPECT_EQ(0, ::close(root_fd));
}

TEST_F(NfsBackendTest, TestKvMetaCapacityPolicyRejectsInvalidOrUnrepresentableProbesWithoutGcPressure) {
    const auto missing_block_size = CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure({1}, 0, 0, 1, false, 0);
    EXPECT_EQ(EC_IO_ERROR, missing_block_size.ec);
    EXPECT_EQ(0u, missing_block_size.reclaim_bytes);
    EXPECT_EQ(0u, missing_block_size.reclaim_objects);

    const auto invalid_value = CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure({0}, 0, 4096, 1, false, 0);
    EXPECT_EQ(EC_BADARGS, invalid_value.ec);
    EXPECT_EQ(0u, invalid_value.reclaim_bytes);
    EXPECT_EQ(0u, invalid_value.reclaim_objects);

    const auto overflow = CapacityPolicyNfsBackend::ComputeKvMetaCapacityPressure(
        {std::numeric_limits<std::uint64_t>::max(), 1}, 0, 1, std::numeric_limits<std::uint64_t>::max(), false, 0);
    EXPECT_EQ(EC_OUT_OF_LIMIT, overflow.ec);
    EXPECT_EQ(0u, overflow.reclaim_bytes);
    EXPECT_EQ(0u, overflow.reclaim_objects);
}

TEST_F(NfsBackendTest, TestKvMetaCapacityProbeFailureDoesNotMasqueradeAsPressure) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    CapacityProbeNfsBackend backend(metrics_registry_, EC_IO_ERROR);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'd');
    const auto result = backend.CreateForKvMeta({object_key}, 123, "probe_error", nullptr);

    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(EC_IO_ERROR, result.front().first);
    EXPECT_FALSE(result.front().second.Valid());
    EXPECT_EQ(1u, backend.capacity_attempts());
}

TEST_F(NfsBackendTest, TestKvMetaCreateCachesRootBarrierUntilBackendReopen) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    AdmissionRootSyncCountingNfsBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "first_open"));
    const std::string first_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'f');
    const std::string second_key = "kvmeta/1/3/" + std::string(kKvMetaObjectNonceBytes, 's');
    ASSERT_EQ(EC_OK, backend.CreateForKvMeta({first_key}, 17, "first", nullptr).front().first);
    ASSERT_EQ(EC_OK, backend.CreateForKvMeta({second_key}, 19, "second", nullptr).front().first);
    EXPECT_EQ(1u, backend.sync_attempts());

    ASSERT_EQ(EC_OK, backend.Close());
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "second_open"));
    ASSERT_EQ(EC_OK, backend.CreateForKvMeta({first_key}, 17, "after_reopen", nullptr).front().first);
    EXPECT_EQ(2u, backend.sync_attempts());
}

TEST_F(NfsBackendTest, TestKvMetaCreateProbesCachedRootOnEveryAdmission) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    AdmissionRootSyncCountingNfsBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    const std::string first_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'a');
    const std::string second_key = "kvmeta/1/3/" + std::string(kKvMetaObjectNonceBytes, 'b');
    ASSERT_EQ(EC_OK, backend.CreateForKvMeta({first_key}, 17, "first", nullptr).front().first);
    ASSERT_EQ(1u, backend.sync_attempts());

    std::error_code remove_ec;
    ASSERT_TRUE(std::filesystem::remove(test_root_, remove_ec));
    ASSERT_FALSE(remove_ec) << remove_ec.message();
    bool callback_called = false;
    const auto missing =
        backend.CreateForKvMeta({second_key}, 19, "missing_after_cache", [&]() { callback_called = true; });
    ASSERT_EQ(1u, missing.size());
    EXPECT_EQ(EC_IO_ERROR, missing.front().first);
    EXPECT_FALSE(missing.front().second.Valid());
    EXPECT_FALSE(callback_called);
    EXPECT_EQ(1u, backend.sync_attempts());
}

TEST_F(NfsBackendTest, TestKvMetaCreateRejectsReplacedRootUntilBackendReopen) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    AdmissionRootSyncCountingNfsBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    const std::string first_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'c');
    const std::string second_key = "kvmeta/1/3/" + std::string(kKvMetaObjectNonceBytes, 'd');
    ASSERT_EQ(EC_OK, backend.CreateForKvMeta({first_key}, 17, "first", nullptr).front().first);

    std::error_code replace_ec;
    ASSERT_TRUE(std::filesystem::remove(test_root_, replace_ec));
    ASSERT_FALSE(replace_ec) << replace_ec.message();
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));
    EXPECT_EQ(EC_IO_ERROR, backend.CreateForKvMeta({second_key}, 19, "replaced", nullptr).front().first);
    EXPECT_EQ(1u, backend.sync_attempts());

    ASSERT_EQ(EC_OK, backend.Close());
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "reopen"));
    EXPECT_EQ(EC_OK, backend.CreateForKvMeta({second_key}, 19, "accepted_after_reopen", nullptr).front().first);
    EXPECT_EQ(2u, backend.sync_attempts());
}

TEST_F(NfsBackendTest, TestKvMetaCreateSerializesTheInitialRootBarrier) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    AdmissionRootSyncCountingNfsBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    constexpr std::size_t kThreadCount = 16;
    std::vector<std::thread> threads;
    std::vector<ErrorCode> results(kThreadCount, EC_ERROR);
    for (std::size_t i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&, i]() {
            const std::string key = "kvmeta/1/" + std::to_string(i + 1) + "/" +
                                    std::string(kKvMetaObjectNonceBytes, static_cast<char>('a' + i));
            results[i] = backend.CreateForKvMeta({key}, i + 1, "concurrent", nullptr).front().first;
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    EXPECT_TRUE(std::all_of(results.begin(), results.end(), [](ErrorCode ec) { return ec == EC_OK; }));
    EXPECT_EQ(1u, backend.sync_attempts());
}

TEST_F(NfsBackendTest, TestKvMetaCreateRejectsBatchingAndUnterminatedRoot) {
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));
    const std::string key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'z');

    auto valid_spec = std::make_shared<NfsStorageSpec>();
    valid_spec->set_key_count_per_file(8);
    valid_spec->set_root_path(test_root_);
    NfsBackend batched(metrics_registry_);
    ASSERT_EQ(EC_OK,
              batched.Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", valid_spec), "open"));
    const auto batched_result = batched.CreateForKvMeta({key, key}, 17, "batch", nullptr);
    ASSERT_EQ(2u, batched_result.size());
    for (const auto &[ec, uri] : batched_result) {
        EXPECT_EQ(EC_BADARGS, ec);
        EXPECT_FALSE(uri.Valid());
    }

    auto bad_spec = std::make_shared<NfsStorageSpec>();
    bad_spec->set_key_count_per_file(1);
    bad_spec->set_root_path(test_root_.substr(0, test_root_.size() - 1));
    NfsBackend unterminated(metrics_registry_);
    ASSERT_EQ(EC_OK,
              unterminated.Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", bad_spec), "open"));
    EXPECT_EQ(EC_BADARGS, unterminated.CreateForKvMeta({key}, 17, "bad_root", nullptr).front().first);
}

TEST_F(NfsBackendTest, TestKvMetaDeleteRemovesAndConfirmsExactFile) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'a');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path.parent_path()));
    {
        std::ofstream output(object_path);
        ASSERT_TRUE(output.good());
        output << "embedding";
    }

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "9");

    // Keep the historical fixed-block path unchanged: only the KVMeta side
    // capability may remove singleton objects.
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), backend.Delete({uri}, "legacy_delete", nullptr));
    EXPECT_TRUE(std::filesystem::exists(object_path));

    bool callback_called = false;
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}),
              backend.DeleteAndConfirmAbsent({uri}, "delete", [&]() { callback_called = true; }));
    EXPECT_TRUE(callback_called);
    EXPECT_FALSE(std::filesystem::exists(object_path));
    EXPECT_FALSE(std::filesystem::exists(object_path.parent_path()));
    EXPECT_FALSE(std::filesystem::exists(object_path.parent_path().parent_path()));
    EXPECT_TRUE(std::filesystem::exists(object_path.parent_path().parent_path().parent_path()));

    // Exact delete is idempotent: an already-absent generation is terminal.
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), backend.DeleteAndConfirmAbsent({uri}, "delete_again", nullptr));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteRejectsFileOutsideConfiguredNamespace) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'b');
    const std::filesystem::path outside_path = std::filesystem::path(outside_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(outside_path.parent_path()));
    {
        std::ofstream output(outside_path);
        ASSERT_TRUE(output.good());
        output << "do-not-delete";
    }

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(outside_path.string());
    uri.SetParam("size", "13");
    bool callback_called = false;
    EXPECT_EQ((std::vector<ErrorCode>{EC_CORRUPTION}),
              backend.DeleteAndConfirmAbsent({uri}, "reject", [&]() { callback_called = true; }));
    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(std::filesystem::exists(outside_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteNeverTraversesAnIntermediateNamespaceSymlink) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 's');
    const std::filesystem::path redirected_object =
        std::filesystem::path(outside_root_) / "1" / "2" / std::string(kKvMetaObjectNonceBytes, 's');
    ASSERT_TRUE(std::filesystem::create_directories(redirected_object.parent_path()));
    {
        std::ofstream output(redirected_object);
        ASSERT_TRUE(output.good());
        output << "must-not-follow";
    }
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));
    std::error_code symlink_ec;
    std::filesystem::create_directory_symlink(outside_root_, std::filesystem::path(test_root_) / "kvmeta", symlink_ec);
    ASSERT_FALSE(symlink_ec) << symlink_ec.message();

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath((std::filesystem::path(test_root_) / object_key).string());
    uri.SetParam("size", "15");
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}),
              backend.DeleteAndConfirmAbsent({uri}, "intermediate_symlink", nullptr));
    EXPECT_TRUE(std::filesystem::exists(redirected_object));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteRejectsASymlinkGenerationWithoutTouchingItsTarget) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'l');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    const std::filesystem::path target_path = std::filesystem::path(outside_root_) / "target";
    ASSERT_TRUE(std::filesystem::create_directories(object_path.parent_path()));
    ASSERT_TRUE(std::filesystem::create_directories(target_path.parent_path()));
    {
        std::ofstream output(target_path);
        ASSERT_TRUE(output.good());
        output << "must-not-delete";
    }
    std::error_code symlink_ec;
    std::filesystem::create_symlink(target_path, object_path, symlink_ec);
    ASSERT_FALSE(symlink_ec) << symlink_ec.message();

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "15");
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}), backend.DeleteAndConfirmAbsent({uri}, "leaf_symlink", nullptr));
    EXPECT_TRUE(std::filesystem::is_symlink(object_path));
    EXPECT_TRUE(std::filesystem::exists(target_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteRejectsAReplacedConfiguredRoot) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'r');
    ASSERT_EQ(EC_OK, backend.CreateForKvMeta({object_key}, 17, "anchor", nullptr).front().first);
    std::error_code rename_ec;
    std::filesystem::rename(test_root_, outside_root_, rename_ec);
    ASSERT_FALSE(rename_ec) << rename_ec.message();

    const std::filesystem::path replacement_object = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(replacement_object.parent_path()));
    {
        std::ofstream output(replacement_object);
        ASSERT_TRUE(output.good());
        output << "replacement";
    }
    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(replacement_object.string());
    uri.SetParam("size", "11");
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}), backend.DeleteAndConfirmAbsent({uri}, "replaced_root", nullptr));
    EXPECT_TRUE(std::filesystem::exists(replacement_object));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteDoesNotRemoveFileWhenLegacyHookFailsOrReturnsShortResult) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'c');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path.parent_path()));
    {
        std::ofstream output(object_path);
        ASSERT_TRUE(output.good());
        output << "preserve-on-uncertainty";
    }

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "23");

    FaultInjectingNfsBackend failing(metrics_registry_, FaultInjectingNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing.Open(storage_config, "open_failing"));
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}), failing.DeleteAndConfirmAbsent({uri}, "failed_delete", nullptr));
    EXPECT_TRUE(std::filesystem::exists(object_path));

    FaultInjectingNfsBackend short_result(metrics_registry_, FaultInjectingNfsBackend::Mode::kShortResult);
    ASSERT_EQ(EC_OK, short_result.Open(storage_config, "open_short"));
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH}),
              short_result.DeleteAndConfirmAbsent({uri}, "short_delete", nullptr));
    EXPECT_TRUE(std::filesystem::exists(object_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteRetainsLedgerWhenDirectoryDurabilityIsUncertain) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'f');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path.parent_path()));
    {
        std::ofstream output(object_path);
        ASSERT_TRUE(output.good());
        output << "deleted-but-not-yet-durable";
    }

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "27");

    DirectorySyncFailingNfsBackend failing(metrics_registry_);
    ASSERT_EQ(EC_OK, failing.Open(storage_config, "open_failing"));
    bool callback_called = false;
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}),
              failing.DeleteAndConfirmAbsent({uri}, "uncertain_sync", [&]() { callback_called = true; }));
    EXPECT_TRUE(callback_called);
    EXPECT_EQ(1u, failing.sync_attempts());
    EXPECT_FALSE(std::filesystem::exists(object_path));
    EXPECT_TRUE(std::filesystem::exists(object_path.parent_path()));

    // The durable tombstone makes retrying an already-unlinked generation
    // safe. A later successful barrier may confirm absence and prune parents.
    NfsBackend retry(metrics_registry_);
    ASSERT_EQ(EC_OK, retry.Open(storage_config, "open_retry"));
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), retry.DeleteAndConfirmAbsent({uri}, "retry", nullptr));
    EXPECT_FALSE(std::filesystem::exists(object_path.parent_path()));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteConfirmsAllocationThatWasNeverMaterialized) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, '0');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_FALSE(std::filesystem::exists(object_path.parent_path().parent_path().parent_path()));

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "64");

    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), backend.DeleteAndConfirmAbsent({uri}, "never_materialized", nullptr));
    EXPECT_FALSE(std::filesystem::exists(object_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteDoesNotTreatAMissingConfiguredRootAsRemoteAbsence) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    ASSERT_FALSE(std::filesystem::exists(test_root_));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'm');
    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath((std::filesystem::path(test_root_) / object_key).string());
    uri.SetParam("size", "64");

    // An unmounted NFS root can look exactly like an absent local path. Keep
    // the durable tombstone instead of releasing quota on that observation.
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}),
              backend.DeleteAndConfirmAbsent({uri}, "missing_configured_root", nullptr));
    EXPECT_FALSE(std::filesystem::exists(test_root_));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteReportsFilesystemFailureWithoutClaimingAbsence) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'd');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path / "child"));

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "1");
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}),
              backend.DeleteAndConfirmAbsent({uri}, "non_empty_directory", nullptr));
    EXPECT_TRUE(std::filesystem::exists(object_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteNeverTreatsAnEmptyDirectoryAsAnObjectGeneration) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'e');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path));

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "1");
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}), backend.DeleteAndConfirmAbsent({uri}, "empty_directory", nullptr));
    EXPECT_TRUE(std::filesystem::is_directory(object_path));
}

TEST_F(NfsBackendTest, TestExistReturnsTrues) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    // TODO(qisa.cb) 没实现
    std::vector<DataStorageUri> uris(5);
    auto res = backend.Exist(uris);
    ASSERT_EQ(res.size(), uris.size());
    for (bool flag : res) {
        ASSERT_TRUE(flag);
    }
}

TEST_F(NfsBackendTest, TestLockAndUnLockReturnOk) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<DataStorageUri> uris(4);
    auto lock_res = backend.Lock(uris);
    auto unlock_res = backend.UnLock(uris);
    ASSERT_EQ(lock_res.size(), uris.size());
    ASSERT_EQ(unlock_res.size(), uris.size());
    for (auto code : lock_res) {
        ASSERT_EQ(code, EC_OK);
    }
    for (auto code : unlock_res) {
        ASSERT_EQ(code, EC_OK);
    }
}

TEST_F(NfsBackendTest, TestCreateHandlesInvalidBatchSize) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(0); // 0 应该被内部处理为 1
    spec->set_root_path("/root/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<std::string> keys = {"k1", "k2"};
    bool cb_called = false;
    auto results = backend.Create(keys, 50, "fake_trace_id_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_EQ(results[0].second.ToUriString(), "file:///root/k1?size=50");
    ASSERT_EQ(results[1].second.ToUriString(), "file:///root/k2?size=50");
}

TEST_F(NfsBackendTest, TestCreateSingleKeyBatch) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(10);
    spec->set_root_path("/root/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<std::string> keys = {"singlekey"};
    bool cb_called = false;
    auto results = backend.Create(keys, 10, "fake_trace_id_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_EQ(results[0].second.ToUriString(), "file:///root/singlekey?blkid=0&size=10");
}
