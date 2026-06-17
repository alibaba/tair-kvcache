#include <atomic>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
namespace kv_cache_manager {

class MetaIndexerManagerTest : public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

    ErrorCode CreateMetaIndexer(const std::string &instance_id, const std::string &storage_type);

private:
    std::shared_ptr<MetaIndexerManager> manager_;
};

void MetaIndexerManagerTest::SetUp() { manager_ = std::make_shared<MetaIndexerManager>(); }

ErrorCode MetaIndexerManagerTest::CreateMetaIndexer(const std::string &instance_id, const std::string &storage_type) {
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    auto backend_config = std::make_shared<MetaStorageBackendConfig>();
    backend_config->storage_type_ = storage_type;
    meta_indexer_config->meta_storage_backend_config_ = backend_config;
    return manager_->CreateMetaIndexer(instance_id, meta_indexer_config);
}

TEST_F(MetaIndexerManagerTest, TestSingleThreadCreateAndGet) {
    // test create
    std::string id_1 = "1";
    std::string id_2 = "2";
    std::string id_3 = "3";
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_1, META_LOCAL_BACKEND_TYPE_STR));
    // ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_2, META_REDIS_BACKEND_TYPE_STR));

    // test get
    ASSERT_TRUE(manager_->GetMetaIndexer(id_1) != nullptr);
    // ASSERT_TRUE(manager_->GetMetaIndexer(id_2) != nullptr);
    ASSERT_TRUE(manager_->GetMetaIndexer(id_3) == nullptr);
    ASSERT_EQ(1, manager_->GetIndexerSize());
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR,
              manager_->GetMetaIndexer(id_1)->backend_manager_->persistent_backend_->GetStorageType());
    // ASSERT_EQ(META_REDIS_BACKEND_TYPE_STR,
    //           manager_->GetMetaIndexer(id_2)->backend_manager_->persistent_backend_->GetStorageType());

    // test delete, TODO
    // ASSERT_EQ(ErrorCode::EC_NOENT, manager_->DeleteMetaIndexer("3"));
    // ASSERT_EQ(2, manager_->GetIndexerSize());
    // ASSERT_EQ(ErrorCode::EC_OK, manager_->DeleteMetaIndexer("2"));
    // ASSERT_EQ(1, manager_->GetIndexerSize());
    // ASSERT_EQ(ErrorCode::EC_OK, manager_->DeleteMetaIndexer("1"));
    // ASSERT_EQ(0, manager_->GetIndexerSize());
}

TEST_F(MetaIndexerManagerTest, TestMultiThreadCreateAndGet) {
    size_t thread_num = 4;
    std::atomic<MetaIndexer *> meta_indexer = nullptr;
    std::atomic<int> new_count = 0;
    std::atomic<int> exist_count = 0;
    std::atomic<bool> go = false;
    auto thread_fcn = [this, &meta_indexer, &new_count, &exist_count, &go]() {
        while (!go.load(std::memory_order_relaxed)) {}
        ErrorCode ec = this->CreateMetaIndexer("unique_instance", META_LOCAL_BACKEND_TYPE_STR);
        MetaIndexer *real = this->manager_->GetMetaIndexer("unique_instance").get();
        if (ec == EC_OK) {
            new_count.fetch_add(1, std::memory_order_relaxed);
        } else if (ec == EC_EXIST) {
            exist_count.fetch_add(1, std::memory_order_relaxed);
        }
        MetaIndexer *expected = nullptr;
        if (!meta_indexer.compare_exchange_strong(expected, real, std::memory_order_acq_rel)) {
            ASSERT_EQ(meta_indexer.load(std::memory_order_relaxed), real);
        }
    };
    for (int i = 0; i < 20; ++i) {
        std::vector<std::thread> threads;
        for (int j = 0; j < thread_num; ++j) {
            threads.push_back(std::thread(thread_fcn));
        }
        go.store(true, std::memory_order_relaxed);
        for (auto &thread : threads) {
            thread.join();
        }
        ASSERT_EQ(1, new_count.load());
        ASSERT_EQ(thread_num - 1, exist_count.load());
        new_count.store(0);
        exist_count.store(0);
        meta_indexer.store(nullptr);
        manager_ = std::make_shared<MetaIndexerManager>();
        go.store(false, std::memory_order_relaxed);
    }
}

TEST_F(MetaIndexerManagerTest, TestCreateFailed) {
    std::string id_1 = "1";
    ASSERT_EQ(ErrorCode::EC_ERROR, CreateMetaIndexer(id_1, "test"));
    ASSERT_EQ(0, manager_->GetIndexerSize());
}

TEST_F(MetaIndexerManagerTest, TestDoCleanup) {
    // 创建多个 indexer
    std::string id_1 = "1";
    std::string id_2 = "2";
    std::string id_3 = "3";
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_1, META_LOCAL_BACKEND_TYPE_STR));
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_2, META_LOCAL_BACKEND_TYPE_STR));
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_3, META_LOCAL_BACKEND_TYPE_STR));

    // 验证创建成功
    ASSERT_TRUE(manager_->GetMetaIndexer(id_1) != nullptr);
    ASSERT_TRUE(manager_->GetMetaIndexer(id_2) != nullptr);
    ASSERT_TRUE(manager_->GetMetaIndexer(id_3) != nullptr);
    ASSERT_EQ(3, manager_->GetIndexerSize());

    // 执行 DoCleanup
    manager_->DoCleanup();

    // 验证清理成功
    ASSERT_TRUE(manager_->GetMetaIndexer(id_1) == nullptr);
    ASSERT_TRUE(manager_->GetMetaIndexer(id_2) == nullptr);
    ASSERT_TRUE(manager_->GetMetaIndexer(id_3) == nullptr);
    ASSERT_EQ(0, manager_->GetIndexerSize());

    // 再次执行 DoCleanup 应该也能成功
    manager_->DoCleanup();
    ASSERT_EQ(0, manager_->GetIndexerSize());

    // 清理后重新创建原来的 indexer
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_1, META_LOCAL_BACKEND_TYPE_STR));
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_2, META_LOCAL_BACKEND_TYPE_STR));
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_3, META_LOCAL_BACKEND_TYPE_STR));

    // 验证重新创建成功
    ASSERT_TRUE(manager_->GetMetaIndexer(id_1) != nullptr);
    ASSERT_TRUE(manager_->GetMetaIndexer(id_2) != nullptr);
    ASSERT_TRUE(manager_->GetMetaIndexer(id_3) != nullptr);
    ASSERT_EQ(3, manager_->GetIndexerSize());
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR,
              manager_->GetMetaIndexer(id_1)->backend_manager_->persistent_backend_->GetStorageType());
}

TEST_F(MetaIndexerManagerTest, TestDoCleanupWithEmptyManager) {
    // 空管理器执行 DoCleanup
    ASSERT_EQ(0, manager_->GetIndexerSize());
    manager_->DoCleanup();
    ASSERT_EQ(0, manager_->GetIndexerSize());
}

TEST_F(MetaIndexerManagerTest, TestDoCleanupMultiThread) {
    // 创建多个 indexer
    std::string id_1 = "1";
    std::string id_2 = "2";
    std::string id_3 = "3";
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_1, META_LOCAL_BACKEND_TYPE_STR));
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_2, META_LOCAL_BACKEND_TYPE_STR));
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(id_3, META_LOCAL_BACKEND_TYPE_STR));

    ASSERT_EQ(3, manager_->GetIndexerSize());

    std::atomic<bool> go = false;
    std::atomic<int> cleanup_count = 0;
    std::atomic<int> get_count = 0;

    // 多线程同时执行 DoCleanup 和 GetMetaIndexer
    auto cleanup_fcn = [this, &go, &cleanup_count]() {
        while (!go.load(std::memory_order_relaxed)) {}
        this->manager_->DoCleanup();
        cleanup_count.fetch_add(1, std::memory_order_relaxed);
    };

    auto get_fcn = [this, &go, &get_count]() {
        while (!go.load(std::memory_order_relaxed)) {}
        for (int i = 0; i < 10; ++i) {
            this->manager_->GetMetaIndexer("1");
            this->manager_->GetMetaIndexer("2");
            this->manager_->GetMetaIndexer("3");
        }
        get_count.fetch_add(1, std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    // 启动多个 cleanup 线程
    for (int i = 0; i < 2; ++i) {
        threads.push_back(std::thread(cleanup_fcn));
    }
    // 启动多个 get 线程
    for (int i = 0; i < 4; ++i) {
        threads.push_back(std::thread(get_fcn));
    }

    go.store(true, std::memory_order_relaxed);

    for (auto &thread : threads) {
        thread.join();
    }

    // 验证：至少有一个 cleanup 执行成功
    ASSERT_GE(cleanup_count.load(), 1);
    // 验证：所有 get 线程都完成
    ASSERT_EQ(4, get_count.load());
    // 验证：最终所有 indexer 都被清理
    ASSERT_EQ(0, manager_->GetIndexerSize());
}

// --- Per-instance boundary override tests ---

TEST_F(MetaIndexerManagerTest, TestCreateWithDefaultBoundaries) {
    // Set global default boundaries
    auto registry = std::make_shared<MetricsRegistry>();
    std::vector<double> default_boundaries = {1, 5, 30, 60};
    manager_->SetRevisitHistogramConfig(registry, default_boundaries);

    // Create indexer without per-instance boundaries (uses default)
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer("inst_default", META_LOCAL_BACKEND_TYPE_STR));
    auto indexer = manager_->GetMetaIndexer("inst_default");
    ASSERT_NE(indexer, nullptr);

    // Verify histogram was created with default boundaries
    auto *backend = dynamic_cast<MetaLocalBackend *>(indexer->backend_manager_->persistent_backend_.get());
    ASSERT_NE(backend, nullptr);
    ASSERT_NE(backend->revisit_histogram_, nullptr);
    EXPECT_EQ(backend->revisit_histogram_->GetBoundaries().size(), 4);
    EXPECT_DOUBLE_EQ(backend->revisit_histogram_->GetBoundaries()[0], 1.0);
    EXPECT_DOUBLE_EQ(backend->revisit_histogram_->GetBoundaries()[3], 60.0);
}

TEST_F(MetaIndexerManagerTest, TestCreateWithPerInstanceBoundaries) {
    // Set global default boundaries
    auto registry = std::make_shared<MetricsRegistry>();
    std::vector<double> default_boundaries = {1, 5, 30, 60};
    manager_->SetRevisitHistogramConfig(registry, default_boundaries);

    // Create indexer with per-instance boundaries (overrides default)
    std::vector<double> custom_boundaries = {1, 10, 60, 300};
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    auto backend_config = std::make_shared<MetaStorageBackendConfig>();
    backend_config->storage_type_ = META_LOCAL_BACKEND_TYPE_STR;
    meta_indexer_config->meta_storage_backend_config_ = backend_config;
    ASSERT_EQ(ErrorCode::EC_OK,
              manager_->CreateMetaIndexer("inst_custom", meta_indexer_config, custom_boundaries));

    auto indexer = manager_->GetMetaIndexer("inst_custom");
    ASSERT_NE(indexer, nullptr);

    // Verify histogram uses per-instance boundaries, NOT the default
    auto *backend = dynamic_cast<MetaLocalBackend *>(indexer->backend_manager_->persistent_backend_.get());
    ASSERT_NE(backend, nullptr);
    ASSERT_NE(backend->revisit_histogram_, nullptr);
    EXPECT_EQ(backend->revisit_histogram_->GetBoundaries().size(), 4);
    EXPECT_DOUBLE_EQ(backend->revisit_histogram_->GetBoundaries()[0], 1.0);
    EXPECT_DOUBLE_EQ(backend->revisit_histogram_->GetBoundaries()[1], 10.0);  // NOT 5.0 (default)
    EXPECT_DOUBLE_EQ(backend->revisit_histogram_->GetBoundaries()[2], 60.0);
    EXPECT_DOUBLE_EQ(backend->revisit_histogram_->GetBoundaries()[3], 300.0);  // NOT 60.0 (default)
}

TEST_F(MetaIndexerManagerTest, TestCreateWithEmptyBoundariesUsesDefault) {
    // Set global default boundaries
    auto registry = std::make_shared<MetricsRegistry>();
    std::vector<double> default_boundaries = {1, 5, 30, 60};
    manager_->SetRevisitHistogramConfig(registry, default_boundaries);

    // Create indexer with empty boundaries vector (should use default)
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    auto backend_config = std::make_shared<MetaStorageBackendConfig>();
    backend_config->storage_type_ = META_LOCAL_BACKEND_TYPE_STR;
    meta_indexer_config->meta_storage_backend_config_ = backend_config;
    std::vector<double> empty_boundaries;
    ASSERT_EQ(ErrorCode::EC_OK,
              manager_->CreateMetaIndexer("inst_fallback", meta_indexer_config, empty_boundaries));

    auto indexer = manager_->GetMetaIndexer("inst_fallback");
    ASSERT_NE(indexer, nullptr);

    // Verify histogram falls back to default boundaries
    auto *backend = dynamic_cast<MetaLocalBackend *>(indexer->backend_manager_->persistent_backend_.get());
    ASSERT_NE(backend, nullptr);
    ASSERT_NE(backend->revisit_histogram_, nullptr);
    EXPECT_EQ(backend->revisit_histogram_->GetBoundaries().size(), 4);
    EXPECT_DOUBLE_EQ(backend->revisit_histogram_->GetBoundaries()[1], 5.0);  // default, not custom
}

TEST_F(MetaIndexerManagerTest, TestExistingInstanceUnaffectedByConfigChange) {
    // Set initial default boundaries
    auto registry = std::make_shared<MetricsRegistry>();
    std::vector<double> initial_boundaries = {1, 5, 30};
    manager_->SetRevisitHistogramConfig(registry, initial_boundaries);

    // Create instance with initial boundaries
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer("inst_stable", META_LOCAL_BACKEND_TYPE_STR));

    // Verify initial boundaries
    auto *backend_before = dynamic_cast<MetaLocalBackend *>(
        manager_->GetMetaIndexer("inst_stable")->backend_manager_->persistent_backend_.get());
    ASSERT_NE(backend_before, nullptr);
    ASSERT_NE(backend_before->revisit_histogram_, nullptr);
    ASSERT_EQ(backend_before->revisit_histogram_->GetBoundaries().size(), 3);
    EXPECT_DOUBLE_EQ(backend_before->revisit_histogram_->GetBoundaries()[1], 5.0);

    // Change the global default boundaries (simulates config update)
    std::vector<double> new_boundaries = {1, 10, 60};
    manager_->SetRevisitHistogramConfig(registry, new_boundaries);

    // Existing instance's histogram MUST still use the original boundaries
    auto *backend_after = dynamic_cast<MetaLocalBackend *>(
        manager_->GetMetaIndexer("inst_stable")->backend_manager_->persistent_backend_.get());
    ASSERT_NE(backend_after, nullptr);
    ASSERT_NE(backend_after->revisit_histogram_, nullptr);
    EXPECT_EQ(backend_after->revisit_histogram_->GetBoundaries().size(), 3);
    EXPECT_DOUBLE_EQ(backend_after->revisit_histogram_->GetBoundaries()[1], 5.0);  // still 5.0, NOT 10.0
}

} // namespace kv_cache_manager
