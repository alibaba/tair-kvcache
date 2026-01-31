#include <atomic>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class MetaSearcherManagerTest : public TESTBASE {
public:
    void SetUp() override {
        registry_manager_ = std::make_shared<RegistryManager>("local://fake", std::make_shared<MetricsRegistry>());
        std::vector<LocationSpecInfo> location_spec_infos = {LocationSpecInfo("tp0", 512),
                                                             LocationSpecInfo("tp1", 512)};
        auto instance_info = std::make_shared<InstanceInfo>(
            "test_quota_group", "test_group", "test_instance", 64, location_spec_infos, ModelDeployment());
        registry_manager_->instance_infos_["test_instance"] = instance_info;
        auto instance_group = std::make_shared<InstanceGroup>();
        registry_manager_->instance_group_configs_["test_group"] = instance_group;
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        instance_group->cache_config_ = std::make_shared<CacheConfig>();
        instance_group->cache_config_->meta_indexer_config_ = meta_indexer_config;
        auto backend_config = std::make_shared<MetaStorageBackendConfig>();
        backend_config->storage_type_ = META_LOCAL_BACKEND_TYPE_STR;
        meta_indexer_config->meta_storage_backend_config_ = backend_config;
        meta_searcher_manager_ =
            std::make_unique<MetaSearcherManager>(registry_manager_, std::make_shared<MetaIndexerManager>());
    }

    void TearDown() override {}

private:
    std::shared_ptr<RegistryManager> registry_manager_;
    std::unique_ptr<MetaSearcherManager> meta_searcher_manager_;
};

TEST_F(MetaSearcherManagerTest, TestMultiThreadCreate) {
    size_t thread_num = 4;
    std::atomic<MetaSearcher *> searcher = nullptr;
    std::atomic<bool> go = false;
    auto thread_fcn = [this, &searcher, &go]() {
        while (!go.load(std::memory_order_relaxed)) {}
        std::shared_ptr<RequestContext> request_context(new RequestContext("test_trace_id"));
        MetaSearcher *real =
            this->meta_searcher_manager_->TryCreateMetaSearcher(request_context.get(), "test_instance");
        MetaSearcher *expected = nullptr;
        if (!searcher.compare_exchange_strong(expected, real, std::memory_order_acq_rel)) {
            ASSERT_EQ(searcher.load(std::memory_order_relaxed), real);
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
        searcher.store(nullptr);
        meta_searcher_manager_ =
            std::make_unique<MetaSearcherManager>(registry_manager_, std::make_shared<MetaIndexerManager>());
        go.store(false, std::memory_order_relaxed);
    }
}

TEST_F(MetaSearcherManagerTest, TestDoCleanup) {
    // 1. 创建 MetaSearcher
    std::shared_ptr<RequestContext> request_context(new RequestContext("test_trace_id"));
    MetaSearcher *searcher = meta_searcher_manager_->TryCreateMetaSearcher(request_context.get(), "test_instance");
    ASSERT_NE(searcher, nullptr);

    // 验证可以获取到创建的 MetaSearcher
    MetaSearcher *retrieved_searcher = meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_EQ(retrieved_searcher, searcher);

    // 2. 调用 DoCleanup
    meta_searcher_manager_->DoCleanup();

    // 3. 验证清理后获取不到 MetaSearcher
    MetaSearcher *after_cleanup_searcher = meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_EQ(after_cleanup_searcher, nullptr);

    // 4. 再次调用 DoCleanup 应该正常工作
    meta_searcher_manager_->DoCleanup();

    // 5. 清理掉MetaIndexer
    meta_searcher_manager_->meta_indexer_manager_->DoCleanup();

    // 6. 重新创建 MetaSearcher
    MetaSearcher *new_searcher = meta_searcher_manager_->TryCreateMetaSearcher(request_context.get(), "test_instance");
    ASSERT_NE(new_searcher, nullptr);

    // 7. 验证可以获取到新创建的 MetaSearcher
    retrieved_searcher = meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_EQ(retrieved_searcher, new_searcher);
}

} // namespace kv_cache_manager
