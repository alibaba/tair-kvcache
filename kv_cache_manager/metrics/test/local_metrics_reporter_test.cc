#include <cstdint>
#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/registry_local_backend.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_manager_metrics_recorder.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/local_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

using namespace kv_cache_manager;

std::size_t MetaIndexer_GetCacheUsage_stub() noexcept { return 16; }

class LocalMetricsReporterTest : public TESTBASE {
public:
    void SetUp() override {
        stub_.set(ADDR(MetaIndexer, GetCacheUsage), MetaIndexer_GetCacheUsage_stub);

        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_ = std::make_unique<LocalMetricsReporter>();
        reporter_->Init(cache_manager_, metrics_registry_, "");
    }

    void TearDown() override { stub_.reset(ADDR(MetaIndexer, GetCacheUsage)); }

    Stub stub_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<LocalMetricsReporter> reporter_;
};

TEST_F(LocalMetricsReporterTest, TestConstructor) { EXPECT_NE(reporter_, nullptr); }

TEST_F(LocalMetricsReporterTest, TestInitWithNull) {
    EXPECT_FALSE(reporter_->Init(nullptr, nullptr, ""));

    EXPECT_FALSE(reporter_->Init(nullptr, metrics_registry_, ""));

    EXPECT_FALSE(reporter_->Init(cache_manager_, nullptr, ""));

    EXPECT_TRUE(reporter_->Init(cache_manager_, metrics_registry_, ""));
}

TEST_F(LocalMetricsReporterTest, TestReportPerQuery00) {
    EXPECT_EQ(3, metrics_registry_->GetSize());
    EXPECT_NO_THROW(reporter_->ReportPerQuery(nullptr));
    EXPECT_EQ(3, metrics_registry_->GetSize());

    {
        DummyMetricsCollector collector;
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
        EXPECT_EQ(3, metrics_registry_->GetSize());
    }

    {
        // simulate the uninitialised case 1
        reporter_->cache_manager_ = nullptr;
        reporter_->metrics_registry_ = std::make_shared<MetricsRegistry>();

        ServiceMetricsCollector collector{metrics_registry_};
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));

        DataStorageMetricsCollector collector2{metrics_registry_};
        collector2.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
    }

    {
        // simulate the uninitialised case 2
        reporter_->cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_->metrics_registry_ = nullptr;

        ServiceMetricsCollector collector{metrics_registry_};
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));

        DataStorageMetricsCollector collector2{metrics_registry_};
        collector2.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
    }

    {
        // simulate the uninitialised case 3
        reporter_->cache_manager_ = nullptr;
        reporter_->metrics_registry_ = nullptr;

        ServiceMetricsCollector collector{metrics_registry_};
        collector.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));

        DataStorageMetricsCollector collector2{metrics_registry_};
        collector2.Init();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportPerQuery(&collector));
    }
}

TEST_F(LocalMetricsReporterTest, TestReportPerQuery01) {
    EXPECT_EQ(3, metrics_registry_->GetSize());

    DataStorageMetricsCollector collector(metrics_registry_);
    collector.Init();

    EXPECT_EQ(7, metrics_registry_->GetSize());

    {
        reporter_->ReportPerQuery(&collector);
        EXPECT_EQ(7, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, data_storage, create_counter, v);
        EXPECT_EQ(1, v);

        GET_METRICS_(&collector, data_storage, create_keys_counter, v);
        EXPECT_EQ(0, v);
    }

    {
        Gauge g;
        COPY_METRICS_(&collector, data_storage, create_keys_qps, g);
        g = 6.0;

        reporter_->ReportPerQuery(&collector);
        EXPECT_EQ(7, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, data_storage, create_counter, v);
        EXPECT_EQ(2, v);

        GET_METRICS_(&collector, data_storage, create_keys_counter, v);
        EXPECT_EQ(6, v);
    }

    {
        Gauge g;
        COPY_METRICS_(&collector, data_storage, create_keys_qps, g);
        g = 2.0;

        reporter_->ReportPerQuery(&collector);
        EXPECT_EQ(7, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, data_storage, create_counter, v);
        EXPECT_EQ(3, v);

        GET_METRICS_(&collector, data_storage, create_keys_counter, v);
        EXPECT_EQ(8, v);
    }
}

TEST_F(LocalMetricsReporterTest, TestReportPerQuery02) {
    EXPECT_EQ(3, metrics_registry_->GetSize());

    ServiceMetricsCollector collector(metrics_registry_);
    collector.Init();

    EXPECT_EQ(3 + 5 + 11 + 5 + 16, metrics_registry_->GetSize());

    {
        reporter_->ReportPerQuery(&collector);

        EXPECT_EQ(3 + 5 + 11 + 5 + 16, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, service, query_counter, v);
        EXPECT_EQ(1, v);

        GET_METRICS_(&collector, service, error_counter, v);
        EXPECT_EQ(0, v);
    }

    {
        Gauge ec;
        COPY_METRICS_(&collector, service, error_code, ec);
        ec = 1.0;

        reporter_->ReportPerQuery(&collector);

        EXPECT_EQ(3 + 5 + 11 + 5 + 16, metrics_registry_->GetSize());

        std::uint64_t v;
        GET_METRICS_(&collector, service, query_counter, v);
        EXPECT_EQ(2, v);

        GET_METRICS_(&collector, service, error_counter, v);
        EXPECT_EQ(1, v);
    }
}

TEST_F(LocalMetricsReporterTest, TestReportInterval00) {
    {
        // simulate the uninitialised case 1
        reporter_->cache_manager_ = nullptr;
        reporter_->metrics_registry_ = std::make_shared<MetricsRegistry>();
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        // simulate the uninitialised case 2
        reporter_->cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_->metrics_registry_ = nullptr;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        // simulate the uninitialised case 3
        reporter_->cache_manager_ = nullptr;
        reporter_->metrics_registry_ = nullptr;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }
}

TEST_F(LocalMetricsReporterTest, TestReportInterval01) {
    {
        registry_manager_->data_storage_manager_ = nullptr;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        auto be = std::make_shared<NfsBackend>(metrics_registry_);
        be->SetOpen(true);
        be->SetAvailable(true);
        registry_manager_->data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
        registry_manager_->data_storage_manager_->storage_map_.emplace("abc", be);
        reporter_->ReportInterval();

        const auto mc_vec = reporter_->data_storage_interval_metrics_collectors_.GetMetricsCollectors();
        EXPECT_EQ(1, mc_vec.size());
        auto p = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(mc_vec.front());
        EXPECT_TRUE(p);
        double v;
        GET_METRICS_(p, data_storage, healthy_status, v);
        EXPECT_DOUBLE_EQ(1., v);
        GET_METRICS_(p, data_storage, storage_usage_ratio, v);
        EXPECT_DOUBLE_EQ(0., v);
    }

    {
        auto be = std::make_shared<NfsBackend>(metrics_registry_);
        be->SetOpen(true);
        be->SetAvailable(false);
        registry_manager_->data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
        registry_manager_->data_storage_manager_->storage_map_.emplace("abc", be);
        reporter_->ReportInterval();

        const auto mc_vec = reporter_->data_storage_interval_metrics_collectors_.GetMetricsCollectors();
        EXPECT_EQ(1, mc_vec.size());
        auto p = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(mc_vec.front());
        EXPECT_TRUE(p);
        double v;
        GET_METRICS_(p, data_storage, healthy_status, v);
        EXPECT_DOUBLE_EQ(0., v);
        GET_METRICS_(p, data_storage, storage_usage_ratio, v);
        EXPECT_DOUBLE_EQ(0., v);
    }
}

TEST_F(LocalMetricsReporterTest, TestReportInterval02) {
    {
        cache_manager_->meta_indexer_manager_ = nullptr;
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    }

    {
        cache_manager_->meta_indexer_manager_ = std::make_shared<MetaIndexerManager>();

        auto midx = std::make_shared<MetaIndexer>();
        midx->key_count_.store(8);
        cache_manager_->meta_indexer_manager_->meta_indexers_.emplace("abc", midx);
        cache_manager_->meta_indexer_manager_->meta_indexers_.emplace("abc2", nullptr);
        EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());

        auto p = std::dynamic_pointer_cast<MetaIndexerAccumulativeMetricsCollector>(
            reporter_->metrics_collector_for_MetaIndexerAccumulative_);
        EXPECT_TRUE(p);

        double v;
        GET_METRICS_(p, meta_indexer, total_key_count, v);
        EXPECT_DOUBLE_EQ(8., v);

        GET_METRICS_(p, meta_indexer, total_cache_usage, v);
        EXPECT_DOUBLE_EQ(16., v);
    }
}

TEST_F(LocalMetricsReporterTest, TestReportIntervalCacheManagerMetrics) {
    cache_manager_->meta_indexer_manager_ = std::make_shared<MetaIndexerManager>();
    cache_manager_->write_location_manager_ = std::make_shared<WriteLocationManager>();
    cache_manager_->metrics_recorder_ = std::make_shared<CacheManagerMetricsRecorder>(
        cache_manager_->meta_indexer_manager_, cache_manager_->write_location_manager_, registry_manager_);

    RequestContext request_context("test_trace");
    InstanceGroup instance_group;
    instance_group.name_ = "test_group";
    instance_group.quota_.capacity_ = 10240;

    registry_manager_->storage_ = std::make_unique<RegistryLocalBackend>();
    ASSERT_EQ(ErrorCode::EC_OK, registry_manager_->storage_->Init({}));
    ASSERT_EQ(ErrorCode::EC_OK, registry_manager_->CreateInstanceGroup(&request_context, instance_group));

    ASSERT_EQ(ErrorCode::EC_OK,
              registry_manager_->RegisterInstance(
                  &request_context, "test_group", "test_instance_id", 8, {LocationSpecInfo("tp0", 1024)}, {}));

    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    ASSERT_EQ(ErrorCode::EC_OK,
              cache_manager_->meta_indexer_manager_->CreateMetaIndexer("test_instance_id", meta_indexer_config));
    auto meta_indexer = cache_manager_->meta_indexer_manager_->GetMetaIndexer("test_instance_id");
    meta_indexer->key_count_.store(5);
    cache_manager_->metrics_recorder_->Start();
    std::this_thread::sleep_for(std::chrono::seconds(6));
    EXPECT_NO_FATAL_FAILURE(reporter_->ReportInterval());
    {
        const auto &vec = reporter_->cache_manager_group_interval_metrics_collectors_.GetMetricsCollectors();
        ASSERT_EQ(1, vec.size());
        auto p = std::dynamic_pointer_cast<CacheManagerGroupMetricsCollector>(vec.at(0));
        EXPECT_TRUE(p);
        double usage_ratio_v;
        GET_METRICS_(p, cache_manager_group, usage_ratio, usage_ratio_v);
        EXPECT_DOUBLE_EQ(0.5, usage_ratio_v);
    }
    {
        const auto &vec = reporter_->cache_manager_instance_interval_metrics_collectors_.GetMetricsCollectors();
        ASSERT_EQ(1, vec.size());
        auto p = std::dynamic_pointer_cast<CacheManagerInstanceMetricsCollector>(vec.at(0));
        EXPECT_TRUE(p);
        double key_count_v, byte_size_v;
        GET_METRICS_(p, cache_manager_instance, key_count, key_count_v);
        GET_METRICS_(p, cache_manager_instance, byte_size, byte_size_v);
        EXPECT_DOUBLE_EQ(5, key_count_v);
        EXPECT_DOUBLE_EQ(5 * 1024, byte_size_v);
    }
}