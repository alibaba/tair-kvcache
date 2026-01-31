#include <memory>
#include <unistd.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

class MetricsCollectorTest : public TESTBASE {
protected:
    void SetUp() override { metrics_registry_ = std::make_shared<MetricsRegistry>(); }

    void TearDown() override {}

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<MetricsCollector> metrics_collector_;
};

#define GET(ptr, group, name) (ptr)->get_##group##_##name##_metrics()

// Test MetaIndexer metrics functionality
TEST_F(MetricsCollectorTest, MetaIndexerMetricsTest) {
    metrics_collector_ = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    metrics_collector_->Init();

    auto p = std::dynamic_pointer_cast<ServiceMetricsCollector>(metrics_collector_);
    ASSERT_NE(nullptr, p);

    // Test initial state
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, query_key_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, get_not_exist_key_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, query_batch_num), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, search_cache_hit_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, search_cache_miss_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, search_cache_hit_ratio), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, io_data_size), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, put_io_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, update_io_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, upsert_io_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, delete_io_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, get_io_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, read_modify_write_put_key_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, read_modify_write_update_key_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, read_modify_write_skip_key_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, read_modify_write_delete_key_count), 0.);

    // Test value setting
    SET_METRICS_(p, meta_indexer, query_key_count, 100.);
    SET_METRICS_(p, meta_indexer, get_not_exist_key_count, 1.);
    SET_METRICS_(p, meta_indexer, query_batch_num, 10.);
    SET_METRICS_(p, meta_indexer, search_cache_hit_count, 50.);
    SET_METRICS_(p, meta_indexer, search_cache_miss_count, 50.);
    SET_METRICS_(p, meta_indexer, search_cache_hit_ratio, 50.);
    SET_METRICS_(p, meta_indexer, io_data_size, 2048.);
    SET_METRICS_(p, meta_indexer, put_io_time_us, 1000.);
    SET_METRICS_(p, meta_indexer, update_io_time_us, 1000.);
    SET_METRICS_(p, meta_indexer, upsert_io_time_us, 1000.);
    SET_METRICS_(p, meta_indexer, delete_io_time_us, 500.);
    SET_METRICS_(p, meta_indexer, get_io_time_us, 100.);
    SET_METRICS_(p, meta_indexer, read_modify_write_put_key_count, 100.);
    SET_METRICS_(p, meta_indexer, read_modify_write_update_key_count, 200.);
    SET_METRICS_(p, meta_indexer, read_modify_write_skip_key_count, 10.);
    SET_METRICS_(p, meta_indexer, read_modify_write_delete_key_count, 10.);

    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, query_key_count), 100.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, get_not_exist_key_count), 1.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, query_batch_num), 10.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, search_cache_hit_count), 50.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, search_cache_miss_count), 50.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, search_cache_hit_ratio), 50.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, io_data_size), 2048.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, put_io_time_us), 1000.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, update_io_time_us), 1000.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, upsert_io_time_us), 1000.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, delete_io_time_us), 500.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, get_io_time_us), 100.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, read_modify_write_put_key_count), 100.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, read_modify_write_update_key_count), 200.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, read_modify_write_skip_key_count), 10.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, read_modify_write_delete_key_count), 10.);
}

// Test MetaSearcher metrics functionality
TEST_F(MetricsCollectorTest, MetaSearcherMetricsTest) {
    metrics_collector_ = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    metrics_collector_->Init();

    auto p = std::dynamic_pointer_cast<ServiceMetricsCollector>(metrics_collector_);
    ASSERT_NE(nullptr, p);

    EXPECT_DOUBLE_EQ(GET(p, meta_searcher, indexer_get_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_searcher, indexer_read_modify_write_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_searcher, index_serialize_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_searcher, index_deserialize_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_searcher, indexer_query_times), 0.);

    SET_METRICS_(p, meta_searcher, index_serialize_time_us, 101.);
    SET_METRICS_(p, meta_searcher, index_deserialize_time_us, 102.);
    SET_METRICS_(p, meta_searcher, indexer_query_times, 103.);
    EXPECT_DOUBLE_EQ(GET(p, meta_searcher, index_serialize_time_us), 101.);
    EXPECT_DOUBLE_EQ(GET(p, meta_searcher, index_deserialize_time_us), 102.);
    EXPECT_DOUBLE_EQ(GET(p, meta_searcher, indexer_query_times), 103.);

    // Test time measurement for indexer get
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, MetaSearcherIndexerGet);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, MetaSearcherIndexerReadModifyWrite);
    usleep(1000); // 1ms
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, MetaSearcherIndexerGet);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, MetaSearcherIndexerReadModifyWrite);
    EXPECT_GE(GET(p, meta_searcher, indexer_get_time_us), 1000.0);
    EXPECT_GE(GET(p, meta_searcher, indexer_read_modify_write_time_us), 1000.0);
}

// Test Manager metrics functionality
TEST_F(MetricsCollectorTest, ManagerMetricsTest) {
    metrics_collector_ = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    metrics_collector_->Init();

    auto p = std::dynamic_pointer_cast<ServiceMetricsCollector>(metrics_collector_);
    ASSERT_NE(nullptr, p);

    EXPECT_DOUBLE_EQ(GET(p, manager, request_key_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, manager, prefix_match_len), 0.);
    EXPECT_DOUBLE_EQ(GET(p, manager, prefix_match_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, manager, lock_write_location_retry_times), 0.);
    EXPECT_DOUBLE_EQ(GET(p, manager, write_cache_io_cost_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, manager, filter_write_cache_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, manager, batch_get_location_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, manager, batch_add_location_time_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, manager, batch_update_location_time_us), 0.);

    SET_METRICS_(p, manager, request_key_count, 10.);
    SET_METRICS_(p, manager, prefix_match_len, 10.);
    SET_METRICS_(p, manager, lock_write_location_retry_times, 5.);
    SET_METRICS_(p, manager, write_cache_io_cost_us, 2000.);
    EXPECT_DOUBLE_EQ(GET(p, manager, request_key_count), 10.);
    EXPECT_DOUBLE_EQ(GET(p, manager, prefix_match_len), 10.);
    EXPECT_DOUBLE_EQ(GET(p, manager, lock_write_location_retry_times), 5.);
    EXPECT_DOUBLE_EQ(GET(p, manager, write_cache_io_cost_us), 2000.);

    // Test time measurement
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, ManagerPrefixMatch);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, ManagerFilterWriteCache);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, ManagerBatchGetLocation);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, ManagerBatchAddLocation);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, ManagerBatchUpdateLocation);
    usleep(1000); // 1ms
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, ManagerPrefixMatch);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, ManagerFilterWriteCache);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, ManagerBatchGetLocation);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, ManagerBatchAddLocation);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, ManagerBatchUpdateLocation);
    EXPECT_GT(GET(p, manager, prefix_match_time_us), 1000.0);
    EXPECT_GT(GET(p, manager, filter_write_cache_time_us), 1000.0);
    EXPECT_GT(GET(p, manager, batch_get_location_time_us), 1000.0);
    EXPECT_GT(GET(p, manager, batch_add_location_time_us), 1000.0);
    EXPECT_GT(GET(p, manager, batch_update_location_time_us), 1000.0);
}

// Test Service metrics functionality
TEST_F(MetricsCollectorTest, ServiceMetricsTest) {
    metrics_collector_ = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    metrics_collector_->Init();

    auto p = std::dynamic_pointer_cast<ServiceMetricsCollector>(metrics_collector_);
    ASSERT_NE(nullptr, p);

    EXPECT_EQ(GET(p, service, query_counter), 0);
    EXPECT_DOUBLE_EQ(GET(p, service, query_rt_us), 0.);
    EXPECT_DOUBLE_EQ(GET(p, service, error_code), 0.);
    EXPECT_EQ(GET(p, service, error_counter), 0);
    EXPECT_DOUBLE_EQ(GET(p, service, request_queue_size), 0.);

    Counter query_counter;
    Counter error_counter;
    COPY_METRICS_(p, service, query_counter, query_counter);
    COPY_METRICS_(p, service, error_counter, error_counter);
    ++query_counter;
    ++error_counter;
    EXPECT_EQ(GET(p, service, query_counter), 1);
    EXPECT_EQ(GET(p, service, error_counter), 1);

    SET_METRICS_(p, service, error_code, 404.);
    SET_METRICS_(p, service, request_queue_size, 10.);
    EXPECT_DOUBLE_EQ(GET(p, service, error_code), 404.);
    EXPECT_DOUBLE_EQ(GET(p, service, request_queue_size), 10.);

    // Test time measurement
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, ServiceQuery);
    usleep(1000); // 1ms
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, ServiceQuery);
    EXPECT_GT(GET(p, service, query_rt_us), 1000.);
}

TEST_F(MetricsCollectorTest, DataStorageMetricsCollectorTest) {
    metrics_collector_ = std::make_shared<DataStorageMetricsCollector>(metrics_registry_);
    metrics_collector_->Init();

    auto p = std::dynamic_pointer_cast<DataStorageMetricsCollector>(metrics_collector_);
    ASSERT_NE(nullptr, p);

    EXPECT_EQ(GET(p, data_storage, create_counter), 0);
    EXPECT_DOUBLE_EQ(GET(p, data_storage, create_keys_qps), 0.);
    EXPECT_EQ(GET(p, data_storage, create_keys_counter), 0);
    EXPECT_DOUBLE_EQ(GET(p, data_storage, create_time_us), 0.);

    Counter create_counter;
    Counter create_keys_counter;
    COPY_METRICS_(p, data_storage, create_counter, create_counter);
    COPY_METRICS_(p, data_storage, create_keys_counter, create_keys_counter);
    ++create_counter;
    ++create_keys_counter;
    EXPECT_EQ(GET(p, data_storage, create_counter), 1);
    EXPECT_EQ(GET(p, data_storage, create_keys_counter), 1);

    SET_METRICS_(p, data_storage, create_keys_qps, 123.);
    EXPECT_DOUBLE_EQ(GET(p, data_storage, create_keys_qps), 123.);

    // Test time measurement
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(p, DataStorageCreate);
    usleep(1000); // 1ms
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(p, DataStorageCreate);
    EXPECT_GT(GET(p, data_storage, create_time_us), 1000.);
}

TEST_F(MetricsCollectorTest, DataStorageHealthMetricsCollectorTest) {
    metrics_collector_ = std::make_shared<DataStorageIntervalMetricsCollector>(metrics_registry_);
    metrics_collector_->Init();

    auto p = std::dynamic_pointer_cast<DataStorageIntervalMetricsCollector>(metrics_collector_);
    ASSERT_NE(nullptr, p);

    EXPECT_DOUBLE_EQ(GET(p, data_storage, healthy_status), 0.);
    SET_METRICS_(p, data_storage, healthy_status, 1.);
    EXPECT_DOUBLE_EQ(GET(p, data_storage, healthy_status), 1.);
}

TEST_F(MetricsCollectorTest, MetaIndexerAccumulativeMetricsCollectorTest) {
    metrics_collector_ = std::make_shared<MetaIndexerAccumulativeMetricsCollector>(metrics_registry_);
    metrics_collector_->Init();

    auto p = std::dynamic_pointer_cast<MetaIndexerAccumulativeMetricsCollector>(metrics_collector_);
    ASSERT_NE(nullptr, p);

    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, total_key_count), 0.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, total_cache_usage), 0.);
    SET_METRICS_(p, meta_indexer, total_key_count, 123.);
    SET_METRICS_(p, meta_indexer, total_cache_usage, 456.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, total_key_count), 123.);
    EXPECT_DOUBLE_EQ(GET(p, meta_indexer, total_cache_usage), 456.);
}
