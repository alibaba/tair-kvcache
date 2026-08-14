#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta//test/meta_indexer_test_base.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_search_cache.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/meta/query_executor.h"
#include "kv_cache_manager/meta/storage_usage_data.h"
#include "kv_cache_manager/meta/types.h"
#include "kv_cache_manager/meta/utils.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

namespace {
// Helper: read the persistent backend's storage type through the new
// MetaStorageBackendManager indirection. MetaIndexer no longer exposes a raw
// `storage_` member - the backend lives inside `backend_manager_`.
std::string GetPersistentStorageType(const MetaIndexer &indexer) {
    return indexer.backend_manager_->persistent_backend_->GetStorageType();
}

class MalformedLocationReadBackend : public MetaLocalBackend {
public:
    enum class Shape {
        kShortOuter,
        kShortInner,
        kNullValueWithOk,
    };

    void SetShape(Shape shape) { shape_ = shape; }

    std::vector<ErrorCode> GetLocations(RequestContext * /*request_context*/,
                                        const KeyTypeVec & /*keys*/,
                                        CacheLocationMapVector &out_locations) noexcept override {
        out_locations.clear();
        return {EC_OK};
    }

    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext * /*request_context*/,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept override {
        if (shape_ == Shape::kShortOuter) {
            out_locations.clear();
            return {};
        }
        if (shape_ == Shape::kShortInner) {
            out_locations.assign(keys.size(), CacheLocationVector{});
            return std::vector<std::vector<ErrorCode>>(keys.size());
        }
        out_locations.resize(keys.size());
        std::vector<std::vector<ErrorCode>> result(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            out_locations[i].assign(location_ids[i].size(), CacheLocationConstPtr{});
            result[i].assign(location_ids[i].size(), EC_OK);
        }
        return result;
    }

    std::vector<ErrorCode> GetLocationValues(RequestContext * /*request_context*/,
                                             const KeyTypeVec &keys,
                                             LocationsPerKey &out_locations) noexcept override {
        if (shape_ == Shape::kShortOuter) {
            out_locations.clear();
            return {EC_OK};
        }
        out_locations.assign(keys.size(), CacheLocationVector{});
        return std::vector<ErrorCode>(keys.size(), EC_OK);
    }

    std::vector<ErrorCode> GetLocationValuesCompact(RequestContext * /*request_context*/,
                                                    const KeyType * /*keys*/,
                                                    size_t key_count,
                                                    CompactLocationsPerKey &out_locations) noexcept override {
        if (shape_ == Shape::kShortOuter) {
            out_locations.Clear();
            return {EC_OK};
        }
        out_locations.Clear(key_count);
        for (size_t i = 0; i < key_count; ++i) {
            out_locations.FinishKey();
        }
        return std::vector<ErrorCode>(key_count, EC_OK);
    }

private:
    Shape shape_ = Shape::kShortOuter;
};
} // namespace

class MetaIndexerTest : public MetaIndexerTestBase, public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

    ErrorCode InitIndexer(const std::string &configStr);
};

void MetaIndexerTest::SetUp() {
    meta_indexer_ = std::make_shared<MetaIndexer>();
    request_context_ = std::make_shared<RequestContext>("test_trace_id");
}

ErrorCode MetaIndexerTest::InitIndexer(const std::string &configStr) {
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->FromJsonString(configStr);
    std::string local_path = GetPrivateTestRuntimeDataPath() + "meta_local_backend_file1";
    meta_indexer_config->meta_storage_backend_config_->SetStorageUri("file://" + local_path);
    return meta_indexer_->Init(/*instance_id*/ "test", meta_indexer_config);
}

TEST_F(MetaIndexerTest, TestInit) {
    // test success
    std::string configStr = R"({
        "max_key_count" : 100, "mutex_shard_num" : 8,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(7, meta_indexer_->mutex_shard_mask_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, GetPersistentStorageType(*meta_indexer_));

    // test failed
    ASSERT_EQ(ErrorCode::EC_BADARGS, meta_indexer_->Init(/*instance_id*/ "test", nullptr));

    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->meta_storage_backend_config_ = nullptr;
    ASSERT_EQ(EC_BADARGS, meta_indexer_->Init(/*instance_id*/ "test", meta_indexer_config));

    configStr = R"({
        "meta_storage_backend_config" : { "storage_type" : "test" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_ERROR, InitIndexer(configStr));

    configStr = R"({
        "max_key_count" : 100, "mutex_shard_num" : 10,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_CONFIG_ERROR, InitIndexer(configStr));

    configStr = R"({
        "max_key_count" : 100, "mutex_shard_num" : 0,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_CONFIG_ERROR, InitIndexer(configStr));

    configStr = R"({
        "max_key_count" : 100, "mutex_shard_num" : 128,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : {}
    })";
    ASSERT_EQ(EC_CONFIG_ERROR, InitIndexer(configStr));
}

TEST_F(MetaIndexerTest, TestProcessErrorCodesRejectsAbnormalResultCount) {
    const KeyVector keys = {11, 22, 33};

    MetaIndexer::Result short_result(keys.size());
    EXPECT_EQ(3, meta_indexer_->ProcessErrorCodes("trace", {EC_OK}, {}, keys, "test_short_result", short_result));
    EXPECT_EQ(EC_MISMATCH, short_result.ec);
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH, EC_MISMATCH, EC_MISMATCH}), short_result.error_codes);

    MetaIndexer::Result long_result(keys.size());
    EXPECT_EQ(3,
              meta_indexer_->ProcessErrorCodes(
                  "trace", {EC_OK, EC_OK, EC_OK, EC_OK}, {}, keys, "test_long_result", long_result));
    EXPECT_EQ(EC_MISMATCH, long_result.ec);
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH, EC_MISMATCH, EC_MISMATCH}), long_result.error_codes);
}

TEST_F(MetaIndexerTest, TestParallelLocalLocationValuesMatchSerialAndPreserveErrors) {
    constexpr std::size_t kKeyCount = 1024;
    meta_indexer_->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 4, /*parallel_threshold*/ 64, /*chunk_size*/ 32, /*queue_capacity*/ 32));
    const std::string config_str = R"({
        "max_key_count" : 2048,
        "mutex_shard_num" : 64,
        "batch_key_size" : 128,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    KVData data;
    MakeKVData(0, kKeyCount, data);
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties).ec);

    KeyVector query_keys = data.keys;
    constexpr std::size_t kMissingIndex = 333;
    query_keys[kMissingIndex] = 100000;
    LocationsPerKey parallel_values;
    const auto parallel_result = meta_indexer_->GetLocationValues(request_context_.get(), query_keys, parallel_values);
    ASSERT_EQ(EC_PARTIAL_OK, parallel_result.ec);
    ASSERT_EQ(kKeyCount, parallel_result.error_codes.size());
    ASSERT_EQ(kKeyCount, parallel_values.size());
    for (std::size_t i = 0; i < kKeyCount; ++i) {
        if (i == kMissingIndex) {
            EXPECT_EQ(EC_NOENT, parallel_result.error_codes[i]);
            EXPECT_TRUE(parallel_values[i].empty());
            continue;
        }
        EXPECT_EQ(EC_OK, parallel_result.error_codes[i]) << "index=" << i;
        ASSERT_EQ(1u, parallel_values[i].size()) << "index=" << i;
        ASSERT_TRUE(parallel_values[i].front());
        EXPECT_EQ("loc_" + std::to_string(query_keys[i]), parallel_values[i].front()->id());
    }

    meta_indexer_->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 1, /*parallel_threshold*/ 64, /*chunk_size*/ 32, /*queue_capacity*/ 1));
    LocationsPerKey serial_values;
    const auto serial_result = meta_indexer_->GetLocationValues(request_context_.get(), query_keys, serial_values);
    EXPECT_EQ(parallel_result.ec, serial_result.ec);
    EXPECT_EQ(parallel_result.error_codes, serial_result.error_codes);
    ASSERT_EQ(parallel_values.size(), serial_values.size());
    for (std::size_t i = 0; i < parallel_values.size(); ++i) {
        ASSERT_EQ(parallel_values[i].size(), serial_values[i].size()) << "index=" << i;
        for (std::size_t j = 0; j < parallel_values[i].size(); ++j) {
            ASSERT_TRUE(serial_values[i][j]);
            EXPECT_EQ(parallel_values[i][j]->id(), serial_values[i][j]->id()) << "index=" << i;
        }
    }
}

TEST_F(MetaIndexerTest, TestCompactPrefixLocationValuesStopsAtFirstMissingChunk) {
    constexpr std::size_t kKeyCount = 16384;
    constexpr std::size_t kChunkSize = 32;
    // Pure-local scans deliberately use a larger bounded metadata-read window
    // than the CPU projection chunk so LRU shard locks are amortized.
    constexpr std::size_t kLocalReadWindow = 4096;
    constexpr std::size_t kMissingIndex = 97;
    // A single worker makes the amount of speculative work deterministic. The
    // parallel case uses the same range callback and can read at most a bounded
    // number of already-claimed chunks beyond the first miss.
    meta_indexer_->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 1, /*parallel_threshold*/ 64, kChunkSize, /*queue_capacity*/ 1));
    const std::string config_str = R"({
        "max_key_count" : 32768,
        "mutex_shard_num" : 64,
        "batch_key_size" : 128,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    KVData data;
    MakeKVData(0, kKeyCount, data);
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties).ec);

    KeyVector query_keys = data.keys;
    query_keys[kMissingIndex] = 100000;
    std::vector<CacheLocationConstPtr> observed(kKeyCount);
    const auto visitor = [&query_keys,
                          &observed](size_t begin, const CompactLocationsPerKey &locations, size_t valid_count) {
        for (size_t i = 0; i < valid_count; ++i) {
            if (locations[i].size() == 1) {
                observed[begin + i] = *locations[i].begin();
            }
        }
        return query_keys.size();
    };
    const auto prefix = meta_indexer_->VisitLocationValuesForPrefix(request_context_.get(), query_keys, visitor);
    EXPECT_EQ(EC_NOENT, prefix.terminal_ec);
    EXPECT_EQ(kMissingIndex, prefix.valid_key_count);
    EXPECT_EQ(kLocalReadWindow, prefix.read_key_count);
    EXPECT_FALSE(prefix.stopped_by_visitor);
    for (std::size_t i = 0; i < kMissingIndex; ++i) {
        ASSERT_TRUE(observed[i]) << "index=" << i;
        EXPECT_EQ("loc_" + std::to_string(query_keys[i]), observed[i]->id()) << "index=" << i;
    }
}

TEST_F(MetaIndexerTest, TestCompactPrefixLocationValuesHonorsVisitorStop) {
    constexpr std::size_t kKeyCount = 16384;
    constexpr std::size_t kChunkSize = 32;
    constexpr std::size_t kLocalReadWindow = 4096;
    constexpr std::size_t kVisitorStop = 97;
    meta_indexer_->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 1, /*parallel_threshold*/ 64, kChunkSize, /*queue_capacity*/ 1));
    const std::string config_str = R"({
        "max_key_count" : 32768,
        "mutex_shard_num" : 64,
        "batch_key_size" : 128,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    KVData data;
    MakeKVData(0, kKeyCount, data);
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties).ec);

    std::atomic<size_t> visited_key_count(0);
    const auto prefix = meta_indexer_->VisitLocationValuesForPrefix(
        request_context_.get(),
        data.keys,
        [&data, &visited_key_count](size_t begin, const CompactLocationsPerKey &, size_t valid_count) {
            visited_key_count.fetch_add(valid_count, std::memory_order_relaxed);
            return begin <= kVisitorStop && kVisitorStop < begin + valid_count ? kVisitorStop : data.keys.size();
        });
    EXPECT_EQ(EC_OK, prefix.terminal_ec);
    EXPECT_EQ(kVisitorStop, prefix.valid_key_count);
    EXPECT_TRUE(prefix.stopped_by_visitor);
    EXPECT_EQ(kLocalReadWindow, prefix.read_key_count);
    EXPECT_EQ(prefix.read_key_count, visited_key_count.load(std::memory_order_relaxed));
}

TEST_F(MetaIndexerTest, TestCompactPrefixLocationValuesReturnsEveryAllHitKey) {
    constexpr std::size_t kKeyCount = 1024;
    meta_indexer_->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 4, /*parallel_threshold*/ 64, /*chunk_size*/ 32, /*queue_capacity*/ 32));
    const std::string config_str = R"({
        "max_key_count" : 2048,
        "mutex_shard_num" : 64,
        "batch_key_size" : 128,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    KVData data;
    MakeKVData(0, kKeyCount, data);
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties).ec);

    std::vector<CacheLocationConstPtr> observed(kKeyCount);
    const auto visitor = [&data, &observed](size_t begin, const CompactLocationsPerKey &locations, size_t valid_count) {
        for (size_t i = 0; i < valid_count; ++i) {
            if (locations[i].size() == 1) {
                observed[begin + i] = *locations[i].begin();
            }
        }
        return data.keys.size();
    };
    const auto prefix = meta_indexer_->VisitLocationValuesForPrefix(request_context_.get(), data.keys, visitor);
    EXPECT_EQ(EC_OK, prefix.terminal_ec);
    EXPECT_EQ(kKeyCount, prefix.valid_key_count);
    EXPECT_EQ(kKeyCount, prefix.read_key_count);
    EXPECT_FALSE(prefix.stopped_by_visitor);
    for (std::size_t i = 0; i < observed.size(); ++i) {
        ASSERT_TRUE(observed[i]) << "index=" << i;
        EXPECT_EQ("loc_" + std::to_string(data.keys[i]), observed[i]->id()) << "index=" << i;
    }
}

TEST_F(MetaIndexerTest, TestCompactPrefixLocationValuesClampsOversizedConfiguredChunk) {
    constexpr std::size_t kKeyCount = 5000;
    meta_indexer_->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 4,
        /*parallel_threshold*/ 1,
        /*chunk_size*/ std::numeric_limits<std::size_t>::max(),
        /*queue_capacity*/ 4));
    const std::string config_str = R"({
        "max_key_count" : 8192,
        "mutex_shard_num" : 64,
        "batch_key_size" : 128,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    KVData data;
    MakeKVData(0, kKeyCount, data);
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties).ec);

    std::vector<size_t> chunk_begins;
    const auto prefix = meta_indexer_->VisitLocationValuesForPrefix(
        request_context_.get(),
        data.keys,
        [&data, &chunk_begins](size_t begin, const CompactLocationsPerKey &locations, size_t valid_count) {
            chunk_begins.push_back(begin);
            EXPECT_EQ(valid_count, locations.size());
            return data.keys.size();
        });
    EXPECT_EQ(EC_OK, prefix.terminal_ec);
    EXPECT_EQ(kKeyCount, prefix.valid_key_count);
    EXPECT_EQ(kKeyCount, prefix.read_key_count);
    EXPECT_EQ((std::vector<size_t>{0, 4096}), chunk_begins);
}

TEST_F(MetaIndexerTest, TestCompactPrefixGetIoMetricExcludesPipelinedVisitorTime) {
    constexpr std::size_t kKeyCount = 5000;
    meta_indexer_->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 1, /*parallel_threshold*/ 64, /*chunk_size*/ 32, /*queue_capacity*/ 1));
    const std::string config_str = R"({
        "max_key_count" : 8192,
        "mutex_shard_num" : 64,
        "batch_key_size" : 128,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    KVData data;
    MakeKVData(0, kKeyCount, data);
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), data.keys, data.location_maps, data.properties).ec);

    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto metrics_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry);
    ASSERT_TRUE(metrics_collector->Init());
    RequestContext metrics_context("prefix_metric_test", metrics_collector);
    size_t visitor_calls = 0;
    const auto begin = std::chrono::steady_clock::now();
    const auto prefix = meta_indexer_->VisitLocationValuesForPrefix(
        &metrics_context, data.keys, [&data, &visitor_calls](size_t, const CompactLocationsPerKey &, size_t) {
            ++visitor_calls;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return data.keys.size();
        });
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();

    EXPECT_EQ(EC_OK, prefix.terminal_ec);
    EXPECT_EQ(2u, visitor_calls);
    const auto backend_wall_us = metrics_collector->get_meta_indexer_get_io_time_us_metrics();
    EXPECT_GT(backend_wall_us, 0);
    EXPECT_GE(elapsed_us - static_cast<int64_t>(backend_wall_us), 30000);
}

TEST_F(MetaIndexerTest, TestCompactPrefixLocationValuesRejectsMalformedShape) {
    meta_indexer_->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 4, /*parallel_threshold*/ 2, /*chunk_size*/ 2, /*queue_capacity*/ 4));
    const std::string config_str = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    auto malformed = std::make_unique<MalformedLocationReadBackend>();
    auto backend_config = std::make_shared<MetaStorageBackendConfig>();
    ASSERT_EQ(EC_OK, malformed->Init("test", backend_config));
    ASSERT_EQ(EC_OK, malformed->Open());
    malformed->SetShape(MalformedLocationReadBackend::Shape::kShortOuter);
    ASSERT_EQ(EC_OK, meta_indexer_->backend_manager_->persistent_backend_->Close());
    meta_indexer_->backend_manager_->persistent_backend_ = std::move(malformed);
    meta_indexer_->backend_manager_->cache_backend_.reset();

    const KeyVector keys{1, 2, 3, 4};
    std::atomic<size_t> visitor_calls(0);
    const auto prefix = meta_indexer_->VisitLocationValuesForPrefix(
        request_context_.get(), keys, [&visitor_calls, &keys](size_t, const CompactLocationsPerKey &, size_t) {
            visitor_calls.fetch_add(1, std::memory_order_relaxed);
            return keys.size();
        });
    EXPECT_EQ(EC_MISMATCH, prefix.terminal_ec);
    EXPECT_EQ(0u, prefix.valid_key_count);
    EXPECT_EQ(0u, visitor_calls.load(std::memory_order_relaxed));
}

TEST_F(MetaIndexerTest, TestReadModifyWriteLocationRejectsMalformedBackendResultShapes) {
    const std::string config_str = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    auto malformed = std::make_unique<MalformedLocationReadBackend>();
    auto backend_config = std::make_shared<MetaStorageBackendConfig>();
    ASSERT_EQ(EC_OK, malformed->Init("test", backend_config));
    ASSERT_EQ(EC_OK, malformed->Open());
    auto *malformed_raw = malformed.get();
    ASSERT_EQ(EC_OK, meta_indexer_->backend_manager_->persistent_backend_->Close());
    meta_indexer_->backend_manager_->persistent_backend_ = std::move(malformed);
    meta_indexer_->backend_manager_->cache_backend_.reset();

    const KeyVector keys{123};
    const LocationIdsPerKey location_ids{{"loc"}};
    size_t modifier_calls = 0;
    const auto modifier =
        [&modifier_calls](
            const std::vector<ErrorCode> &, const LocationIdVector &ids, size_t, CacheLocationVector &, PropertyMap &) {
            ++modifier_calls;
            return LocationModifierResult{ModifierAction::MA_SKIP, std::vector<ErrorCode>(ids.size(), EC_OK)};
        };

    auto result = meta_indexer_->ReadModifyWriteLocation(request_context_.get(), keys, location_ids, modifier);
    EXPECT_EQ(EC_ERROR, result.ec);
    ASSERT_EQ(1u, result.per_location_error_codes.size());
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH}), result.per_location_error_codes[0]);
    EXPECT_EQ(0u, modifier_calls);

    malformed_raw->SetShape(MalformedLocationReadBackend::Shape::kShortInner);
    result = meta_indexer_->ReadModifyWriteLocation(request_context_.get(), keys, location_ids, modifier);
    EXPECT_EQ(EC_ERROR, result.ec);
    ASSERT_EQ(1u, result.per_location_error_codes.size());
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH}), result.per_location_error_codes[0]);
    EXPECT_EQ(0u, modifier_calls);

    malformed_raw->SetShape(MalformedLocationReadBackend::Shape::kNullValueWithOk);
    result = meta_indexer_->ReadModifyWriteLocation(request_context_.get(), keys, location_ids, modifier);
    EXPECT_EQ(EC_ERROR, result.ec);
    ASSERT_EQ(1u, result.per_location_error_codes.size());
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH}), result.per_location_error_codes[0]);
    EXPECT_EQ(1u, modifier_calls);

    LocationsPerKey locations;
    auto get_result = meta_indexer_->GetLocations(request_context_.get(), keys, location_ids, locations);
    EXPECT_EQ(EC_ERROR, get_result.ec);
    ASSERT_EQ(1u, get_result.per_location_error_codes.size());
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH}), get_result.per_location_error_codes[0]);
    ASSERT_EQ(1u, locations.size());
    ASSERT_EQ(1u, locations[0].size());
    EXPECT_FALSE(locations[0][0]);

    CacheLocationMapVector location_maps;
    const auto get_all_result = meta_indexer_->GetLocations(request_context_.get(), keys, location_maps);
    EXPECT_EQ(EC_ERROR, get_all_result.ec);
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH}), get_all_result.error_codes);
    ASSERT_EQ(1u, location_maps.size());
    EXPECT_TRUE(location_maps[0].empty());

    malformed_raw->SetShape(MalformedLocationReadBackend::Shape::kShortOuter);
    LocationsPerKey location_values;
    const KeyVector two_keys{123, 124};
    const auto get_values_result = meta_indexer_->GetLocationValues(request_context_.get(), two_keys, location_values);
    EXPECT_EQ(EC_ERROR, get_values_result.ec);
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH, EC_MISMATCH}), get_values_result.error_codes);
    ASSERT_EQ(2u, location_values.size());
    EXPECT_TRUE(location_values[0].empty());
    EXPECT_TRUE(location_values[1].empty());
}

TEST_F(MetaIndexerTest, TestReadModifyWriteLocationPreservesPartialModifierResult) {
    const std::string config_str = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));

    const KeyVector keys{124};
    const LocationIdsPerKey location_ids{{"good", "bad"}};
    const auto modifier = [](const std::vector<ErrorCode> &get_ecs,
                             const LocationIdVector &ids,
                             size_t,
                             CacheLocationVector &locations,
                             PropertyMap &) {
        EXPECT_EQ((std::vector<ErrorCode>{EC_NOENT, EC_NOENT}), get_ecs);
        auto good = std::make_shared<CacheLocation>();
        good->set_id(ids[0]);
        good->set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
        good->set_status(CLS_SERVING);
        locations[0] = std::move(good);
        return LocationModifierResult{MA_OK, {EC_OK, EC_BADARGS}};
    };

    const auto result = meta_indexer_->ReadModifyWriteLocation(request_context_.get(), keys, location_ids, modifier);
    // The RMW itself succeeded. A per-location validation failure is surfaced
    // in the aligned result without turning the whole batch into an
    // infrastructure failure.
    EXPECT_EQ(EC_OK, result.ec);
    ASSERT_EQ(1u, result.per_location_error_codes.size());
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK, EC_BADARGS}), result.per_location_error_codes[0]);

    LocationsPerKey stored_locations;
    const auto get_result = meta_indexer_->GetLocations(request_context_.get(), keys, location_ids, stored_locations);
    EXPECT_EQ(EC_PARTIAL_OK, get_result.ec);
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}), get_result.per_location_error_codes[0]);
    ASSERT_TRUE(stored_locations[0][0]);
    EXPECT_EQ("good", stored_locations[0][0]->id());
    EXPECT_FALSE(stored_locations[0][1]);
}

TEST_F(MetaIndexerTest, TestSingleTargetRmwPreservesCapacityAndExistingKeySemantics) {
    const std::string config_str = R"({
        "max_key_count" : 2,
        "mutex_shard_num" : 2,
        "batch_key_size" : 100,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(config_str));
    ASSERT_TRUE(meta_indexer_->SupportsSingleLocationRmw());

    const std::string target_id = "target";
    const std::string sibling_id = "sibling";
    auto old_target = std::make_shared<CacheLocation>();
    old_target->set_id(target_id);
    auto sibling = std::make_shared<CacheLocation>();
    sibling->set_id(sibling_id);
    CacheLocationMapVector seed_locations(2);
    seed_locations[0].emplace(target_id, old_target);
    seed_locations[1].emplace(sibling_id, sibling);
    PropertyMapVector seed_properties(2);
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), {1, 2}, seed_locations, seed_properties).ec);
    ASSERT_EQ(2u, meta_indexer_->GetKeyCount());

    const KeyVector keys{1, 2, 3};
    const LocationIdRefVector target_ids{&target_id, &target_id, &target_id};
    std::vector<CacheLocationConstPtr> replacements(keys.size());
    const long old_target_use_count = old_target.use_count();
    const auto modifier = [&replacements, &old_target, old_target_use_count](ErrorCode get_ec,
                                                                             const LocationId &location_id,
                                                                             size_t key_index,
                                                                             const CacheLocation *existing_location,
                                                                             CacheLocationConstPtr &out_location) {
        if (key_index == 0) {
            EXPECT_EQ(EC_OK, get_ec);
            EXPECT_EQ(old_target.get(), existing_location);
            // The fused local read borrows the immutable value from the pinned
            // cache item; it must not increment the shared_ptr control block.
            EXPECT_EQ(old_target_use_count, old_target.use_count());
        } else {
            EXPECT_EQ(EC_NOENT, get_ec);
            EXPECT_EQ(nullptr, existing_location);
        }
        auto replacement = std::make_shared<CacheLocation>();
        replacement->set_id(location_id);
        replacement->set_status(CLS_SERVING);
        replacements[key_index] = replacement;
        out_location = std::move(replacement);
        return ModifierResult{MA_OK, EC_OK};
    };

    const auto result =
        meta_indexer_->ReadModifyWriteSingleTargetLocations(request_context_.get(), keys, target_ids, modifier);
    EXPECT_EQ(EC_PARTIAL_OK, result.ec);
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_NOSPC}), result.error_codes);
    EXPECT_EQ(2u, meta_indexer_->GetKeyCount());

    CacheLocationMapVector stored_locations;
    const auto get_result = meta_indexer_->GetLocations(request_context_.get(), keys, stored_locations);
    EXPECT_EQ(EC_PARTIAL_OK, get_result.ec);
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_NOENT}), get_result.error_codes);
    ASSERT_EQ(1u, stored_locations[0].size());
    EXPECT_EQ(replacements[0], stored_locations[0].at(target_id));
    ASSERT_EQ(2u, stored_locations[1].size());
    EXPECT_EQ(sibling, stored_locations[1].at(sibling_id));
    EXPECT_EQ(replacements[1], stored_locations[1].at(target_id));
    EXPECT_TRUE(stored_locations[2].empty());

    size_t skip_calls = 0;
    const auto skip_result =
        meta_indexer_->ReadModifyWriteSingleTargetLocations(request_context_.get(),
                                                            {3},
                                                            LocationIdRefVector{&target_id},
                                                            [&skip_calls](ErrorCode get_ec,
                                                                          const LocationId &,
                                                                          size_t,
                                                                          const CacheLocation *existing_location,
                                                                          CacheLocationConstPtr &) {
                                                                ++skip_calls;
                                                                EXPECT_EQ(EC_NOENT, get_ec);
                                                                EXPECT_EQ(nullptr, existing_location);
                                                                return ModifierResult{MA_SKIP, EC_OK};
                                                            });
    EXPECT_EQ(EC_OK, skip_result.ec);
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), skip_result.error_codes);
    EXPECT_EQ(1u, skip_calls);
    EXPECT_EQ(2u, meta_indexer_->GetKeyCount());

    size_t duplicate_modifier_calls = 0;
    const auto duplicate_result = meta_indexer_->ReadModifyWriteSingleTargetLocations(
        request_context_.get(),
        {4, 4},
        LocationIdRefVector{&target_id, &target_id},
        [&duplicate_modifier_calls](
            ErrorCode, const LocationId &, size_t, const CacheLocation *, CacheLocationConstPtr &) {
            ++duplicate_modifier_calls;
            return ModifierResult{MA_SKIP, EC_OK};
        });
    EXPECT_EQ(EC_BADARGS, duplicate_result.ec);
    EXPECT_TRUE(duplicate_result.error_codes.empty());
    EXPECT_EQ(0u, duplicate_modifier_calls);
}

// Verifies the invariants of MakeBatches() that callers rely on, regardless
// of the exact shard distribution (which is now hash-driven and therefore
// not deterministic across keys):
//   * every input key appears in exactly one batch;
//   * batch_indexs preserves the original positions in `keys`;
//   * within a batch, all keys belong to the shards listed in batch_shard_indexs;
//   * each batch_shard_indexs entry is a distinct shard.
TEST_F(MetaIndexerTest, TestMakeBatches) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "batch_key_size" : 2,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(7, meta_indexer_->mutex_shard_mask_);
    ASSERT_EQ(2, meta_indexer_->batch_key_size_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, GetPersistentStorageType(*meta_indexer_));

    KeyVector keys = {0, 1, 2, 3, 4, 8, 9, 80, 800};
    LocationIdsPerKey empty_location_ids;
    CacheLocationMapVector empty_locations;
    PropertyMapVector empty_properties;
    auto batches = meta_indexer_->MakeBatches(keys, empty_location_ids, empty_locations, empty_properties);

    std::vector<int32_t> covered_indexs;
    for (const auto &batch : batches) {
        std::set<int32_t> shards_in_batch(batch.batch_shard_indexs.begin(), batch.batch_shard_indexs.end());
        ASSERT_EQ(shards_in_batch.size(), batch.batch_shard_indexs.size()) << "duplicate shard in one batch";
        ASSERT_EQ(batch.batch_keys.size(), batch.batch_indexs.size());
        for (size_t j = 0; j < batch.batch_keys.size(); ++j) {
            const int32_t origin_idx = batch.batch_indexs[j];
            ASSERT_EQ(keys[origin_idx], batch.batch_keys[j]);
            const int32_t shard = meta_indexer_->GetMutexShardIndex(batch.batch_keys[j]);
            ASSERT_TRUE(shards_in_batch.count(shard) > 0)
                << "key " << batch.batch_keys[j] << " hashed to shard " << shard
                << " but the batch only locked shards declared in batch_shard_indexs";
            covered_indexs.push_back(origin_idx);
        }
        ASSERT_TRUE(batch.batch_properties.empty());
        ASSERT_TRUE(batch.batch_locations.empty());
    }
    std::sort(covered_indexs.begin(), covered_indexs.end());
    std::vector<int32_t> expected_indexs(keys.size());
    std::iota(expected_indexs.begin(), expected_indexs.end(), 0);
    ASSERT_EQ(expected_indexs, covered_indexs);
}

TEST_F(MetaIndexerTest, TestPureLocalMutexShardsReuseLruHashSeed) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 16,
        "batch_key_size" : 4,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));

    auto *local_backend = dynamic_cast<MetaLocalBackend *>(meta_indexer_->backend_manager_->persistent_backend_.get());
    ASSERT_NE(nullptr, local_backend);
    uint32_t lru_hash_seed = 0;
    ASSERT_TRUE(local_backend->GetCacheHashSeed(lru_hash_seed));
    ASSERT_EQ(static_cast<uint64_t>(lru_hash_seed), meta_indexer_->mutex_shard_hash_seed_);

    const KeyVector keys = {KeyType{0},
                            KeyType{1},
                            KeyType{2},
                            KeyType{17},
                            KeyType{1'000},
                            KeyType{94'422},
                            std::numeric_limits<KeyType>::max()};
    for (KeyType key : keys) {
        const uint64_t lru_hash = Hash64(reinterpret_cast<const char *>(&key), sizeof(key), lru_hash_seed);
        EXPECT_EQ(static_cast<int32_t>(lru_hash & meta_indexer_->mutex_shard_mask_),
                  meta_indexer_->GetMutexShardIndex(key));
    }
}

TEST_F(MetaIndexerTest, TestMakeBatches2) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 16,
        "batch_key_size" : 3,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(15, meta_indexer_->mutex_shard_mask_);
    ASSERT_EQ(3, meta_indexer_->batch_key_size_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, GetPersistentStorageType(*meta_indexer_));

    KeyVector keys = {0, 4, 7, 16, 20, 32, 33, 34, 35, 64};
    PropertyMapVector properties = {{{"uri", "0"}},
                                    {{"uri", "4"}},
                                    {{"uri", "7"}},
                                    {{"uri", "16"}},
                                    {{"uri", "20"}},
                                    {{"uri", "32"}},
                                    {{"uri", "33"}},
                                    {{"uri", "34"}},
                                    {{"uri", "35"}},
                                    {{"uri", "64"}}};
    LocationIdsPerKey empty_location_ids;
    CacheLocationMapVector empty_locations;
    auto batches = meta_indexer_->MakeBatches(keys, empty_location_ids, empty_locations, properties);

    std::vector<int32_t> covered_indexs;
    for (const auto &batch : batches) {
        std::set<int32_t> shards_in_batch(batch.batch_shard_indexs.begin(), batch.batch_shard_indexs.end());
        ASSERT_EQ(shards_in_batch.size(), batch.batch_shard_indexs.size()) << "duplicate shard in one batch";
        ASSERT_EQ(batch.batch_keys.size(), batch.batch_indexs.size());
        ASSERT_EQ(batch.batch_keys.size(), batch.batch_properties.size());
        for (size_t j = 0; j < batch.batch_keys.size(); ++j) {
            const int32_t origin_idx = batch.batch_indexs[j];
            ASSERT_EQ(keys[origin_idx], batch.batch_keys[j]);
            const int32_t shard = meta_indexer_->GetMutexShardIndex(batch.batch_keys[j]);
            ASSERT_TRUE(shards_in_batch.count(shard) > 0);
            ASSERT_EQ(std::to_string(keys[origin_idx]), batch.batch_properties[j].at("uri"));
            covered_indexs.push_back(origin_idx);
        }
        ASSERT_TRUE(batch.batch_locations.empty());
    }
    std::sort(covered_indexs.begin(), covered_indexs.end());
    std::vector<int32_t> expected_indexs(keys.size());
    std::iota(expected_indexs.begin(), expected_indexs.end(), 0);
    ASSERT_EQ(expected_indexs, covered_indexs);
}

TEST_F(MetaIndexerTest, TestLocalSimple) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,        
        "batch_key_size" : 2,
        "meta_storage_backend_config" : {
            "storage_type" : "local"
        },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(100, meta_indexer_->max_key_count_);
    ASSERT_EQ(7, meta_indexer_->mutex_shard_mask_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, GetPersistentStorageType(*meta_indexer_));
    DoSimpleTest();
}

TEST_F(MetaIndexerTest, TestMultiThread) {
    std::string configStr = R"({
        "max_key_count" : 10000,
        "mutex_shard_num" : 16,        
        "batch_key_size" : 4,
        "meta_storage_backend_config" : {
            "storage_type" : "local"
        },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    ASSERT_EQ(EC_OK, InitIndexer(configStr));
    ASSERT_EQ(10000, meta_indexer_->max_key_count_);
    ASSERT_EQ(15, meta_indexer_->mutex_shard_mask_);
    ASSERT_EQ(4, meta_indexer_->batch_key_size_);
    ASSERT_EQ(META_LOCAL_BACKEND_TYPE_STR, GetPersistentStorageType(*meta_indexer_));
    DoMultiThreadTest();
}

TEST_F(MetaIndexerTest, TestMetadataPersistAndRecover) {
    const std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "persist_metadata_interval_time_ms" : 0,
        "meta_storage_backend_config" : { "storage_type" : "dummy" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";
    const auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->FromJsonString(configStr);
    const std::string path = GetPrivateTestRuntimeDataPath() + "meta_dummy_backend_file";
    meta_indexer_config->meta_storage_backend_config_->SetStorageUri("file://" + path);

    // verify fresh init behavior
    {
        meta_indexer_ = std::make_shared<MetaIndexer>();
        ASSERT_EQ(ErrorCode::EC_OK, meta_indexer_->Init(/* instance_id */ "test_instance_01", meta_indexer_config));

        ASSERT_EQ(0, meta_indexer_->GetKeyCount());
        for (auto &v : meta_indexer_->storage_usage_data_.storage_usage_by_type_) {
            ASSERT_EQ(0, v.load());
        }
    }

    // persist
    meta_indexer_->key_count_.store(3);
    const std::vector<std::uint64_t> expected_usage_vec{1, 100, 200, 300, 400, 500, 600, 700, 800};
    ASSERT_EQ(expected_usage_vec.size(), meta_indexer_->storage_usage_data_.storage_usage_by_type_.size());
    for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
        meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).store(expected_usage_vec.at(i));
    }
    meta_indexer_->PersistMetaData();

    // verify recovery behavior
    {
        meta_indexer_ = std::make_shared<MetaIndexer>();
        ASSERT_EQ(ErrorCode::EC_OK, meta_indexer_->Init(/* instance_id */ "test_instance_01", meta_indexer_config));

        ASSERT_EQ(3, meta_indexer_->GetKeyCount());
        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }
    }
}

TEST_F(MetaIndexerTest, TestStorageUsageDataManipulation) {
    std::string configStr = R"({
        "max_key_count" : 100,
        "mutex_shard_num" : 8,
        "persist_metadata_interval_time_ms" : 0,
        "meta_storage_backend_config" : { "storage_type" : "local" },
        "meta_cache_policy_config" : { "capacity" : 0 }
    })";

    ASSERT_EQ(EC_OK, InitIndexer(configStr));

    // test get/set
    {
        meta_indexer_->storage_usage_data_.Reset();
        ASSERT_EQ(0, meta_indexer_->GetStorageUsage());

        auto type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        std::vector<std::uint64_t> expected_usage_vec{0, 100, 200, 300, 400, 0, 0, 0, 0};

        type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));

        type = DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));

        type = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));

        type = DataStorageType::DATA_STORAGE_TYPE_NFS;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));

        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }

        type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));

        type = DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));

        type = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));

        type = DataStorageType::DATA_STORAGE_TYPE_NFS;
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));

        std::uint64_t expect_usage = 0;
        for (const auto &v : expected_usage_vec) {
            expect_usage += v;
        }
        ASSERT_EQ(expect_usage, meta_indexer_->GetStorageUsage());
    }

    // test add/sub
    {
        meta_indexer_->storage_usage_data_.Reset();
        auto type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        std::vector<std::uint64_t> expected_usage_vec{0, 100, 200, 300, 400, 0, 0, 0, 0};

        type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));
        meta_indexer_->AddStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)) + 16,
                  meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 1024);         // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(type)); // expect to be proper handled

        type = DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));
        meta_indexer_->AddStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)) + 16,
                  meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 1024);         // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(type)); // expect to be proper handled

        type = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));
        meta_indexer_->AddStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)) + 16,
                  meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 1024);         // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(type)); // expect to be proper handled

        type = DataStorageType::DATA_STORAGE_TYPE_NFS;
        meta_indexer_->SetStorageUsageByType(type, expected_usage_vec.at(static_cast<std::size_t>(type)));
        meta_indexer_->AddStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)) + 16,
                  meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 16);
        ASSERT_EQ(expected_usage_vec.at(static_cast<std::size_t>(type)), meta_indexer_->GetStorageUsageByType(type));
        meta_indexer_->SubStorageUsageByType(type, 1024);         // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(type)); // expect to be proper handled
    }

    // test special case: DATA_STORAGE_TYPE_VCNS_HF3FS behavior as DATA_STORAGE_TYPE_HF3FS
    {
        meta_indexer_->storage_usage_data_.Reset();
        std::vector<std::uint64_t> expected_usage_vec{0, 128, 0, 0, 0, 0, 0, 0, 0};

        meta_indexer_->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 128);
        ASSERT_EQ(128, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        ASSERT_EQ(128, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }

        meta_indexer_->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 16);
        ASSERT_EQ(128 + 16, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        ASSERT_EQ(128 + 16, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));

        meta_indexer_->SubStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 16);
        ASSERT_EQ(128, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        ASSERT_EQ(128, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }

        meta_indexer_->SubStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 1024); // would underflow
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        ASSERT_EQ(0, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        expected_usage_vec[1] = 0;
        for (std::size_t i = 0; i != meta_indexer_->storage_usage_data_.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), meta_indexer_->storage_usage_data_.storage_usage_by_type_.at(i).load());
        }
    }
}

TEST_F(MetaIndexerTest, TestStorageUsageDataSeriDeseri) {
    StorageUsageData storage_usage_data;

    // Successful round-trip: serialize then deserialize
    {
        std::vector<std::uint64_t> expected_usage_vec{1, 100, 200, 300, 400, 500, 600, 700, 800};

        storage_usage_data.Reset();
        for (std::size_t i = 0; i != expected_usage_vec.size(); ++i) {
            storage_usage_data.storage_usage_by_type_.at(i).store(expected_usage_vec.at(i));
        }

        std::string serialized = storage_usage_data.Serialize();
        ASSERT_EQ(
            R"({"unknown":1,"hf3fs":100,"mooncake":200,"pace":300,"file":400,"vcns_hf3fs":500,"dummy":600,"event_report_l1p5":700,"event_report_l2":800})",
            serialized);

        storage_usage_data.Reset();
        ASSERT_EQ(EC_OK, storage_usage_data.Deserialize(serialized));
        for (std::size_t i = 0; i != storage_usage_data.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), storage_usage_data.storage_usage_by_type_.at(i).load());
        }
    }

    // Legal input: keys in different order
    {
        const std::vector<std::uint64_t> expected_usage_vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
        storage_usage_data.Reset();
        ASSERT_EQ(
            EC_OK,
            storage_usage_data.Deserialize(
                R"({"event_report_l2":9,"event_report_l1p5":8,"dummy":7,"vcns_hf3fs":6,"file":5,"pace":4,"mooncake":3,"hf3fs":2,"unknown":1})"));
        for (std::size_t i = 0; i != storage_usage_data.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), storage_usage_data.storage_usage_by_type_.at(i).load());
        }
    }

    // Legal input: partial JSON (missing keys default to 0)
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 99);
        ASSERT_EQ(EC_OK, storage_usage_data.Deserialize(R"({"hf3fs":100,"mooncake":200})"));
        ASSERT_EQ(100, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        ASSERT_EQ(200, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
        ASSERT_EQ(0, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
        ASSERT_EQ(0, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
        ASSERT_EQ(0, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_DUMMY));
    }

    // Legal input: whitespace-padded JSON
    {
        const std::vector<std::uint64_t> expected_usage_vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
        storage_usage_data.Reset();
        ASSERT_EQ(EC_OK,
                  storage_usage_data.Deserialize("  "
                                                 "{\"unknown\":1,\"hf3fs\":2,\"mooncake\":3,\"pace\":4,\"file\":5,"
                                                 "\"vcns_hf3fs\":6,\"dummy\":7,\"event_report_l1p5\":8,"
                                                 "\"event_report_l2\":9}  "));
        for (std::size_t i = 0; i != storage_usage_data.storage_usage_by_type_.size(); ++i) {
            ASSERT_EQ(expected_usage_vec.at(i), storage_usage_data.storage_usage_by_type_.at(i).load());
        }
    }

    // Empty string
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 888);
        ASSERT_EQ(EC_ERROR, storage_usage_data.Deserialize(""));
        // Original data must not be modified on error
        ASSERT_EQ(888, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Malformed data: not valid JSON (old comma-separated format)
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 777);
        ASSERT_EQ(EC_ERROR, storage_usage_data.Deserialize("0,abc,2,3,4,5"));
        // Original data must not be modified on error
        ASSERT_EQ(777, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Malformed data: JSON array instead of object
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 777);
        ASSERT_EQ(EC_ERROR, storage_usage_data.Deserialize("[1,2,3,4,5,6]"));
        // Original data must not be modified on error
        ASSERT_EQ(777, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Malformed data: non-integer value for a key
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 777);
        ASSERT_EQ(EC_ERROR,
                  storage_usage_data.Deserialize(
                      R"({"unknown":0,"hf3fs":"not_a_number","mooncake":2,"pace":3,"file":4,"vcns_hf3fs":5})"));
        // Original data must not be modified on error
        ASSERT_EQ(777, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Malformed data: floating-point value
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 777);
        ASSERT_EQ(EC_ERROR,
                  storage_usage_data.Deserialize(
                      R"({"unknown":0,"hf3fs":1.5,"mooncake":2,"pace":3,"file":4,"vcns_hf3fs":5})"));
        // Original data must not be modified on error
        ASSERT_EQ(777, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // Unrecognized key: must be rejected and leave data unchanged
    {
        storage_usage_data.Reset();
        storage_usage_data.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 888);
        ASSERT_EQ(EC_ERROR, storage_usage_data.Deserialize(R"({"future_type": 999, "hf3fs": 10})"));
        // Original data must not be modified on error
        ASSERT_EQ(888, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }

    // "unknown" is a legitimate key and must still be accepted.
    {
        storage_usage_data.Reset();
        ASSERT_EQ(EC_OK, storage_usage_data.Deserialize(R"({"unknown":7,"hf3fs":10})"));
        ASSERT_EQ(7u,
                  storage_usage_data.storage_usage_by_type_
                      .at(static_cast<std::size_t>(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN))
                      .load());
        ASSERT_EQ(10, storage_usage_data.GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    }
}
