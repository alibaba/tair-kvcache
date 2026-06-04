#include "kv_cache_manager/meta/meta_indexer.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "kv_cache_manager/common/common.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/meta/utils.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {
#define PREFIX_INDEXER_LOG(LEVEL, format, args...)                                                                     \
    KVCM_LOG_##LEVEL("trace_id[%s] instance[%s] | " format, trace_id.c_str(), instance_id_.c_str(), ##args);

namespace {
static constexpr const char *kPutMetaOperation = "put";
static constexpr const char *kRmwMetaOperation = "read_modify_write";
static constexpr const char *kRmwUpsertMetaOperation = "read_modify_write_upsert";
static constexpr const char *kRmwDeleteMetaOperation = "read_modify_write_delete";
static constexpr const char *kDeleteMetaOperation = "delete";
static constexpr const char *kExistMetaOperation = "exist";
static constexpr const char *kGetMetaOperation = "get";
} // namespace

class MetaIndexer::ScopedBatchLock {
public:
    // If `out_lock_wait_time_us` is non-null, accumulates the elapsed time
    // spent acquiring all shard mutexes (in microseconds).
    ScopedBatchLock(MetaIndexer &indexer,
                    const std::vector<int32_t> &shard_indexs,
                    int64_t *out_lock_wait_time_us = nullptr)
        : indexer_(indexer), shard_indexs_(shard_indexs) {
        const int64_t begin = TimestampUtil::GetCurrentTimeUs();
        for (const int32_t shardIdx : shard_indexs_) {
            indexer_.mutex_shards_[shardIdx]->lock();
        }
        if (out_lock_wait_time_us != nullptr) {
            *out_lock_wait_time_us += TimestampUtil::GetCurrentTimeUs() - begin;
        }
    }
    ~ScopedBatchLock() {
        for (const int32_t shardIdx : shard_indexs_) {
            indexer_.mutex_shards_[shardIdx]->unlock();
        }
    }

    ScopedBatchLock(const ScopedBatchLock &) = delete;
    ScopedBatchLock &operator=(const ScopedBatchLock &) = delete;

private:
    MetaIndexer &indexer_;
    std::vector<int32_t> shard_indexs_;
};

MetaIndexer::~MetaIndexer() {
    // try to persist metadata when quit gracefully
    if (backend_manager_) {
        PersistMetaData();
    }
}

ErrorCode MetaIndexer::Init(const std::string &instance_id, const std::shared_ptr<MetaIndexerConfig> &config) noexcept {
    if (!config || !config->GetMetaStorageBackendConfig()) {
        KVCM_LOG_ERROR("instance[%s] meta indexer init failed, config is invalid", instance_id.c_str());
        return EC_BADARGS;
    }
    max_key_count_ = config->GetMaxKeyCount();
    const size_t mutex_shard_num = config->GetMutexShardNum();
    batch_key_size_ = config->GetBatchKeySize();
    persist_metadata_interval_time_ms_ = config->GetPersistMetaDataIntervalTimeMs();
    if (mutex_shard_num > max_key_count_ || (mutex_shard_num & (mutex_shard_num - 1)) || mutex_shard_num <= 0) {
        KVCM_LOG_ERROR(
            "instance[%s] meta indexer init failed, config is invalid, mutex shard num[%lu] max key count[%lu]",
            instance_id.c_str(),
            mutex_shard_num,
            max_key_count_);
        return EC_CONFIG_ERROR;
    }
    mutex_shard_mask_ = mutex_shard_num - 1;
    for (size_t i = 0; i < mutex_shard_num; ++i) {
        mutex_shards_.emplace_back(std::make_unique<std::mutex>());
    }

    instance_id_ = instance_id;
    auto storage_backend_config = config->GetMetaStorageBackendConfig();

    backend_manager_ = std::make_unique<MetaStorageBackendManager>();
    auto ec = backend_manager_->Init(instance_id_, storage_backend_config);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("instance[%s] meta storage backend manager init failed, ec[%d]", instance_id_.c_str(), ec);
        backend_manager_.reset();
        return ec;
    }
    ec = backend_manager_->Open();
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("instance[%s] meta storage backend manager open failed, ec[%d]", instance_id_.c_str(), ec);
        backend_manager_.reset();
        return ec;
    }

    storage_usage_data_.Reset();
    ec = RecoverMetaData();
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR("instance[%s] recover metadata failed, ec[%d]", instance_id_.c_str(), ec);
        return ec;
    }
    KVCM_LOG_INFO("instance[%s] meta indexer init success, mutex shard num[%lu], max key count[%lu], "
                  "batch key size[%lu], key_count[%lu], persist_metadata_interval_time_ms[%zu], storage usage data[%s]",
                  instance_id_.c_str(),
                  mutex_shard_num,
                  max_key_count_,
                  batch_key_size_,
                  key_count_.load(),
                  persist_metadata_interval_time_ms_,
                  storage_usage_data_.ToJsonString().c_str());
    return EC_OK;
}

MetaIndexer::Result MetaIndexer::Put(RequestContext *request_context,
                                     const KeyVector &keys,
                                     CacheLocationMapVector &location_maps,
                                     PropertyMapVector &properties) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    const auto &trace_id = request_context->trace_id();
    if ((!location_maps.empty() && keys.size() != location_maps.size()) ||
        (!properties.empty() && keys.size() != properties.size())) {
        PREFIX_INDEXER_LOG(ERROR,
                           "Put keys size[%lu], location_maps size[%lu], properties size[%lu] not equal",
                           keys.size(),
                           location_maps.size(),
                           properties.size());
        return Result(EC_ERROR);
    }
    if (keys.size() + GetKeyCount() > max_key_count_) {
        PREFIX_INDEXER_LOG(ERROR,
                           "Put keys count[%lu] + current key count[%lu] > max key count[%lu]",
                           keys.size(),
                           GetKeyCount(),
                           max_key_count_);
        return Result(EC_NOSPC);
    }

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());

    static LocationIdsPerKey empty_location_ids;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_location_ids, location_maps, properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    Result result(keys.size());
    int32_t error_count = 0;
    int64_t put_io_time_us = 0;
    int64_t lock_wait_time_us = 0;
    int64_t cache_backend_put_time_us = 0;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs, &lock_wait_time_us);
        int64_t begin_put_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = backend_manager_->Put(request_context, batch);
        put_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_put_io_time;
        int64_t v = 0;
        KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, cache_backend_put_time_us, v);
        cache_backend_put_time_us += v;
        error_count += ProcessErrorCodes(trace_id, error_codes, batch.batch_indexs, keys, kPutMetaOperation, result);
    }
    AdjustKeyCountMeta(keys.size() - error_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, put_io_time_us, put_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, lock_wait_time_us, lock_wait_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, cache_backend_put_time_us, cache_backend_put_time_us);
    ProcessErrorResult(trace_id, kPutMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::Delete(RequestContext *request_context, const KeyVector &keys) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();
    static LocationIdsPerKey empty_location_ids;
    static CacheLocationMapVector empty_locations;
    static PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_location_ids, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t lock_wait_time_us = 0;
    int64_t cache_backend_delete_time_us = 0;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs, &lock_wait_time_us);
        std::vector<ErrorCode> error_codes = backend_manager_->Delete(request_context, batch.batch_keys);
        int64_t v = 0;
        KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, cache_backend_delete_time_us, v);
        cache_backend_delete_time_us += v;
        error_count += ProcessErrorCodes(trace_id, error_codes, batch.batch_indexs, keys, kDeleteMetaOperation, result);
    }
    AdjustKeyCountMeta(error_count - keys.size());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, lock_wait_time_us, lock_wait_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, cache_backend_delete_time_us, cache_backend_delete_time_us);
    ProcessErrorResult(trace_id, kDeleteMetaOperation, error_count, keys.size(), result);
    return result;
}

std::pair<int32_t, int32_t> MetaIndexer::ExecuteRmwUpsert(const std::string &trace_id,
                                                          RequestContext *request_context,
                                                          BatchMetaData &upsert_batch,
                                                          const std::vector<int32_t> &put_global_indexs,
                                                          const KeyVector &all_keys,
                                                          RmwStats &stats,
                                                          Result &result) noexcept {
    if (upsert_batch.batch_keys.empty()) {
        return {0, 0};
    }

    stats.put_key_count += static_cast<int64_t>(put_global_indexs.size());
    stats.update_key_count += static_cast<int64_t>(upsert_batch.batch_keys.size() - put_global_indexs.size());

    std::vector<ErrorCode> upsert_ecs;
    if (put_global_indexs.size() + GetKeyCount() > max_key_count_) {
        PREFIX_INDEXER_LOG(ERROR,
                           "ReadModifyWrite put keys count[%lu] + current key count[%lu] > max key count[%lu]",
                           put_global_indexs.size(),
                           GetKeyCount(),
                           max_key_count_);
        upsert_ecs.assign(upsert_batch.batch_keys.size(), EC_NOSPC);
    } else {
        const int64_t begin = TimestampUtil::GetCurrentTimeUs();
        upsert_ecs = backend_manager_->Upsert(request_context, upsert_batch);
        stats.upsert_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin;
        int64_t v = 0;
        auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
        KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_searcher, index_serialize_time_us, v);
        stats.index_serialize_time_us += v;
        v = 0;
        KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, async_enqueue_timeout_key_count, v);
        stats.async_enqueue_timeout_key_count += v;
        v = 0;
        KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, async_enqueue_time_us, v);
        stats.async_enqueue_time_us += v;
        v = 0;
        KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, cache_backend_upsert_time_us, v);
        stats.cache_backend_upsert_time_us += v;
    }

    const int32_t error_count =
        ProcessErrorCodes(trace_id, upsert_ecs, upsert_batch.batch_indexs, all_keys, kRmwUpsertMetaOperation, result);
    int32_t put_success_count = 0;
    if (error_count == 0) {
        put_success_count = static_cast<int32_t>(put_global_indexs.size());
    } else {
        for (const int32_t idx : put_global_indexs) {
            if (result.error_codes[idx] == EC_OK) {
                ++put_success_count;
            }
        }
    }
    return {error_count, put_success_count};
}

std::pair<int32_t, int32_t> MetaIndexer::ExecuteRmwDelete(const std::string &trace_id,
                                                          RequestContext *request_context,
                                                          const BatchMetaData &delete_batch,
                                                          const KeyVector &all_keys,
                                                          RmwStats &stats,
                                                          Result &result) noexcept {
    if (delete_batch.batch_keys.empty()) {
        return {0, 0};
    }
    stats.delete_key_count += static_cast<int64_t>(delete_batch.batch_keys.size());

    const int64_t begin = TimestampUtil::GetCurrentTimeUs();
    std::vector<ErrorCode> delete_ecs;
    int32_t reclaimed_count = 0;
    if (delete_batch.batch_location_ids.empty()) {
        delete_ecs = backend_manager_->Delete(request_context, delete_batch.batch_keys);
    } else {
        delete_ecs = backend_manager_->Delete(
            request_context, delete_batch.batch_keys, delete_batch.batch_location_ids, reclaimed_count);
    }
    stats.delete_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin;
    int64_t v = 0;
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, async_enqueue_timeout_key_count, v);
    stats.async_enqueue_timeout_key_count += v;
    v = 0;
    KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, async_enqueue_time_us, v);
    stats.async_enqueue_time_us += v;
    v = 0;
    KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, cache_backend_delete_time_us, v);
    stats.cache_backend_delete_time_us += v;

    const int32_t error_count =
        ProcessErrorCodes(trace_id, delete_ecs, delete_batch.batch_indexs, all_keys, kRmwDeleteMetaOperation, result);
    // For whole-key deletes, success count = keys - errors.
    // For location deletes, reclaimed_count reflects empty blocks auto-removed.
    const int32_t delete_success_count =
        delete_batch.batch_location_ids.empty() ? delete_batch.batch_keys.size() - error_count : reclaimed_count;
    return {error_count, delete_success_count};
}

void MetaIndexer::EmitRmwMetrics(MetricsCollector *metrics_collector,
                                 const RmwStats &stats,
                                 size_t total_key_count) const noexcept {
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(metrics_collector);
    const bool has_upsert = stats.put_key_count + stats.update_key_count > 0;
    const bool has_delete = stats.delete_key_count > 0;

    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, rmw_get_io_time_us, stats.get_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, lock_wait_time_us, stats.lock_wait_time_us);

    if (has_upsert) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, upsert_io_time_us, stats.upsert_io_time_us);
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, cache_backend_upsert_time_us, stats.cache_backend_upsert_time_us);
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_searcher, index_serialize_time_us, stats.index_serialize_time_us);
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, read_modify_write_update_key_count, stats.update_key_count);
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, read_modify_write_put_key_count, stats.put_key_count);
    }

    if (has_delete) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, delete_io_time_us, stats.delete_io_time_us);
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, cache_backend_delete_time_us, stats.cache_backend_delete_time_us);
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, read_modify_write_delete_key_count, stats.delete_key_count);
    }

    if (has_upsert || has_delete) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, async_enqueue_timeout_key_count, stats.async_enqueue_timeout_key_count);
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_indexer, async_enqueue_time_us, stats.async_enqueue_time_us);
    }

    if (stats.has_index_deserialize) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            service_metrics_collector, meta_searcher, index_deserialize_time_us, stats.index_deserialize_time_us);
    }

    const int64_t skip_key_count =
        static_cast<int64_t>(total_key_count) - stats.update_key_count - stats.put_key_count - stats.delete_key_count;
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_skip_key_count, skip_key_count);
}

MetaIndexer::Result MetaIndexer::ReadModifyWriteBlock(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      const BlockIdsOnlyModifierFunc &modifier) noexcept {
    if (keys.empty()) {
        return Result(EC_OK);
    }
    const auto &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    std::shared_ptr<MetricsRegistry> ephemeral_metrics_registry = std::make_shared<MetricsRegistry>();
    std::shared_ptr<MetricsCollector> ephemeral_metrics_collector =
        std::make_shared<ServiceMetricsCollector>(ephemeral_metrics_registry);
    ephemeral_metrics_collector->Init();
    auto ephemeral_request_context =
        std::make_shared<RequestContext>("read_modify_write_block", ephemeral_metrics_collector);

    static LocationIdsPerKey empty_location_ids;
    static CacheLocationMapVector empty_locations;
    static PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_location_ids, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    Result result(keys.size());
    int32_t error_count = 0;
    RmwStats stats;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs, &stats.lock_wait_time_us);

        // 1. Read each key's existing location id list (no value deserialization)
        const auto &batch_keys = batch.batch_keys;
        LocationIdsPerKey batch_location_ids;
        const int64_t begin_get = TimestampUtil::GetCurrentTimeUs();
        std::vector<ErrorCode> get_ecs =
            backend_manager_->GetLocationIds(ephemeral_request_context.get(), batch_keys, batch_location_ids);
        stats.get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get;

        // 2. Modify -> bucket each key into upsert_batch / delete_batch.
        BatchMetaData upsert_batch;
        BatchMetaData delete_batch;
        std::vector<int32_t> put_global_indexs; // brand-new keys (subset of upsert_batch)
        for (size_t i = 0; i < batch_keys.size(); ++i) {
            const KeyType key = batch_keys[i];

            const LocationIdVector &existing_location_ids = batch_location_ids[i];
            const ErrorCode get_ec = get_ecs[i];
            const int32_t global_idx = batch.batch_indexs[i];
            PropertyMap upsert_property_map;
            CacheLocationMap out_new_locations;
            const auto [action, modifier_ec] = modifier(
                existing_location_ids, get_ec, static_cast<size_t>(global_idx), upsert_property_map, out_new_locations);
            if (action == MA_OK) {
                if (get_ec != EC_OK && get_ec != EC_NOENT) {
                    result.error_codes[global_idx] = get_ec;
                    ++error_count;
                    continue;
                }
                upsert_batch.batch_keys.emplace_back(key);
                upsert_batch.batch_indexs.emplace_back(global_idx);
                upsert_batch.batch_locations.emplace_back(std::move(out_new_locations));
                upsert_batch.batch_properties.emplace_back(std::move(upsert_property_map));
                if (get_ec == EC_NOENT) {
                    put_global_indexs.emplace_back(global_idx);
                }
            } else if (action == MA_DELETE && modifier_ec == EC_OK) {
                delete_batch.batch_keys.emplace_back(key);
                delete_batch.batch_indexs.emplace_back(global_idx);
            } else {
                // MA_FAIL / MA_SKIP / unknown: surface modifier_ec if any.
                if (modifier_ec != EC_OK) {
                    result.error_codes[global_idx] = modifier_ec;
                    ++error_count;
                }
            }
        }

        // 3. Dispatch upsert and delete sub-batches.
        const auto [upsert_errs, put_success_count] = ExecuteRmwUpsert(
            trace_id, ephemeral_request_context.get(), upsert_batch, put_global_indexs, keys, stats, result);
        const auto [delete_errs, delete_success_count] =
            ExecuteRmwDelete(trace_id, ephemeral_request_context.get(), delete_batch, keys, stats, result);
        error_count += upsert_errs + delete_errs;
        AdjustKeyCountMeta(put_success_count - delete_success_count);
    }

    EmitRmwMetrics(request_context->metrics_collector(), stats, keys.size());
    ProcessErrorResult(trace_id, kRmwMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::LocationResult MetaIndexer::ReadModifyWriteLocation(RequestContext *request_context,
                                                                 const KeyVector &keys,
                                                                 const LocationIdsPerKey &location_ids,
                                                                 const LocationModifierFunc &modifier) noexcept {
    const auto &trace_id = request_context->trace_id();
    if (keys.empty()) {
        return LocationResult(EC_OK);
    }
    if (keys.size() != location_ids.size()) {
        PREFIX_INDEXER_LOG(ERROR,
                           "ReadModifyWriteLocation keys size[%lu] != location_ids size[%lu]",
                           keys.size(),
                           location_ids.size());
        return LocationResult(EC_BADARGS);
    }

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    std::shared_ptr<MetricsRegistry> ephemeral_metrics_registry = std::make_shared<MetricsRegistry>();
    std::shared_ptr<MetricsCollector> ephemeral_metrics_collector =
        std::make_shared<ServiceMetricsCollector>(ephemeral_metrics_registry);
    ephemeral_metrics_collector->Init();
    auto ephemeral_request_context =
        std::make_shared<RequestContext>("read_modify_write_location", ephemeral_metrics_collector);

    static CacheLocationMapVector empty_locations;
    static PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, location_ids, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    LocationResult location_result(location_ids);
    Result rmw_result(keys.size());
    int32_t error_count = 0;
    RmwStats stats;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs, &stats.lock_wait_time_us);

        // 1. One batched read for every (key, location_id) return deserialised CacheLocation
        const auto &batch_keys = batch.batch_keys;
        LocationsPerKey batch_locations_per_key;
        const int64_t begin_get = TimestampUtil::GetCurrentTimeUs();
        std::vector<std::vector<ErrorCode>> get_ecs_per_key = backend_manager_->GetLocations(
            ephemeral_request_context.get(), batch_keys, batch.batch_location_ids, batch_locations_per_key);
        stats.get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get;
        int64_t v = 0;
        auto *ephemeral_service_metrics_collector =
            dynamic_cast<ServiceMetricsCollector *>(ephemeral_request_context->metrics_collector());
        KVCM_METRICS_COLLECTOR_GET_METRICS(
            ephemeral_service_metrics_collector, meta_searcher, index_deserialize_time_us, v);
        stats.index_deserialize_time_us += v;
        stats.has_index_deserialize = true;

        // 2. Per-key modifier dispatch -> bucket each key into the upsert sub-batch or the delete sub-batch.
        BatchMetaData upsert_batch;
        BatchMetaData delete_batch;
        std::vector<std::vector<int32_t>> upsert_location_indexs(keys.size());
        std::vector<std::vector<int32_t>> delete_location_indexs(keys.size());
        for (size_t i = 0; i < batch_keys.size(); ++i) {
            const int32_t global_idx = batch.batch_indexs[i];
            const KeyType key = batch_keys[i];

            const std::vector<ErrorCode> &get_ecs = get_ecs_per_key[i];
            const LocationIdVector &loc_ids = batch.batch_location_ids[i];
            CacheLocationVector &loc_values = batch_locations_per_key[i];
            PropertyMap upsert_property_map;
            auto [action, modifier_ecs] =
                modifier(get_ecs, loc_ids, static_cast<size_t>(global_idx), loc_values, upsert_property_map);
            if (modifier_ecs.size() != loc_ids.size()) {
                modifier_ecs.assign(loc_ids.size(), EC_ERROR);
                action = MA_FAIL;
            }
            if (action == MA_OK) {
                CacheLocationMap upsert_loc_map;
                for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
                    if (modifier_ecs[loc_index] != EC_OK) {
                        location_result.per_location_error_codes[global_idx][loc_index] = modifier_ecs[loc_index];
                        continue;
                    }
                    const LocationId &loc_id = loc_ids[loc_index];
                    const CacheLocationConstPtr &working_loc = loc_values[loc_index];
                    assert(working_loc && loc_id == working_loc->id());
                    upsert_loc_map.emplace(loc_id, working_loc);
                    upsert_location_indexs[global_idx].emplace_back(loc_index);
                }
                if (!upsert_loc_map.empty() || !upsert_property_map.empty()) {
                    upsert_batch.batch_keys.emplace_back(key);
                    upsert_batch.batch_indexs.emplace_back(global_idx);
                    upsert_batch.batch_locations.emplace_back(std::move(upsert_loc_map));
                    upsert_batch.batch_properties.emplace_back(std::move(upsert_property_map));
                }
            } else if (action == MA_DELETE) {
                LocationIdVector alive_ids;
                for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
                    if (modifier_ecs[loc_index] != EC_OK) {
                        location_result.per_location_error_codes[global_idx][loc_index] = modifier_ecs[loc_index];
                        continue;
                    }
                    alive_ids.emplace_back(loc_ids[loc_index]);
                    delete_location_indexs[global_idx].emplace_back(loc_index);
                }
                if (!alive_ids.empty()) {
                    delete_batch.batch_keys.emplace_back(key);
                    delete_batch.batch_indexs.emplace_back(global_idx);
                    delete_batch.batch_location_ids.emplace_back(std::move(alive_ids));
                }
            } else {
                // MA_FAIL / MA_SKIP / unknown: surface modifier_ec if any.
                if (action == MA_FAIL) {
                    ++error_count;
                }
                location_result.per_location_error_codes[global_idx] = std::move(modifier_ecs);
            }
        }

        // 3. Dispatch upsert and delete sub-batches.
        static std::vector<int32_t> empty_put_global_indexs;
        const auto [upsert_errs, put_success_count] = ExecuteRmwUpsert(
            trace_id, ephemeral_request_context.get(), upsert_batch, empty_put_global_indexs, keys, stats, rmw_result);
        for (const auto &global_index : upsert_batch.batch_indexs) {
            for (const auto &location_index : upsert_location_indexs[global_index]) {
                location_result.per_location_error_codes[global_index][location_index] =
                    rmw_result.error_codes[global_index];
            }
        }
        const auto [delete_errs, delete_success_count] =
            ExecuteRmwDelete(trace_id, ephemeral_request_context.get(), delete_batch, keys, stats, rmw_result);
        for (const auto &global_index : delete_batch.batch_indexs) {
            for (const auto &location_index : delete_location_indexs[global_index]) {
                location_result.per_location_error_codes[global_index][location_index] =
                    rmw_result.error_codes[global_index];
            }
        }
        error_count += upsert_errs + delete_errs;
        AdjustKeyCountMeta(put_success_count - delete_success_count);
    }

    EmitRmwMetrics(request_context->metrics_collector(), stats, keys.size());
    if (error_count == keys.size()) {
        location_result.ec = EC_ERROR;
        PREFIX_INDEXER_LOG(DEBUG, "all locations rmw failed, error count[%d]", error_count);
    } else if (error_count > 0) {
        location_result.ec = EC_PARTIAL_OK;
        PREFIX_INDEXER_LOG(
            DEBUG, "partial locations rmw failed, keys count[%lu] failed count[%d]", keys.size(), error_count);
    }
    return location_result;
}

MetaIndexer::Result
MetaIndexer::Exist(RequestContext *request_context, const KeyVector &keys, std::vector<bool> &out_exists) noexcept {
    const auto &trace_id = request_context->trace_id();
    out_exists.reserve(keys.size());
    std::vector<ErrorCode> error_codes = backend_manager_->Exists(request_context, keys, out_exists);

    Result result(keys.size());
    int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kExistMetaOperation, result);
    ProcessErrorResult(trace_id, kExistMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::Get(RequestContext *request_context,
                                     const KeyVector &keys,
                                     CacheLocationMapVector &out_location_maps,
                                     PropertyMapVector &out_properties) noexcept {
    if (keys.empty()) {
        out_location_maps.clear();
        out_properties.clear();
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();

    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->Get(request_context, keys, out_location_maps, out_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);

    Result result(keys.size());
    int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kGetMetaOperation, result);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::GetLocations(RequestContext *request_context,
                                              const KeyVector &keys,
                                              CacheLocationMapVector &out_location_maps) noexcept {
    if (keys.empty()) {
        out_location_maps.clear();
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();

    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->GetLocations(request_context, keys, out_location_maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);

    Result result(keys.size());
    int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kGetMetaOperation, result);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::LocationResult MetaIndexer::GetLocations(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      const LocationIdsPerKey &location_ids,
                                                      LocationsPerKey &out_locations) noexcept {
    assert(keys.size() == location_ids.size());
    if (keys.empty()) {
        out_locations.clear();
        return LocationResult(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();

    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto per_location_ecs = backend_manager_->GetLocations(request_context, keys, location_ids, out_locations);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);

    LocationResult result(location_ids);
    result.per_location_error_codes = std::move(per_location_ecs);

    int64_t total_slots = 0;
    int64_t error_slots = 0;
    for (size_t i = 0; i < result.per_location_error_codes.size(); ++i) {
        for (size_t j = 0; j < result.per_location_error_codes[i].size(); ++j) {
            ++total_slots;
            const ErrorCode ec = result.per_location_error_codes[i][j];
            if (ec != EC_OK) {
                ++error_slots;
                if (ec != EC_NOENT) {
                    PREFIX_INDEXER_LOG(ERROR,
                                       "meta indexer get_locations failed, key[%ld] location_id[%s] ec[%d]",
                                       keys[i],
                                       location_ids[i][j].c_str(),
                                       ec);
                }
            }
        }
    }
    if (total_slots > 0 && error_slots == total_slots) {
        result.ec = EC_ERROR;
    } else if (error_slots > 0) {
        result.ec = EC_PARTIAL_OK;
    }
    return result;
}

MetaIndexer::Result MetaIndexer::GetProperties(RequestContext *request_context,
                                               const KeyVector &keys,
                                               const std::vector<std::string> &property_names,
                                               PropertyMapVector &out_properties) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();
    out_properties.reserve(keys.size());
    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->GetProperties(request_context, keys, property_names, out_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    Result result(keys.size());
    int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kGetMetaOperation, result);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

ErrorCode MetaIndexer::Scan(RequestContext *request_context,
                            const std::string &cursor,
                            const size_t limit,
                            std::string &out_next_cursor,
                            KeyVector &out_keys) noexcept {
    out_keys.reserve(limit);
    ErrorCode ec = backend_manager_->ListKeys(request_context, cursor, limit, out_next_cursor, out_keys);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR(
            "instance[%s] meta indexer scan failed, cursor[%s] limit[%lu] next cursor[%s] scan key size[%lu]",
            instance_id_.c_str(),
            cursor.c_str(),
            limit,
            out_next_cursor.c_str(),
            out_keys.size());
    }
    return ec;
}

ErrorCode
MetaIndexer::RandomSample(RequestContext *request_context, const size_t count, KeyVector &out_keys) const noexcept {
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    out_keys.reserve(count);
    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    ErrorCode ec = backend_manager_->RandomSample(request_context, count, out_keys);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector,
                                       meta_indexer,
                                       rand_io_time_us,
                                       TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("instance[%s] meta indexer random sample failed, count[%lu] sample key size[%lu]",
                       instance_id_.c_str(),
                       count,
                       out_keys.size());
    }
    return ec;
}

ErrorCode MetaIndexer::SampleReclaimKeys(RequestContext *request_context,
                                         const int64_t count,
                                         KeyVector &out_keys) const noexcept {
    out_keys.clear();
    out_keys.reserve(count);
    ErrorCode ec = backend_manager_->SampleReclaimKeys(request_context, count, out_keys);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("instance[%s] meta indexer sample reclaim keys failed, count[%lu] sample key size[%lu]",
                       instance_id_.c_str(),
                       count,
                       out_keys.size());
    }
    return ec;
}

size_t MetaIndexer::GetKeyCount() const noexcept { return key_count_.load(); }

size_t MetaIndexer::GetMaxKeyCount() const noexcept { return max_key_count_; }

size_t MetaIndexer::GetMemUsage() const noexcept { return backend_manager_->GetMemUsage(); }

bool MetaIndexer::Sync(const KeyVector &keys) noexcept { return backend_manager_->Sync(keys); }

MetaStorageBackend::AsyncWriteStats MetaIndexer::GetAsyncWriteStats() noexcept {
    return backend_manager_->GetAsyncWriteStats();
}

int64_t MetaIndexer::GetOldestAccessTime() const noexcept { return backend_manager_->GetOldestAccessTime(); }

std::uint64_t MetaIndexer::GetStorageUsage() const noexcept { return storage_usage_data_.GetStorageUsage(); }

std::uint64_t MetaIndexer::GetStorageUsageByType(const DataStorageType &type) const noexcept {
    return storage_usage_data_.GetStorageUsageByType(type);
}

void MetaIndexer::SetStorageUsageByType(const DataStorageType &type, const std::uint64_t value) noexcept {
    storage_usage_data_.SetStorageUsageByType(type, value);
}

std::uint64_t MetaIndexer::AddStorageUsageByType(const DataStorageType &type, const std::uint64_t value) noexcept {
    return storage_usage_data_.AddStorageUsageByType(type, value);
}

std::uint64_t MetaIndexer::SubStorageUsageByType(const DataStorageType &type, const std::uint64_t value) noexcept {
    return storage_usage_data_.SubStorageUsageByType(type, value);
}

std::vector<BatchMetaData> MetaIndexer::MakeBatches(const KeyVector &keys,
                                                    const LocationIdsPerKey &location_ids,
                                                    CacheLocationMapVector &locations,
                                                    PropertyMapVector &properties) const noexcept {
    std::vector<BatchMetaData> result;

    std::map<int32_t, std::vector<int32_t>> shard_map;
    for (int32_t i = 0; i < static_cast<int32_t>(keys.size()); ++i) {
        const int32_t shard_idx = GetShardIndex(keys[i], mutex_shard_mask_);
        shard_map[shard_idx].push_back(i);
    }
    if (shard_map.empty()) {
        return result;
    }

    BatchMetaData current;
    size_t current_batch_size = 0;
    size_t shards_emitted = 0;
    const size_t total_shards = shard_map.size();

    for (auto &shard_kv : shard_map) {
        const int32_t shard_index = shard_kv.first;
        const auto &index_list = shard_kv.second;

        current.batch_shard_indexs.emplace_back(shard_index);
        for (const int32_t idx : index_list) {
            current.batch_indexs.emplace_back(idx);
            current.batch_keys.emplace_back(keys[idx]);
            if (!properties.empty()) {
                assert(idx < static_cast<int32_t>(properties.size()));
                current.batch_properties.emplace_back(std::move(properties[idx]));
            }
            if (!locations.empty()) {
                assert(idx < static_cast<int32_t>(locations.size()));
                current.batch_locations.emplace_back(std::move(locations[idx]));
            }
            if (!location_ids.empty()) {
                assert(idx < static_cast<int32_t>(location_ids.size()));
                current.batch_location_ids.emplace_back(location_ids[idx]);
            }
        }
        current_batch_size += index_list.size();
        ++shards_emitted;

        // Flush on soft-limit, or after the last shard so the tail batch is kept.
        if (current_batch_size >= batch_key_size_ || shards_emitted == total_shards) {
            result.emplace_back(std::move(current));
            current = BatchMetaData{};
            current_batch_size = 0;
        }
    }
    return result;
}

ErrorCode MetaIndexer::RecoverMetaData() noexcept {
    PropertyMap metadata_map;
    ErrorCode ec = backend_manager_->GetMetaData(metadata_map);
    if (ec == EC_NOENT) {
        KVCM_LOG_INFO("there is no metadata key in storage backend, no need to recover metadata");
        return ec;
    }
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("meta indexer read metadata from storage backend failed, ec[%d]", ec);
        return ec;
    }

    // METADATA_PROPERTY_KEY_COUNT *must* always be presented
    std::string key_count_str = metadata_map[METADATA_PROPERTY_KEY_COUNT];
    int64_t key_count;
    bool is_valid = StringUtil::StrToInt64(key_count_str.c_str(), key_count);
    if (!is_valid) {
        KVCM_LOG_ERROR("meta indexer convert metadata from string to int64 failed, key_count[%s]",
                       key_count_str.c_str());
        return EC_ERROR;
    }
    key_count_ = key_count;

    if (const auto it = metadata_map.find(METADATA_PROPERTY_STORAGE_USAGE_DATA); it != metadata_map.end()) {
        if (storage_usage_data_.Deserialize(it->second) != EC_OK) {
            KVCM_LOG_ERROR("meta indexer deserialize storage usage data failed, str: [%s]", it->second.c_str());
            return EC_ERROR;
        }
    }

    return EC_OK;
}

// 定时持久化key count等meta data，failover时可能因持久化不及时，key count与真实值会发生偏差
void MetaIndexer::PersistMetaData() noexcept {
    int64_t current_time = TimestampUtil::GetSteadyTimeMs();
    if (current_time >= last_persist_metadata_time_ + persist_metadata_interval_time_ms_) {
        std::map<std::string, std::string> metadata_map;
        metadata_map[METADATA_PROPERTY_KEY_COUNT] = std::to_string(key_count_);
        metadata_map[METADATA_PROPERTY_STORAGE_USAGE_DATA] = storage_usage_data_.Serialize();
        ErrorCode ec = backend_manager_->PutMetaData(metadata_map);
        if (ec != EC_OK) {
            KVCM_LOG_WARN("meta indexer persist metadata failed, ec[%d]", ec);
        }
        last_persist_metadata_time_ = current_time;
    }
}

void MetaIndexer::AdjustKeyCountMeta(const int32_t delta) noexcept {
    if (delta >= 0) {
        key_count_ += delta;
        return;
    }
    int64_t expected = key_count_;
    int64_t desired;
    do {
        desired = std::max(expected + delta, 0L);
    } while (!key_count_.compare_exchange_weak(expected, desired, std::memory_order_relaxed));
}

int32_t MetaIndexer::ProcessErrorCodes(const std::string &trace_id,
                                       const std::vector<ErrorCode> &error_codes,
                                       const std::vector<int32_t> &indexs,
                                       const KeyVector &keys,
                                       const std::string &op_name,
                                       Result &result) const noexcept {
    assert(indexs.size() == error_codes.size() || indexs.empty());
    int32_t error_count = 0;
    for (int32_t i = 0; i < error_codes.size(); ++i) {
        int32_t index = i;
        if (!indexs.empty()) {
            index = indexs[i];
        }
        if (error_codes[i] != EC_OK) {
            if (error_codes[i] != EC_NOENT) {
                PREFIX_INDEXER_LOG(
                    ERROR, "meta indexer %s failed, key[%lu] ec[%d]", op_name.c_str(), keys[index], error_codes[i]);
            }
            result.error_codes[index] = error_codes[i];
            ++error_count;
        }
    }
    return error_count;
}

void MetaIndexer::ProcessErrorResult(const std::string &trace_id,
                                     const std::string &op_name,
                                     const int32_t error_count,
                                     const int32_t key_count,
                                     Result &result) const noexcept {
    if (error_count == key_count) {
        result.ec = EC_ERROR;
        PREFIX_INDEXER_LOG(DEBUG, "all keys %s failed, key count[%d]", op_name.c_str(), key_count);
    } else if (error_count > 0) {
        result.ec = EC_PARTIAL_OK;
        PREFIX_INDEXER_LOG(
            DEBUG, "partial keys %s failed, key count[%d] failed count[%d]", op_name.c_str(), key_count, error_count);
    }
}

} // namespace kv_cache_manager
