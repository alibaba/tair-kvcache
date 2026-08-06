#include "kv_cache_manager/meta/meta_indexer.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <exception>
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
// Pure-local prefix scans group enough keys to amortize the 1024-shard LRU's
// lookup/release locks. This is independent from the smaller CPU-projection
// chunk configured on QueryExecutor. The first range remains bounded so a
// short prefix still cancels a million-key suffix promptly.
static constexpr size_t kLocalPrefixReadChunkSize = 4096;
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

void MetaIndexer::SetRevisitHistogram(std::shared_ptr<RevisitIntervalHistogram> histogram) {
    if (backend_manager_) {
        backend_manager_->SetRevisitHistogram(histogram);
    }
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
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, cache_backend_put_time_us, cache_backend_put_time_us);
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
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, cache_backend_delete_time_us, cache_backend_delete_time_us);
    ProcessErrorResult(trace_id, kDeleteMetaOperation, error_count, keys.size(), result);
    return result;
}

std::pair<int32_t, int32_t> MetaIndexer::ExecuteRmwUpsert(const std::string &trace_id,
                                                          RequestContext *request_context,
                                                          BatchMetaData &upsert_batch,
                                                          const std::vector<int32_t> &put_global_indexs,
                                                          const KeyVector &all_keys,
                                                          RmwStats &stats,
                                                          Result &result,
                                                          bool preserve_existing_updates_when_full) noexcept {
    if (upsert_batch.batch_keys.empty()) {
        return {0, 0};
    }

    stats.put_key_count += static_cast<int64_t>(put_global_indexs.size());
    stats.update_key_count += static_cast<int64_t>(upsert_batch.batch_keys.size() - put_global_indexs.size());

    std::vector<ErrorCode> upsert_ecs;
    BatchMetaData existing_update_batch;
    std::vector<size_t> existing_update_positions;
    BatchMetaData *backend_batch = &upsert_batch;
    const bool capacity_exceeded = put_global_indexs.size() + GetKeyCount() > max_key_count_;
    if (capacity_exceeded) {
        PREFIX_INDEXER_LOG(ERROR,
                           "ReadModifyWrite put keys count[%lu] + current key count[%lu] > max key count[%lu]",
                           put_global_indexs.size(),
                           GetKeyCount(),
                           max_key_count_);
        if (!preserve_existing_updates_when_full) {
            upsert_ecs.assign(upsert_batch.batch_keys.size(), EC_NOSPC);
        } else {
            // Fused targeted RMW can mix brand-new keys and updates to keys
            // that already count toward capacity. Reject only the former;
            // the old two-phase merge still admitted the latter at capacity.
            std::vector<bool> is_new_key(all_keys.size(), false);
            for (const int32_t global_index : put_global_indexs) {
                if (global_index >= 0 && static_cast<size_t>(global_index) < is_new_key.size()) {
                    is_new_key[global_index] = true;
                }
            }
            upsert_ecs.assign(upsert_batch.batch_keys.size(), EC_NOSPC);
            existing_update_positions.reserve(upsert_batch.batch_keys.size());
            for (size_t i = 0; i < upsert_batch.batch_keys.size(); ++i) {
                const int32_t global_index = upsert_batch.batch_indexs[i];
                if (global_index < 0 || static_cast<size_t>(global_index) >= is_new_key.size() ||
                    is_new_key[global_index]) {
                    continue;
                }
                existing_update_positions.push_back(i);
                existing_update_batch.batch_keys.push_back(upsert_batch.batch_keys[i]);
                existing_update_batch.batch_indexs.push_back(global_index);
                existing_update_batch.batch_locations.push_back(upsert_batch.batch_locations[i]);
                existing_update_batch.batch_properties.push_back(upsert_batch.batch_properties[i]);
            }
            backend_batch = &existing_update_batch;
        }
    }
    if (upsert_ecs.empty() || !existing_update_positions.empty()) {
        const int64_t begin = TimestampUtil::GetCurrentTimeUs();
        std::vector<ErrorCode> backend_ecs = backend_manager_->Upsert(request_context, *backend_batch);
        stats.upsert_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin;
        if (existing_update_positions.empty()) {
            upsert_ecs = std::move(backend_ecs);
        } else if (backend_ecs.size() != existing_update_positions.size()) {
            PREFIX_INDEXER_LOG(ERROR,
                               "ReadModifyWrite existing update results[%lu] mismatch keys[%lu]",
                               backend_ecs.size(),
                               existing_update_positions.size());
            for (const size_t original_position : existing_update_positions) {
                upsert_ecs[original_position] = EC_MISMATCH;
            }
        } else {
            for (size_t i = 0; i < backend_ecs.size(); ++i) {
                upsert_ecs[existing_update_positions[i]] = backend_ecs[i];
            }
        }
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
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector,
                                           meta_indexer,
                                           async_enqueue_timeout_key_count,
                                           stats.async_enqueue_timeout_key_count);
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
        if (get_ecs.size() != batch_keys.size() || batch_location_ids.size() != batch_keys.size()) {
            PREFIX_INDEXER_LOG(ERROR,
                               "ReadModifyWrite GetLocationIds result size mismatch, keys[%lu], ecs[%lu], ids[%lu]",
                               batch_keys.size(),
                               get_ecs.size(),
                               batch_location_ids.size());
            for (const int32_t global_idx : batch.batch_indexs) {
                result.error_codes[global_idx] = EC_MISMATCH;
            }
            error_count += static_cast<int32_t>(batch.batch_indexs.size());
            result.ec = EC_MISMATCH;
            continue;
        }

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
                                                                 const LocationModifierFunc &modifier,
                                                                 bool adjust_reclaimed_key_count,
                                                                 bool refresh_cache_from_persistent) noexcept {
    return ReadModifyWriteLocationImpl(request_context,
                                       keys,
                                       location_ids,
                                       modifier,
                                       adjust_reclaimed_key_count,
                                       false,
                                       refresh_cache_from_persistent);
}

MetaIndexer::LocationResult MetaIndexer::ReadModifyWriteTargetLocations(RequestContext *request_context,
                                                                        const KeyVector &keys,
                                                                        const LocationIdsPerKey &location_ids,
                                                                        const LocationModifierFunc &modifier) noexcept {
    return ReadModifyWriteLocationImpl(request_context, keys, location_ids, modifier, false, true, false);
}

MetaIndexer::LocationResult MetaIndexer::ReadModifyWriteLocationImpl(RequestContext *request_context,
                                                                     const KeyVector &keys,
                                                                     const LocationIdsPerKey &location_ids,
                                                                     const LocationModifierFunc &modifier,
                                                                     bool adjust_reclaimed_key_count,
                                                                     bool track_created_key_count,
                                                                     bool refresh_cache_from_persistent) noexcept {
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
    auto ephemeral_request_context = std::make_shared<RequestContext>(
        track_created_key_count ? "read_modify_write_target_locations" : "read_modify_write_location",
        ephemeral_metrics_collector);

    static CacheLocationMapVector empty_locations;
    static PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, location_ids, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    LocationResult location_result(location_ids);
    Result rmw_result(keys.size());
    // The aggregate result reports whether the RMW machinery completed for
    // each key. Per-location semantic outcomes (for example, a CAS mismatch
    // or a rejected task in an otherwise valid batch) remain in
    // per_location_error_codes and do not fail the whole operation. Track
    // structural/read/modifier/write failures separately so malformed backend
    // responses still fail closed without changing that established contract.
    std::vector<bool> key_level_failures(keys.size(), false);
    RmwStats stats;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs, &stats.lock_wait_time_us);

        // 1. One batched read for every (key, location_id) return deserialised CacheLocation
        const auto &batch_keys = batch.batch_keys;
        LocationsPerKey batch_locations_per_key;
        std::vector<ErrorCode> batch_key_get_ecs;
        const int64_t begin_get = TimestampUtil::GetCurrentTimeUs();
        std::vector<ErrorCode> refresh_results;
        if (refresh_cache_from_persistent) {
            // Maintenance candidates originate from the persistent scan. Under
            // the same shard lock as the CAS, replace a missing or stale hot
            // cache entry with the complete source-of-truth key first.
            refresh_results =
                backend_manager_->RefreshCacheFromPersistent(ephemeral_request_context.get(), batch_keys);
        }
        std::vector<std::vector<ErrorCode>> get_ecs_per_key;
        if (track_created_key_count) {
            get_ecs_per_key = backend_manager_->GetLocationsWithKeyStatus(ephemeral_request_context.get(),
                                                                          batch_keys,
                                                                          batch.batch_location_ids,
                                                                          batch_locations_per_key,
                                                                          batch_key_get_ecs);
        } else {
            get_ecs_per_key = backend_manager_->GetLocations(
                ephemeral_request_context.get(), batch_keys, batch.batch_location_ids, batch_locations_per_key);
        }
        stats.get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get;
        int64_t v = 0;
        auto *ephemeral_service_metrics_collector =
            dynamic_cast<ServiceMetricsCollector *>(ephemeral_request_context->metrics_collector());
        KVCM_METRICS_COLLECTOR_GET_METRICS(
            ephemeral_service_metrics_collector, meta_searcher, index_deserialize_time_us, v);
        stats.index_deserialize_time_us += v;
        stats.has_index_deserialize = true;

        if (get_ecs_per_key.size() != batch_keys.size() || batch_locations_per_key.size() != batch_keys.size() ||
            (track_created_key_count && batch_key_get_ecs.size() != batch_keys.size()) ||
            (refresh_cache_from_persistent && refresh_results.size() != batch_keys.size())) {
            PREFIX_INDEXER_LOG(ERROR,
                               "ReadModifyWriteLocation result size mismatch, keys[%lu], ecs[%lu], locations[%lu], "
                               "key_ecs[%lu]",
                               batch_keys.size(),
                               get_ecs_per_key.size(),
                               batch_locations_per_key.size(),
                               batch_key_get_ecs.size());
            for (const int32_t global_idx : batch.batch_indexs) {
                location_result.per_location_error_codes[global_idx].assign(location_ids[global_idx].size(),
                                                                            EC_MISMATCH);
                key_level_failures[global_idx] = true;
            }
            continue;
        }

        if (refresh_cache_from_persistent) {
            for (size_t i = 0; i < batch_keys.size(); ++i) {
                if (refresh_results[i] == EC_OK) {
                    continue;
                }
                get_ecs_per_key[i].assign(batch.batch_location_ids[i].size(), refresh_results[i]);
                batch_locations_per_key[i].assign(batch.batch_location_ids[i].size(), nullptr);
            }
        }

        // 2. Per-key modifier dispatch -> bucket each key into the upsert sub-batch or the delete sub-batch.
        BatchMetaData upsert_batch;
        BatchMetaData delete_batch;
        std::vector<int32_t> put_global_indexs;
        for (size_t i = 0; i < batch_keys.size(); ++i) {
            const int32_t global_idx = batch.batch_indexs[i];
            const KeyType key = batch_keys[i];

            std::vector<ErrorCode> &get_ecs = get_ecs_per_key[i];
            const LocationIdVector &loc_ids = batch.batch_location_ids[i];
            CacheLocationVector &loc_values = batch_locations_per_key[i];
            if (get_ecs.size() != loc_ids.size() || loc_values.size() != loc_ids.size()) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "ReadModifyWriteLocation per-key result size mismatch, key[%ld], ids[%lu], "
                                   "ecs[%lu], locations[%lu]",
                                   key,
                                   loc_ids.size(),
                                   get_ecs.size(),
                                   loc_values.size());
                location_result.per_location_error_codes[global_idx].assign(loc_ids.size(), EC_MISMATCH);
                key_level_failures[global_idx] = true;
                continue;
            }
            const ErrorCode key_get_ec = track_created_key_count ? batch_key_get_ecs[i] : EC_OK;
            if (key_get_ec != EC_OK && key_get_ec != EC_NOENT) {
                location_result.per_location_error_codes[global_idx].assign(loc_ids.size(), key_get_ec);
                key_level_failures[global_idx] = true;
                continue;
            }
            // EC_OK promises a usable value for the requested id. Treat a
            // null or mis-keyed value as corruption and never let a modifier
            // turn it into a write based on fabricated state.
            for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
                if (get_ecs[loc_index] == EC_OK &&
                    (!loc_values[loc_index] || loc_values[loc_index]->id() != loc_ids[loc_index])) {
                    PREFIX_INDEXER_LOG(ERROR,
                                       "ReadModifyWriteLocation invalid EC_OK value, key[%ld], requested id[%s]",
                                       key,
                                       loc_ids[loc_index].c_str());
                    get_ecs[loc_index] = EC_MISMATCH;
                    loc_values[loc_index].reset();
                }
                if (get_ecs[loc_index] != EC_OK && get_ecs[loc_index] != EC_NOENT) {
                    key_level_failures[global_idx] = true;
                }
            }
            if (track_created_key_count && key_get_ec == EC_NOENT &&
                std::any_of(get_ecs.begin(), get_ecs.end(), [](ErrorCode ec) { return ec == EC_OK; })) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "ReadModifyWriteTargetLocations key[%ld] reported missing with an existing "
                                   "target location",
                                   key);
                location_result.per_location_error_codes[global_idx].assign(loc_ids.size(), EC_MISMATCH);
                key_level_failures[global_idx] = true;
                continue;
            }
            PropertyMap upsert_property_map;
            auto [action, modifier_ecs] =
                modifier(get_ecs, loc_ids, static_cast<size_t>(global_idx), loc_values, upsert_property_map);
            if (modifier_ecs.size() != loc_ids.size()) {
                modifier_ecs.assign(loc_ids.size(), EC_ERROR);
                action = MA_FAIL;
                key_level_failures[global_idx] = true;
            }
            // A read error other than NOENT is not a valid basis for an RMW.
            // Force it back into the corresponding result slot even if a
            // buggy modifier accidentally returned EC_OK.
            for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
                if (get_ecs[loc_index] != EC_OK && get_ecs[loc_index] != EC_NOENT) {
                    modifier_ecs[loc_index] = get_ecs[loc_index];
                }
            }
            if (action == MA_FAIL &&
                std::all_of(modifier_ecs.begin(), modifier_ecs.end(), [](ErrorCode ec) { return ec == EC_OK; })) {
                modifier_ecs.assign(loc_ids.size(), EC_ERROR);
                key_level_failures[global_idx] = true;
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
                    if (!working_loc || loc_id != working_loc->id()) {
                        location_result.per_location_error_codes[global_idx][loc_index] = EC_MISMATCH;
                        key_level_failures[global_idx] = true;
                        continue;
                    }
                    upsert_loc_map.emplace(loc_id, working_loc);
                }
                if (!upsert_loc_map.empty() || !upsert_property_map.empty()) {
                    upsert_batch.batch_keys.emplace_back(key);
                    upsert_batch.batch_indexs.emplace_back(global_idx);
                    upsert_batch.batch_locations.emplace_back(std::move(upsert_loc_map));
                    upsert_batch.batch_properties.emplace_back(std::move(upsert_property_map));
                    if (track_created_key_count && key_get_ec == EC_NOENT) {
                        put_global_indexs.emplace_back(global_idx);
                    }
                }
            } else if (action == MA_DELETE) {
                LocationIdVector alive_ids;
                for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
                    if (modifier_ecs[loc_index] != EC_OK) {
                        location_result.per_location_error_codes[global_idx][loc_index] = modifier_ecs[loc_index];
                        continue;
                    }
                    alive_ids.emplace_back(loc_ids[loc_index]);
                }
                if (!alive_ids.empty()) {
                    delete_batch.batch_keys.emplace_back(key);
                    delete_batch.batch_indexs.emplace_back(global_idx);
                    delete_batch.batch_location_ids.emplace_back(std::move(alive_ids));
                }
            } else {
                // MA_FAIL / MA_SKIP / unknown: surface modifier_ec if any.
                if (action == MA_FAIL) {
                    key_level_failures[global_idx] = true;
                } else if (action != MA_SKIP) {
                    modifier_ecs.assign(loc_ids.size(), EC_ERROR);
                    key_level_failures[global_idx] = true;
                }
                location_result.per_location_error_codes[global_idx] = std::move(modifier_ecs);
            }
        }

        // 3. Dispatch upsert and delete sub-batches.
        const auto [upsert_errs, put_success_count] = ExecuteRmwUpsert(trace_id,
                                                                       ephemeral_request_context.get(),
                                                                       upsert_batch,
                                                                       put_global_indexs,
                                                                       keys,
                                                                       stats,
                                                                       rmw_result,
                                                                       track_created_key_count);
        (void)upsert_errs;
        for (const auto &global_index : upsert_batch.batch_indexs) {
            if (rmw_result.error_codes[global_index] != EC_OK) {
                key_level_failures[global_index] = true;
            }
            for (auto &location_ec : location_result.per_location_error_codes[global_index]) {
                if (location_ec == EC_OK) {
                    location_ec = rmw_result.error_codes[global_index];
                }
            }
        }
        const auto [delete_errs, delete_success_count] =
            ExecuteRmwDelete(trace_id, ephemeral_request_context.get(), delete_batch, keys, stats, rmw_result);
        (void)delete_errs;
        for (const auto &global_index : delete_batch.batch_indexs) {
            if (rmw_result.error_codes[global_index] != EC_OK) {
                key_level_failures[global_index] = true;
            }
            for (auto &location_ec : location_result.per_location_error_codes[global_index]) {
                if (location_ec == EC_OK) {
                    location_ec = rmw_result.error_codes[global_index];
                }
            }
        }
        AdjustKeyCountMeta(put_success_count - (adjust_reclaimed_key_count ? delete_success_count : 0));
    }

    EmitRmwMetrics(request_context->metrics_collector(), stats, keys.size());
    const size_t failed_key_count =
        static_cast<size_t>(std::count(key_level_failures.begin(), key_level_failures.end(), true));
    if (failed_key_count == keys.size()) {
        location_result.ec = EC_ERROR;
        PREFIX_INDEXER_LOG(DEBUG, "all locations rmw failed, error count[%lu]", failed_key_count);
    } else if (failed_key_count > 0) {
        location_result.ec = EC_PARTIAL_OK;
        PREFIX_INDEXER_LOG(
            DEBUG, "partial locations rmw failed, keys count[%lu] failed count[%lu]", keys.size(), failed_key_count);
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
    if (error_codes.size() != keys.size() || out_location_maps.size() != keys.size()) {
        PREFIX_INDEXER_LOG(ERROR,
                           "GetLocations result size mismatch, keys[%lu], ecs[%lu], locations[%lu]",
                           keys.size(),
                           error_codes.size(),
                           out_location_maps.size());
        error_codes.assign(keys.size(), EC_MISMATCH);
        out_location_maps.assign(keys.size(), CacheLocationMap{});
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);

    Result result(keys.size());
    int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kGetMetaOperation, result);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::GetLocationsFromPersistent(RequestContext *request_context,
                                                            const KeyVector &keys,
                                                            CacheLocationMapVector &out_location_maps) noexcept {
    if (keys.empty()) {
        out_location_maps.clear();
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();

    const int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->GetLocationsFromPersistent(request_context, keys, out_location_maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);

    Result result(keys.size());
    const int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kGetMetaOperation, result);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::GetLocationValues(RequestContext *request_context,
                                                   const KeyVector &keys,
                                                   LocationsPerKey &out_locations) noexcept {
    if (keys.empty()) {
        out_locations.clear();
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();

    const int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    std::vector<ErrorCode> error_codes;
    const bool use_parallel_local_read = query_executor_ && query_executor_->worker_count() > 1 &&
                                         keys.size() >= query_executor_->parallel_threshold() &&
                                         backend_manager_->SupportsConcurrentLocationValueReads();
    if (use_parallel_local_read) {
        error_codes.assign(keys.size(), EC_ERROR);
        out_locations.clear();
        out_locations.resize(keys.size());
        const bool completed = query_executor_->ParallelFor(
            keys.size(), [this, &keys, &error_codes, &out_locations](std::size_t begin, std::size_t end) {
                KeyVector chunk_keys(keys.begin() + begin, keys.begin() + end);
                LocationsPerKey chunk_locations;
                auto chunk_errors = backend_manager_->GetLocationValues(nullptr, chunk_keys, chunk_locations);
                const std::size_t expected = end - begin;
                if (chunk_errors.size() != expected || chunk_locations.size() != expected) {
                    KVCM_LOG_ERROR(
                        "parallel local location read size mismatch: errors[%zu] locations[%zu] expected[%zu]",
                        chunk_errors.size(),
                        chunk_locations.size(),
                        expected);
                    return;
                }
                for (std::size_t i = 0; i < expected; ++i) {
                    error_codes[begin + i] = chunk_errors[i];
                    out_locations[begin + i] = std::move(chunk_locations[i]);
                }
            });
        if (!completed) {
            KVCM_LOG_ERROR("trace_id[%s] instance[%s] | parallel local location read callback failed",
                           trace_id.c_str(),
                           instance_id_.c_str());
        }
    } else {
        error_codes = backend_manager_->GetLocationValues(request_context, keys, out_locations);
    }
    if (error_codes.size() != keys.size() || out_locations.size() != keys.size()) {
        KVCM_LOG_ERROR("trace_id[%s] instance[%s] | location value result size mismatch: errors[%zu] "
                       "locations[%zu] keys[%zu]",
                       trace_id.c_str(),
                       instance_id_.c_str(),
                       error_codes.size(),
                       out_locations.size(),
                       keys.size());
        // Errors and values are one positional response. If either outer
        // shape is malformed, no apparent in-range EC_OK/value pair can be
        // trusted to refer to the requested key.
        error_codes.assign(keys.size(), EC_MISMATCH);
        out_locations.assign(keys.size(), CacheLocationVector{});
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);

    Result result(keys.size());
    int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kGetMetaOperation, result);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::PrefixLocationResult MetaIndexer::VisitLocationValuesForPrefix(
    RequestContext *request_context, const KeyVector &keys, const PrefixLocationVisitor &visitor) noexcept {
    PrefixLocationResult prefix_result;
    if (keys.empty()) {
        return prefix_result;
    }

    // Preserve every non-local backend's existing batching/recovery behavior.
    // Only the pure-local path below performs progressive concurrent reads.
    if (!backend_manager_->SupportsConcurrentLocationValueReads()) {
        LocationsPerKey locations;
        auto result = GetLocationValues(request_context, keys, locations);
        prefix_result.read_key_count = keys.size();
        if (result.error_codes.size() != keys.size() || locations.size() != keys.size()) {
            prefix_result.terminal_ec = EC_MISMATCH;
            return prefix_result;
        }

        size_t metadata_stop = 0;
        while (metadata_stop < keys.size() && result.error_codes[metadata_stop] == EC_OK) {
            ++metadata_stop;
        }
        size_t location_count = 0;
        for (size_t i = 0; i < metadata_stop; ++i) {
            location_count += locations[i].size();
        }
        CompactLocationsPerKey compact;
        compact.Clear(metadata_stop, location_count);
        for (size_t i = 0; i < metadata_stop; ++i) {
            compact.values.insert(compact.values.end(), locations[i].begin(), locations[i].end());
            compact.FinishKey();
        }

        size_t requested_stop = keys.size();
        if (metadata_stop != 0 && visitor) {
            try {
                requested_stop = std::min(keys.size(), visitor(0, compact, metadata_stop));
            } catch (const std::exception &e) {
                KVCM_LOG_ERROR("location prefix visitor failed: %s", e.what());
                prefix_result.terminal_ec = EC_ERROR;
                return prefix_result;
            } catch (...) {
                KVCM_LOG_ERROR("location prefix visitor failed with unknown exception");
                prefix_result.terminal_ec = EC_ERROR;
                return prefix_result;
            }
        }
        prefix_result.valid_key_count = std::min(metadata_stop, requested_stop);
        prefix_result.stopped_by_visitor = requested_stop <= metadata_stop && requested_stop < keys.size();
        if (metadata_stop < requested_stop && metadata_stop < keys.size()) {
            prefix_result.terminal_ec = result.error_codes[metadata_stop];
        }
        return prefix_result;
    }

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();
    const int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    const size_t chunk_size =
        std::max(kLocalPrefixReadChunkSize, query_executor_ ? query_executor_->chunk_size() : size_t{256});
    const size_t chunk_count = 1 + (keys.size() - 1) / chunk_size;

    struct ChunkTerminal {
        size_t index = 0;
        ErrorCode ec = EC_OK;
        bool present = false;
    };
    std::vector<ChunkTerminal> terminals(chunk_count);
    std::atomic<size_t> metadata_stop(keys.size());
    std::atomic<size_t> visitor_stop(keys.size());
    std::atomic<size_t> read_key_count(0);

    auto reduce_stop_index = [](std::atomic<size_t> &target, size_t candidate) {
        size_t current = target.load(std::memory_order_relaxed);
        while (candidate < current && !target.compare_exchange_weak(
                                          current, candidate, std::memory_order_release, std::memory_order_relaxed)) {}
    };
    auto current_stop = [&metadata_stop, &visitor_stop]() {
        return std::min(metadata_stop.load(std::memory_order_acquire), visitor_stop.load(std::memory_order_acquire));
    };
    auto read_one_chunk = [this,
                           &keys,
                           &visitor,
                           &terminals,
                           &metadata_stop,
                           &visitor_stop,
                           &read_key_count,
                           &reduce_stop_index,
                           &current_stop,
                           chunk_size](size_t chunk_begin, size_t count) {
        if (chunk_begin >= current_stop()) {
            return;
        }

        CompactLocationsPerKey locations;
        auto errors = backend_manager_->GetLocationValuesCompact(nullptr, keys.data() + chunk_begin, count, locations);
        read_key_count.fetch_add(count, std::memory_order_relaxed);

        size_t successful_count = 0;
        ErrorCode terminal_ec = EC_OK;
        if (errors.size() != count || !locations.IsValid(count)) {
            KVCM_LOG_ERROR("compact local location read size mismatch: errors[%zu] locations[%zu] expected[%zu]",
                           errors.size(),
                           locations.size(),
                           count);
            terminal_ec = EC_MISMATCH;
        } else {
            while (successful_count < count && errors[successful_count] == EC_OK) {
                ++successful_count;
            }
            if (successful_count < count) {
                terminal_ec = errors[successful_count];
            }
        }

        if (successful_count != 0 && visitor) {
            const size_t requested_stop = std::min(keys.size(), visitor(chunk_begin, locations, successful_count));
            reduce_stop_index(visitor_stop, requested_stop);
        }
        if (terminal_ec != EC_OK) {
            auto &terminal = terminals[chunk_begin / chunk_size];
            terminal.index = chunk_begin + successful_count;
            terminal.ec = terminal_ec;
            terminal.present = true;
            reduce_stop_index(metadata_stop, terminal.index);
        }
    };
    auto read_ranges = [&read_one_chunk, &current_stop, &keys, chunk_size](size_t begin, size_t end) {
        for (size_t chunk_begin = begin; chunk_begin < end; chunk_begin += chunk_size) {
            if (chunk_begin >= current_stop()) {
                return;
            }
            const size_t chunk_end = std::min(end, std::min(keys.size(), chunk_begin + chunk_size));
            read_one_chunk(chunk_begin, chunk_end - chunk_begin);
        }
    };

    bool completed = true;
    const size_t first_chunk_size = std::min(chunk_size, keys.size());
    try {
        // Candidate hosts are derived from key zero. Visiting this chunk before
        // scheduling the suffix makes every later callback independent and
        // allows an early host miss to cancel all suffix metadata reads.
        read_one_chunk(0, first_chunk_size);
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("first compact local location read failed: %s", e.what());
        completed = false;
    } catch (...) {
        KVCM_LOG_ERROR("first compact local location read failed with unknown exception");
        completed = false;
    }

    if (completed && first_chunk_size < keys.size() && first_chunk_size < current_stop()) {
        const size_t remaining_count = keys.size() - first_chunk_size;
        auto read_remaining_ranges = [&read_ranges, first_chunk_size](size_t begin, size_t end) {
            read_ranges(first_chunk_size + begin, first_chunk_size + end);
        };
        if (query_executor_) {
            completed = query_executor_->ParallelForWithChunkSize(remaining_count, chunk_size, read_remaining_ranges);
        } else {
            try {
                read_remaining_ranges(0, remaining_count);
            } catch (const std::exception &e) {
                KVCM_LOG_ERROR("serial compact local location read failed: %s", e.what());
                completed = false;
            } catch (...) {
                KVCM_LOG_ERROR("serial compact local location read failed with unknown exception");
                completed = false;
            }
        }
    }

    prefix_result.read_key_count = read_key_count.load(std::memory_order_relaxed);
    if (!completed) {
        prefix_result.terminal_ec = EC_ERROR;
    } else {
        const size_t first_metadata_error = metadata_stop.load(std::memory_order_acquire);
        const size_t first_visitor_stop = visitor_stop.load(std::memory_order_acquire);
        prefix_result.valid_key_count = std::min(first_metadata_error, first_visitor_stop);
        prefix_result.stopped_by_visitor =
            first_visitor_stop <= first_metadata_error && first_visitor_stop < keys.size();
        if (first_metadata_error < first_visitor_stop && first_metadata_error < keys.size()) {
            const auto &terminal = terminals[first_metadata_error / chunk_size];
            if (!terminal.present || terminal.index != first_metadata_error) {
                prefix_result.terminal_ec = EC_MISMATCH;
                prefix_result.valid_key_count = 0;
            } else {
                prefix_result.terminal_ec = terminal.ec;
                if (prefix_result.terminal_ec != EC_NOENT) {
                    PREFIX_INDEXER_LOG(ERROR,
                                       "meta indexer prefix get failed, key[%lu] ec[%d]",
                                       keys[first_metadata_error],
                                       prefix_result.terminal_ec);
                }
            }
        }
    }

    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    return prefix_result;
}

MetaIndexer::LocationResult MetaIndexer::GetLocations(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      const LocationIdsPerKey &location_ids,
                                                      LocationsPerKey &out_locations) noexcept {
    if (keys.size() != location_ids.size()) {
        out_locations.clear();
        KVCM_LOG_ERROR("instance[%s] | GetLocations keys size[%lu] != location_ids size[%lu]",
                       instance_id_.c_str(),
                       keys.size(),
                       location_ids.size());
        return LocationResult(EC_BADARGS);
    }
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
    if (per_location_ecs.size() != keys.size() || out_locations.size() != keys.size()) {
        PREFIX_INDEXER_LOG(ERROR,
                           "GetLocations result size mismatch, keys[%lu], ecs[%lu], locations[%lu]",
                           keys.size(),
                           per_location_ecs.size(),
                           out_locations.size());
        out_locations.assign(keys.size(), CacheLocationVector{});
        for (size_t i = 0; i < keys.size(); ++i) {
            out_locations[i].resize(location_ids[i].size());
            result.per_location_error_codes[i].assign(location_ids[i].size(), EC_MISMATCH);
        }
    } else {
        for (size_t i = 0; i < keys.size(); ++i) {
            const size_t expected = location_ids[i].size();
            if (per_location_ecs[i].size() != expected || out_locations[i].size() != expected) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "GetLocations per-key result size mismatch, key[%ld], ids[%lu], ecs[%lu], "
                                   "locations[%lu]",
                                   keys[i],
                                   expected,
                                   per_location_ecs[i].size(),
                                   out_locations[i].size());
                out_locations[i].assign(expected, CacheLocationConstPtr{});
                result.per_location_error_codes[i].assign(expected, EC_MISMATCH);
                continue;
            }
            result.per_location_error_codes[i] = std::move(per_location_ecs[i]);
            for (size_t j = 0; j < expected; ++j) {
                if (result.per_location_error_codes[i][j] == EC_OK &&
                    (!out_locations[i][j] || out_locations[i][j]->id() != location_ids[i][j])) {
                    PREFIX_INDEXER_LOG(ERROR,
                                       "GetLocations invalid EC_OK value, key[%ld], requested id[%s]",
                                       keys[i],
                                       location_ids[i][j].c_str());
                    out_locations[i][j].reset();
                    result.per_location_error_codes[i][j] = EC_MISMATCH;
                }
            }
        }
    }

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

bool MetaIndexer::ParallelForQuery(std::size_t count, const QueryExecutor::RangeFunction &fn) const noexcept {
    if (query_executor_) {
        return query_executor_->ParallelFor(count, fn);
    }
    if (count == 0) {
        return true;
    }
    try {
        fn(0, count);
        return true;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("serial query callback threw exception: %s", e.what());
    } catch (...) { KVCM_LOG_ERROR("serial query callback threw unknown exception"); }
    return false;
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

ErrorCode MetaIndexer::ScanLocationsForMaintenance(RequestContext *request_context,
                                                   const std::string &cursor,
                                                   const size_t limit,
                                                   MaintenanceScanBatch &out) noexcept {
    out.Clear();
    if (limit == 0 || limit > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        KVCM_LOG_ERROR("instance[%s] maintenance scan invalid limit[%zu], cursor[%s]",
                       instance_id_.c_str(),
                       limit,
                       cursor.c_str());
        return EC_BADARGS;
    }
    ErrorCode ec =
        backend_manager_->ScanLocationsForMaintenance(request_context, cursor, static_cast<int64_t>(limit), out);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("instance[%s] maintenance scan failed, cursor[%s] limit[%zu] ec[%d]",
                       instance_id_.c_str(),
                       cursor.c_str(),
                       limit,
                       ec);
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
    const size_t expected_count = indexs.empty() ? keys.size() : indexs.size();
    if (error_codes.size() != expected_count) {
        PREFIX_INDEXER_LOG(ERROR,
                           "meta indexer %s result size mismatch, expect[%lu], actual[%lu]",
                           op_name.c_str(),
                           expected_count,
                           error_codes.size());
        int32_t mismatch_count = 0;
        for (size_t i = 0; i < expected_count; ++i) {
            const int32_t index = indexs.empty() ? static_cast<int32_t>(i) : indexs[i];
            if (index < 0 || static_cast<size_t>(index) >= result.error_codes.size()) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "meta indexer %s result index out of range, index[%d], result size[%lu]",
                                   op_name.c_str(),
                                   index,
                                   result.error_codes.size());
                continue;
            }
            result.error_codes[index] = EC_MISMATCH;
            ++mismatch_count;
        }
        result.ec = EC_MISMATCH;
        return mismatch_count;
    }

    int32_t error_count = 0;
    for (size_t i = 0; i < error_codes.size(); ++i) {
        int32_t index = static_cast<int32_t>(i);
        if (!indexs.empty()) {
            index = indexs[i];
        }
        if (index < 0 || static_cast<size_t>(index) >= result.error_codes.size() ||
            static_cast<size_t>(index) >= keys.size()) {
            PREFIX_INDEXER_LOG(ERROR,
                               "meta indexer %s result index out of range, index[%d], keys[%lu], result[%lu]",
                               op_name.c_str(),
                               index,
                               keys.size(),
                               result.error_codes.size());
            result.ec = EC_MISMATCH;
            ++error_count;
            continue;
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
    if (result.ec != EC_OK) {
        return;
    }
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
