#include "kv_cache_manager/meta/meta_indexer.h"

#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/common.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/simple/simple_kv_cache_backend.h"
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
// Pure-local prefix scans first inspect a bounded window so a short prefix can
// cancel a million-key suffix promptly. Once that probe succeeds, larger
// ranges amortize the 1024-shard LRU's lookup/release locks while preserving
// enough independent work for the query executor.
static constexpr size_t kLocalPrefixProbeKeyCount = 4096;
static constexpr size_t kLocalPrefixParallelReadChunkSize = 16384;
static constexpr size_t kPrefixStateWordBits = 64;
static_assert(kLocalPrefixProbeKeyCount % kPrefixStateWordBits == 0);
static_assert(kLocalPrefixParallelReadChunkSize % kPrefixStateWordBits == 0);

void FillBatchKeys(const std::vector<int32_t> &global_indices, const KeyVector &all_keys, KeyVector &batch_keys) {
    batch_keys.clear();
    batch_keys.reserve(global_indices.size());
    for (const int32_t global_index : global_indices) {
        batch_keys.push_back(all_keys[global_index]);
    }
}

struct PrefixLocationScratch {
    bool in_use = false;
    CompactLocationsPerKey locations;
};

thread_local PrefixLocationScratch tls_prefix_location_scratch;

class PrefixLocationScratchLease {
public:
    explicit PrefixLocationScratchLease(size_t key_count) {
        if (key_count <= kLocalPrefixParallelReadChunkSize && !tls_prefix_location_scratch.in_use) {
            scratch_ = &tls_prefix_location_scratch;
            scratch_->in_use = true;
            uses_thread_local_ = true;
        } else {
            local_.emplace();
            scratch_ = &*local_;
        }
    }

    ~PrefixLocationScratchLease() {
        if (uses_thread_local_) {
            // Retain only allocation capacity. Holding shared_ptr values until
            // this worker's next query would pin replaced CacheLocations and
            // make the scratch cache an accidental object cache.
            scratch_->locations.Clear();
            scratch_->in_use = false;
        }
    }

    CompactLocationsPerKey &locations() noexcept { return scratch_->locations; }

private:
    std::optional<PrefixLocationScratch> local_;
    PrefixLocationScratch *scratch_ = nullptr;
    bool uses_thread_local_ = false;
};
} // namespace

class MetaIndexer::ScopedBatchLock {
public:
    // Optional outputs accumulate time spent waiting for and holding all
    // shard mutexes in this batch.
    ScopedBatchLock(MetaIndexer &indexer,
                    const std::vector<int32_t> &shard_indexs,
                    int64_t *out_lock_wait_time_us,
                    int64_t *out_lock_hold_time_us)
        : indexer_(indexer), shard_indexs_(shard_indexs), out_lock_hold_time_us_(out_lock_hold_time_us) {
        const int64_t begin = TimestampUtil::GetCurrentTimeUs();
        for (const int32_t shardIdx : shard_indexs_) {
            indexer_.mutex_shards_[shardIdx]->lock();
        }
        lock_acquired_time_us_ = TimestampUtil::GetCurrentTimeUs();
        if (out_lock_wait_time_us != nullptr) {
            *out_lock_wait_time_us += lock_acquired_time_us_ - begin;
        }
    }
    ~ScopedBatchLock() {
        if (out_lock_hold_time_us_ != nullptr) {
            *out_lock_hold_time_us_ += std::max<int64_t>(0, TimestampUtil::GetCurrentTimeUs() - lock_acquired_time_us_);
        }
        for (const int32_t shardIdx : shard_indexs_) {
            indexer_.mutex_shards_[shardIdx]->unlock();
        }
    }

    ScopedBatchLock(const ScopedBatchLock &) = delete;
    ScopedBatchLock &operator=(const ScopedBatchLock &) = delete;

private:
    MetaIndexer &indexer_;
    const std::vector<int32_t> &shard_indexs_;
    int64_t *out_lock_hold_time_us_ = nullptr;
    int64_t lock_acquired_time_us_ = 0;
};

MetaIndexer::MetaIndexer(std::shared_ptr<const InstanceInfo> instance_info) : instance_info_(std::move(instance_info)) {
    storage_usage_data_.Reset();
}

bool MetaIndexer::SupportsWriteSession() const noexcept {
    return !instance_info_ || !instance_info_->disable_write_session();
}

MetaIndexer::~MetaIndexer() {
    // try to persist metadata when quit gracefully
    if (backend_manager_) {
        if (SupportsWriteSession()) {
            PersistMetaData();
        } else {
            const ErrorCode ec = SaveMetaData();
            if (ec != EC_OK) {
                KVCM_LOG_WARN("instance[%s] shutdown metadata save failed, ec[%d]", instance_id_.c_str(), ec);
            }
            if (!backend_manager_->SyncAll()) {
                KVCM_LOG_WARN("instance[%s] shutdown sync failed", instance_id_.c_str());
            }
        }
    }
}

bool MetaIndexer::IsRecovering() const noexcept {
    return backend_manager_ && backend_manager_->GetRecoverState() == MetaStorageBackendManager::RecoverState::kRecover;
}

ErrorCode MetaIndexer::Retire() noexcept {
    if (!backend_manager_) {
        return EC_OK;
    }
    if (!backend_manager_->SyncAll()) {
        return EC_ERROR;
    }
    const ErrorCode ec = backend_manager_->DeleteMetaData();
    if (ec != EC_OK) {
        return ec;
    }
    backend_manager_.reset();
    return EC_OK;
}

ErrorCode MetaIndexer::Init(const std::string &instance_id, const std::shared_ptr<MetaIndexerConfig> &config) noexcept {
    if (!config || !config->GetMetaStorageBackendConfig()) {
        KVCM_LOG_ERROR("instance[%s] meta indexer init failed, config is invalid", instance_id.c_str());
        return EC_BADARGS;
    }
    if (instance_info_ && instance_info_->instance_id() != instance_id) {
        return EC_BADARGS;
    }
    std::unique_ptr<MetaCacheBaseBackend> local;
    if (!SupportsWriteSession()) {
        const std::vector<LocationSpecInfo> &specs = instance_info_->location_spec_infos();
        if (specs.size() != 1 || specs.front().name().empty()) {
            KVCM_LOG_ERROR("instance[%s] simple cache requires one named spec", instance_id.c_str());
            return EC_BADARGS;
        }
        const std::shared_ptr<MetaStorageBackendConfig> &backend_config = config->GetMetaStorageBackendConfig();
        const std::string &storage_type = backend_config->GetStorageType();
        if ((storage_type != META_LOCAL_BACKEND_TYPE_STR && storage_type != META_CACHED_BACKEND_TYPE_STR) ||
            backend_config->GetMemoryPrimary() || config->GetBatchKeySize() == 0) {
            KVCM_LOG_ERROR(
                "instance[%s] simple metadata requires local/cached, memory_primary=false and nonzero batch size",
                instance_id.c_str());
            return EC_CONFIG_ERROR;
        }
        const bool cached = storage_type == META_CACHED_BACKEND_TYPE_STR;
        const std::string &storage_uri = backend_config->GetStorageUri();
        StandardUri uri;
        if ((cached || !storage_uri.empty()) && (!uri.Parse(storage_uri) || !uri.Valid())) {
            return EC_BADARGS;
        }
        if (cached) {
            const std::string persistent_type = uri.GetParam("persistent_type");
            const std::string cache_type = uri.GetParam("cache_type");
            if ((!persistent_type.empty() && persistent_type != META_REDIS_BACKEND_TYPE_STR &&
                 persistent_type != META_ASYNC_REDIS_BACKEND_TYPE_STR) ||
                (!cache_type.empty() && cache_type != META_LOCAL_BACKEND_TYPE_STR)) {
                KVCM_LOG_ERROR("instance[%s] unsupported simple cached backend types", instance_id.c_str());
                return EC_CONFIG_ERROR;
            }
        }
        auto spec_name = std::make_shared<const std::string>(specs.front().name());
        local = std::make_unique<SimpleKVCacheBackend>(std::move(spec_name));
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
    const std::shared_ptr<MetaStorageBackendConfig> &storage_backend_config = config->GetMetaStorageBackendConfig();

    backend_manager_ = std::make_unique<MetaStorageBackendManager>();
    ErrorCode ec = backend_manager_->Init(instance_id_, storage_backend_config, std::move(local));
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("instance[%s] meta storage backend manager init failed, ec[%d]", instance_id_.c_str(), ec);
        backend_manager_.reset();
        return ec;
    }
    mutex_shard_hash_seed_ = kDefaultMetaShardHashSeed;
    uint32_t local_cache_hash_seed = 0;
    if (backend_manager_->GetPureLocalCacheHashSeed(local_cache_hash_seed)) {
        // Pure-memory RMW batches already hold one metadata mutex shard at a
        // time. Reusing the LRU's host-specific hash seed makes each such
        // batch touch only the corresponding subset of LRU shards, avoiding
        // thousands of redundant LRU lock/unlock cycles for large reports.
        mutex_shard_hash_seed_ = local_cache_hash_seed;
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
        // Failed initialization must not persist partial/default counters in
        // the destructor over the Redis recovery baseline.
        backend_manager_.reset();
        return ec;
    }
    KVCM_LOG_INFO("instance[%s] meta indexer init success, mutex shard num[%lu], mutex hash seed[%" PRIu64
                  "], max key count[%lu], "
                  "batch key size[%lu], key_count[%lu], persist_metadata_interval_time_ms[%zu], storage usage data[%s]",
                  instance_id_.c_str(),
                  mutex_shard_num,
                  mutex_shard_hash_seed_,
                  max_key_count_,
                  batch_key_size_,
                  key_count_.load(),
                  persist_metadata_interval_time_ms_,
                  storage_usage_data_.ToJsonString().c_str());
    return EC_OK;
}

ErrorCode MetaIndexer::ValidateSimpleLocation(const CacheLocationConstPtr &location, bool validate_uri) const noexcept {
    if (!location || location->status() != CLS_SERVING || location->spec_size() != 1 ||
        location->location_specs().size() != 1) {
        return EC_BADARGS;
    }
    std::string_view storage_type;
    std::string_view medium;
    std::string_view host;
    if (!SnapshotUriUtils::ParseEventReportLocationIdView(location->id(), storage_type, medium, host)) {
        return EC_BADARGS;
    }
    const DataStorageType expected_type = storage_type == "event_report_l1p5"
                                              ? DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5
                                              : DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2;
    const LocationSpec &spec = location->location_specs().front();
    if (location->type() != expected_type || !spec.name_handle() ||
        spec.name() != instance_info_->location_spec_infos().front().name()) {
        return EC_BADARGS;
    }
    if (spec.uri().size() > std::numeric_limits<uint32_t>::max()) {
        return EC_OUT_OF_RANGE;
    }
    if (validate_uri && !location->HasValidatedLocationSpecs()) {
        StandardUri uri;
        if (!uri.Parse(spec.uri()) || !uri.Valid()) {
            return EC_BADARGS;
        }
    }
    return EC_OK;
}

void MetaIndexer::ValidateSimpleReadResults(size_t key_count,
                                            CacheLocationMapVector &locations,
                                            std::vector<ErrorCode> &errors) const noexcept {
    if (SupportsWriteSession()) {
        return;
    }
    if (errors.size() != key_count || locations.size() != key_count) {
        errors.assign(key_count, EC_MISMATCH);
        locations.assign(key_count, CacheLocationMap{});
        return;
    }
    for (size_t i = 0; i < key_count; ++i) {
        if (errors[i] == EC_OK) {
            for (const CacheLocationMap::value_type &entry : locations[i]) {
                errors[i] = entry.second && entry.first == entry.second->id()
                                ? ValidateSimpleLocation(entry.second, false)
                                : EC_MISMATCH;
                if (errors[i] != EC_OK) {
                    break;
                }
            }
        }
        if (errors[i] != EC_OK) {
            locations[i].clear();
        }
    }
}

void MetaIndexer::ValidateSimpleReadResults(size_t key_count,
                                            LocationsPerKey &locations,
                                            std::vector<ErrorCode> &errors) const noexcept {
    if (SupportsWriteSession()) {
        return;
    }
    if (errors.size() != key_count || locations.size() != key_count) {
        errors.assign(key_count, EC_MISMATCH);
        locations.assign(key_count, CacheLocationVector{});
        return;
    }
    for (size_t i = 0; i < key_count; ++i) {
        if (errors[i] == EC_OK) {
            for (const CacheLocationConstPtr &location : locations[i]) {
                errors[i] = ValidateSimpleLocation(location, false);
                if (errors[i] != EC_OK) {
                    break;
                }
            }
        }
        if (errors[i] != EC_OK) {
            locations[i].clear();
        }
    }
}

void MetaIndexer::SetRevisitHistogram(std::shared_ptr<RevisitIntervalHistogram> histogram) {
    if (backend_manager_) {
        backend_manager_->SetRevisitHistogram(histogram);
    }
}

int32_t MetaIndexer::GetMutexShardIndex(KeyType key) const noexcept {
    return GetShardIndex(key, mutex_shard_mask_, mutex_shard_hash_seed_);
}

MetaIndexer::Result MetaIndexer::Put(RequestContext *request_context,
                                     const KeyVector &keys,
                                     CacheLocationMapVector &location_maps,
                                     PropertyMapVector &properties) noexcept {
    if (!SupportsWriteSession()) {
        return Result(EC_UNIMPLEMENTED);
    }
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

    std::vector<IndexBatch> batches = MakeBatches(keys);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    Result result(keys.size());
    int32_t error_count = 0;
    int64_t put_io_time_us = 0;
    int64_t lock_wait_time_us = 0;
    int64_t lock_hold_time_us = 0;
    int64_t cache_backend_put_time_us = 0;
    BatchMetaData backend_batch;
    for (const auto &batch : batches) {
        FillBatchKeys(batch.global_indices, keys, backend_batch.batch_keys);
        backend_batch.batch_locations.clear();
        backend_batch.batch_properties.clear();
        if (!location_maps.empty()) {
            backend_batch.batch_locations.reserve(batch.global_indices.size());
            for (const int32_t global_index : batch.global_indices) {
                backend_batch.batch_locations.push_back(std::move(location_maps[global_index]));
            }
        }
        if (!properties.empty()) {
            backend_batch.batch_properties.reserve(batch.global_indices.size());
            for (const int32_t global_index : batch.global_indices) {
                backend_batch.batch_properties.push_back(std::move(properties[global_index]));
            }
        }
        ScopedBatchLock lock(*this, batch.shard_indices, &lock_wait_time_us, &lock_hold_time_us);
        int64_t begin_put_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = backend_manager_->Put(request_context, backend_batch);
        put_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_put_io_time;
        int64_t v = 0;
        KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, cache_backend_put_time_us, v);
        cache_backend_put_time_us += v;
        error_count += ProcessErrorCodes(trace_id, error_codes, batch.global_indices, keys, kPutMetaOperation, result);
    }
    AdjustKeyCountMeta(keys.size() - error_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, put_io_time_us, put_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, lock_wait_time_us, lock_wait_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, lock_hold_time_us, lock_hold_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, cache_backend_put_time_us, cache_backend_put_time_us);
    ProcessErrorResult(trace_id, kPutMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::Delete(RequestContext *request_context, const KeyVector &keys) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    ServiceMetricsCollector *service_metrics_collector =
        dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const std::string &trace_id = request_context->trace_id();
    std::vector<IndexBatch> batches = MakeBatches(keys);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t lock_wait_time_us = 0;
    int64_t lock_hold_time_us = 0;
    int64_t cache_backend_delete_time_us = 0;
    KeyVector batch_keys;
    for (const IndexBatch &batch : batches) {
        FillBatchKeys(batch.global_indices, keys, batch_keys);
        ScopedBatchLock lock(*this, batch.shard_indices, &lock_wait_time_us, &lock_hold_time_us);
        std::vector<ErrorCode> error_codes = backend_manager_->Delete(request_context, batch_keys);
        int64_t v = 0;
        KVCM_METRICS_COLLECTOR_GET_METRICS(service_metrics_collector, meta_indexer, cache_backend_delete_time_us, v);
        cache_backend_delete_time_us += v;
        error_count +=
            ProcessErrorCodes(trace_id, error_codes, batch.global_indices, keys, kDeleteMetaOperation, result);
    }
    AdjustKeyCountMeta(error_count - keys.size());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, lock_wait_time_us, lock_wait_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, lock_hold_time_us, lock_hold_time_us);
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

    std::unordered_set<KeyType> unique_put_keys;
    unique_put_keys.reserve(put_global_indexs.size());
    for (const int32_t global_index : put_global_indexs) {
        if (global_index >= 0 && static_cast<size_t>(global_index) < all_keys.size()) {
            unique_put_keys.insert(all_keys[global_index]);
        }
    }
    const size_t unique_put_key_count = unique_put_keys.size();
    stats.put_key_count += static_cast<int64_t>(unique_put_key_count);
    stats.update_key_count += static_cast<int64_t>(upsert_batch.batch_keys.size() - unique_put_key_count);

    std::vector<ErrorCode> upsert_ecs;
    BatchMetaData existing_update_batch;
    std::vector<size_t> existing_update_positions;
    BatchMetaData *backend_batch = &upsert_batch;
    const bool capacity_exceeded = unique_put_key_count + GetKeyCount() > max_key_count_;
    if (capacity_exceeded) {
        PREFIX_INDEXER_LOG(ERROR,
                           "ReadModifyWrite put keys count[%lu] + current key count[%lu] > max key count[%lu]",
                           unique_put_key_count,
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
                const bool requires_secondary_admission =
                    std::binary_search(upsert_batch.batch_secondary_admission_indices.begin(),
                                       upsert_batch.batch_secondary_admission_indices.end(),
                                       i);
                const size_t filtered_index = existing_update_batch.batch_keys.size();
                existing_update_positions.push_back(i);
                existing_update_batch.batch_keys.push_back(upsert_batch.batch_keys[i]);
                existing_update_batch.batch_indexs.push_back(global_index);
                existing_update_batch.batch_locations.push_back(std::move(upsert_batch.batch_locations[i]));
                existing_update_batch.batch_properties.push_back(std::move(upsert_batch.batch_properties[i]));
                if (requires_secondary_admission) {
                    existing_update_batch.batch_secondary_admission_indices.push_back(filtered_index);
                }
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
    int32_t put_success_count = static_cast<int32_t>(unique_put_key_count);
    if (error_count != 0) {
        // Reuse the existing buckets to avoid another allocation on the
        // partial-failure path. The all-success path needs no second scan.
        unique_put_keys.clear();
        for (const int32_t global_index : put_global_indexs) {
            if (global_index >= 0 && static_cast<size_t>(global_index) < all_keys.size() &&
                result.error_codes[global_index] == EC_OK) {
                unique_put_keys.insert(all_keys[global_index]);
            }
        }
        put_success_count = static_cast<int32_t>(unique_put_keys.size());
    }
    return {error_count, put_success_count};
}

std::pair<int32_t, int32_t> MetaIndexer::ExecuteRmwDelete(const std::string &trace_id,
                                                          RequestContext *request_context,
                                                          const BatchMetaData &delete_batch,
                                                          const KeyVector &all_keys,
                                                          RmwStats &stats,
                                                          Result &result,
                                                          bool maintenance_no_touch,
                                                          bool maintenance_write_local_primary) noexcept {
    if (delete_batch.batch_keys.empty()) {
        return {0, 0};
    }
    stats.delete_key_count += static_cast<int64_t>(delete_batch.batch_keys.size());

    const int64_t begin = TimestampUtil::GetCurrentTimeUs();
    std::vector<ErrorCode> delete_ecs;
    int32_t reclaimed_count = 0;
    if (delete_batch.batch_location_ids.empty()) {
        delete_ecs = backend_manager_->Delete(request_context, delete_batch.batch_keys);
    } else if (maintenance_no_touch) {
        delete_ecs = backend_manager_->DeleteLocationsForMaintenance(request_context,
                                                                     delete_batch.batch_keys,
                                                                     delete_batch.batch_location_ids,
                                                                     reclaimed_count,
                                                                     maintenance_write_local_primary);
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
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, lock_hold_time_us, stats.lock_hold_time_us);

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
    if (!SupportsWriteSession()) {
        return Result(EC_UNIMPLEMENTED);
    }
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

    std::vector<IndexBatch> batches = MakeBatches(keys);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    Result result(keys.size());
    int32_t error_count = 0;
    RmwStats stats;
    KeyVector batch_keys;
    for (const auto &batch : batches) {
        FillBatchKeys(batch.global_indices, keys, batch_keys);
        ScopedBatchLock lock(*this, batch.shard_indices, &stats.lock_wait_time_us, &stats.lock_hold_time_us);

        // 1. Read each key's existing location id list (no value deserialization)
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
            for (const int32_t global_idx : batch.global_indices) {
                result.error_codes[global_idx] = EC_MISMATCH;
            }
            error_count += static_cast<int32_t>(batch.global_indices.size());
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
            const int32_t global_idx = batch.global_indices[i];
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
        const std::pair<int32_t, int32_t> upsert_result = ExecuteRmwUpsert(
            trace_id, ephemeral_request_context.get(), upsert_batch, put_global_indexs, keys, stats, result, false);
        const std::pair<int32_t, int32_t> delete_result = ExecuteRmwDelete(
            trace_id, ephemeral_request_context.get(), delete_batch, keys, stats, result, false, false);
        error_count += upsert_result.first + delete_result.first;
        AdjustKeyCountMeta(upsert_result.second - delete_result.second);
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
                                                                 bool fill_missing_cache) noexcept {
    return ReadModifyWriteLocationImpl(
        request_context, keys, location_ids, modifier, adjust_reclaimed_key_count, false, fill_missing_cache, false);
}

MetaIndexer::LocationResult
MetaIndexer::ReadModifyWriteLocationsForMaintenance(RequestContext *request_context,
                                                    const KeyVector &keys,
                                                    const LocationIdsPerKey &location_ids,
                                                    const LocationModifierFunc &modifier,
                                                    bool adjust_reclaimed_key_count) noexcept {
    return ReadModifyWriteLocationImpl(
        request_context, keys, location_ids, modifier, adjust_reclaimed_key_count, false, false, true);
}

MetaIndexer::LocationResult MetaIndexer::ReadModifyWriteTargetLocations(RequestContext *request_context,
                                                                        const KeyVector &keys,
                                                                        const LocationIdsPerKey &location_ids,
                                                                        const LocationModifierFunc &modifier) noexcept {
    return ReadModifyWriteLocationImpl(request_context, keys, location_ids, modifier, false, true, false, false);
}

MetaIndexer::LocationResult MetaIndexer::ReadModifyWriteLocationImpl(RequestContext *request_context,
                                                                     const KeyVector &keys,
                                                                     const LocationIdsPerKey &location_ids,
                                                                     const LocationModifierFunc &modifier,
                                                                     bool adjust_reclaimed_key_count,
                                                                     bool track_created_key_count,
                                                                     bool fill_missing_cache,
                                                                     bool maintenance_no_touch) noexcept {
    const std::string &trace_id = request_context->trace_id();
    const bool simple_mode = !SupportsWriteSession();
    const bool convert_deleting = simple_mode && !track_created_key_count && !maintenance_no_touch;
    // Ordinary simple writes fill a missing Recover key before validation.
    // Maintenance retains its explicit hot-only read policy.
    const bool fill_cache = fill_missing_cache || convert_deleting;
    if (keys.empty()) {
        return LocationResult(EC_OK);
    }
    if (simple_mode && (!modifier || keys.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))) {
        return LocationResult(!modifier ? EC_BADARGS : EC_OUT_OF_RANGE);
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

    std::vector<IndexBatch> batches = MakeBatches(keys);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    LocationResult location_result(location_ids);
    Result rmw_result(keys.size());
    Result delete_result(convert_deleting ? keys.size() : size_t{0});
    Result &deletion_results = convert_deleting ? delete_result : rmw_result;
    struct ConvertedDelete {
        int32_t key_index;
        size_t location_index;
        CacheLocationConstPtr location;
    };
    std::vector<ConvertedDelete> converted_deletes;
    // The aggregate result reports whether the RMW machinery completed for
    // each key. Per-location semantic outcomes (for example, a CAS mismatch
    // or a rejected task in an otherwise valid batch) remain in
    // per_location_error_codes and do not fail the whole operation. Track
    // structural/read/modifier/write failures separately so malformed backend
    // responses still fail closed without changing that established contract.
    std::vector<bool> key_level_failures(keys.size(), false);
    RmwStats stats;
    KeyVector batch_keys;
    LocationIdsPerKey batch_location_ids;
    for (const IndexBatch &batch : batches) {
        converted_deletes.clear();
        FillBatchKeys(batch.global_indices, keys, batch_keys);
        batch_location_ids.clear();
        batch_location_ids.reserve(batch.global_indices.size());
        for (const int32_t global_index : batch.global_indices) {
            batch_location_ids.push_back(location_ids[global_index]);
        }
        ScopedBatchLock lock(*this, batch.shard_indices, &stats.lock_wait_time_us, &stats.lock_hold_time_us);

        // 1. One batched read for every (key, location_id) return deserialised CacheLocation
        // Cached maintenance always validates hot data. Only its write route
        // is fixed for this batch, even if recovery completes before deletion.
        const bool maintenance_write_local_primary =
            maintenance_no_touch && backend_manager_->UseLocalPrimaryForMaintenance();
        if (maintenance_no_touch && !backend_manager_->SyncBeforeMaintenanceRead(batch_keys)) {
            // A single async metadata backend may still have an accepted
            // same-key mutation queued. Reading before that mutation reaches
            // the persistent view could authorize a stale expected-value
            // delete which is then queued behind, and removes, the newer
            // mutation. Drain accepted writes while the shard fence prevents
            // another same-key mutation from being admitted.
            PREFIX_INDEXER_LOG(WARN, "maintenance pre-read Sync failed for keys[%lu]", batch_keys.size());
            for (const int32_t global_idx : batch.global_indices) {
                location_result.per_location_error_codes[global_idx].assign(location_ids[global_idx].size(),
                                                                            EC_TIMEOUT);
                rmw_result.error_codes[global_idx] = EC_TIMEOUT;
                key_level_failures[global_idx] = true;
            }
            continue;
        }
        LocationsPerKey batch_locations_per_key;
        std::vector<ErrorCode> batch_key_get_ecs;
        const int64_t begin_get = TimestampUtil::GetCurrentTimeUs();
        std::vector<ErrorCode> fill_results;
        if (fill_cache) {
            // Ordinary GC may fill a missing Recover key under the same
            // shard lock as its CAS. Existing hot keys are never overwritten;
            // Running and tombstoned keys do not fall back to Redis.
            fill_results = backend_manager_->EnsureCacheForGc(ephemeral_request_context.get(), batch_keys);
        }
        std::vector<std::vector<ErrorCode>> get_ecs_per_key;
        if (track_created_key_count) {
            get_ecs_per_key = backend_manager_->GetLocationsWithKeyStatus(ephemeral_request_context.get(),
                                                                          batch_keys,
                                                                          batch_location_ids,
                                                                          batch_locations_per_key,
                                                                          batch_key_get_ecs);
        } else {
            get_ecs_per_key =
                (maintenance_no_touch || (fill_cache && backend_manager_->HasCacheBackend()))
                    ? backend_manager_->GetLocationsForMaintenance(
                          ephemeral_request_context.get(), batch_keys, batch_location_ids, batch_locations_per_key)
                    : backend_manager_->GetLocations(
                          ephemeral_request_context.get(), batch_keys, batch_location_ids, batch_locations_per_key);
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
            (fill_cache && fill_results.size() != batch_keys.size())) {
            PREFIX_INDEXER_LOG(ERROR,
                               "ReadModifyWriteLocation result size mismatch, keys[%lu], ecs[%lu], locations[%lu], "
                               "key_ecs[%lu]",
                               batch_keys.size(),
                               get_ecs_per_key.size(),
                               batch_locations_per_key.size(),
                               batch_key_get_ecs.size());
            for (const int32_t global_idx : batch.global_indices) {
                location_result.per_location_error_codes[global_idx].assign(location_ids[global_idx].size(),
                                                                            EC_MISMATCH);
                key_level_failures[global_idx] = true;
            }
            continue;
        }

        if (fill_cache) {
            for (size_t i = 0; i < batch_keys.size(); ++i) {
                if (fill_results[i] == EC_OK) {
                    continue;
                }
                get_ecs_per_key[i].assign(batch_location_ids[i].size(), fill_results[i]);
                batch_locations_per_key[i].assign(batch_location_ids[i].size(), nullptr);
            }
        }

        // 2. Per-key modifier dispatch -> bucket each key into the upsert sub-batch or the delete sub-batch.
        BatchMetaData upsert_batch;
        BatchMetaData delete_batch;
        std::vector<int32_t> put_global_indexs;
        std::unordered_map<KeyType, std::unordered_set<std::string_view>> requested_ids;
        std::unordered_map<KeyType, bool> invalid_keys;
        std::unordered_map<KeyType, size_t> upsert_by_key;
        std::unordered_map<KeyType, size_t> delete_by_key;
        std::vector<std::vector<int32_t>> upsert_positions;
        std::vector<std::vector<int32_t>> delete_positions;
        if (simple_mode) {
            for (size_t i = 0; i < batch_keys.size(); ++i) {
                std::unordered_set<std::string_view> &seen = requested_ids[batch_keys[i]];
                for (const LocationId &id : batch_location_ids[i]) {
                    if (id.empty() || !seen.insert(id).second) {
                        invalid_keys[batch_keys[i]] = true;
                    }
                }
            }
        }
        auto append_delete = [&delete_batch, &delete_by_key, &delete_positions, simple_mode](
                                 KeyType key, int32_t global_index, LocationIdVector ids) {
            if (ids.empty()) {
                return;
            }
            if (simple_mode) {
                const std::unordered_map<KeyType, size_t>::const_iterator existing = delete_by_key.find(key);
                if (existing != delete_by_key.end()) {
                    LocationIdVector &target = delete_batch.batch_location_ids[existing->second];
                    target.insert(target.end(), ids.begin(), ids.end());
                    delete_positions[existing->second].push_back(global_index);
                    return;
                }
                [[maybe_unused]] const bool inserted =
                    delete_by_key.emplace(key, delete_batch.batch_keys.size()).second;
                assert(inserted);
                delete_positions.push_back({global_index});
            }
            delete_batch.batch_keys.push_back(key);
            delete_batch.batch_indexs.push_back(global_index);
            delete_batch.batch_location_ids.push_back(std::move(ids));
        };
        for (size_t i = 0; i < batch_keys.size(); ++i) {
            const int32_t global_idx = batch.global_indices[i];
            const KeyType key = batch_keys[i];
            if (simple_mode && invalid_keys.count(key) != 0) {
                location_result.per_location_error_codes[global_idx].assign(location_ids[global_idx].size(),
                                                                            EC_BADARGS);
                key_level_failures[global_idx] = true;
                continue;
            }

            std::vector<ErrorCode> &get_ecs = get_ecs_per_key[i];
            const LocationIdVector &loc_ids = batch_location_ids[i];
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
                if (simple_mode && get_ecs[loc_index] == EC_OK) {
                    get_ecs[loc_index] = ValidateSimpleLocation(loc_values[loc_index], false);
                    if (get_ecs[loc_index] != EC_OK) {
                        loc_values[loc_index].reset();
                    }
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
            const CacheLocationVector previous_locations = convert_deleting ? loc_values : CacheLocationVector{};
            PropertyMap upsert_property_map;
            LocationModifierResult modified =
                modifier(get_ecs, loc_ids, static_cast<size_t>(global_idx), loc_values, upsert_property_map);
            ModifierAction action = modified.first;
            std::vector<ErrorCode> &modifier_ecs = modified.second;
            if (modifier_ecs.size() != loc_ids.size() || (simple_mode && loc_values.size() != loc_ids.size())) {
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
            if (simple_mode && !upsert_property_map.empty()) {
                modifier_ecs.assign(loc_ids.size(), EC_UNIMPLEMENTED);
                action = MA_FAIL;
                key_level_failures[global_idx] = true;
            }
            if (action == MA_FAIL &&
                std::all_of(modifier_ecs.begin(), modifier_ecs.end(), [](ErrorCode ec) { return ec == EC_OK; })) {
                modifier_ecs.assign(loc_ids.size(), EC_ERROR);
                key_level_failures[global_idx] = true;
            }
            if (action == MA_OK) {
                CacheLocationMap upsert_loc_map;
                LocationIdVector deleting_ids;
                bool requires_secondary_admission = false;
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
                    if (convert_deleting && working_loc->status() == CLS_DELETING) {
                        if (get_ecs[loc_index] == EC_OK) {
                            deleting_ids.push_back(loc_id);
                            converted_deletes.push_back({global_idx, loc_index, previous_locations[loc_index]});
                        } else {
                            location_result.per_location_error_codes[global_idx][loc_index] = EC_NOENT;
                        }
                        continue;
                    }
                    if (simple_mode) {
                        const ErrorCode ec = ValidateSimpleLocation(working_loc, true);
                        if (ec != EC_OK) {
                            location_result.per_location_error_codes[global_idx][loc_index] = ec;
                            key_level_failures[global_idx] = true;
                            continue;
                        }
                    }
                    requires_secondary_admission =
                        requires_secondary_admission || working_loc->status() == CacheLocationStatus::CLS_DELETING;
                    upsert_loc_map.emplace(loc_id, working_loc);
                }
                append_delete(key, global_idx, std::move(deleting_ids));
                if (!upsert_loc_map.empty() || !upsert_property_map.empty()) {
                    if (simple_mode) {
                        const std::unordered_map<KeyType, size_t>::const_iterator existing = upsert_by_key.find(key);
                        if (existing != upsert_by_key.end()) {
                            upsert_batch.batch_locations[existing->second].merge(upsert_loc_map);
                            assert(upsert_loc_map.empty());
                            upsert_positions[existing->second].push_back(global_idx);
                            continue;
                        }
                        [[maybe_unused]] const bool inserted =
                            upsert_by_key.emplace(key, upsert_batch.batch_keys.size()).second;
                        assert(inserted);
                        upsert_positions.push_back({global_idx});
                    }
                    const size_t upsert_index = upsert_batch.batch_keys.size();
                    upsert_batch.batch_keys.emplace_back(key);
                    upsert_batch.batch_indexs.emplace_back(global_idx);
                    upsert_batch.batch_locations.emplace_back(std::move(upsert_loc_map));
                    upsert_batch.batch_properties.emplace_back(std::move(upsert_property_map));
                    if (requires_secondary_admission) {
                        upsert_batch.batch_secondary_admission_indices.push_back(upsert_index);
                    }
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
                append_delete(key, global_idx, std::move(alive_ids));
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

        const bool mixed_writes = !converted_deletes.empty() && !upsert_batch.batch_keys.empty();
        auto propagate_errors = [&location_result, &location_ids, &key_level_failures, simple_mode, mixed_writes](
                                    const BatchMetaData &writes,
                                    const std::vector<std::vector<int32_t>> &positions,
                                    const Result &write_result) {
            for (size_t i = 0; i < writes.batch_indexs.size(); ++i) {
                const int32_t source = writes.batch_indexs[i];
                auto apply = [&location_result,
                              &location_ids,
                              &key_level_failures,
                              &writes,
                              &write_result,
                              mixed_writes,
                              i,
                              source](int32_t global) {
                    if (write_result.error_codes[source] != EC_OK) {
                        key_level_failures[global] = true;
                    }
                    for (size_t j = 0; j < location_ids[global].size(); ++j) {
                        const LocationId &id = location_ids[global][j];
                        if (mixed_writes &&
                            (writes.batch_location_ids.empty() ? writes.batch_locations[i].count(id) == 0
                                                               : std::find(writes.batch_location_ids[i].begin(),
                                                                           writes.batch_location_ids[i].end(),
                                                                           id) == writes.batch_location_ids[i].end())) {
                            continue;
                        }
                        ErrorCode &ec = location_result.per_location_error_codes[global][j];
                        if (ec == EC_OK) {
                            ec = write_result.error_codes[source];
                        }
                    }
                };
                if (simple_mode) {
                    for (const int32_t global : positions[i]) {
                        apply(global);
                    }
                } else {
                    apply(source);
                }
            }
        };

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
        propagate_errors(upsert_batch, upsert_positions, rmw_result);
        const bool maintenance_delete = maintenance_no_touch || !converted_deletes.empty();
        const std::pair<int32_t, int32_t> deleted = ExecuteRmwDelete(trace_id,
                                                                     ephemeral_request_context.get(),
                                                                     delete_batch,
                                                                     keys,
                                                                     stats,
                                                                     deletion_results,
                                                                     maintenance_delete,
                                                                     maintenance_write_local_primary);
        const int32_t delete_success_count = deleted.second;
        const bool maintenance_delete_synced = !maintenance_delete || delete_batch.batch_keys.empty() ||
                                               !backend_manager_->RequiresMaintenancePostDeleteSync() ||
                                               backend_manager_->Sync(delete_batch.batch_keys);
        if (!maintenance_delete_synced) {
            // Keep the per-key shard fence until the accepted maintenance
            // delete reaches its async backend queue barrier. A timeout is an
            // unknown persistence outcome, so surface a key-level hard result
            // and let a later GC round reconcile the authoritative metadata
            // again. Preserve the accepted per-location results and
            // accounting: ordinary async RMW uses enqueue acceptance as its
            // accounting boundary, and replacing those results with TIMEOUT
            // would permanently leak usage when the delete has already been
            // applied and is therefore absent from the next scan.
            PREFIX_INDEXER_LOG(
                WARN, "maintenance post-delete Sync failed for keys[%lu]", delete_batch.batch_keys.size());
            for (const int32_t global_idx : delete_batch.batch_indexs) {
                key_level_failures[global_idx] = true;
            }
            for (const std::vector<int32_t> &positions : delete_positions) {
                for (const int32_t global : positions) {
                    key_level_failures[global] = true;
                }
            }
        }
        propagate_errors(delete_batch, delete_positions, deletion_results);
        // Explicit deletes keep their caller's accounting contract.
        for (const ConvertedDelete &entry : converted_deletes) {
            if (location_result.per_location_error_codes[entry.key_index][entry.location_index] != EC_OK) {
                continue;
            }
            uint64_t bytes = 0;
            if (!entry.location->GetValidatedTotalSize(bytes)) {
                StandardUri uri;
                if (uri.Parse(entry.location->location_specs().front().uri()) && uri.Valid()) {
                    uri.GetParamAs("size", bytes);
                }
            }
            const uint64_t remaining = SubStorageUsageByType(entry.location->type(), bytes);
            PREFIX_INDEXER_LOG(DEBUG, "Delete storage usage remaining[%" PRIu64 "]", remaining);
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

bool MetaIndexer::SupportsSingleLocationRmw() const noexcept {
    return backend_manager_ && backend_manager_->SupportsSingleLocationRmw();
}

MetaIndexer::SingleLocationResult
MetaIndexer::ReadModifyWriteSingleTargetLocations(RequestContext *request_context,
                                                  const KeyVector &keys,
                                                  const LocationIdRefVector &location_ids,
                                                  const SingleLocationModifierFunc &modifier) noexcept {
    if (!SupportsWriteSession()) {
        return SingleLocationResult(EC_UNIMPLEMENTED);
    }
    const auto &trace_id = request_context->trace_id();
    if (keys.empty()) {
        return SingleLocationResult(EC_OK);
    }
    if (keys.size() != location_ids.size() || !modifier ||
        std::any_of(location_ids.begin(), location_ids.end(), [](const LocationId *id) {
            return id == nullptr || id->empty();
        })) {
        PREFIX_INDEXER_LOG(
            ERROR, "single target RMW invalid inputs, keys[%lu], location ids[%lu]", keys.size(), location_ids.size());
        return SingleLocationResult(EC_BADARGS);
    }
    bool has_duplicate_keys = false;
    if (std::is_sorted(keys.begin(), keys.end())) {
        has_duplicate_keys = std::adjacent_find(keys.begin(), keys.end()) != keys.end();
    } else {
        std::unordered_set<KeyType> seen_keys;
        seen_keys.reserve(keys.size());
        for (const KeyType key : keys) {
            if (!seen_keys.insert(key).second) {
                has_duplicate_keys = true;
                break;
            }
        }
    }
    if (has_duplicate_keys) {
        PREFIX_INDEXER_LOG(ERROR, "single target RMW requires unique keys");
        return SingleLocationResult(EC_BADARGS);
    }
    if (!SupportsSingleLocationRmw()) {
        PREFIX_INDEXER_LOG(ERROR, "single target RMW requires a pure local metadata backend");
        return SingleLocationResult(EC_UNIMPLEMENTED);
    }

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());

    std::vector<IndexBatch> batches = MakeBatches(keys);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    // Prepare backend-owned lookup/release workspace once, before acquiring
    // any metadata shard mutex. The same storage is reused by the read and
    // upsert halves of every batch in this synchronous request.
    const size_t max_batch_size =
        std::max_element(batches.begin(), batches.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.global_indices.size() < rhs.global_indices.size();
        })->global_indices.size();
    SingleLocationRmwScratch backend_scratch;
    backend_manager_->PrepareSingleLocationRmwScratch(max_batch_size, backend_scratch);

    SingleLocationResult result(keys.size());
    std::vector<bool> key_level_failures(keys.size(), false);
    RmwStats stats;
    stats.has_index_deserialize = true;

    KeyVector batch_keys;
    LocationIdRefVector batch_location_ids;
    CacheLocationViewVector batch_existing_locations;
    std::vector<ErrorCode> batch_get_ecs;
    std::vector<ErrorCode> batch_key_get_ecs;
    KeyVector upsert_keys;
    LocationIdRefVector upsert_location_ids;
    CacheLocationVector upsert_locations;
    std::vector<int32_t> upsert_global_indices;
    std::vector<bool> upsert_is_new_key;
    std::vector<size_t> upsert_read_indices;
    batch_keys.reserve(max_batch_size);
    batch_location_ids.reserve(max_batch_size);
    batch_existing_locations.reserve(max_batch_size);
    batch_get_ecs.reserve(max_batch_size);
    batch_key_get_ecs.reserve(max_batch_size);
    upsert_keys.reserve(max_batch_size);
    upsert_location_ids.reserve(max_batch_size);
    upsert_locations.reserve(max_batch_size);
    upsert_global_indices.reserve(max_batch_size);
    upsert_is_new_key.reserve(max_batch_size);
    upsert_read_indices.reserve(max_batch_size);

    struct DeferredLocationRelease {
        CacheLocationVector &locations;
        ~DeferredLocationRelease() { locations.clear(); }
    };

    for (const auto &batch : batches) {
        // Reuse request-shaped buffers across internal batches. Clear happens
        // before the next ScopedBatchLock, so releasing replacement locations
        // and growing these vectors never extends metadata lock hold time.
        FillBatchKeys(batch.global_indices, keys, batch_keys);
        batch_location_ids.clear();
        batch_existing_locations.clear();
        batch_get_ecs.clear();
        batch_key_get_ecs.clear();
        upsert_keys.clear();
        upsert_location_ids.clear();
        upsert_locations.clear();
        upsert_global_indices.clear();
        upsert_is_new_key.clear();
        upsert_read_indices.clear();
        for (const int32_t global_index : batch.global_indices) {
            batch_location_ids.push_back(location_ids[global_index]);
        }
        // Allocate all request-shaped scratch vectors before taking metadata
        // shard locks. Large ReportEvent requests otherwise extend every lock
        // hold with allocator work and can amplify allocator futex contention.
        backend_scratch.retired_locations.clear();
        // Construct this guard before the shard lock. Reverse destruction
        // releases the lock first, then drops replaced CacheLocations and
        // their URI strings outside the metadata critical section.
        DeferredLocationRelease deferred_location_release{backend_scratch.retired_locations};
        ScopedBatchLock lock(*this, batch.shard_indices, &stats.lock_wait_time_us, &stats.lock_hold_time_us);

        const int64_t begin_get = TimestampUtil::GetCurrentTimeUs();
        backend_manager_->GetSingleLocationViewsWithKeyStatusInto(nullptr,
                                                                  batch_keys,
                                                                  batch_location_ids,
                                                                  batch_existing_locations,
                                                                  batch_key_get_ecs,
                                                                  batch_get_ecs,
                                                                  backend_scratch);
        stats.get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get;
        if (batch_get_ecs.size() != batch_keys.size() || batch_key_get_ecs.size() != batch_keys.size() ||
            batch_existing_locations.size() != batch_keys.size()) {
            PREFIX_INDEXER_LOG(ERROR,
                               "single target RMW result size mismatch, keys[%lu], ecs[%lu], locations[%lu], "
                               "key ecs[%lu]",
                               batch_keys.size(),
                               batch_get_ecs.size(),
                               batch_existing_locations.size(),
                               batch_key_get_ecs.size());
            for (const int32_t global_index : batch.global_indices) {
                result.error_codes[global_index] = EC_MISMATCH;
                key_level_failures[global_index] = true;
            }
            backend_scratch.ReleaseRetainedHandles();
            continue;
        }

        for (size_t i = 0; i < batch_keys.size(); ++i) {
            const int32_t global_index = batch.global_indices[i];
            ErrorCode get_ec = batch_get_ecs[i];
            const ErrorCode key_get_ec = batch_key_get_ecs[i];
            if (key_get_ec != EC_OK && key_get_ec != EC_NOENT) {
                result.error_codes[global_index] = key_get_ec;
                key_level_failures[global_index] = true;
                continue;
            }
            if (get_ec == EC_OK &&
                (!batch_existing_locations[i] || batch_existing_locations[i]->id() != *batch_location_ids[i])) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "single target RMW invalid EC_OK value, key[%ld], requested id[%s]",
                                   batch_keys[i],
                                   batch_location_ids[i]->c_str());
                batch_existing_locations[i] = nullptr;
                get_ec = EC_MISMATCH;
            }
            if (key_get_ec == EC_NOENT && get_ec == EC_OK) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "single target RMW key[%ld] reported missing with an existing target location",
                                   batch_keys[i]);
                result.error_codes[global_index] = EC_MISMATCH;
                key_level_failures[global_index] = true;
                continue;
            }
            if (get_ec != EC_OK && get_ec != EC_NOENT) {
                key_level_failures[global_index] = true;
            }

            CacheLocationConstPtr new_location;
            auto [action, modifier_ec] = modifier(get_ec,
                                                  *batch_location_ids[i],
                                                  static_cast<size_t>(global_index),
                                                  batch_existing_locations[i],
                                                  new_location);
            if (get_ec != EC_OK && get_ec != EC_NOENT) {
                modifier_ec = get_ec;
            }
            if (action == MA_OK && modifier_ec == EC_OK) {
                if (!new_location || new_location->id() != *batch_location_ids[i]) {
                    result.error_codes[global_index] = EC_MISMATCH;
                    key_level_failures[global_index] = true;
                    continue;
                }
                upsert_keys.push_back(batch_keys[i]);
                upsert_location_ids.push_back(batch_location_ids[i]);
                upsert_locations.push_back(std::move(new_location));
                upsert_global_indices.push_back(global_index);
                upsert_is_new_key.push_back(key_get_ec == EC_NOENT);
                upsert_read_indices.push_back(i);
            } else {
                if (action == MA_OK && modifier_ec != EC_OK) {
                    action = MA_SKIP;
                }
                if (action == MA_FAIL || action == MA_DELETE || (action != MA_SKIP && action != MA_OK)) {
                    key_level_failures[global_index] = true;
                }
                result.error_codes[global_index] =
                    action == MA_SKIP ? modifier_ec : (modifier_ec == EC_OK ? EC_ERROR : modifier_ec);
            }
        }

        const size_t new_key_count =
            static_cast<size_t>(std::count(upsert_is_new_key.begin(), upsert_is_new_key.end(), true));
        stats.put_key_count += static_cast<int64_t>(new_key_count);
        stats.update_key_count += static_cast<int64_t>(upsert_keys.size() - new_key_count);
        if (upsert_keys.empty()) {
            backend_scratch.ReleaseRetainedHandles();
            continue;
        }

        const bool capacity_exceeded = new_key_count + GetKeyCount() > max_key_count_;
        if (!capacity_exceeded) {
            const int64_t begin_upsert = TimestampUtil::GetCurrentTimeUs();
            // The read result is dead after modifier evaluation. Reuse its
            // capacity for writer status instead of allocating another
            // request-sized vector under the metadata lock.
            backend_manager_->UpsertSingleLocationsUsingRetainedHandlesInto(nullptr,
                                                                            upsert_keys,
                                                                            upsert_location_ids,
                                                                            upsert_locations,
                                                                            upsert_read_indices,
                                                                            batch_get_ecs,
                                                                            backend_scratch);
            stats.upsert_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_upsert;
        } else {
            PREFIX_INDEXER_LOG(ERROR,
                               "single target RMW put keys count[%lu] + current key count[%lu] > max key count[%lu]",
                               new_key_count,
                               GetKeyCount(),
                               max_key_count_);

            // Compact existing-key updates in place. The old implementation
            // allocated four subset vectors while all metadata shard locks
            // were held. New-key positions are already final EC_NOSPC, while
            // the compacted global indices retain the mapping needed below.
            for (size_t i = 0; i < upsert_is_new_key.size(); ++i) {
                if (upsert_is_new_key[i]) {
                    const int32_t global_index = upsert_global_indices[i];
                    result.error_codes[global_index] = EC_NOSPC;
                    key_level_failures[global_index] = true;
                }
            }
            size_t existing_count = 0;
            for (size_t i = 0; i < upsert_keys.size(); ++i) {
                if (upsert_is_new_key[i]) {
                    continue;
                }
                if (existing_count != i) {
                    upsert_keys[existing_count] = upsert_keys[i];
                    upsert_location_ids[existing_count] = upsert_location_ids[i];
                    upsert_locations[existing_count] = std::move(upsert_locations[i]);
                    upsert_global_indices[existing_count] = upsert_global_indices[i];
                    upsert_read_indices[existing_count] = upsert_read_indices[i];
                }
                ++existing_count;
            }
            upsert_keys.resize(existing_count);
            upsert_location_ids.resize(existing_count);
            upsert_locations.resize(existing_count);
            upsert_global_indices.resize(existing_count);
            upsert_read_indices.resize(existing_count);

            if (existing_count > 0) {
                const int64_t begin_upsert = TimestampUtil::GetCurrentTimeUs();
                // The get result vector is dead after modifier evaluation and
                // already has sufficient capacity, so reuse it for writes.
                backend_manager_->UpsertSingleLocationsUsingRetainedHandlesInto(nullptr,
                                                                                upsert_keys,
                                                                                upsert_location_ids,
                                                                                upsert_locations,
                                                                                upsert_read_indices,
                                                                                batch_get_ecs,
                                                                                backend_scratch);
                stats.upsert_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_upsert;
                if (batch_get_ecs.size() != existing_count) {
                    batch_get_ecs.assign(existing_count, EC_MISMATCH);
                }
            } else {
                backend_scratch.ReleaseRetainedHandles();
            }

            // New keys cannot be admitted, but existing locations retain
            // ordinary update semantics. Finish this exceptional batch here
            // because its arrays were compacted in place.
            for (size_t i = 0; i < existing_count; ++i) {
                const int32_t global_index = upsert_global_indices[i];
                const ErrorCode ec = batch_get_ecs[i];
                result.error_codes[global_index] = ec;
                if (ec != EC_OK) {
                    key_level_failures[global_index] = true;
                    PREFIX_INDEXER_LOG(
                        ERROR, "single target RMW upsert failed, key[%ld], ec[%d]", keys[global_index], ec);
                }
            }
            continue;
        }

        int32_t successful_new_keys = 0;
        if (batch_get_ecs.size() != upsert_keys.size()) {
            PREFIX_INDEXER_LOG(ERROR,
                               "single target RMW upsert result size[%lu] mismatch keys[%lu]",
                               batch_get_ecs.size(),
                               upsert_keys.size());
            batch_get_ecs.assign(upsert_keys.size(), EC_MISMATCH);
        }
        for (size_t i = 0; i < batch_get_ecs.size(); ++i) {
            const int32_t global_index = upsert_global_indices[i];
            result.error_codes[global_index] = batch_get_ecs[i];
            if (batch_get_ecs[i] != EC_OK) {
                key_level_failures[global_index] = true;
                PREFIX_INDEXER_LOG(
                    ERROR, "single target RMW upsert failed, key[%ld], ec[%d]", keys[global_index], batch_get_ecs[i]);
            } else if (upsert_is_new_key[i]) {
                ++successful_new_keys;
            }
        }
        AdjustKeyCountMeta(successful_new_keys);
    }

    EmitRmwMetrics(request_context->metrics_collector(), stats, keys.size());
    const size_t failed_key_count =
        static_cast<size_t>(std::count(key_level_failures.begin(), key_level_failures.end(), true));
    if (failed_key_count == keys.size()) {
        result.ec = EC_ERROR;
    } else if (failed_key_count > 0) {
        result.ec = EC_PARTIAL_OK;
    }
    return result;
}

MetaIndexer::Result
MetaIndexer::Exist(RequestContext *request_context, const KeyVector &keys, std::vector<bool> &out_exists) noexcept {
    if (!SupportsWriteSession()) {
        out_exists.clear();
        return Result(EC_UNIMPLEMENTED);
    }
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
    if (!SupportsWriteSession()) {
        out_location_maps.clear();
        out_properties.clear();
        return Result(EC_UNIMPLEMENTED);
    }
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
    ValidateSimpleReadResults(keys.size(), out_location_maps, error_codes);
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

MetaIndexer::Result MetaIndexer::GetLocationsFromPrimary(RequestContext *request_context,
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
    auto error_codes = backend_manager_->GetLocationsFromPrimary(request_context, keys, out_location_maps);
    ValidateSimpleReadResults(keys.size(), out_location_maps, error_codes);
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
    ValidateSimpleReadResults(keys.size(), out_locations, error_codes);
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

    if (!SupportsWriteSession()) {
        const std::string &trace_id = request_context->trace_id();
        CompactLocationsPerKey locations;
        size_t batch_count = 0;
        int64_t io_time_us = 0;
        int64_t deserialize_time_us = 0;
        ServiceMetricsCollector *query_metrics =
            dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
        for (size_t begin = 0; begin < keys.size();) {
            const size_t count = std::min(batch_key_size_, keys.size() - begin);
            KVCM_METRICS_COLLECTOR_SET_METRICS(query_metrics, meta_searcher, index_deserialize_time_us, 0);
            const int64_t start = TimestampUtil::GetCurrentTimeUs();
            std::vector<ErrorCode> errors =
                backend_manager_->GetLocationValuesCompact(request_context, keys.data() + begin, count, locations);
            io_time_us += TimestampUtil::GetCurrentTimeUs() - start;
            int64_t chunk_deserialize_us = 0;
            KVCM_METRICS_COLLECTOR_GET_METRICS(
                query_metrics, meta_searcher, index_deserialize_time_us, chunk_deserialize_us);
            deserialize_time_us += chunk_deserialize_us;
            ++batch_count;
            prefix_result.read_key_count += count;
            if (errors.size() != count || !locations.IsValid(count)) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "simple prefix invalid result, begin[%zu] expected[%zu] errors[%zu]",
                                   begin,
                                   count,
                                   errors.size());
                prefix_result.terminal_ec = EC_MISMATCH;
                break;
            }
            size_t valid = 0;
            for (; valid < count && errors[valid] == EC_OK; ++valid) {
                for (const CacheLocationConstPtr &location : locations[valid]) {
                    errors[valid] = ValidateSimpleLocation(location, false);
                    if (errors[valid] != EC_OK) {
                        break;
                    }
                }
                if (errors[valid] != EC_OK) {
                    break;
                }
            }
            size_t requested_stop = keys.size();
            if (valid != 0 && visitor) {
                requested_stop = std::min(keys.size(), visitor(begin, locations, valid));
            }
            prefix_result.valid_key_count = std::min(begin + valid, requested_stop);
            if (requested_stop <= begin + valid && requested_stop < keys.size()) {
                prefix_result.stopped_by_visitor = true;
                break;
            }
            if (valid != count) {
                prefix_result.terminal_ec = errors[valid];
                if (prefix_result.terminal_ec != EC_NOENT) {
                    PREFIX_INDEXER_LOG(ERROR,
                                       "simple prefix read failed, index[%zu] ec[%d]",
                                       begin + valid,
                                       prefix_result.terminal_ec);
                }
                break;
            }
            begin += count;
        }
        KVCM_METRICS_COLLECTOR_SET_METRICS(query_metrics, meta_indexer, query_key_count, keys.size());
        KVCM_METRICS_COLLECTOR_SET_METRICS(query_metrics, meta_indexer, query_batch_num, batch_count);
        KVCM_METRICS_COLLECTOR_SET_METRICS(query_metrics, meta_indexer, get_io_time_us, io_time_us);
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            query_metrics, meta_searcher, index_deserialize_time_us, deserialize_time_us);
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
    // Backend reads and projection are pipelined. Track the union of backend
    // call intervals so get_io_time_us does not accidentally include visitor
    // CPU time or double-count overlapping worker reads.
    std::atomic<size_t> active_backend_reads(0);
    std::atomic<int64_t> backend_read_interval_start_us(0);
    std::atomic<int64_t> backend_read_wall_time_us(0);
    const size_t configured_chunk_size =
        std::max(kLocalPrefixParallelReadChunkSize, query_executor_ ? query_executor_->chunk_size() : size_t{256});
    // Keep callback boundaries on 64-key words. The Mamba projection stores
    // per-host state as disjoint key bit ranges, so concurrent callbacks never
    // update the same word.
    const size_t aligned_chunk_size =
        configured_chunk_size > std::numeric_limits<size_t>::max() - (kPrefixStateWordBits - 1)
            ? configured_chunk_size
            : (configured_chunk_size + kPrefixStateWordBits - 1) & ~(kPrefixStateWordBits - 1);
    // A configuration larger than the request still means one suffix range;
    // clamp it so absolute range arithmetic cannot wrap around size_t.
    const size_t parallel_chunk_size = std::min(aligned_chunk_size, keys.size());
    const size_t first_chunk_size = std::min(kLocalPrefixProbeKeyCount, keys.size());
    const size_t remaining_chunk_count =
        first_chunk_size == keys.size() ? 0 : 1 + (keys.size() - first_chunk_size - 1) / parallel_chunk_size;

    struct ChunkTerminal {
        size_t index = 0;
        ErrorCode ec = EC_OK;
        bool present = false;
    };
    std::vector<ChunkTerminal> terminals(1 + remaining_chunk_count);
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
                           &active_backend_reads,
                           &backend_read_interval_start_us,
                           &backend_read_wall_time_us,
                           &reduce_stop_index,
                           &current_stop](size_t chunk_begin, size_t count, size_t terminal_slot) {
        if (chunk_begin >= current_stop()) {
            return;
        }

        PrefixLocationScratchLease scratch_lease(count);
        auto &locations = scratch_lease.locations();
        const int64_t backend_read_begin_us = TimestampUtil::GetCurrentTimeUs();
        if (active_backend_reads.fetch_add(1, std::memory_order_acq_rel) == 0) {
            backend_read_interval_start_us.store(backend_read_begin_us, std::memory_order_release);
        }
        auto errors = backend_manager_->GetLocationValuesCompact(nullptr, keys.data() + chunk_begin, count, locations);
        const int64_t backend_read_end_us = TimestampUtil::GetCurrentTimeUs();
        const int64_t interval_start_us = backend_read_interval_start_us.load(std::memory_order_acquire);
        if (active_backend_reads.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            backend_read_wall_time_us.fetch_add(std::max<int64_t>(backend_read_end_us - interval_start_us, 0),
                                                std::memory_order_relaxed);
        }
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
            auto &terminal = terminals[terminal_slot];
            terminal.index = chunk_begin + successful_count;
            terminal.ec = terminal_ec;
            terminal.present = true;
            reduce_stop_index(metadata_stop, terminal.index);
        }
    };
    auto read_ranges = [&read_one_chunk, &current_stop, &keys, first_chunk_size, parallel_chunk_size](size_t begin,
                                                                                                      size_t end) {
        for (size_t chunk_begin = begin; chunk_begin < end;) {
            if (chunk_begin >= current_stop()) {
                return;
            }
            const size_t chunk_end = chunk_begin + std::min(parallel_chunk_size, end - chunk_begin);
            const size_t terminal_slot = 1 + (chunk_begin - first_chunk_size) / parallel_chunk_size;
            read_one_chunk(chunk_begin, chunk_end - chunk_begin, terminal_slot);
            chunk_begin = chunk_end;
        }
    };

    bool completed = true;
    try {
        // Candidate hosts are derived from key zero. Visiting this chunk before
        // scheduling the suffix makes every later callback independent and
        // allows an early host miss to cancel all suffix metadata reads.
        read_one_chunk(0, first_chunk_size, 0);
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
            completed =
                query_executor_->ParallelForWithChunkSize(remaining_count, parallel_chunk_size, read_remaining_ranges);
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
            const size_t terminal_slot = first_metadata_error < first_chunk_size
                                             ? 0
                                             : 1 + (first_metadata_error - first_chunk_size) / parallel_chunk_size;
            const auto &terminal = terminals[terminal_slot];
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

    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector,
                                       meta_indexer,
                                       get_io_time_us,
                                       backend_read_wall_time_us.load(std::memory_order_relaxed));
    return prefix_result;
}

MetaIndexer::LocationResult MetaIndexer::GetLocations(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      const LocationIdsPerKey &location_ids,
                                                      LocationsPerKey &out_locations) noexcept {
    if (!SupportsWriteSession()) {
        out_locations.clear();
        return LocationResult(EC_UNIMPLEMENTED);
    }
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
    } catch (...) {
        KVCM_LOG_ERROR("serial query callback threw unknown exception");
    }
    return false;
}

MetaIndexer::Result MetaIndexer::GetLocationMapsForMaintenance(RequestContext *request_context,
                                                               const KeyVector &keys,
                                                               CacheLocationMapVector &out_locations) noexcept {
    auto codes = backend_manager_->GetLocationMapsForMaintenance(request_context, keys, out_locations);
    ValidateSimpleReadResults(keys.size(), out_locations, codes);
    Result result(keys.size());
    if (keys.empty()) {
        result.ec = EC_OK;
        return result;
    }
    const auto &trace_id = request_context->trace_id();
    const auto errors = ProcessErrorCodes(trace_id, codes, {}, keys, kGetMetaOperation, result);
    ProcessErrorResult(trace_id, kGetMetaOperation, errors, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::GetProperties(RequestContext *request_context,
                                               const KeyVector &keys,
                                               const std::vector<std::string> &property_names,
                                               PropertyMapVector &out_properties) noexcept {
    if (!SupportsWriteSession()) {
        out_properties.assign(keys.size(), PropertyMap{});
        Result result(keys.size());
        result.ec = EC_UNIMPLEMENTED;
        result.error_codes.assign(keys.size(), EC_UNIMPLEMENTED);
        return result;
    }
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
    if (!SupportsWriteSession()) {
        out_keys.clear();
        out_next_cursor = SCAN_BASE_CURSOR;
        if (limit == 0 || limit > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            return EC_BADARGS;
        }
    }
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

ErrorCode MetaIndexer::TrimResidues(RequestContext *request_context, const size_t scan_batch_size) noexcept {
    if (!SupportsWriteSession() && scan_batch_size == 0) {
        return EC_BADARGS;
    }
    if (!backend_manager_->IsMemoryPrimary() ||
        backend_manager_->GetRecoverState() != MetaStorageBackendManager::RecoverState::kRunning) {
        return EC_OK;
    }

    std::string cursor = SCAN_BASE_CURSOR;
    do {
        std::string next_cursor;
        KeyVector keys;
        const ErrorCode list_ec = backend_manager_->ListPersistentKeys(
            request_context, cursor, static_cast<int64_t>(scan_batch_size), next_cursor, keys);
        if (list_ec != EC_OK) {
            KVCM_LOG_ERROR("instance[%s] list persistent keys failed, cursor[%s], ec[%d]",
                           instance_id_.c_str(),
                           cursor.c_str(),
                           list_ec);
            return list_ec;
        }

        KeyVector batch_keys;
        KeyVector trimmed_keys;
        KeyVector page_trimmed_keys;
        page_trimmed_keys.reserve(keys.size());
        ErrorCode page_ec = EC_OK;
        for (const auto &batch : MakeBatches(keys)) {
            FillBatchKeys(batch.global_indices, keys, batch_keys);
            {
                ScopedBatchLock lock(*this, batch.shard_indices, nullptr, nullptr);
                page_ec = backend_manager_->TrimPersistentOrphans(request_context, batch_keys, trimmed_keys);
            }
            page_trimmed_keys.insert(page_trimmed_keys.end(), trimmed_keys.begin(), trimmed_keys.end());
            if (page_ec != EC_OK) {
                break;
            }
        }
        if (!page_trimmed_keys.empty() && !backend_manager_->Sync(page_trimmed_keys)) {
            KVCM_LOG_ERROR("instance[%s] sync persistent orphan deletes failed", instance_id_.c_str());
            return EC_ERROR;
        }
        if (page_ec != EC_OK) {
            return page_ec;
        }
        cursor = std::move(next_cursor);
    } while (cursor != SCAN_BASE_CURSOR);
    return EC_OK;
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
    if (ec == EC_OK) {
        ValidateSimpleReadResults(out.keys.size(), out.locations, out.location_results);
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

bool MetaIndexer::PreferSingleTaskReclaimSampling() const noexcept {
    return backend_manager_ && backend_manager_->PreferSingleTaskReclaimSampling();
}

size_t MetaIndexer::TouchKeysForMaintenance(const KeyVector &keys) const noexcept {
    return backend_manager_ ? backend_manager_->TouchKeysForMaintenance(keys) : 0;
}

ErrorCode MetaIndexer::SampleReclaimCandidates(RequestContext *request_context,
                                               const int64_t count,
                                               ReclaimCandidateVector &out_candidates,
                                               bool require_read_success) const noexcept {
    out_candidates.clear();
    if (count > 0) {
        out_candidates.reserve(static_cast<size_t>(count));
    }
    const ErrorCode ec =
        backend_manager_->SampleReclaimCandidates(request_context, count, out_candidates, require_read_success);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("instance[%s] meta indexer sample reclaim candidates failed, count[%lu] candidate size[%lu]",
                       instance_id_.c_str(),
                       count,
                       out_candidates.size());
    }
    return ec;
}

size_t MetaIndexer::GetKeyCount() const noexcept { return key_count_.load(); }

size_t MetaIndexer::GetMaxKeyCount() const noexcept { return max_key_count_; }

size_t MetaIndexer::GetMemUsage() const noexcept { return backend_manager_->GetMemUsage(); }

bool MetaIndexer::Sync(const KeyVector &keys) noexcept { return backend_manager_->Sync(keys); }

bool MetaIndexer::SyncAll() noexcept { return backend_manager_->SyncAll(); }

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

std::vector<std::vector<int32_t>> MetaIndexer::GetWriteBatches(const KeyVector &keys) const {
    std::vector<IndexBatch> batches = MakeBatches(keys);
    std::vector<std::vector<int32_t>> result;
    result.reserve(batches.size());
    for (IndexBatch &batch : batches) {
        std::sort(batch.global_indices.begin(), batch.global_indices.end());
        result.push_back(std::move(batch.global_indices));
    }
    return result;
}

std::vector<MetaIndexer::IndexBatch> MetaIndexer::MakeBatches(const KeyVector &keys) const noexcept {
    std::vector<IndexBatch> result;

    std::map<int32_t, std::vector<int32_t>> shard_map;
    for (int32_t i = 0; i < static_cast<int32_t>(keys.size()); ++i) {
        const int32_t shard_idx = GetMutexShardIndex(keys[i]);
        shard_map[shard_idx].push_back(i);
    }
    if (shard_map.empty()) {
        return result;
    }

    IndexBatch current;
    size_t current_batch_size = 0;
    size_t shards_emitted = 0;
    const size_t total_shards = shard_map.size();

    for (auto &shard_kv : shard_map) {
        const int32_t shard_index = shard_kv.first;
        const auto &index_list = shard_kv.second;

        current.shard_indices.emplace_back(shard_index);
        current.global_indices.insert(current.global_indices.end(), index_list.begin(), index_list.end());
        current_batch_size += index_list.size();
        ++shards_emitted;

        // Flush on soft-limit, or after the last shard so the tail batch is kept.
        if (current_batch_size >= batch_key_size_ || shards_emitted == total_shards) {
            result.emplace_back(std::move(current));
            current = IndexBatch{};
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
    int64_t key_count = 0;
    bool is_valid = false;
    if (SupportsWriteSession()) {
        is_valid = StringUtil::StrToInt64(key_count_str.c_str(), key_count);
    } else {
        const char *begin = key_count_str.data();
        const char *end = begin + key_count_str.size();
        const std::from_chars_result parsed = std::from_chars(begin, end, key_count);
        is_valid = parsed.ec == std::errc{} && parsed.ptr == end && key_count >= 0;
    }
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

ErrorCode MetaIndexer::SaveMetaData() noexcept {
    FieldMap metadata{{METADATA_PROPERTY_KEY_COUNT, std::to_string(key_count_.load())},
                      {METADATA_PROPERTY_STORAGE_USAGE_DATA, storage_usage_data_.Serialize()}};
    return backend_manager_->PutMetaData(metadata);
}

// 定时持久化key count等meta data，failover时可能因持久化不及时，key count与真实值会发生偏差
void MetaIndexer::PersistMetaData() noexcept {
    if (!backend_manager_) {
        return;
    }
    int64_t current_time = TimestampUtil::GetSteadyTimeMs();
    if (current_time >= last_persist_metadata_time_ + persist_metadata_interval_time_ms_) {
        const ErrorCode ec = SaveMetaData();
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

size_t MetaIndexer::GetLocalLocationCount() const noexcept {
    return backend_manager_ ? backend_manager_->GetLocalLocationCount() : 0;
}

size_t MetaIndexer::GetLocalKeyCount() const noexcept {
    return backend_manager_ ? backend_manager_->GetLocalKeyCount() : 0;
}

} // namespace kv_cache_manager
