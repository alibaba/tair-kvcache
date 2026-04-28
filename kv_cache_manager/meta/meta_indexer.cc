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
#include "kv_cache_manager/meta/meta_search_cache.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/meta/utils.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {
#define PREFIX_INDEXER_LOG(LEVEL, format, args...)                                                                     \
    KVCM_LOG_##LEVEL("trace_id[%s] instance[%s] | " format, trace_id.c_str(), instance_id_.c_str(), ##args);

static constexpr const char *kPutMetaOperation = "put";
static constexpr const char *kUpdateMetaOperation = "update";
static constexpr const char *kRmwMetaOperation = "read_modify_write";
static constexpr const char *kRmwUpsertMetaOperation = "read_modify_write_upsert";
static constexpr const char *kRmwDeleteMetaOperation = "read_modify_write_delete";
static constexpr const char *kDeleteMetaOperation = "delete";
static constexpr const char *kExistMetaOperation = "exist";
static constexpr const char *kGetMetaOperation = "get";

class MetaIndexer::ScopedBatchLock {
public:
    ScopedBatchLock(MetaIndexer &indexer, const std::vector<int32_t> &shard_indexs)
        : indexer_(indexer), shard_indexs_(shard_indexs) {
        for (const int32_t shardIdx : shard_indexs_) {
            indexer_.mutex_shards_[shardIdx]->lock();
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

std::uint64_t MetaIndexer::StorageUsageData::GetStorageUsage() const noexcept {
    std::uint64_t storage_usage = 0;
    for (size_t_ i = 0; i != storage_usage_by_type_.size(); ++i) {
        if (i == static_cast<size_t_>(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) ||
            i == static_cast<size_t_>(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS)) {
            continue;
        }
        storage_usage += storage_usage_by_type_.at(i).load();
    }
    return storage_usage;
}

std::uint64_t MetaIndexer::StorageUsageData::GetStorageUsageByType(const DataStorageType &type) const noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      storage_usage_by_type_.size(),
                      idx);
        return 0;
    }
    return storage_usage_by_type_.at(idx).load();
}

void MetaIndexer::StorageUsageData::Reset() noexcept {
    // array.fill(0) won't work here due to the deleted operator= of the
    // std::atomic type, explicitly assign 0 to all elements in the
    // array instead
    for (auto &v : storage_usage_by_type_) {
        v.store(0);
    }
}

void MetaIndexer::StorageUsageData::SetStorageUsageByType(const DataStorageType &type,
                                                          const std::uint64_t value) noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      storage_usage_by_type_.size(),
                      idx);
        return;
    }
    storage_usage_by_type_.at(idx).store(value);
}

std::uint64_t MetaIndexer::StorageUsageData::AddStorageUsageByType(const DataStorageType &type,
                                                                   const std::uint64_t value) noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      storage_usage_by_type_.size(),
                      idx);
        return 0;
    }
    return storage_usage_by_type_.at(idx).fetch_add(value);
}

std::uint64_t MetaIndexer::StorageUsageData::SubStorageUsageByType(const DataStorageType &type,
                                                                   const std::uint64_t value) noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      storage_usage_by_type_.size(),
                      idx);
        return 0;
    }

    auto &ref = storage_usage_by_type_.at(idx);
    std::uint64_t expected = ref.load(), desired = 0;
    bool underflow = false;
    do {
        if (expected < value) {
            underflow = true;
            desired = 0;
        } else {
            desired = expected - value;
        }
    } while (!ref.compare_exchange_weak(expected, desired));
    if (underflow) {
        KVCM_LOG_WARN("storage usage underflow for type [%zu]: "
                      "current [%" PRIu64 "] < subtract [%" PRIu64 "], clamped to 0",
                      idx,
                      expected,
                      value);
    }
    return desired;
}

void MetaIndexer::StorageUsageData::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    for (size_t_ i = 0; i != storage_usage_by_type_.size(); ++i) {
        const auto type = static_cast<DataStorageType>(i);
        const std::string key = ToString(type);
        Put(writer, key, storage_usage_by_type_.at(i).load());
    }
}

bool MetaIndexer::StorageUsageData::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        return false;
    }

    // parse into a temporary buffer first to avoid partial updates
    std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> buf{};
    for (auto it = rapid_value.MemberBegin(); it != rapid_value.MemberEnd(); ++it) {
        const std::string key(it->name.GetString(), it->name.GetStringLength());
        const DataStorageType type = ToDataStorageType(key);
        if (ToString(type) != key) {
            // round-trip mismatch: key is not a recognized type
            // prevent ToDataStorageType silently maps every unknown
            // string to DATA_STORAGE_TYPE_UNKNOWN
            KVCM_LOG_ERROR("deserialize storage usage data failed: unrecognized storage type [%s]", key.c_str());
            return false;
        }
        const size_t_ idx = ToIndex(type);
        if (it->value.IsUint64()) {
            buf.at(idx) = it->value.GetUint64();
        } else {
            KVCM_LOG_ERROR("deserialize storage usage data failed: non-integer value for key [%s]", key.c_str());
            return false;
        }
    }

    // all values parsed successfully, apply to the actual array
    for (size_t_ i = 0; i != storage_usage_by_type_.size(); ++i) {
        storage_usage_by_type_.at(i).store(buf.at(i));
    }
    return true;
}

std::string MetaIndexer::StorageUsageData::Serialize() const noexcept {
    const std::string str = ToJsonString();
    KVCM_LOG_DEBUG("serializing storage usage data into: [%s]", str.c_str());
    return str;
}

ErrorCode MetaIndexer::StorageUsageData::Deserialize(const std::string &str) noexcept {
    KVCM_LOG_DEBUG("deserializing storage usage data: [%s]", str.c_str());
    std::string str_copy{str};
    StringUtil::Trim(str_copy);
    if (str_copy.empty()) {
        KVCM_LOG_ERROR("deserialize storage usage data failed: input string is empty");
        return ErrorCode::EC_ERROR;
    }
    if (!FromJsonString(str_copy)) {
        KVCM_LOG_ERROR("deserialize storage usage data failed: invalid JSON [%s]", str_copy.c_str());
        return ErrorCode::EC_ERROR;
    }
    return ErrorCode::EC_OK;
}

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
    // mutex_shard_num is validated as a power of two above, so (num - 1) is a
    // contiguous low-bit mask suitable for hash & mask shard lookup.
    mutex_shard_mask_ = mutex_shard_num - 1;
    for (size_t i = 0; i < mutex_shard_num; ++i) {
        mutex_shards_.emplace_back(std::make_unique<std::mutex>());
    }

    instance_id_ = instance_id;
    auto storage_backend_config = config->GetMetaStorageBackendConfig();

    // The dual-backend manager (persistent + local cache) owns both the
    // source-of-truth persistent backend and the in-memory hot cache. It is
    // initialised from MetaStorageBackendConfig so existing deployments need
    // no extra configuration.
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
    if (config->GetMetaCachePolicyConfig()->GetCapacity() > 0) {
        cache_ = std::make_shared<MetaSearchCache>();
        auto ec = cache_->Init(config->GetMetaCachePolicyConfig());
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("instance[%s] init search cache failed, ec[%d]", instance_id_.c_str(), ec);
            return ec;
        }
    }

    storage_usage_data_.Reset();
    ec = RecoverMetaData();
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR("instance[%s] recover metadata failed, ec[%d]", instance_id_.c_str(), ec);
        return ec;
    }
    KVCM_LOG_INFO(
        "instance[%s] meta indexer init success, mutex shard num[%lu], max key count[%lu], "
        "batch key size[%lu], search cache size[%lu], key_count[%lu], persist_metadata_interval_time_ms[%zu], "
        "storage usage data[%s], instance version[%" PRIu8 "]",
        instance_id_.c_str(),
        mutex_shard_num,
        max_key_count_,
        batch_key_size_,
        config->GetMetaCachePolicyConfig()->GetCapacity(),
        key_count_.load(),
        persist_metadata_interval_time_ms_,
        version_ == InstanceVersion::VERSION_1 ? storage_usage_data_.ToJsonString().c_str() : "N/A",
        static_cast<std::uint8_t>(version_));
    return EC_OK;
}


// Put / Update now consume LocationMapVector instead of UriVector. The dual-
// backend manager (backend_manager_) is responsible for serialising every
// CacheLocation into the per-key PropertyMap before issuing the actual write
// to the persistent + local backends, so this layer only needs to assemble
// BatchMetaData and own the per-shard locks (the manager itself never grabs
// shard mutexes).
MetaIndexer::Result MetaIndexer::Put(RequestContext *request_context,
                                     const KeyVector &keys,
                                     LocationMapVector &location_maps,
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

    std::vector<BatchMetaData> batches = MakeBatches(keys, location_maps, properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t put_io_time_us = 0;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);
        int64_t begin_put_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = backend_manager_->Put(batch);
        put_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_put_io_time;
        error_count += ProcessErrorCodes(trace_id, error_codes, batch.batch_indexs, keys, kPutMetaOperation, result);
    }
    AdjustKeyCountMeta(keys.size() - error_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, put_io_time_us, put_io_time_us);
    ProcessErrorResult(trace_id, kPutMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::Update(RequestContext *request_context,
                                        const KeyVector &keys,
                                        LocationMapVector &location_maps,
                                        PropertyMapVector &properties) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    const auto &trace_id = request_context->trace_id();
    if ((!location_maps.empty() && keys.size() != location_maps.size()) || 
        (!properties.empty() && keys.size() != properties.size())) {
        PREFIX_INDEXER_LOG(ERROR,
                           "Update keys size[%lu], location_maps size[%lu], properties size[%lu] not equal",
                           keys.size(),
                           location_maps.size(),
                           properties.size());
        return Result(EC_ERROR);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());

    std::vector<BatchMetaData> batches = MakeBatches(keys, location_maps, properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t update_io_time_us = 0;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);
        int64_t begin_update_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = backend_manager_->UpdateFields(batch);
        update_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_update_io_time;
        error_count += ProcessErrorCodes(trace_id, error_codes, batch.batch_indexs, keys, kUpdateMetaOperation, result);
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, update_io_time_us, update_io_time_us);
    ProcessErrorResult(trace_id, kUpdateMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::Delete(RequestContext *request_context, const KeyVector &keys) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();
    LocationMapVector empty_locations;
    PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());
    Result result(keys.size());
    int32_t error_count = 0;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);
        auto error_codes = backend_manager_->Delete(batch);
        error_count += ProcessErrorCodes(trace_id, error_codes, batch.batch_indexs, keys, kDeleteMetaOperation, result);
    }
    AdjustKeyCountMeta(error_count - keys.size());
    ProcessErrorResult(trace_id, kDeleteMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result
MetaIndexer::Exist(RequestContext *request_context, const KeyVector &keys, std::vector<bool> &out_exists) noexcept {
    const auto &trace_id = request_context->trace_id();
    out_exists.reserve(keys.size());
    auto error_codes = backend_manager_->Exists(keys, out_exists);

    Result result(keys.size());
    int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kExistMetaOperation, result);
    ProcessErrorResult(trace_id, kExistMetaOperation, error_count, keys.size(), result);
    return result;
}


MetaIndexer::Result
MetaIndexer::Get(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept {
    out_uris.resize(keys.size());
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    if (cache_) {
        return DoGetWithCache(request_context, keys, out_uris);
    } else {
        return DoGetWithoutCache(request_context, keys, out_uris);
    }
}

MetaIndexer::Result MetaIndexer::Get(RequestContext *request_context,
                                     const KeyVector &keys,
                                     UriVector &out_uris,
                                     PropertyMapVector &out_properties) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();
    out_uris.reserve(keys.size());
    out_properties.reserve(keys.size());
    PropertyMapVector maps;
    Result result(keys.size());
    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->GetAllFields(keys, maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    int32_t error_count = 0;
    for (int32_t i = 0; i < keys.size(); ++i) {
        auto &map = maps[i];
        out_uris.emplace_back(std::move(map[PROPERTY_URI]));
        for (auto it = map.begin(); it != map.end();) {
            if (it->first.rfind(PROPERTY_INNER_PREFIX, 0) == 0) {
                it = map.erase(it);
            } else {
                ++it;
            }
        }
        out_properties.emplace_back(std::move(map));
        if (error_codes[i] != EC_OK) {
            if (error_codes[i] != EC_NOENT) {
                PREFIX_INDEXER_LOG(ERROR, "meta indexer get failed, key[%ld] ec[%d]", keys[i], error_codes[i]);
            }
            result.error_codes[i] = error_codes[i];
            ++error_count;
        }
    }
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

// When get properties, maybe a key exists but its properties in property_names do not exist.
// To ensure consistent semantics, EC_OK is returned even if the property map is empty.
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
    auto error_codes = backend_manager_->Get(keys, property_names, out_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    Result result(keys.size());
    int32_t error_count = ProcessErrorCodes(trace_id, error_codes, {}, keys, kGetMetaOperation, result);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

ErrorCode MetaIndexer::Scan(const std::string &cursor,
                            const size_t limit,
                            std::string &out_next_cursor,
                            KeyVector &out_keys) noexcept {
    out_keys.reserve(limit);
    auto ec = backend_manager_->ListKeys(cursor, limit, out_next_cursor, out_keys);
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
    auto ec = backend_manager_->RandomSample(count, out_keys);
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
    auto ec = backend_manager_->SampleReclaimKeys(count, out_keys);
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

size_t MetaIndexer::GetCacheUsage() const noexcept {
    if (cache_) {
        return cache_->GetCacheUsage();
    }
    return 0;
}

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

MetaIndexer::InstanceVersion MetaIndexer::GetVersion() const noexcept { return version_; }

std::vector<MetaIndexer::BatchMetaData> MetaIndexer::MakeBatches(const KeyVector &keys,
                                                                 LocationMapVector &locations,
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
    auto ec = backend_manager_->GetMetaData(metadata_map);
    if (ec == EC_NOENT) {
        KVCM_LOG_INFO("there is no metadata key in storage backend, no need to recover metadata");
        // no metadata to recover (new instance or persistence turned
        // off), safe to set to the newest version
        // version_ remains InstanceVersion::VERSION_1
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

    // METADATA_PROPERTY_STORAGE_USAGE_DATA decides the behavior version
    if (const auto it = metadata_map.find(METADATA_PROPERTY_STORAGE_USAGE_DATA); it != metadata_map.end()) {
        if (storage_usage_data_.Deserialize(it->second) != EC_OK) {
            KVCM_LOG_ERROR("meta indexer deserialize storage usage data failed, str: [%s]", it->second.c_str());
            return EC_ERROR;
        }
        // version_ remains InstanceVersion::VERSION_1
    } else {
        // METADATA_PROPERTY_STORAGE_USAGE_DATA do not exist
        version_ = InstanceVersion::VERSION_0;
    }

    return EC_OK;
}

// 定时持久化key count等meta data，failover时可能因持久化不及时，key count与真实值会发生偏差
void MetaIndexer::PersistMetaData() noexcept {
    int64_t current_time = TimestampUtil::GetSteadyTimeMs();
    if (current_time >= last_persist_metadata_time_ + persist_metadata_interval_time_ms_) {
        std::map<std::string, std::string> metadata_map;
        metadata_map[METADATA_PROPERTY_KEY_COUNT] = std::to_string(key_count_);
        if (version_ == InstanceVersion::VERSION_1) {
            metadata_map[METADATA_PROPERTY_STORAGE_USAGE_DATA] = storage_usage_data_.Serialize();
        }
        auto ec = backend_manager_->PutMetaData(metadata_map);
        if (ec != EC_OK) {
            KVCM_LOG_WARN("meta indexer persist metadata failed, ec[%d]", ec);
        }
        last_persist_metadata_time_ = current_time;
    }
}

// 如果key重复，put时key count将重复计算，将比真实值偏大
// KV Cache场景key是累积hash的计算值，同一个请求中不会存在重复值
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

MetaIndexer::Result
MetaIndexer::DoGetWithCache(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept {
    // get from cache first
    const auto &trace_id = request_context->trace_id();
    KeyVector miss_keys;
    std::vector<int32_t> miss_indexs;
    for (int32_t i = 0; i < keys.size(); ++i) {
        std::string uri;
        auto ec = cache_->Get(keys[i], &uri);
        if (ec == EC_OK) {
            out_uris[i] = std::move(uri);
        } else {
            miss_keys.push_back(keys[i]);
            miss_indexs.push_back(i);
        }
    }
    assert(miss_keys.size() == miss_indexs.size());
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    size_t cache_key_hit_count = keys.size() - miss_keys.size();
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, search_cache_miss_count, miss_keys.size());
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, search_cache_hit_count, cache_key_hit_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, search_cache_hit_ratio, cache_key_hit_count * 100.0 / keys.size());

    Result result(keys.size());
    if (miss_keys.empty()) {
        return result;
    }

    // for cache miss keys, get from storage backend, and put into cache
    const std::vector<std::string> property_names = {PROPERTY_URI};
    PropertyMapVector maps;
    int32_t error_count = 0;
    int32_t not_exist_key_count = 0;
    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->Get(miss_keys, property_names, maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    size_t io_data_size = 0;
    for (int32_t i = 0; i < miss_keys.size(); ++i) {
        int32_t index = miss_indexs[i];
        out_uris[index] = std::move(maps[i][PROPERTY_URI]);
        io_data_size += out_uris[index].size();
        if (out_uris[index].empty()) {
            error_codes[i] = EC_NOENT;
            ++not_exist_key_count;
        }
        if (error_codes[i] == EC_OK) {
            auto ec = cache_->Put(miss_keys[i], out_uris[index]);
            PREFIX_INDEXER_LOG(DEBUG, "meta indexer put cache, key[%ld] ec[%d]", miss_keys[i], ec);
        } else {
            if (error_codes[i] != EC_NOENT) {
                PREFIX_INDEXER_LOG(ERROR, "meta indexer get failed, key[%ld] ec[%d]", miss_keys[i], error_codes[i]);
            }
            result.error_codes[index] = error_codes[i];
            ++error_count;
        }
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, io_data_size, io_data_size);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_not_exist_key_count, not_exist_key_count);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result
MetaIndexer::DoGetWithoutCache(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept {
    // for cache miss keys, get from storage backend, and put into cache
    const auto &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    const std::vector<std::string> property_names = {PROPERTY_URI};
    PropertyMapVector maps;
    int32_t error_count = 0;
    int32_t not_exist_key_count = 0;
    Result result(keys.size());
    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->Get(keys, property_names, maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    size_t io_data_size = 0;
    for (int32_t i = 0; i < keys.size(); ++i) {
        out_uris[i] = std::move(maps[i][PROPERTY_URI]);
        io_data_size += out_uris[i].size();
        if (out_uris[i].empty()) {
            error_codes[i] = EC_NOENT;
            ++not_exist_key_count;
        }
        if (error_codes[i] != EC_OK) {
            if (error_codes[i] != EC_NOENT) {
                PREFIX_INDEXER_LOG(ERROR, "meta indexer get failed, key[%ld] ec[%d]", keys[i], error_codes[i]);
            }
            result.error_codes[i] = error_codes[i];
            ++error_count;
        }
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, io_data_size, io_data_size);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_not_exist_key_count, not_exist_key_count);
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
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
        } else if (error_codes[i] == EC_OK && op_name != kGetMetaOperation && op_name != kExistMetaOperation &&
                   cache_) {
            // todo: need to update cache after delete?
            cache_->Delete(keys[index]);
            PREFIX_INDEXER_LOG(DEBUG, "meta indexer %s delete cache, key[%lu]", op_name.c_str(), keys[index]);
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

MetaIndexer::Result MetaIndexer::Get(RequestContext *request_context,
                                     const KeyVector &keys,
                                     LocationMapVector &out_location_maps,
                                     PropertyMapVector &out_properties) noexcept {
    if (keys.empty()) {
        out_location_maps.clear();
        out_properties.clear();
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();

    out_location_maps.assign(keys.size(), LocationMap{});
    out_properties.assign(keys.size(), PropertyMap{});

    FieldMapVec field_maps;
    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->GetAllFields(keys, field_maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);

    Result result(keys.size());
    int32_t error_count = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (error_codes[i] != EC_OK) {
            if (error_codes[i] != EC_NOENT) {
                PREFIX_INDEXER_LOG(ERROR, "meta indexer get failed, key[%ld] ec[%d]", keys[i], error_codes[i]);
            }
            result.error_codes[i] = error_codes[i];
            ++error_count;
            continue;
        }
        // Split each field into either a CacheLocation (LOCATION_PREFIX) or a
        // user property (skip internal "__"-prefixed fields entirely).
        for (auto &kv : field_maps[i]) {
            if (kv.first.find(LOCATION_PREFIX, 0) == 0) {
                // Empty value = tombstoned location, skip.
                if (kv.second.empty()) {
                    continue;
                }
                CacheLocation location;
                if (!location.FromJsonString(kv.second)) {
                    PREFIX_INDEXER_LOG(WARN,
                                       "deserialize CacheLocation failed, key[%ld] field[%s]",
                                       keys[i],
                                       kv.first.c_str());
                    continue;
                }
                out_location_maps[i].emplace(location.id(), std::move(location));
            } else {
                out_properties[i].emplace(kv.first, std::move(kv.second));
            }
        }
    }
    ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::GetLocations(RequestContext *request_context,
                                              const KeyVector &keys,
                                              LocationMapVector &out_location_maps) noexcept {
    if (keys.empty()) {
        out_location_maps.clear();
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();

    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = backend_manager_->GetLocations(keys, out_location_maps);
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
    auto per_location_ecs = backend_manager_->GetLocations(keys, location_ids, out_locations);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);

    LocationResult result(location_ids);
    result.per_location_error_codes = std::move(per_location_ecs);

    // Aggregate to a single top-level ec: EC_OK if every slot is OK, EC_ERROR
    // if every slot failed, otherwise EC_PARTIAL_OK. EC_NOENT slots are
    // counted as failures so callers see partial success when only some
    // locations were missing.
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

// ============================================================================
// ReadModifyWrite (block-level) implementations.
//
// Both overloads share the same overall flow, mirroring the original
// MetaIndexer::ReadModifyWrite (now removed) and routed through
// backend_manager_ (dual persistent + local backend):
//
//   1. Partition keys into BatchMetaData and acquire shard locks per batch.
//   2. GetAllFields(batch) once to fetch every key's current state.
//   3. Reconstruct (LocationMap or LocationIdVector, PropertyMap) from the
//      field map and invoke the modifier under the shard lock.
//   4. Collect upsert / delete sub-batches across the batch and dispatch them
//      via backend_manager_->Upsert / Delete.
//   5. Adjust key_count_ and metrics accordingly.
//
// The two overloads only differ in step 3 (what the modifier sees) and in
// how the to-be-written LocationMap is sourced. To keep the two functions
// readable we share the per-key (field_map -> existing locations) decoding
// helper at file scope and inline the rest of the flow in each overload.
// ============================================================================

namespace {

// Split a raw field map (as returned by GetAllFields) into:
//   - existing_locations: deserialized CacheLocation per location_id (only
//     populated when `decode_locations` is true).
//   - existing_location_ids: list of location_id portions of every "__loc__"
//     field that still has a non-empty value.
//   - existing_properties: business property map (every internal "__"-prefixed
//     field is dropped so the modifier only sees user properties). Currently
//     unused by the public modifier signatures but kept here as a hook for
//     callers that want to surface previous business properties.
void SplitFieldMap(const FieldMap &field_map,
                   bool decode_locations,
                   LocationMap &existing_locations,
                   LocationIdVector &existing_location_ids,
                   PropertyMap &existing_properties) {
    for (const auto &kv : field_map) {
        if (kv.first.rfind(LOCATION_PREFIX, 0) == 0) {
            // Empty value means a previously deleted (tombstoned) location;
            // skip it so the modifier never sees zombies.
            if (kv.second.empty()) {
                continue;
            }
            const std::string loc_id = kv.first.substr(LOCATION_PREFIX.size());
            existing_location_ids.emplace_back(loc_id);
            if (decode_locations) {
                CacheLocation loc;
                if (loc.FromJsonString(kv.second)) {
                    if (loc.id().empty()) {
                        loc.set_id(loc_id);
                    }
                    existing_locations.emplace(loc_id, std::move(loc));
                }
            }
        } else if (kv.first.rfind(PROPERTY_INNER_PREFIX, 0) != 0) {
            existing_properties.emplace(kv.first, kv.second);
        }
    }
}

} // namespace

MetaIndexer::Result MetaIndexer::ReadModifyWriteBlock(RequestContext *request_context,
                                                 const KeyVector &keys,
                                                 const BlockModifierFunc &modifier) noexcept {
    if (keys.empty()) {
        return Result(EC_OK);
    }
    const auto &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());

    LocationMapVector empty_locations;
    PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    Result result(keys.size());
    int32_t error_count = 0;
    int64_t get_io_time_us = 0;
    int64_t upsert_io_time_us = 0;
    int64_t delete_io_time_us = 0;
    int64_t put_key_count = 0;
    int64_t update_key_count = 0;
    int64_t delete_key_count = 0;

    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);

        // 1. Get
        FieldMapVec batch_field_maps;
        int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
        // tianran: todo 修改get
        std::vector<ErrorCode> get_ecs = backend_manager_->GetAllFields(batch, batch_field_maps);
        get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get_io_time;
        if (batch_field_maps.size() != batch.batch_keys.size()) {
            for (size_t j = 0; j < batch.batch_keys.size(); ++j) {
                const int32_t idx = batch.batch_indexs[j];
                result.error_codes[idx] = EC_ERROR;
                ++error_count;
            }
            continue;
        }

        // 2. Modify
        BatchMetaData upsert_batch;
        BatchMetaData delete_batch;
        std::vector<int32_t> put_global_indexs;

        for (size_t j = 0; j < batch.batch_keys.size(); ++j) {
            const ErrorCode get_ec = get_ecs[j];
            const int32_t global_idx = batch.batch_indexs[j];
            const KeyType key = batch.batch_keys[j];

            LocationMap existing_locations;
            LocationIdVector existing_ids;
            PropertyMap existing_properties;
            SplitFieldMap(
                batch_field_maps[j], /*decode_locations=*/true, existing_locations, existing_ids, existing_properties);

            PropertyMap upsert_property_map;
            // Hand the modifier the existing locations as its working buffer;
            // on MA_OK whatever it leaves behind becomes the upsert payload.
            auto [action, modifier_ec] =
                modifier(existing_locations, get_ec, static_cast<size_t>(global_idx), upsert_property_map);

            if (action == MA_FAIL || action == MA_SKIP) {
                if (modifier_ec != EC_OK) {
                    result.error_codes[global_idx] = modifier_ec;
                    ++error_count;
                }
                continue;
            }
            if (action == MA_DELETE && modifier_ec == EC_OK) {
                delete_batch.batch_keys.emplace_back(key);
                delete_batch.batch_indexs.emplace_back(global_idx);
                continue;
            }
            if (action == MA_OK) {
                if (get_ec != EC_OK && get_ec != EC_NOENT) {
                    result.error_codes[global_idx] = get_ec;
                    ++error_count;
                    continue;
                }
                upsert_batch.batch_keys.emplace_back(key);
                upsert_batch.batch_indexs.emplace_back(global_idx);
                upsert_batch.batch_locations.emplace_back(std::move(existing_locations));
                upsert_batch.batch_properties.emplace_back(std::move(upsert_property_map));
                if (get_ec == EC_NOENT) {
                    put_global_indexs.emplace_back(global_idx);
                }
                continue;
            }
            // Unknown action: surface modifier_ec if any.
            if (modifier_ec != EC_OK) {
                result.error_codes[global_idx] = modifier_ec;
                ++error_count;
            }
        }

        // 3. Upsert
        int32_t put_success_count = 0;
        if (!upsert_batch.batch_keys.empty()) {
            put_key_count += put_global_indexs.size();
            update_key_count += upsert_batch.batch_keys.size() - put_global_indexs.size();

            std::vector<ErrorCode> upsert_ecs;
            if (put_global_indexs.size() + GetKeyCount() > max_key_count_) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "ReadModifyWrite put keys count[%lu] + current key count[%lu] > max key count[%lu]",
                                   put_global_indexs.size(),
                                   GetKeyCount(),
                                   max_key_count_);
                upsert_ecs.assign(upsert_batch.batch_keys.size(), EC_NOSPC);
            } else {
                int64_t begin_upsert_io_time = TimestampUtil::GetCurrentTimeUs();
                upsert_ecs = backend_manager_->Upsert(upsert_batch);
                upsert_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_upsert_io_time;
            }
            int32_t upsert_error_count = ProcessErrorCodes(
                trace_id, upsert_ecs, upsert_global_indexs, keys, kRmwUpsertMetaOperation, result);
            if (upsert_error_count == 0) {
                put_success_count = put_indexs.size();
            } else {
                for (int32_t i = 0; i < put_indexs.size(); ++i) {
                    if (result.error_codes[put_indexs[i]] == EC_OK) {
                        ++put_success_count;
                    }
                }
            }
            error_count += upsert_error_count;
        }

        // 4. Delete
        int32_t delete_success_count = 0;
        if (!delete_batch.batch_keys.empty()) {
            delete_key_count += delete_batch.batch_keys.size();
            int64_t begin_delete_io_time = TimestampUtil::GetCurrentTimeUs();
            std::vector<ErrorCode> delete_ecs = backend_manager_->Delete(delete_batch);
            delete_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_delete_io_time;
            int32_t delete_error_count =
                ProcessErrorCodes(trace_id, delete_ecs, delete_batch.batch_indexs, keys, kRmwDeleteMetaOperation, result);
            error_count += delete_error_count;
            delete_success_count += delete_batch.batch_keys.size() - delete_error_count;
        }

        // 4. Adjust key_count
        AdjustKeyCountMeta(put_success_count - delete_success_count);
    }

    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, get_io_time_us, get_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, upsert_io_time_us, upsert_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, delete_io_time_us, delete_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_update_key_count, update_key_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_put_key_count, put_key_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_delete_key_count, delete_key_count);
    int64_t skip_key_count =
        static_cast<int64_t>(keys.size()) - update_key_count - put_key_count - delete_key_count;
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_skip_key_count, skip_key_count);
    ProcessErrorResult(trace_id, kRmwMetaOperation, error_count, keys.size(), result);
    return result;
}

// tianran todo
MetaIndexer::Result MetaIndexer::ReadModifyWriteBlock(RequestContext *request_context,
                                                 const KeyVector &keys,
                                                 const BlockIdsOnlyModifierFunc &modifier) noexcept {
    if (keys.empty()) {
        return Result(EC_OK);
    }
    const auto &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());

    LocationMapVector empty_locations;
    PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    Result result(keys.size());
    int32_t error_count = 0;
    int64_t get_io_time_us = 0;
    int64_t upsert_io_time_us = 0;
    int64_t delete_io_time_us = 0;
    int64_t put_key_count = 0;
    int64_t update_key_count = 0;
    int64_t delete_key_count = 0;

    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);

        // 1. Get (we still need GetAllFields because the manager exposes no
        //    dedicated "field-names only" primitive yet; the modifier itself only
        //    sees the existing id list, never the deserialized values).
        FieldMapVec batch_field_maps;
        int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
        std::vector<ErrorCode> get_ecs = backend_manager_->GetAllFields(batch, batch_field_maps);
        get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get_io_time;
        if (batch_field_maps.size() != batch.batch_keys.size()) {
            for (size_t j = 0; j < batch.batch_keys.size(); ++j) {
                const int32_t idx = batch.batch_indexs[j];
                result.error_codes[idx] = EC_ERROR;
                ++error_count;
            }
            continue;
        }

        // 2. Modify
        BatchMetaData upsert_batch;
        BatchMetaData delete_batch;
        std::vector<int32_t> put_global_indexs;

        for (size_t j = 0; j < batch.batch_keys.size(); ++j) {
            const ErrorCode get_ec = get_ecs[j];
            const int32_t global_idx = batch.batch_indexs[j];
            const KeyType key = batch.batch_keys[j];

            LocationMap unused_locations;
            LocationIdVector existing_ids;
            PropertyMap existing_properties;
            SplitFieldMap(
                batch_field_maps[j], /*decode_locations=*/false, unused_locations, existing_ids, existing_properties);

            PropertyMap upsert_property_map;
            LocationMap out_new_locations;
            // The lightweight modifier writes any new CacheLocations directly
            // into out_new_locations, which becomes the upsert payload as-is.
            auto [action, modifier_ec] = modifier(
                existing_ids, get_ec, static_cast<size_t>(global_idx), upsert_property_map, out_new_locations);

            if (action == MA_FAIL || action == MA_SKIP) {
                if (modifier_ec != EC_OK) {
                    result.error_codes[global_idx] = modifier_ec;
                    ++error_count;
                }
                continue;
            }
            if (action == MA_DELETE && modifier_ec == EC_OK) {
                delete_batch.batch_keys.emplace_back(key);
                delete_batch.batch_indexs.emplace_back(global_idx);
                continue;
            }
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
                continue;
            }
            // Unknown action: surface modifier_ec if any.
            if (modifier_ec != EC_OK) {
                result.error_codes[global_idx] = modifier_ec;
                ++error_count;
            }
        }

        // 3. Upsert
        int32_t put_success_count = 0;
        if (!upsert_batch.batch_keys.empty()) {
            put_key_count += put_global_indexs.size();
            update_key_count += upsert_batch.batch_keys.size() - put_global_indexs.size();

            std::vector<ErrorCode> upsert_ecs;
            if (put_global_indexs.size() + GetKeyCount() > max_key_count_) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "ReadModifyWrite put keys count[%lu] + current key count[%lu] > max key count[%lu]",
                                   put_global_indexs.size(),
                                   GetKeyCount(),
                                   max_key_count_);
                upsert_ecs.assign(upsert_batch.batch_keys.size(), EC_NOSPC);
            } else {
                int64_t begin_upsert_io_time = TimestampUtil::GetCurrentTimeUs();
                upsert_ecs = backend_manager_->Upsert(upsert_batch);
                upsert_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_upsert_io_time;
            }
            int32_t upsert_error_count = ProcessErrorCodes(
                trace_id, upsert_ecs, upsert_batch.batch_indexs, keys, kRmwUpsertMetaOperation, result);
            if (upsert_error_count == 0) {
                put_success_count = put_global_indexs.size();
            } else {
                for (size_t i = 0; i < put_global_indexs.size(); ++i) {
                    if (result.error_codes[put_global_indexs[i]] == EC_OK) {
                        ++put_success_count;
                    }
                }
            }
            error_count += upsert_error_count;
        }

        // 4. Delete
        int32_t delete_success_count = 0;
        if (!delete_batch.batch_keys.empty()) {
            delete_key_count += delete_batch.batch_keys.size();
            int64_t begin_delete_io_time = TimestampUtil::GetCurrentTimeUs();
            std::vector<ErrorCode> delete_ecs = backend_manager_->Delete(delete_batch);
            delete_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_delete_io_time;
            int32_t delete_error_count =
                ProcessErrorCodes(trace_id, delete_ecs, delete_batch.batch_indexs, keys, kRmwDeleteMetaOperation, result);
            error_count += delete_error_count;
            delete_success_count += delete_batch.batch_keys.size() - delete_error_count;
        }

        // 5. Adjust key_count
        AdjustKeyCountMeta(put_success_count - delete_success_count);
    }

    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, get_io_time_us, get_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, upsert_io_time_us, upsert_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, delete_io_time_us, delete_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_update_key_count, update_key_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_put_key_count, put_key_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_delete_key_count, delete_key_count);
    int64_t skip_key_count =
        static_cast<int64_t>(keys.size()) - update_key_count - put_key_count - delete_key_count;
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_skip_key_count, skip_key_count);
    ProcessErrorResult(trace_id, kRmwMetaOperation, error_count, keys.size(), result);
    return result;
}

// tianran todo
MetaIndexer::LocationResult MetaIndexer::ReadModifyWriteLocation(RequestContext *request_context,
                                                                 const KeyVector &keys,
                                                                 const LocationIdsPerKey &location_ids,
                                                                 const LocationModifierFunc &modifier) noexcept {
    if (keys.empty()) {
        return LocationResult(EC_OK);
    }
    if (keys.size() != location_ids.size()) {
        const auto &trace_id = request_context->trace_id();
        PREFIX_INDEXER_LOG(ERROR,
                           "ReadModifyWriteLocation keys size[%lu] != location_ids size[%lu]",
                           keys.size(),
                           location_ids.size());
        return LocationResult(EC_ERROR);
    }

    const auto &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());

    // Pre-size the result tensor so every (key, location_id) slot has a
    // well-defined ec even when a key is skipped or fails before the modifier
    // is invoked.
    LocationResult result(location_ids);

    int32_t total_targets = 0;
    for (const auto &ids : location_ids) {
        total_targets += static_cast<int32_t>(ids.size());
    }
    if (total_targets == 0) {
        return result;
    }

    int32_t error_count = 0;
    int64_t get_io_time_us = 0;
    int64_t upsert_io_time_us = 0;
    int64_t delete_io_time_us = 0;

    LocationMapVector dummy_locations;
    PropertyMapVector dummy_properties;

    // Drive one single-key BatchMetaData per input key. Different keys
    // typically target different (location_id set, modifier output) tuples,
    // so batching them together would force a join we cannot express through
    // the manager's batch-level field upsert/delete primitives.
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i].empty()) {
            continue;
        }

        KeyVector single_key{keys[i]};
        std::vector<BatchMetaData> single_key_batches = MakeBatches(single_key, dummy_locations, dummy_properties);
        if (single_key_batches.empty()) {
            // Defensive: MakeBatches should always emit one batch when keys
            // is non-empty (Init guarantees mutex_shard_mask_ + 1 > 0).
            result.per_location_error_codes[i].assign(location_ids[i].size(), EC_ERROR);
            error_count += static_cast<int32_t>(location_ids[i].size());
            continue;
        }
        BatchMetaData &batch = single_key_batches.front();
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);

        // 1. Get all fields of this key once; a missing key is OK (the
        //    modifier may decide to create new locations from scratch).
        FieldMapVec key_field_maps;
        int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
        std::vector<ErrorCode> get_ecs = backend_manager_->GetAllFields(batch, key_field_maps);
        get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get_io_time;

        const ErrorCode key_get_ec = get_ecs.empty() ? EC_ERROR : get_ecs.front();
        if (key_get_ec != EC_OK && key_get_ec != EC_NOENT) {
            // Whole-key get failed - we cannot safely run any modifier on it.
            for (size_t j = 0; j < location_ids[i].size(); ++j) {
                result.per_location_error_codes[i][j] = key_get_ec;
                ++error_count;
            }
            if (key_get_ec != EC_NOENT) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "ReadModifyWriteLocation get failed, key[%ld] ec[%d]",
                                   keys[i],
                                   key_get_ec);
            }
            continue;
        }

        // Reconstruct the full LocationMap once so subsequent modifier calls
        // for the same key can see siblings (e.g. for cross-location checks).
        LocationMap existing_locations;
        LocationIdVector existing_ids;
        PropertyMap existing_properties_unused;
        if (!key_field_maps.empty()) {
            SplitFieldMap(key_field_maps.front(),
                          /*decode_locations=*/true,
                          existing_locations,
                          existing_ids,
                          existing_properties_unused);
        }

        // 2. Per-location modifier loop.
        LocationMap upsert_loc_map;
        std::vector<std::string> delete_location_ids;
        // Track which result slot to mark for upsert/delete outcomes after the
        // backend call returns.
        std::vector<size_t> upsert_loc_slots; // index into location_ids[i] for each upsert_loc_map entry
        std::vector<size_t> delete_loc_slots; // index into location_ids[i] for each delete_location_ids entry
        PropertyMap upsert_property_map;
        bool any_upsert_payload_property = false;

        for (size_t j = 0; j < location_ids[i].size(); ++j) {
            const LocationId &loc_id = location_ids[i][j];
            const auto loc_it = existing_locations.find(loc_id);
            const ErrorCode loc_get_ec = (loc_it != existing_locations.end()) ? EC_OK : EC_NOENT;

            // Modifier sees a mutable CacheLocation: either the deserialized
            // existing one or a default-constructed placeholder for creation.
            CacheLocation working_loc;
            if (loc_it != existing_locations.end()) {
                working_loc = loc_it->second;
            }
            if (working_loc.id().empty()) {
                working_loc.set_id(loc_id);
            }

            auto [action, modifier_ec] = modifier(working_loc, loc_get_ec, i, loc_id, upsert_property_map);

            if (action == MA_FAIL || action == MA_SKIP) {
                if (modifier_ec != EC_OK) {
                    result.per_location_error_codes[i][j] = modifier_ec;
                    ++error_count;
                } // else: keep EC_OK default, behave like a no-op success
                continue;
            }
            if (action == MA_DELETE && modifier_ec == EC_OK) {
                if (loc_get_ec == EC_NOENT) {
                    // Nothing to delete; surface as no-op success.
                    continue;
                }
                delete_location_ids.emplace_back(loc_id);
                delete_loc_slots.emplace_back(j);
                continue;
            }
            if (action == MA_OK) {
                // Make sure the serialised payload carries the canonical id so
                // later GetLocations calls observe a consistent view.
                if (working_loc.id().empty()) {
                    working_loc.set_id(loc_id);
                }
                upsert_loc_map.emplace(loc_id, std::move(working_loc));
                upsert_loc_slots.emplace_back(j);
                continue;
            }
            // Unknown action: treat as failure with whatever ec the modifier set.
            if (modifier_ec != EC_OK) {
                result.per_location_error_codes[i][j] = modifier_ec;
                ++error_count;
            }
        }

        any_upsert_payload_property = !upsert_property_map.empty();

        // 3. Upsert all MA_OK locations of this key in a single backend call.
        //    The manager's Upsert serializes upsert_batch.batch_locations into
        //    the per-key PropertyMap and calls UpdateFields/HSET style under
        //    the hood, so existing untouched __loc__{id} fields are preserved.
        bool key_was_brand_new = (key_get_ec == EC_NOENT);
        int32_t put_success_count = 0;
        int32_t delete_success_count = 0;
        if (!upsert_loc_map.empty() || any_upsert_payload_property) {
            // Capacity guard mirrors the block-level RMW path: a brand-new key
            // counts towards key_count_ only if upsert succeeds.
            int64_t additional_keys = key_was_brand_new ? 1 : 0;
            BatchMetaData upsert_batch;
            upsert_batch.batch_shard_indexs = batch.batch_shard_indexs;
            upsert_batch.batch_keys = {keys[i]};
            upsert_batch.batch_indexs = {static_cast<int32_t>(i)};
            upsert_batch.batch_locations = {std::move(upsert_loc_map)};
            upsert_batch.batch_properties = {std::move(upsert_property_map)};

            std::vector<ErrorCode> upsert_ecs;
            if (additional_keys + GetKeyCount() > max_key_count_) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "ReadModifyWriteLocation put key count[%ld] + current key count[%lu] > max key "
                                   "count[%lu]",
                                   additional_keys,
                                   GetKeyCount(),
                                   max_key_count_);
                upsert_ecs.assign(1, EC_NOSPC);
            } else {
                int64_t begin_upsert_io_time = TimestampUtil::GetCurrentTimeUs();
                upsert_ecs = backend_manager_->Upsert(upsert_batch);
                upsert_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_upsert_io_time;
            }
            const ErrorCode upsert_ec = upsert_ecs.empty() ? EC_ERROR : upsert_ecs.front();
            if (upsert_ec != EC_OK) {
                for (const size_t slot : upsert_loc_slots) {
                    result.per_location_error_codes[i][slot] = upsert_ec;
                    ++error_count;
                }
                if (upsert_ec != EC_NOENT) {
                    PREFIX_INDEXER_LOG(ERROR,
                                       "ReadModifyWriteLocation upsert failed, key[%ld] ec[%d]",
                                       keys[i],
                                       upsert_ec);
                }
            } else if (key_was_brand_new && !upsert_loc_slots.empty()) {
                put_success_count = 1;
                key_was_brand_new = false; // accounted for; subsequent paths must not double-count
            }
        }

        // 4. Field-level delete for all MA_DELETE locations.
        if (!delete_location_ids.empty()) {
            int64_t begin_delete_io_time = TimestampUtil::GetCurrentTimeUs();
            std::vector<ErrorCode> delete_ecs = backend_manager_->Delete(batch, delete_location_ids);
            delete_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_delete_io_time;
            const ErrorCode delete_ec = delete_ecs.empty() ? EC_ERROR : delete_ecs.front();
            if (delete_ec != EC_OK) {
                for (const size_t slot : delete_loc_slots) {
                    result.per_location_error_codes[i][slot] = delete_ec;
                    if (delete_ec != EC_NOENT) {
                        ++error_count;
                    }
                }
                if (delete_ec != EC_NOENT) {
                    PREFIX_INDEXER_LOG(ERROR,
                                       "ReadModifyWriteLocation delete failed, key[%ld] ec[%d]",
                                       keys[i],
                                       delete_ec);
                }
            }
        }

        // 5. Reclaim the whole block_key when no live location remains. We
        //    re-read every field after the upsert+delete pair so the residual
        //    check observes both effects in order.
        if (!delete_location_ids.empty()) {
            FieldMapVec residual_field_maps;
            std::vector<ErrorCode> residual_ecs = backend_manager_->GetAllFields(batch, residual_field_maps);
            const ErrorCode residual_ec = residual_ecs.empty() ? EC_ERROR : residual_ecs.front();
            if (residual_ec == EC_OK && !residual_field_maps.empty()) {
                bool any_live_location = false;
                for (const auto &kv : residual_field_maps.front()) {
                    if (kv.first.rfind(LOCATION_PREFIX, 0) == 0 && !kv.second.empty()) {
                        any_live_location = true;
                        break;
                    }
                }
                if (!any_live_location) {
                    std::vector<ErrorCode> whole_delete_ecs = backend_manager_->Delete(batch);
                    const ErrorCode whole_ec = whole_delete_ecs.empty() ? EC_ERROR : whole_delete_ecs.front();
                    if (whole_ec == EC_OK) {
                        delete_success_count = 1;
                        if (cache_) {
                            cache_->Delete(keys[i]);
                        }
                    } else if (whole_ec != EC_NOENT) {
                        PREFIX_INDEXER_LOG(WARN,
                                           "ReadModifyWriteLocation delete empty block_key failed, key[%ld] ec[%d]",
                                           keys[i],
                                           whole_ec);
                    }
                }
            }
        }

        // 6. Adjust key_count
        AdjustKeyCountMeta(put_success_count - delete_success_count);
    }

    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, get_io_time_us, get_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, upsert_io_time_us, upsert_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, delete_io_time_us, delete_io_time_us);

    if (error_count == total_targets) {
        result.ec = EC_ERROR;
        PREFIX_INDEXER_LOG(DEBUG, "all locations rmw failed, target count[%d]", total_targets);
    } else if (error_count > 0) {
        result.ec = EC_PARTIAL_OK;
        PREFIX_INDEXER_LOG(DEBUG,
                           "partial locations rmw failed, target count[%d] failed count[%d]",
                           total_targets,
                           error_count);
    }
    return result;
}

// tianran todo
MetaIndexer::LocationResult MetaIndexer::DeleteLocations(RequestContext *request_context,
                                                         const KeyVector &keys,
                                                         const LocationIdsPerKey &location_ids,
                                                         LocationsPerKey &out_deleted_locations) noexcept {
    if (keys.empty()) {
        return LocationResult(EC_OK);
    }
    if (keys.size() != location_ids.size()) {
        const auto &trace_id = request_context->trace_id();
        PREFIX_INDEXER_LOG(ERROR,
                           "DeleteLocations keys size[%lu] != location_ids size[%lu]",
                           keys.size(),
                           location_ids.size());
        return LocationResult(EC_ERROR);
    }

    const auto &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());

    // Initialise per-(key, location_id) result/output containers up-front so
    // every slot has a well-defined value even if a key is skipped below.
    LocationResult result(location_ids);
    out_deleted_locations.assign(location_ids.size(), CacheLocationVector{});
    for (size_t i = 0; i < location_ids.size(); ++i) {
        out_deleted_locations[i].resize(location_ids[i].size());
    }

    int32_t error_count = 0;
    int32_t whole_key_deletes = 0;
    int64_t delete_io_time_us = 0;

    // The dual-backend manager's location-field Delete operates on a single
    // (batch, location_ids) pair: every key in the batch loses the same set of
    // fields. Different keys here may target different location_ids, so the
    // simplest correct strategy is to drive one single-key BatchMetaData per
    // input key. The shard-level locking surface stays identical to other
    // write paths in this class.
    LocationMapVector dummy_locations;
    PropertyMapVector dummy_properties;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i].empty()) {
            continue;
        }

        KeyVector single_key{keys[i]};
        std::vector<BatchMetaData> single_key_batches = MakeBatches(single_key, dummy_locations, dummy_properties);
        if (single_key_batches.empty()) {
            // GetShardIndex would only ever skip a key if the shard count
            // (mutex_shard_mask_ + 1) is zero, which Init() already rejects,
            // so this is purely defensive.
            result.per_location_error_codes[i].assign(location_ids[i].size(), EC_ERROR);
            error_count += static_cast<int32_t>(location_ids[i].size());
            continue;
        }
        BatchMetaData &batch = single_key_batches.front();
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);

        // Stage 1: snapshot the targeted location fields so the caller can
        // observe what was removed (storage usage accounting, audit logs, ...).
        std::vector<std::string> location_field_names;
        location_field_names.reserve(location_ids[i].size());
        for (const auto &loc_id : location_ids[i]) {
            location_field_names.emplace_back(LOCATION_PREFIX + loc_id);
        }
        FieldMapVec snapshot_field_maps;
        std::vector<ErrorCode> get_ecs = backend_manager_->Get(batch, location_field_names, snapshot_field_maps);
        const ErrorCode get_ec = get_ecs.empty() ? EC_ERROR : get_ecs.front();
        const FieldMap &snapshot_map =
            (!snapshot_field_maps.empty()) ? snapshot_field_maps.front() : FieldMap{};
        for (size_t j = 0; j < location_ids[i].size(); ++j) {
            CacheLocation &slot = out_deleted_locations[i][j];
            slot.set_id(location_ids[i][j]);
            const auto field_it = snapshot_map.find(location_field_names[j]);
            if (get_ec == EC_OK && field_it != snapshot_map.end() && !field_it->second.empty()) {
                // Best-effort deserialise; if the stored payload is malformed we
                // still return the id so the caller can audit the deletion.
                (void)slot.FromJsonString(field_it->second);
            }
        }

        // Stage 2: actually remove the location fields.
        int64_t begin_delete_io_time = TimestampUtil::GetCurrentTimeUs();
        std::vector<ErrorCode> delete_ecs = backend_manager_->Delete(batch, location_ids[i]);
        delete_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_delete_io_time;
        const ErrorCode delete_ec = delete_ecs.empty() ? EC_ERROR : delete_ecs.front();
        for (size_t j = 0; j < location_ids[i].size(); ++j) {
            result.per_location_error_codes[i][j] = delete_ec;
            if (delete_ec != EC_OK) {
                ++error_count;
            }
        }
        if (delete_ec != EC_OK) {
            if (delete_ec != EC_NOENT) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "meta indexer delete locations failed, key[%ld] ec[%d]",
                                   keys[i],
                                   delete_ec);
            }
            continue;
        }

        // Stage 3: reclaim the whole block_key when no live location remains.
        // We re-read every field and drop the key only if every non-inner
        // field is empty (location fields tombstoned by the field-delete) and
        // there is no location field with a non-empty payload left.
        FieldMapVec residual_field_maps;
        std::vector<ErrorCode> residual_ecs = backend_manager_->GetAllFields(batch, residual_field_maps);
        const ErrorCode residual_ec = residual_ecs.empty() ? EC_ERROR : residual_ecs.front();
        if (residual_ec != EC_OK || residual_field_maps.empty()) {
            continue;
        }
        bool any_live_location = false;
        for (const auto &kv : residual_field_maps.front()) {
            if (kv.first.rfind(LOCATION_PREFIX, 0) == 0 && !kv.second.empty()) {
                any_live_location = true;
                break;
            }
        }
        if (!any_live_location) {
            std::vector<ErrorCode> whole_delete_ecs = backend_manager_->Delete(batch);
            const ErrorCode whole_ec = whole_delete_ecs.empty() ? EC_ERROR : whole_delete_ecs.front();
            if (whole_ec == EC_OK) {
                ++whole_key_deletes;
                if (cache_) {
                    cache_->Delete(keys[i]);
                }
            } else if (whole_ec != EC_NOENT) {
                PREFIX_INDEXER_LOG(WARN,
                                   "meta indexer delete empty block_key failed, key[%ld] ec[%d]",
                                   keys[i],
                                   whole_ec);
            }
        }
    }

    AdjustKeyCountMeta(-whole_key_deletes);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, delete_io_time_us, delete_io_time_us);

    // Aggregate the per-(key, location_id) errors into the request-level ec.
    int64_t total_targets = 0;
    for (const auto &ids : location_ids) {
        total_targets += static_cast<int64_t>(ids.size());
    }
    if (total_targets > 0) {
        if (error_count == total_targets) {
            result.ec = EC_ERROR;
            PREFIX_INDEXER_LOG(DEBUG, "all locations delete failed, target count[%ld]", total_targets);
        } else if (error_count > 0) {
            result.ec = EC_PARTIAL_OK;
            PREFIX_INDEXER_LOG(DEBUG,
                               "partial locations delete failed, target count[%ld] failed count[%d]",
                               total_targets,
                               error_count);
        }
    }
    return result;
}

} // namespace kv_cache_manager
