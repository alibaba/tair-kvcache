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

    static LocationIdsPerKey empty_location_ids;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_location_ids, location_maps, properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t put_io_time_us = 0;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);
        int64_t begin_put_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = backend_manager_->Put(request_context, batch);
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
    static LocationIdsPerKey empty_location_ids;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_location_ids, location_maps, properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t update_io_time_us = 0;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);
        int64_t begin_update_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = backend_manager_->UpdateFields(request_context, batch);
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
    static LocationIdsPerKey empty_location_ids;
    static LocationMapVector empty_locations;
    static PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_location_ids, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());
    Result result(keys.size());
    int32_t error_count = 0;
    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);
        auto error_codes = backend_manager_->Delete(batch.batch_keys);
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

// MetaIndexer::Result
// MetaIndexer::Get(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept {
//     out_uris.resize(keys.size());
//     auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
//     KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
//     if (cache_) {
//         return DoGetWithCache(request_context, keys, out_uris);
//     } else {
//         return DoGetWithoutCache(request_context, keys, out_uris);
//     }
// }

// MetaIndexer::Result MetaIndexer::Get(RequestContext *request_context,
//                                      const KeyVector &keys,
//                                      UriVector &out_uris,
//                                      PropertyMapVector &out_properties) noexcept {
//     if (keys.size() == 0) {
//         return Result(EC_OK);
//     }
//     auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
//     KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
//     const auto &trace_id = request_context->trace_id();
//     out_uris.reserve(keys.size());
//     out_properties.reserve(keys.size());
//     PropertyMapVector maps;
//     Result result(keys.size());
//     int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
//     auto error_codes = backend_manager_->GetAllFields(keys, maps);
//     KVCM_METRICS_COLLECTOR_SET_METRICS(
//         service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() -
//         begin_get_io_time);
//     int32_t error_count = 0;
//     for (int32_t i = 0; i < keys.size(); ++i) {
//         auto &map = maps[i];
//         out_uris.emplace_back(std::move(map[PROPERTY_URI]));
//         for (auto it = map.begin(); it != map.end();) {
//             if (it->first.rfind(PROPERTY_INNER_PREFIX, 0) == 0) {
//                 it = map.erase(it);
//             } else {
//                 ++it;
//             }
//         }
//         out_properties.emplace_back(std::move(map));
//         if (error_codes[i] != EC_OK) {
//             if (error_codes[i] != EC_NOENT) {
//                 PREFIX_INDEXER_LOG(ERROR, "meta indexer get failed, key[%ld] ec[%d]", keys[i], error_codes[i]);
//             }
//             result.error_codes[i] = error_codes[i];
//             ++error_count;
//         }
//     }
//     ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
//     return result;
// }

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

std::vector<BatchMetaData> MetaIndexer::MakeBatches(const KeyVector &keys,
                                                    const LocationIdsPerKey &location_ids,
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

// MetaIndexer::Result
// MetaIndexer::DoGetWithCache(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept {
//     // get from cache first
//     const auto &trace_id = request_context->trace_id();
//     KeyVector miss_keys;
//     std::vector<int32_t> miss_indexs;
//     for (int32_t i = 0; i < keys.size(); ++i) {
//         std::string uri;
//         auto ec = cache_->Get(keys[i], &uri);
//         if (ec == EC_OK) {
//             out_uris[i] = std::move(uri);
//         } else {
//             miss_keys.push_back(keys[i]);
//             miss_indexs.push_back(i);
//         }
//     }
//     assert(miss_keys.size() == miss_indexs.size());
//     auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
//     size_t cache_key_hit_count = keys.size() - miss_keys.size();
//     KVCM_METRICS_COLLECTOR_SET_METRICS(
//         service_metrics_collector, meta_indexer, search_cache_miss_count, miss_keys.size());
//     KVCM_METRICS_COLLECTOR_SET_METRICS(
//         service_metrics_collector, meta_indexer, search_cache_hit_count, cache_key_hit_count);
//     KVCM_METRICS_COLLECTOR_SET_METRICS(
//         service_metrics_collector, meta_indexer, search_cache_hit_ratio, cache_key_hit_count * 100.0 / keys.size());

//     Result result(keys.size());
//     if (miss_keys.empty()) {
//         return result;
//     }

//     // for cache miss keys, get from storage backend, and put into cache
//     const std::vector<std::string> property_names = {PROPERTY_URI};
//     PropertyMapVector maps;
//     int32_t error_count = 0;
//     int32_t not_exist_key_count = 0;
//     int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
//     auto error_codes = backend_manager_->Get(miss_keys, property_names, maps);
//     KVCM_METRICS_COLLECTOR_SET_METRICS(
//         service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() -
//         begin_get_io_time);
//     size_t io_data_size = 0;
//     for (int32_t i = 0; i < miss_keys.size(); ++i) {
//         int32_t index = miss_indexs[i];
//         out_uris[index] = std::move(maps[i][PROPERTY_URI]);
//         io_data_size += out_uris[index].size();
//         if (out_uris[index].empty()) {
//             error_codes[i] = EC_NOENT;
//             ++not_exist_key_count;
//         }
//         if (error_codes[i] == EC_OK) {
//             auto ec = cache_->Put(miss_keys[i], out_uris[index]);
//             PREFIX_INDEXER_LOG(DEBUG, "meta indexer put cache, key[%ld] ec[%d]", miss_keys[i], ec);
//         } else {
//             if (error_codes[i] != EC_NOENT) {
//                 PREFIX_INDEXER_LOG(ERROR, "meta indexer get failed, key[%ld] ec[%d]", miss_keys[i], error_codes[i]);
//             }
//             result.error_codes[index] = error_codes[i];
//             ++error_count;
//         }
//     }
//     KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, io_data_size, io_data_size);
//     KVCM_METRICS_COLLECTOR_SET_METRICS(
//         service_metrics_collector, meta_indexer, get_not_exist_key_count, not_exist_key_count);
//     ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
//     return result;
// }

// MetaIndexer::Result
// MetaIndexer::DoGetWithoutCache(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept
// {
//     // for cache miss keys, get from storage backend, and put into cache
//     const auto &trace_id = request_context->trace_id();
//     auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
//     const std::vector<std::string> property_names = {PROPERTY_URI};
//     PropertyMapVector maps;
//     int32_t error_count = 0;
//     int32_t not_exist_key_count = 0;
//     Result result(keys.size());
//     int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
//     auto error_codes = backend_manager_->Get(keys, property_names, maps);
//     KVCM_METRICS_COLLECTOR_SET_METRICS(
//         service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() -
//         begin_get_io_time);
//     size_t io_data_size = 0;
//     for (int32_t i = 0; i < keys.size(); ++i) {
//         out_uris[i] = std::move(maps[i][PROPERTY_URI]);
//         io_data_size += out_uris[i].size();
//         if (out_uris[i].empty()) {
//             error_codes[i] = EC_NOENT;
//             ++not_exist_key_count;
//         }
//         if (error_codes[i] != EC_OK) {
//             if (error_codes[i] != EC_NOENT) {
//                 PREFIX_INDEXER_LOG(ERROR, "meta indexer get failed, key[%ld] ec[%d]", keys[i], error_codes[i]);
//             }
//             result.error_codes[i] = error_codes[i];
//             ++error_count;
//         }
//     }
//     KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, io_data_size, io_data_size);
//     KVCM_METRICS_COLLECTOR_SET_METRICS(
//         service_metrics_collector, meta_indexer, get_not_exist_key_count, not_exist_key_count);
//     ProcessErrorResult(trace_id, kGetMetaOperation, error_count, keys.size(), result);
//     return result;
// }

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
                    PREFIX_INDEXER_LOG(
                        WARN, "deserialize CacheLocation failed, key[%ld] field[%s]", keys[i], kv.first.c_str());
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

namespace {

// Pull the location ids of a key out of a raw field map (as returned by
// GetAllFields). Tombstoned (empty-value) location fields and any internal
// "__"-prefixed property are skipped so the modifier never sees zombies.
LocationIdVector ExtractLocationIds(const FieldMap &field_map) {
    LocationIdVector ids;
    ids.reserve(field_map.size());
    for (const auto &kv : field_map) {
        if (kv.first.rfind(LOCATION_PREFIX, 0) != 0) {
            continue;
        }
        if (kv.second.empty()) {
            continue;
        }
        ids.emplace_back(kv.first.substr(LOCATION_PREFIX.size()));
    }
    return ids;
}

} // namespace

std::pair<int32_t, int32_t> MetaIndexer::ExecuteRmwUpsert(RequestContext *request_context,
                                                          BatchMetaData &upsert_batch,
                                                          const std::vector<int32_t> &put_global_indexs,
                                                          const KeyVector &all_keys,
                                                          RmwStats &stats,
                                                          Result &result) noexcept {
    if (upsert_batch.batch_keys.empty()) {
        return {0, 0};
    }

    const auto &trace_id = request_context->trace_id();
    stats.put_key_count += static_cast<int64_t>(put_global_indexs.size());
    stats.update_key_count += static_cast<int64_t>(upsert_batch.batch_keys.size() - put_global_indexs.size());

    // Capacity guard: brand-new keys count against max_key_count_. We short-
    // circuit the backend call if introducing put_global_indexs.size() new
    // keys would breach the limit; the per-key result is EC_NOSPC for every
    // key in the upsert batch (existing keys included, since we cannot
    // partially apply a single Upsert and still maintain consistency).
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
    }

    const int32_t error_count =
        ProcessErrorCodes(trace_id, upsert_ecs, upsert_batch.batch_indexs, all_keys, kRmwUpsertMetaOperation, result);

    // Count brand-new keys that actually succeeded; everything in
    // put_global_indexs ended up in upsert_batch and shares a single Upsert
    // result, so the fast path is "no errors -> all new keys succeeded".
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
        delete_ecs = backend_manager_->Delete(delete_batch.batch_keys);
    } else {
        delete_ecs =
            backend_manager_->Delete(delete_batch.batch_keys, delete_batch.batch_location_ids, reclaimed_count);
    }
    stats.delete_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin;

    const int32_t error_count =
        ProcessErrorCodes(trace_id, delete_ecs, delete_batch.batch_indexs, all_keys, kRmwDeleteMetaOperation, result);
    // For whole-key deletes, success count = keys - errors.
    // For location deletes, reclaimed_count reflects empty blocks auto-removed.
    const int32_t delete_success_count =
        delete_batch.batch_location_ids.empty() ? delete_batch.batch_keys.size() - error_count : reclaimed_count;
    return {error_count, delete_success_count};
}

void MetaIndexer::EmitBlockRmwMetrics(MetricsCollector *metrics_collector,
                                      const RmwStats &stats,
                                      size_t total_key_count) const noexcept {
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(metrics_collector);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, get_io_time_us, stats.get_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, upsert_io_time_us, stats.upsert_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, delete_io_time_us, stats.delete_io_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_update_key_count, stats.update_key_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_put_key_count, stats.put_key_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_delete_key_count, stats.delete_key_count);
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

    static LocationIdsPerKey empty_location_ids;
    static LocationMapVector empty_locations;
    static PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, empty_location_ids, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    Result result(keys.size());
    int32_t error_count = 0;
    RmwStats stats;

    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);

        const auto &batch_keys = batch.batch_keys;
        // 1. Read each key's current state. We only need the existing location
        //    id list; values are not deserialized. The backend exposes no
        //    "field-names only" primitive, so we still pay one GetAllFields.
        FieldMapVec batch_field_maps;
        const int64_t begin_get = TimestampUtil::GetCurrentTimeUs();
        // TODO:(tianran) 实现一个GetLocationId，无需反序列化location
        std::vector<ErrorCode> get_ecs = backend_manager_->GetAllFields(batch_keys, batch_field_maps);
        stats.get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get;
        if (batch_field_maps.size() != batch_keys.size()) {
            // Backend protocol violation: surface a per-key error and skip.
            for (const int32_t idx : batch.batch_indexs) {
                result.error_codes[idx] = EC_ERROR;
                ++error_count;
            }
            continue;
        }

        // 2. Modify -> bucket each key into upsert_batch / delete_batch.
        BatchMetaData upsert_batch;
        BatchMetaData delete_batch;
        std::vector<int32_t> put_global_indexs; // brand-new keys (subset of upsert_batch)

        for (size_t j = 0; j < batch_keys.size(); ++j) {
            const ErrorCode get_ec = get_ecs[j];
            const int32_t global_idx = batch.batch_indexs[j];
            const KeyType key = batch_keys[j];

            const LocationIdVector existing_ids = ExtractLocationIds(batch_field_maps[j]);

            PropertyMap upsert_property_map;
            LocationMap out_new_locations;
            const auto [action, modifier_ec] =
                modifier(existing_ids, get_ec, static_cast<size_t>(global_idx), upsert_property_map, out_new_locations);

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
        const auto [upsert_errs, put_success_count] =
            ExecuteRmwUpsert(request_context, upsert_batch, put_global_indexs, keys, stats, result);
        const auto [delete_errs, delete_success_count] = ExecuteRmwDelete(trace_id, delete_batch, keys, stats, result);
        error_count += upsert_errs + delete_errs;

        // 4. Adjust key_count for net key population change of this batch.
        AdjustKeyCountMeta(put_success_count - delete_success_count);
    }

    EmitBlockRmwMetrics(request_context->metrics_collector(), stats, keys.size());
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
    // The contract is strict per-key alignment: location_ids[i] is the set of
    // (key=keys[i]) locations to RMW. A size mismatch means the caller built
    // the wrong fan-out and we cannot recover safely.
    if (keys.size() != location_ids.size()) {
        PREFIX_INDEXER_LOG(ERROR,
                           "ReadModifyWriteLocation keys size[%lu] != location_ids size[%lu]",
                           keys.size(),
                           location_ids.size());
        return LocationResult(EC_BADARGS);
    }

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());

    // Pre-size the result tensor so every (key, location_id) slot has a
    // well-defined ec even when a key is skipped or fails before the modifier
    // is invoked. Keys with an empty location_ids[i] yield an empty inner
    // vector here, which the caller can detect as "no targets for this key".
    LocationResult result(location_ids);
    Result rmw_result(keys.size());

    // Mirror ReadModifyWriteBlock: partition the request into shard-aligned
    // batches and drive one batch at a time. `location_ids` is not threaded
    // through MakeBatches (it carries no key-level payload), we re-project it
    // per batch via batch.batch_indexs below.
    static LocationMapVector empty_locations;
    static PropertyMapVector empty_properties;
    std::vector<BatchMetaData> batches = MakeBatches(keys, location_ids, empty_locations, empty_properties);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_batch_num, batches.size());

    int32_t error_count = 0;
    RmwStats stats; // get/upsert/delete IO timings; key counters are unused here
    std::vector<std::vector<int32_t>> upsert_location_indexs(keys.size());
    std::vector<std::vector<int32_t>> delete_location_indexs(keys.size());

    for (auto &batch : batches) {
        ScopedBatchLock lock(*this, batch.batch_shard_indexs);

        const auto &batch_keys = batch.batch_keys;
        // 1. One batched read for every (key, location_id) pair in this
        //    batch. The backend returns a per-(key, location) ec matrix and
        //    fills out_locations[j][k] with the deserialised CacheLocation
        //    when the slot is OK; NOENT slots leave a default-constructed
        //    CacheLocation that we never read.
        LocationsPerKey batch_locations_per_key;
        const int64_t begin_get = TimestampUtil::GetCurrentTimeUs();
        std::vector<std::vector<ErrorCode>> get_ecs_per_key = backend_manager_->GetLocations(
            request_context, batch_keys, batch.batch_location_ids, batch_locations_per_key);
        stats.get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get;

        // 2. Per-key modifier dispatch -> bucket each key into the upsert
        //    sub-batch or the delete sub-batch. The new modifier contract is
        //    "one call per key, sees the entire location vector at once" -
        //    so each key contributes at most one entry to each sub-batch.
        BatchMetaData upsert_batch;
        BatchMetaData delete_batch;
        std::vector<int32_t> put_global_indexs; // brand-new keys (subset of upsert_global_indexs)

        for (size_t j = 0; j < batch_keys.size(); ++j) {
            const int32_t global_idx = batch.batch_indexs[j];
            const KeyType key = batch_keys[j];
            const std::vector<ErrorCode> &get_ecs = get_ecs_per_key[j];
            const LocationIdVector &loc_ids = batch.batch_location_ids[j];
            CacheLocationVector &loc_values = batch_locations_per_key[j];

            PropertyMap upsert_property_map;
            // Modifier returns a per-location ec vector: modifier_ecs[k] is the
            // authoritative outcome for (key, loc_ids[k]) before the backend
            // upsert/delete is applied. EC_OK slots participate in the action,
            // others are skipped and surfaced as-is on result.
            auto [action, modifier_ecs] =
                modifier(loc_values, get_ecs, static_cast<size_t>(global_idx), loc_ids, upsert_property_map);
            if (modifier_ecs.size() != loc_ids.size()) {
                modifier_ecs.assign(loc_ids.size(), EC_ERROR);
                action = MA_FAIL;
            }

            if (action == MA_OK) {
                LocationMap upsert_loc_map;
                for (size_t k = 0; k < loc_ids.size(); ++k) {
                    if (modifier_ecs[k] != EC_OK) {
                        result.per_location_error_codes[global_idx][k] = modifier_ecs[k];
                        continue;
                    }
                    const LocationId &loc_id = loc_ids[k];
                    CacheLocation &working_loc = loc_values[k];
                    assert(loc_id == working_loc.id());
                    upsert_loc_map.emplace(loc_id, std::move(working_loc));
                    upsert_location_indexs[global_idx].emplace_back(k);
                }
                if (!upsert_loc_map.empty() || !upsert_property_map.empty()) {
                    upsert_batch.batch_keys.emplace_back(key);
                    upsert_batch.batch_indexs.emplace_back(global_idx);
                    upsert_batch.batch_locations.emplace_back(std::move(upsert_loc_map));
                    upsert_batch.batch_properties.emplace_back(std::move(upsert_property_map));
                }
            } else if (action == MA_DELETE) {
                LocationIdVector live_ids;
                for (size_t k = 0; k < loc_ids.size(); ++k) {
                    if (modifier_ecs[k] != EC_OK) {
                        result.per_location_error_codes[global_idx][k] = modifier_ecs[k];
                        continue;
                    }
                    live_ids.emplace_back(loc_ids[k]);
                    delete_location_indexs[global_idx].emplace_back(k);
                }
                if (!live_ids.empty()) {
                    delete_batch.batch_keys.emplace_back(key);
                    delete_batch.batch_indexs.emplace_back(global_idx);
                    delete_batch.batch_location_ids.emplace_back(std::move(live_ids));
                }
            } else {
                // MA_FAIL / MA_SKIP / unknown: surface modifier_ec if any.
                if (action == MA_FAIL) {
                    ++error_count;
                }
                result.per_location_error_codes[global_idx] = std::move(modifier_ecs);
            }
        }

        // 3. Dispatch upsert and delete sub-batches. The sub-batch ec is a
        //    single key-level value; we apply it only to slots that the
        //    modifier marked EC_OK (the ones we actually sent to the backend).

        const auto [upsert_errs, put_success_count] =
            ExecuteRmwUpsert(request_context, upsert_batch, put_global_indexs, keys, stats, rmw_result);
        for (const auto &global_index : upsert_batch.batch_indexs) {
            for (const auto &location_index : upsert_location_indexs[global_index]) {
                result.per_location_error_codes[global_index][location_index] = rmw_result.error_codes[global_index];
            }
        }

        const auto [delete_errs, delete_success_count] =
            ExecuteRmwDelete(trace_id, delete_batch, keys, stats, rmw_result);
        for (const auto &global_index : delete_batch.batch_indexs) {
            for (const auto &location_index : delete_location_indexs[global_index]) {
                result.per_location_error_codes[global_index][location_index] = rmw_result.error_codes[global_index];
            }
        }

        error_count += upsert_errs + delete_errs;
        AdjustKeyCountMeta(put_success_count - delete_success_count);
    }

    EmitBlockRmwMetrics(request_context->metrics_collector(), stats, keys.size());
    if (error_count == keys.size()) {
        result.ec = EC_ERROR;
        PREFIX_INDEXER_LOG(DEBUG, "all locations rmw failed, error count[%d]", error_count);
    } else if (error_count > 0) {
        result.ec = EC_PARTIAL_OK;
        PREFIX_INDEXER_LOG(
            DEBUG, "partial locations rmw failed, keys count[%lu] failed count[%d]", keys.size(), error_count);
    }
    return result;
}

} // namespace kv_cache_manager
