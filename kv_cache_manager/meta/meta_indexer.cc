#include "kv_cache_manager/meta/meta_indexer.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/common.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_location.h"
#include "kv_cache_manager/meta/meta_search_cache.h"
#include "kv_cache_manager/meta/meta_storage_backend_factory.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {
#define PREFIX_INDEXER_LOG(LEVEL, format, args...)                                                                     \
    KVCM_LOG_##LEVEL("trace_id[%s] instance[%s] | " format, trace_id.c_str(), instance_id_.c_str(), ##args);

namespace {

// V8 §2.1.1 read-side helper: given the full HASH field map returned by
// storage_->GetAllFields / Get, gather every L#{loc_id} entry into a single
// BlockCacheLocationsMeta and serialize it to a JSON string. The string is
// what older callers received as PROPERTY_URI ("__uri__") so we keep the
// modifier API source-compatible.
//
// Dual-mode: if no L# fields are present, fall back to the legacy
// PROPERTY_URI value. This keeps storage-backend unit tests (which write
// opaque strings into `__uri__`) working alongside V8 production data.
std::string SynthesizeUriFromFields(const std::map<std::string, std::string> &field_map) {
    BlockCacheLocationsMeta meta;
    bool any_location_field = false;
    for (const auto &kv : field_map) {
        if (kv.first.rfind(PROPERTY_LOCATION_PREFIX, 0) == 0) {
            any_location_field = true;
            break;
        }
    }
    if (any_location_field) {
        // Best-effort: a malformed L# entry returns the partially-decoded meta
        // instead of failing the whole read; matches the lenient v7 contract.
        meta.FromFieldMap(field_map);
        return meta.ToJsonString();
    }
    if (auto it = field_map.find(PROPERTY_URI); it != field_map.end()) {
        return it->second;
    }
    return std::string();
}

// V8 §2.1.1 write-side helper: take a property map prepared by callers (which
// historically wrote the BlockCacheLocationsMeta JSON under PROPERTY_URI) and
// expand the URI string into individual L#{loc_id} fields. PROPERTY_URI is
// removed from the map when expansion succeeds so the on-disk layout is
// purely BP#/L#/P#.
//
// If the URI does not parse as a BlockCacheLocationsMeta JSON, we leave the
// raw string under PROPERTY_URI ("__uri__") instead. This dual-mode behavior
// preserves backwards compatibility with storage-backend tests that store
// opaque strings, while still routing V8 production payloads through the L#
// decomposition. Returns true unconditionally because both branches are
// considered valid writes.
bool ExpandUriIntoFields(std::map<std::string, std::string> &field_map) {
    auto it = field_map.find(PROPERTY_URI);
    if (it == field_map.end()) {
        return true;
    }
    std::string uri_json = std::move(it->second);
    if (uri_json.empty()) {
        field_map.erase(it);
        return true;
    }
    BlockCacheLocationsMeta meta;
    if (!meta.FromJsonString(uri_json)) {
        // Legacy / opaque payload: keep it under PROPERTY_URI as-is. This
        // path is exercised by tests that store arbitrary strings; production
        // callers always emit valid BlockCacheLocationsMeta JSON.
        it->second = std::move(uri_json);
        return true;
    }
    field_map.erase(it);
    meta.ToFieldMap(field_map);
    return true;
}

} // anonymous namespace

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
    if (storage_) {
        PersistMetaData();
    }
}

ErrorCode MetaIndexer::Init(const std::string &instance_id, const std::shared_ptr<MetaIndexerConfig> &config) noexcept {
    if (!config || !config->GetMetaStorageBackendConfig()) {
        KVCM_LOG_ERROR("instance[%s] meta indexer init failed, config is invalid", instance_id.c_str());
        return EC_BADARGS;
    }
    max_key_count_ = config->GetMaxKeyCount();
    mutex_shard_num_ = config->GetMutexShardNum();
    batch_key_size_ = config->GetBatchKeySize();
    persist_metadata_interval_time_ms_ = config->GetPersistMetaDataIntervalTimeMs();
    if (mutex_shard_num_ > max_key_count_ || (mutex_shard_num_ & (mutex_shard_num_ - 1)) || mutex_shard_num_ <= 0) {
        KVCM_LOG_ERROR(
            "instance[%s] meta indexer init failed, config is invalid, mutex shard num[%lu] max key count[%lu]",
            instance_id.c_str(),
            mutex_shard_num_,
            max_key_count_);
        return EC_CONFIG_ERROR;
    }
    for (int32_t i = 0; i < mutex_shard_num_; ++i) {
        mutex_shards_.emplace_back(std::make_unique<std::mutex>());
    }

    instance_id_ = instance_id;
    auto storage_backend_config = config->GetMetaStorageBackendConfig();
    storage_ = MetaStorageBackendFactory::CreateAndInitStorageBackend(instance_id, storage_backend_config);
    if (!storage_) {
        KVCM_LOG_ERROR("instance[%s] create storage backend failed, storage backend type[%s]",
                       instance_id_.c_str(),
                       config->GetMetaStorageBackendConfig()->GetStorageType().c_str());
        return EC_ERROR;
    }
    auto ec = storage_->Open();
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("instance[%s] meta storage backend open failed, type[%s], ec[%d]",
                       instance_id_.c_str(),
                       storage_->GetStorageType().c_str(),
                       ec);
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
        "instance[%s] meta indexer init success, storage backend type[%s], mutex shard num[%lu], max key count[%lu], "
        "batch key size[%lu], search cache size[%lu], key_count[%lu], persist_metadata_interval_time_ms[%zu], "
        "storage usage data[%s], instance version[%" PRIu8 "]",
        instance_id_.c_str(),
        storage_->GetStorageType().c_str(),
        mutex_shard_num_,
        max_key_count_,
        batch_key_size_,
        config->GetMetaCachePolicyConfig()->GetCapacity(),
        key_count_.load(),
        persist_metadata_interval_time_ms_,
        version_ == InstanceVersion::VERSION_1 ? storage_usage_data_.ToJsonString().c_str() : "N/A",
        static_cast<std::uint8_t>(version_));
    return EC_OK;
}

// Put, Update will move uris and properties to storage
MetaIndexer::Result MetaIndexer::Put(RequestContext *request_context,
                                     const KeyVector &keys,
                                     UriVector &uris,
                                     PropertyMapVector &properties) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    const auto &trace_id = request_context->trace_id();
    if (keys.size() != uris.size() || keys.size() != properties.size()) {
        PREFIX_INDEXER_LOG(ERROR,
                           "Put keys size[%lu], uris size[%lu], properties size[%lu] not equal",
                           keys.size(),
                           uris.size(),
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
    for (int32_t i = 0; i < keys.size(); ++i) {
        PropertyMap &map = properties[i];
        map[PROPERTY_URI] = std::move(uris[i]);
    }

    BatchMetaData batch_datas;
    MakeBatches(keys, properties, batch_datas);
    // V8: each batch row carries a synthetic PROPERTY_URI built from the
    // BlockCacheLocationsMeta JSON; expand it into L#{loc_id} fields before
    // writing to storage. Bad JSON short-circuits with EC_ERROR per row.
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t put_io_time_us = 0;
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, query_batch_num, batch_datas.batch_keys.size());
    for (int32_t i = 0; i < batch_datas.batch_keys.size(); ++i) {
        ScopedBatchLock lock(*this, batch_datas.batch_shard_indexs[i]);
        // V8: rewrite each batch row's PROPERTY_URI to BP/L/P field layout
        // before talking to storage_. Opaque non-JSON payloads stay under
        // PROPERTY_URI for back-compat with storage-backend tests.
        for (auto &fm : batch_datas.batch_properties[i]) {
            ExpandUriIntoFields(fm);
        }
        int64_t begin_put_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = storage_->Put(batch_datas.batch_keys[i], batch_datas.batch_properties[i]);
        put_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_put_io_time;
        error_count +=
            ProcessErrorCodes(trace_id, error_codes, batch_datas.batch_indexs[i], keys, kPutMetaOperation, result);
    }
    AdjustKeyCountMeta(keys.size() - error_count);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, put_io_time_us, put_io_time_us);
    ProcessErrorResult(trace_id, kPutMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result
MetaIndexer::Update(RequestContext *request_context, const KeyVector &keys, PropertyMapVector &properties) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    const auto &trace_id = request_context->trace_id();
    if (keys.size() != properties.size()) {
        PREFIX_INDEXER_LOG(
            ERROR, "Update keys size[%lu], properties size[%lu] not equal", keys.size(), properties.size());
        return Result(EC_ERROR);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    BatchMetaData batch_datas;
    MakeBatches(keys, properties, batch_datas);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, query_batch_num, batch_datas.batch_keys.size());
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t update_io_time_us = 0;
    for (int32_t i = 0; i < batch_datas.batch_keys.size(); ++i) {
        ScopedBatchLock lock(*this, batch_datas.batch_shard_indexs[i]);
        int64_t begin_update_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = storage_->UpdateFields(batch_datas.batch_keys[i], batch_datas.batch_properties[i]);
        update_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_update_io_time;
        error_count +=
            ProcessErrorCodes(trace_id, error_codes, batch_datas.batch_indexs[i], keys, kUpdateMetaOperation, result);
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, update_io_time_us, update_io_time_us);
    ProcessErrorResult(trace_id, kUpdateMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::Update(RequestContext *request_context,
                                        const KeyVector &keys,
                                        UriVector &uris,
                                        PropertyMapVector &properties) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    const auto &trace_id = request_context->trace_id();
    if (keys.size() != uris.size() || keys.size() != properties.size()) {
        PREFIX_INDEXER_LOG(ERROR,
                           "Update keys size[%lu], uris size[%lu], properties size[%lu] not equal",
                           keys.size(),
                           uris.size(),
                           properties.size());
        return Result(EC_ERROR);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    for (int32_t i = 0; i < keys.size(); ++i) {
        PropertyMap &map = properties[i];
        map[PROPERTY_URI] = std::move(uris[i]);
    }

    BatchMetaData batch_datas;
    MakeBatches(keys, properties, batch_datas);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, query_batch_num, batch_datas.batch_keys.size());
    Result result(keys.size());
    int32_t error_count = 0;
    int64_t update_io_time_us = 0;
    for (int32_t i = 0; i < batch_datas.batch_keys.size(); ++i) {
        ScopedBatchLock lock(*this, batch_datas.batch_shard_indexs[i]);
        // Same URI-to-L# expansion as Put. UpdateFields is field-additive
        // (HSET-only), but expanding still keeps the on-disk layout uniform.
        for (auto &fm : batch_datas.batch_properties[i]) {
            ExpandUriIntoFields(fm);
        }
        int64_t begin_update_io_time = TimestampUtil::GetCurrentTimeUs();
        auto error_codes = storage_->UpdateFields(batch_datas.batch_keys[i], batch_datas.batch_properties[i]);
        update_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_update_io_time;
        error_count +=
            ProcessErrorCodes(trace_id, error_codes, batch_datas.batch_indexs[i], keys, kUpdateMetaOperation, result);
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, update_io_time_us, update_io_time_us);
    ProcessErrorResult(trace_id, kUpdateMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::ReadModifyWrite(RequestContext *request_context,
                                                 const KeyVector &keys,
                                                 const ModifierFunc &modifier) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();
    BatchMetaData batch_datas;
    PropertyMapVector empty_properties;
    MakeBatches(keys, empty_properties, batch_datas);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, query_batch_num, batch_datas.batch_keys.size());
    Result result(keys.size());
    int32_t error_count = 0;

    int64_t get_io_time_us = 0;
    int64_t upsert_io_time_us = 0;
    int64_t delete_io_time_us = 0;
    int64_t update_key_count = 0;
    int64_t put_key_count = 0;
    int64_t delete_key_count = 0;
    for (int32_t i = 0; i < batch_datas.batch_keys.size(); ++i) {
        ScopedBatchLock lock(*this, batch_datas.batch_shard_indexs[i]);
        // 1. get -- V8 reads ALL fields (BP#/L#/P# plus any legacy data) so we
        //    can both synthesize the BlockCacheLocationsMeta JSON for the
        //    modifier AND preserve unmodified BP#/P# fields on write-back.
        MetaStorageBackend::FieldMapVec all_field_maps;
        int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
        auto get_error_codes = storage_->GetAllFields(batch_datas.batch_keys[i], all_field_maps);
        get_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_get_io_time;
        // 2. modify
        KeyVector upsert_keys, delete_keys;
        std::vector<int32_t> put_indexs, upsert_indexs, delete_indexs;
        PropertyMapVector upsert_properties;
        for (int32_t j = 0; j < static_cast<int32_t>(get_error_codes.size()); ++j) {
            auto get_ec = get_error_codes[j];
            int32_t idx = batch_datas.batch_indexs[i][j];
            // For NOENT keys all_field_maps[j] is empty; that's the expected
            // "first write" state.
            std::string uri = SynthesizeUriFromFields(all_field_maps[j]);
            PropertyMap map;
            auto [action, ec] = modifier(uri, get_ec, idx, map);
            if (action == MA_FAIL || action == MA_SKIP) {
                if (ec != EC_OK) {
                    ++error_count;
                }
                result.error_codes[idx] = ec;
                continue;
            }
            if (action == MA_DELETE && ec == EC_OK) {
                delete_keys.push_back(keys[idx]);
                delete_indexs.push_back(idx);
                continue;
            }
            if (get_ec != EC_OK && get_ec != EC_NOENT) {
                ++error_count;
                result.error_codes[idx] = get_ec;
                continue;
            }
            // Build the post-modify field map. Steps:
            //   (a) start from existing all_field_maps[j] (preserves BP#/P#)
            //   (b) erase every existing L# entry; the new uri is the
            //       authoritative location set
            //   (c) overlay modifier-supplied props
            //   (d) drop the synthetic PROPERTY_URI key by expanding it into
            //       L#{loc_id} fields
            MetaStorageBackend::FieldMap final_map = std::move(all_field_maps[j]);
            for (auto it = final_map.begin(); it != final_map.end();) {
                if (it->first.rfind(PROPERTY_LOCATION_PREFIX, 0) == 0) {
                    it = final_map.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto &kv : map) {
                final_map[kv.first] = std::move(kv.second);
            }
            final_map[PROPERTY_URI] = std::move(uri);
            ExpandUriIntoFields(final_map);
            if (final_map.empty()) {
                // Defensive: a modifier that returns MA_OK but produces zero
                // fields would result in an HSET with no field/value pairs,
                // which Redis rejects. Convert to a delete instead.
                delete_keys.push_back(keys[idx]);
                delete_indexs.push_back(idx);
                continue;
            }
            upsert_keys.push_back(keys[idx]);
            upsert_indexs.push_back(idx);
            upsert_properties.push_back(std::move(final_map));
            if (get_ec == EC_NOENT) {
                put_indexs.push_back(idx);
            }
        }
        // 3. upsert -- atomically replace each Hash via Put (DEL+HSET). We use
        //    Put (not Upsert/HSET-only) so removed L#{loc_id} fields actually
        //    leave Redis; preservation of unrelated BP#/P# fields was already
        //    done in step 2 by carrying them through final_map.
        if (!upsert_keys.empty()) {
            put_key_count += put_indexs.size();
            update_key_count += upsert_keys.size() - put_indexs.size();
            std::vector<ErrorCode> error_codes;
            if (put_indexs.size() + GetKeyCount() > max_key_count_) {
                PREFIX_INDEXER_LOG(ERROR,
                                   "ReadModifyWrite put keys count[%lu] + current key count[%lu] > max key count[%lu]",
                                   put_indexs.size(),
                                   GetKeyCount(),
                                   max_key_count_);
                error_codes = std::vector<ErrorCode>(upsert_keys.size(), EC_NOSPC);
            } else {
                int64_t begin_upsert_io_time = TimestampUtil::GetCurrentTimeUs();
                error_codes = storage_->Put(upsert_keys, upsert_properties);
                upsert_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_upsert_io_time;
            }
            int32_t upsert_error_count =
                ProcessErrorCodes(trace_id, error_codes, upsert_indexs, keys, kRmwUpsertMetaOperation, result);
            int32_t put_success_count = 0;
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
            AdjustKeyCountMeta(put_success_count);
        }
        // 4. delete
        if (!delete_keys.empty()) {
            delete_key_count += delete_keys.size();
            int64_t begin_delete_io_time = TimestampUtil::GetCurrentTimeUs();
            auto error_codes = storage_->Delete(delete_keys);
            delete_io_time_us += TimestampUtil::GetCurrentTimeUs() - begin_delete_io_time;
            int32_t delete_error_count =
                ProcessErrorCodes(trace_id, error_codes, delete_indexs, keys, kRmwDeleteMetaOperation, result);
            error_count += delete_error_count;
            AdjustKeyCountMeta(delete_error_count - delete_keys.size());
        }
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
    int64_t skip_key_count = keys.size() - update_key_count - put_key_count - delete_key_count;
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, read_modify_write_skip_key_count, skip_key_count);
    ProcessErrorResult(trace_id, kRmwMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result MetaIndexer::Delete(RequestContext *request_context, const KeyVector &keys) noexcept {
    if (keys.size() == 0) {
        return Result(EC_OK);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_indexer, query_key_count, keys.size());
    const auto &trace_id = request_context->trace_id();
    BatchMetaData batch_datas;
    PropertyMapVector empty_properties;
    MakeBatches(keys, empty_properties, batch_datas);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, query_batch_num, batch_datas.batch_keys.size());
    Result result(keys.size());
    int32_t error_count = 0;
    for (int32_t i = 0; i < batch_datas.batch_keys.size(); ++i) {
        ScopedBatchLock lock(*this, batch_datas.batch_shard_indexs[i]);
        auto error_codes = storage_->Delete(batch_datas.batch_keys[i]);
        error_count +=
            ProcessErrorCodes(trace_id, error_codes, batch_datas.batch_indexs[i], keys, kDeleteMetaOperation, result);
    }
    AdjustKeyCountMeta(error_count - keys.size());
    ProcessErrorResult(trace_id, kDeleteMetaOperation, error_count, keys.size(), result);
    return result;
}

MetaIndexer::Result
MetaIndexer::Exist(RequestContext *request_context, const KeyVector &keys, std::vector<bool> &out_exists) noexcept {
    const auto &trace_id = request_context->trace_id();
    out_exists.reserve(keys.size());
    auto error_codes = storage_->Exists(keys, out_exists);

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
    auto error_codes = storage_->GetAllFields(keys, maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    int32_t error_count = 0;
    for (int32_t i = 0; i < keys.size(); ++i) {
        auto &map = maps[i];
        // V8: synthesize PROPERTY_URI from the L# field family so callers
        // continue to receive a single BlockCacheLocationsMeta JSON.
        out_uris.emplace_back(SynthesizeUriFromFields(map));
        // Strip every internal field (BP#/L#/P# and legacy __) before handing
        // properties to user code.
        for (auto it = map.begin(); it != map.end();) {
            if (IsInternalPropertyName(it->first)) {
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
    auto error_codes = storage_->Get(keys, property_names, out_properties);
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
    auto ec = storage_->ListKeys(cursor, limit, out_next_cursor, out_keys);
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
    auto ec = storage_->RandomSample(count, out_keys);
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
    auto ec = storage_->SampleReclaimKeys(count, out_keys);
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

// Combining batch size and lock granularity size to assemble batch data
void MetaIndexer::MakeBatches(const KeyVector &keys,
                              PropertyMapVector &properties,
                              BatchMetaData &batch_data) const noexcept {
    std::map<int32_t, std::vector<int32_t>> shard_map;
    std::set<int32_t> shard_set;
    for (int32_t i = 0; i < keys.size(); ++i) {
        int32_t shard_idx = GetShardIndex(keys[i]);
        shard_map[shard_idx].push_back(i);
        shard_set.insert(shard_idx);
    }

    std::vector<int32_t> batch_shard_index;
    std::vector<int32_t> batch_index;
    KeyVector batch_key;
    PropertyMapVector batch_property;
    int32_t current_batch_size = 0;
    int32_t shard_count = 0;
    for (const int32_t shard_index : shard_set) {
        current_batch_size += shard_map[shard_index].size();
        batch_shard_index.emplace_back(shard_index);
        for (const int32_t idx : shard_map[shard_index]) {
            batch_index.emplace_back(idx);
            batch_key.emplace_back(keys[idx]);
            // maybe properties is empty, e.g. delete, get
            if (!properties.empty()) {
                assert(idx < properties.size());
                batch_property.emplace_back(std::move(properties[idx]));
            }
        }
        ++shard_count;
        if (current_batch_size >= batch_key_size_ || shard_count == shard_set.size()) {
            batch_data.batch_shard_indexs.emplace_back(std::move(batch_shard_index));
            batch_data.batch_indexs.emplace_back(std::move(batch_index));
            batch_data.batch_keys.emplace_back(std::move(batch_key));
            if (!batch_property.empty()) {
                batch_data.batch_properties.emplace_back(std::move(batch_property));
            }
            batch_shard_index.clear();
            batch_index.clear();
            batch_key.clear();
            batch_property.clear();
            current_batch_size = 0;
        }
    }
}

ErrorCode MetaIndexer::RecoverMetaData() noexcept {
    PropertyMap metadata_map;
    auto ec = storage_->GetMetaData(metadata_map);
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
        auto ec = storage_->PutMetaData(metadata_map);
        if (ec != EC_OK) {
            KVCM_LOG_WARN("meta indexer persist metadata failed, ec[%d]", ec);
        }
        last_persist_metadata_time_ = current_time;
    }
}

int32_t MetaIndexer::GetShardIndex(KeyType key) const noexcept {
    return static_cast<int32_t>(key) & (mutex_shard_num_ - 1);
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

    // V8: pull every Hash field and synthesize the URI from the L# family.
    // GetAllFields is intrinsically larger than the old single-field HMGET,
    // but each block_key carries at most O(replicas) L#s plus a handful of
    // BP#/P# entries -- this is acceptable for the read path and keeps the
    // implementation symmetric with ReadModifyWrite.
    PropertyMapVector maps;
    int32_t error_count = 0;
    int32_t not_exist_key_count = 0;
    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = storage_->GetAllFields(miss_keys, maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    size_t io_data_size = 0;
    for (int32_t i = 0; i < miss_keys.size(); ++i) {
        int32_t index = miss_indexs[i];
        out_uris[index] = SynthesizeUriFromFields(maps[i]);
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
    const auto &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    PropertyMapVector maps;
    int32_t error_count = 0;
    int32_t not_exist_key_count = 0;
    Result result(keys.size());
    int64_t begin_get_io_time = TimestampUtil::GetCurrentTimeUs();
    auto error_codes = storage_->GetAllFields(keys, maps);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_indexer, get_io_time_us, TimestampUtil::GetCurrentTimeUs() - begin_get_io_time);
    size_t io_data_size = 0;
    for (int32_t i = 0; i < keys.size(); ++i) {
        out_uris[i] = SynthesizeUriFromFields(maps[i]);
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

} // namespace kv_cache_manager
