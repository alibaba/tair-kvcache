#include "kv_cache_manager/meta/meta_redis_backend.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/utils.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

MetaRedisBackend::~MetaRedisBackend() { [[maybe_unused]] ErrorCode _ = Close(); }

std::string MetaRedisBackend::GetStorageType() noexcept { return META_REDIS_BACKEND_TYPE_STR; }

std::shared_ptr<RedisClient> MetaRedisBackend::CreateRedisClient() const {
    return std::make_shared<RedisClient>(storage_uri_);
}

ErrorCode MetaRedisBackend::Init(const std::string &instance_id,
                                 const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("fail to init meta redis backend, invalid empty instance id");
        return EC_BADARGS;
    }
    instance_id_ = instance_id;
    cache_key_prefix_ = "kvcache:instance_" + instance_id_ + ":cache_";
    metadata_key_ = "kvcache:instance_" + instance_id_ + ":metadata";

    if (!config) {
        KVCM_LOG_ERROR("fail to init meta redis backend, invalid nullptr config");
        return EC_BADARGS;
    }

    if (config->GetStorageUri().empty()) {
        KVCM_LOG_ERROR("fail to init meta redis backend, invalid empty storage uri, instance[%s]",
                       instance_id_.c_str());
        return EC_BADARGS;
    }

    storage_uri_ = StandardUri::FromUri(config->GetStorageUri());
    int64_t tmp_timeout_ms = 0;
    storage_uri_.GetParamAs("timeout_ms", tmp_timeout_ms);
    if (tmp_timeout_ms > 0) {
        timeout_ms_ = tmp_timeout_ms;
    }
    int64_t db_index = 0;
    storage_uri_.GetParamAs("db", db_index);

    KVCM_LOG_INFO("meta redis backend init successfully, instance[%s], storage uri[%s], timeout ms[%ld], redis db[%ld]",
                  instance_id_.c_str(),
                  config->GetStorageUri().c_str(),
                  timeout_ms_,
                  db_index);
    return EC_OK;
}

ErrorCode MetaRedisBackend::Open() noexcept {
    constexpr int32_t DEFAULT_CLIENT_MAX_POOL_SIZE = 16;
    constexpr int32_t DEFAULT_CLIENT_MIN_POOL_SIZE = 0;
    int32_t client_max_pool_size = DEFAULT_CLIENT_MAX_POOL_SIZE;
    int32_t client_min_pool_size = DEFAULT_CLIENT_MIN_POOL_SIZE;
    int64_t tmp_client_max_pool_size = 0;
    storage_uri_.GetParamAs("client_max_pool_size", tmp_client_max_pool_size);
    if (tmp_client_max_pool_size > 0) {
        client_max_pool_size = tmp_client_max_pool_size;
    }
    int64_t tmp_client_min_pool_size = 0;
    storage_uri_.GetParamAs("client_min_pool_size", tmp_client_min_pool_size);
    if (tmp_client_min_pool_size > 0) {
        client_min_pool_size = tmp_client_min_pool_size;
    }

    client_pool_ = std::make_shared<DynamicClientPool<RedisClient>>(
        [this]() -> std::shared_ptr<RedisClient> {
            auto client = this->CreateRedisClient();
            if (!client || !client->Open()) {
                return nullptr;
            }
            return client;
        },
        client_min_pool_size,
        client_max_pool_size);
    if (!client_pool_->Initialize()) {
        KVCM_LOG_ERROR("meta redis client_pool init faild");
        return EC_ERROR;
    }
    KVCM_LOG_INFO("meta redis backend open successfully, redis client pool size min[%d], max[%d], instance[%s]",
                  client_min_pool_size,
                  client_max_pool_size,
                  instance_id_.c_str());
    return EC_OK;
}

ErrorCode MetaRedisBackend::Close() noexcept {
    client_pool_.reset();
    KVCM_LOG_INFO("meta redis backend close successfully, instance[%s]", instance_id_.c_str());
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Write operations
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaRedisBackend::Put(RequestContext *request_context,
                                             const KeyTypeVec &keys,
                                             const CacheLocationMapVector &locations,
                                             const PropertyMapVector &properties) noexcept {
    const int64_t serde_begin = TimestampUtil::GetCurrentTimeUs();
    FieldMapVec field_maps(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_maps[i] = SerializeToFieldMap(locations[i], properties[i]);
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_serialize_time_us, serde_us);

    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "put fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    return handle->Set(full_keys, field_maps);
}

std::vector<ErrorCode> MetaRedisBackend::Upsert(RequestContext *request_context,
                                                const KeyTypeVec &keys,
                                                const CacheLocationMapVector &locations,
                                                const PropertyMapVector &properties) noexcept {
    const int64_t serde_begin = TimestampUtil::GetCurrentTimeUs();
    FieldMapVec field_maps(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_maps[i] = SerializeToFieldMap(locations[i], properties[i]);
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_serialize_time_us, serde_us);

    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "upsert fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    return handle->Upsert(full_keys, field_maps);
}

std::vector<ErrorCode> MetaRedisBackend::Delete(RequestContext * /*request_context*/, const KeyTypeVec &keys) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "delete fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    return handle->Delete(full_keys);
}

std::vector<ErrorCode> MetaRedisBackend::DeleteLocations(RequestContext * /*request_context*/,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdsPerKey &location_ids) noexcept {
    // Convert location ids to field names with LOCATION_PREFIX for Redis DeleteFields.
    std::vector<std::vector<std::string>> field_names_vec(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_names_vec[i].reserve(location_ids[i].size());
        for (const auto &loc_id : location_ids[i]) {
            field_names_vec[i].push_back(LOCATION_PREFIX + loc_id);
        }
    }

    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "delete locations fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    return handle->DeleteFields(full_keys, field_names_vec);
}

// ---------------------------------------------------------------------------
// Read operations
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaRedisBackend::Get(RequestContext *request_context,
                                             const KeyTypeVec &keys,
                                             CacheLocationMapVector &out_locations,
                                             PropertyMapVector &out_properties) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "get fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    FieldMapVec field_maps;
    std::vector<ErrorCode> results = handle->GetAllFields(full_keys, field_maps);

    const int64_t serde_begin = TimestampUtil::GetCurrentTimeUs();
    out_locations.resize(keys.size());
    out_properties.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        ErrorCode ec = DeserializeFieldMap(field_maps[i], out_locations[i], out_properties[i]);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("get deserialize failed, key[%ld], instance[%s]", keys[i], instance_id_.c_str());
            results[i] = ec;
        }
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_deserialize_time_us, serde_us);
    return results;
}

std::vector<ErrorCode> MetaRedisBackend::GetLocations(RequestContext *request_context,
                                                      const KeyTypeVec &keys,
                                                      CacheLocationMapVector &out_locations) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "get locations fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    FieldMapVec field_maps;
    std::vector<ErrorCode> results = handle->GetAllFields(full_keys, field_maps);

    const int64_t serde_begin = TimestampUtil::GetCurrentTimeUs();
    out_locations.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        ErrorCode ec = DeserializeLocations(field_maps[i], out_locations[i]);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("get locations deserialize failed, key[%ld], instance[%s]", keys[i], instance_id_.c_str());
            results[i] = ec;
        }
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_deserialize_time_us, serde_us);
    return results;
}

std::vector<std::vector<ErrorCode>> MetaRedisBackend::GetLocations(RequestContext *request_context,
                                                                   const KeyTypeVec &keys,
                                                                   const LocationIdsPerKey &location_ids,
                                                                   LocationsPerKey &out_locations) noexcept {
    assert(keys.size() == location_ids.size());

    // Build per-key field names to fetch only requested locations.
    std::vector<std::vector<std::string>> field_names_vec(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_names_vec[i].reserve(location_ids[i].size());
        for (const auto &loc_id : location_ids[i]) {
            field_names_vec[i].push_back(LOCATION_PREFIX + loc_id);
        }
    }

    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "get locations by id fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        std::vector<std::vector<ErrorCode>> results(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i].assign(location_ids[i].size(), EC_TIMEOUT);
        }
        return results;
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    FieldMapVec field_maps;
    std::vector<ErrorCode> key_results = handle->Get(full_keys, field_names_vec, field_maps);

    const int64_t serde_begin = TimestampUtil::GetCurrentTimeUs();
    std::vector<std::vector<ErrorCode>> results(keys.size());
    out_locations.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        out_locations[i].resize(location_ids[i].size());
        if (key_results[i] != EC_OK) {
            results[i].assign(location_ids[i].size(), key_results[i]);
            continue;
        }
        results[i].resize(location_ids[i].size());
        for (size_t j = 0; j < location_ids[i].size(); ++j) {
            auto it = field_maps[i].find(field_names_vec[i][j]);
            if (it == field_maps[i].end() || it->second.empty()) {
                results[i][j] = EC_NOENT;
                continue;
            }
            auto location = std::make_shared<CacheLocation>();
            if (!location->FromJsonString(it->second)) {
                results[i][j] = EC_CORRUPTION;
                continue;
            }
            out_locations[i][j] = std::move(location);
            results[i][j] = EC_OK;
        }
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_deserialize_time_us, serde_us);
    return results;
}

std::vector<ErrorCode> MetaRedisBackend::GetLocationIds(RequestContext * /*request_context*/,
                                                        const KeyTypeVec &keys,
                                                        LocationIdsPerKey &out_location_ids) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "get location ids fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    std::vector<std::vector<std::string>> raw_field_names_vec;
    std::vector<ErrorCode> results = handle->GetFieldNamesWithPrefix(full_keys, LOCATION_PREFIX, raw_field_names_vec);

    out_location_ids.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        out_location_ids[i].reserve(raw_field_names_vec[i].size());
        for (const auto &field_name : raw_field_names_vec[i]) {
            if (field_name.size() > LOCATION_PREFIX.size() && field_name.rfind(LOCATION_PREFIX, 0) == 0) {
                out_location_ids[i].push_back(field_name.substr(LOCATION_PREFIX.size()));
            }
        }
    }
    return results;
}

std::vector<ErrorCode> MetaRedisBackend::GetProperties(RequestContext * /*request_context*/,
                                                       const KeyTypeVec &keys,
                                                       const std::vector<std::string> &field_names,
                                                       PropertyMapVector &out_properties) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "get properties fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    FieldMapVec field_maps;
    std::vector<ErrorCode> results = handle->Get(full_keys, field_names, field_maps);

    out_properties.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        // Properties are stored as plain fields (no LOCATION_PREFIX).
        out_properties[i] = std::move(field_maps[i]);
    }
    return results;
}

// ---------------------------------------------------------------------------
// Key-level operations
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaRedisBackend::Exists(RequestContext * /*request_context*/,
                                                const KeyTypeVec &keys,
                                                std::vector<bool> &out_is_exist_vec) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "exists fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    return handle->Exists(full_keys, out_is_exist_vec);
}

std::vector<ErrorCode> MetaRedisBackend::ExistsLocation(RequestContext * /*request_context*/,
                                                        const KeyTypeVec &keys,
                                                        std::vector<bool> &out_exists) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "exists location fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    return handle->ExistsFieldWithPrefix(full_keys, LOCATION_PREFIX, out_exists);
}

ErrorCode MetaRedisBackend::ListKeys(RequestContext * /*request_context*/,
                                     const std::string &cursor,
                                     const int64_t limit,
                                     std::string &out_next_cursor,
                                     KeyTypeVec &out_keys) noexcept {
    out_keys.clear();
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "list keys fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return EC_TIMEOUT;
    }
    std::vector<std::string> full_keys;
    ErrorCode ec = handle->Scan(cache_key_prefix_, cursor, limit, out_next_cursor, full_keys);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("list keys fail, scan redis fail, instance[%s]", instance_id_.c_str());
        return ec;
    }
    if (!StripPrefixInKeys(cache_key_prefix_, instance_id_, full_keys, out_keys)) {
        KVCM_LOG_ERROR("list keys fail, strip key prefix fail, instance[%s]", instance_id_.c_str());
        out_keys.clear();
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode MetaRedisBackend::RandomSample(RequestContext * /*request_context*/,
                                         const int64_t count,
                                         KeyTypeVec &out_keys) noexcept {
    out_keys.clear();
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "random fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return EC_TIMEOUT;
    }
    std::vector<std::string> full_keys;
    ErrorCode ec = handle->Rand(cache_key_prefix_, count, full_keys);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("random fail, rand redis fail, instance[%s]", instance_id_.c_str());
        return ec;
    }
    if (!StripPrefixInKeys(cache_key_prefix_, instance_id_, full_keys, out_keys)) {
        KVCM_LOG_ERROR("random fail, strip key prefix fail, instance[%s]", instance_id_.c_str());
        out_keys.clear();
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode MetaRedisBackend::SampleReclaimKeys(RequestContext * /*request_context*/,
                                              const int64_t count,
                                              KeyTypeVec &out_keys) noexcept {
    return RandomSample(nullptr, count, out_keys);
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

ErrorCode MetaRedisBackend::PutMetaData(const FieldMap &field_map) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "put metadata fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return EC_TIMEOUT;
    }
    auto error_codes = handle->Set({metadata_key_}, {field_map});
    assert(error_codes.size() == 1);
    return error_codes[0];
}

ErrorCode MetaRedisBackend::GetMetaData(FieldMap &out_field_map) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "get fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return EC_TIMEOUT;
    }
    FieldMapVec maps;
    auto error_codes = handle->GetAllFields({metadata_key_}, maps);
    assert(error_codes.size() == 1);
    if (error_codes[0] == EC_OK) {
        assert(maps.size() == 1);
        out_field_map = std::move(maps[0]);
    }
    return error_codes[0];
}

} // namespace kv_cache_manager
