#include "kv_cache_manager/meta/meta_async_redis_backend.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/utils.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

MetaAsyncRedisBackend::~MetaAsyncRedisBackend() { [[maybe_unused]] ErrorCode _ = Close(); }

std::string MetaAsyncRedisBackend::GetStorageType() noexcept { return META_ASYNC_REDIS_BACKEND_TYPE_STR; }

// ---------------------------------------------------------------------------
// Init / Open / Close
// ---------------------------------------------------------------------------

std::shared_ptr<RedisClient> MetaAsyncRedisBackend::CreateRedisClient() const {
    return std::make_shared<RedisClient>(storage_uri_);
}

ErrorCode MetaAsyncRedisBackend::Init(const std::string &instance_id,
                                      const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("fail to init meta async redis backend, invalid empty instance id");
        return EC_BADARGS;
    }
    instance_id_ = instance_id;
    cache_key_prefix_ = "kvcache:instance_" + instance_id_ + ":cache_";
    metadata_key_ = "kvcache:instance_" + instance_id_ + ":metadata";

    if (!config) {
        KVCM_LOG_ERROR("fail to init meta async redis backend, invalid nullptr config");
        return EC_BADARGS;
    }
    if (config->GetStorageUri().empty()) {
        KVCM_LOG_ERROR("fail to init meta async redis backend, invalid empty storage uri, instance[%s]",
                       instance_id_.c_str());
        return EC_BADARGS;
    }

    storage_uri_ = StandardUri::FromUri(config->GetStorageUri());

    auto parse_config_param = [&](const char *key, auto &target) {
        int64_t value = static_cast<int64_t>(target);
        storage_uri_.GetParamAs(key, value);
        target = static_cast<std::remove_reference_t<decltype(target)>>(value);
    };
    parse_config_param("timeout_ms", timeout_ms_);
    parse_config_param("async_queue_count", queue_count_);
    parse_config_param("async_max_batch", max_batch_size_);
    parse_config_param("async_wait_us", batch_wait_timeout_us_);
    parse_config_param("async_max_size", queue_max_size_);
    parse_config_param("async_enqueue_timeout_ms", enqueue_timeout_ms_);
    parse_config_param("async_sync_timeout_ms", sync_timeout_ms_);
    parse_config_param("async_drain_ms", drain_timeout_ms_);

    constexpr int32_t kMaxQueueCount = 2048;
    if (queue_count_ <= 0 || queue_count_ > kMaxQueueCount || max_batch_size_ <= 0 || batch_wait_timeout_us_ <= 0 ||
        queue_max_size_ <= 0 || enqueue_timeout_ms_ < 0 || sync_timeout_ms_ <= 0 || drain_timeout_ms_ <= 0) {
        KVCM_LOG_ERROR("fail to init meta async redis backend, invalid async config, instance[%s], queue_count[%d], "
                       "max_batch[%ld], wait_timeout_us[%ld], max_size[%ld], enqueue_timeout_ms[%ld], "
                       "sync_timeout_ms[%ld], drain_timeout_ms[%ld]",
                       instance_id_.c_str(),
                       queue_count_,
                       max_batch_size_,
                       batch_wait_timeout_us_,
                       queue_max_size_,
                       enqueue_timeout_ms_,
                       sync_timeout_ms_,
                       drain_timeout_ms_);
        return EC_BADARGS;
    }

    KVCM_LOG_INFO("meta async redis backend init ok, instance[%s], queue_count[%d], max_batch[%ld], "
                  "wait_timeout_us[%ld], max_size[%ld], enqueue_timeout_ms[%ld], sync_timeout_ms[%ld], "
                  "drain_timeout_ms[%ld]",
                  instance_id_.c_str(),
                  queue_count_,
                  max_batch_size_,
                  batch_wait_timeout_us_,
                  queue_max_size_,
                  enqueue_timeout_ms_,
                  sync_timeout_ms_,
                  drain_timeout_ms_);
    return EC_OK;
}

ErrorCode MetaAsyncRedisBackend::Open() noexcept {
    consumer_clients_.resize(queue_count_);
    for (int i = 0; i < queue_count_; ++i) {
        auto client = CreateRedisClient();
        if (!client || !client->Open()) {
            KVCM_LOG_ERROR("async redis backend open failed, cannot create consumer client[%d], instance[%s]",
                           i,
                           instance_id_.c_str());
            CleanupResources();
            return EC_ERROR;
        }
        consumer_clients_[i] = std::move(client);
    }

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
    read_client_pool_ = std::make_shared<DynamicClientPool<RedisClient>>(
        [this]() -> std::shared_ptr<RedisClient> {
            auto client = this->CreateRedisClient();
            if (!client || !client->Open()) {
                return nullptr;
            }
            return client;
        },
        client_min_pool_size,
        client_max_pool_size);
    if (!read_client_pool_->Initialize()) {
        KVCM_LOG_ERROR("async redis backend open failed, read_client_pool init failed, instance[%s]",
                       instance_id_.c_str());
        CleanupResources();
        return EC_ERROR;
    }

    queues_.resize(queue_count_);
    for (int i = 0; i < queue_count_; ++i) {
        queues_[i] = std::make_unique<MpscWriteQueue>();
    }

    is_running_.store(true, std::memory_order_release);
    consumer_threads_.reserve(queue_count_);
    for (int i = 0; i < queue_count_; ++i) {
        consumer_threads_.emplace_back(&MetaAsyncRedisBackend::ConsumerLoop, this, i);
    }

    KVCM_LOG_INFO("meta async redis backend open ok, instance[%s], %d consumer threads started, "
                  "read client pool size min[%d], max[%d]",
                  instance_id_.c_str(),
                  queue_count_,
                  client_min_pool_size,
                  client_max_pool_size);
    return EC_OK;
}

ErrorCode MetaAsyncRedisBackend::Close() noexcept {
    if (!is_running_.exchange(false, std::memory_order_acq_rel)) {
        return EC_OK;
    }

    for (auto &q : queues_) {
        if (q) {
            q->NotifyConsumer();
        }
    }

    for (auto &t : consumer_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    consumer_threads_.clear();

    CleanupResources();

    KVCM_LOG_INFO("meta async redis backend closed, instance[%s]", instance_id_.c_str());
    return EC_OK;
}

void MetaAsyncRedisBackend::CleanupResources() noexcept {
    for (auto &client : consumer_clients_) {
        if (client) {
            client->Close();
        }
    }
    consumer_clients_.clear();

    read_client_pool_.reset();
    queues_.clear();
}

int MetaAsyncRedisBackend::GetQueueIndexForKey(KeyType key) const noexcept {
    return static_cast<int>(HashKey(key) % queue_count_);
}

// ==================== Write Operations (async enqueue) ====================

bool MetaAsyncRedisBackend::WaitForQueueCapacity(int queue_id, int64_t incoming_key_count) {
    if (queues_[queue_id]->WaitForCapacity(queue_max_size_, incoming_key_count, enqueue_timeout_ms_ * 1000)) {
        return true;
    }
    KVCM_INTERVAL_LOG_WARN(10,
                           "async redis enqueue timeout, queue[%d] key_size[%ld] incoming_keys[%ld], instance[%s]",
                           queue_id,
                           queues_[queue_id]->GetKeySize(),
                           incoming_key_count,
                           instance_id_.c_str());
    return false;
}

std::vector<ErrorCode> MetaAsyncRedisBackend::EnqueueWriteOp(RequestContext *request_context, WriteOp op) {
    if (op.keys.empty()) {
        return {};
    }
    const int64_t enqueue_begin_us = TimestampUtil::GetSteadyTimeUs();

    std::unordered_map<int, std::vector<size_t>> queue_to_indices;
    for (size_t i = 0; i < op.keys.size(); ++i) {
        queue_to_indices[GetQueueIndexForKey(op.keys[i])].push_back(i);
    }

    int64_t enqueue_timeout_key_count = 0;
    std::vector<ErrorCode> error_codes(op.keys.size(), EC_OK);
    for (auto &[qi, indices] : queue_to_indices) {
        WriteOp sub_op;
        sub_op.type = op.type;
        sub_op.keys.reserve(indices.size());
        if (!op.field_maps.empty()) {
            sub_op.field_maps.reserve(indices.size());
        }
        if (!op.field_names_vec.empty()) {
            sub_op.field_names_vec.reserve(indices.size());
        }
        for (size_t idx : indices) {
            sub_op.keys.push_back(op.keys[idx]);
            if (!op.field_maps.empty()) {
                sub_op.field_maps.push_back(std::move(op.field_maps[idx]));
            }
            if (!op.field_names_vec.empty()) {
                sub_op.field_names_vec.push_back(std::move(op.field_names_vec[idx]));
            }
        }

        const int64_t incoming_key_count = static_cast<int64_t>(indices.size());
        if (!WaitForQueueCapacity(qi, incoming_key_count)) {
            enqueue_timeout_key_count += incoming_key_count;
            for (size_t idx : indices) {
                error_codes[idx] = EC_TIMEOUT;
            }
            continue;
        }
        queues_[qi]->Push(QueueItem{std::move(sub_op)});
    }

    const int64_t enqueue_time_us = TimestampUtil::GetSteadyTimeUs() - enqueue_begin_us;

    if (request_context) {
        auto *mc = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            mc, meta_indexer, async_enqueue_timeout_key_count, enqueue_timeout_key_count);
        KVCM_METRICS_COLLECTOR_SET_METRICS(mc, meta_indexer, async_enqueue_time_us, enqueue_time_us);
    }

    return error_codes;
}

std::vector<ErrorCode> MetaAsyncRedisBackend::Put(RequestContext *request_context,
                                                  const KeyTypeVec &keys,
                                                  const CacheLocationMapVector &locations,
                                                  const PropertyMapVector &properties) noexcept {
    const int64_t serde_begin = TimestampUtil::GetCurrentTimeUs();
    WriteOp op;
    op.type = WriteOpType::kPut;
    op.keys = keys;
    op.field_maps.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        op.field_maps[i] = SerializeToFieldMap(locations[i], properties[i]);
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_serialize_time_us, serde_us);
    return EnqueueWriteOp(request_context, std::move(op));
}

std::vector<ErrorCode> MetaAsyncRedisBackend::Upsert(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const CacheLocationMapVector &locations,
                                                     const PropertyMapVector &properties) noexcept {
    const int64_t serde_begin = TimestampUtil::GetCurrentTimeUs();
    WriteOp op;
    op.type = WriteOpType::kUpsert;
    op.keys = keys;
    op.field_maps.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        op.field_maps[i] = SerializeToFieldMap(locations[i], properties[i]);
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_serialize_time_us, serde_us);
    return EnqueueWriteOp(request_context, std::move(op));
}

std::vector<ErrorCode> MetaAsyncRedisBackend::Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept {
    WriteOp op;
    op.type = WriteOpType::kDelete;
    op.keys = keys;
    return EnqueueWriteOp(request_context, std::move(op));
}

std::vector<ErrorCode> MetaAsyncRedisBackend::DeleteLocations(RequestContext *request_context,
                                                              const KeyTypeVec &keys,
                                                              const LocationIdsPerKey &location_ids) noexcept {
    std::vector<std::vector<std::string>> field_names_vec(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_names_vec[i].reserve(location_ids[i].size());
        for (const auto &loc_id : location_ids[i]) {
            field_names_vec[i].push_back(PROPERTY_LOCATION_PREFIX + loc_id);
        }
    }

    WriteOp op;
    op.type = WriteOpType::kDeleteLocations;
    op.keys = keys;
    op.field_names_vec = std::move(field_names_vec);
    return EnqueueWriteOp(request_context, std::move(op));
}

// ==================== Read Operations (sync passthrough) ====================

std::vector<ErrorCode> MetaAsyncRedisBackend::Get(RequestContext *request_context,
                                                  const KeyTypeVec &keys,
                                                  CacheLocationMapVector &out_locations,
                                                  PropertyMapVector &out_properties) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis get fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
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
            KVCM_LOG_ERROR("async redis get deserialize failed, key[%ld], instance[%s]", keys[i], instance_id_.c_str());
            results[i] = ec;
        }
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_deserialize_time_us, serde_us);
    return results;
}

std::vector<ErrorCode> MetaAsyncRedisBackend::GetLocations(RequestContext *request_context,
                                                           const KeyTypeVec &keys,
                                                           CacheLocationMapVector &out_locations) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis get locations fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
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
            KVCM_LOG_ERROR(
                "async redis get locations deserialize failed, key[%ld], instance[%s]", keys[i], instance_id_.c_str());
            results[i] = ec;
        }
    }
    const int64_t serde_us = TimestampUtil::GetCurrentTimeUs() - serde_begin;
    auto *service_metrics_collector =
        request_context ? dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector()) : nullptr;
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, meta_searcher, index_deserialize_time_us, serde_us);
    return results;
}

std::vector<std::vector<ErrorCode>> MetaAsyncRedisBackend::GetLocations(RequestContext *request_context,
                                                                        const KeyTypeVec &keys,
                                                                        const LocationIdsPerKey &location_ids,
                                                                        LocationsPerKey &out_locations) noexcept {
    std::vector<std::vector<std::string>> field_names_vec(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_names_vec[i].reserve(location_ids[i].size());
        for (const auto &loc_id : location_ids[i]) {
            field_names_vec[i].push_back(PROPERTY_LOCATION_PREFIX + loc_id);
        }
    }

    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10,
                               "async redis get locations by id fail, fail to acquire read client, instance[%s]",
                               instance_id_.c_str());
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

std::vector<ErrorCode> MetaAsyncRedisBackend::GetLocationIds(RequestContext * /*request_context*/,
                                                             const KeyTypeVec &keys,
                                                             LocationIdsPerKey &out_location_ids) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis get location ids fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    std::vector<std::vector<std::string>> raw_field_names_vec;
    std::vector<ErrorCode> results =
        handle->GetFieldNamesWithPrefix(full_keys, PROPERTY_LOCATION_PREFIX, raw_field_names_vec);

    out_location_ids.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        out_location_ids[i].reserve(raw_field_names_vec[i].size());
        for (const auto &field_name : raw_field_names_vec[i]) {
            if (field_name.size() > PROPERTY_LOCATION_PREFIX.size() &&
                field_name.rfind(PROPERTY_LOCATION_PREFIX, 0) == 0) {
                out_location_ids[i].push_back(field_name.substr(PROPERTY_LOCATION_PREFIX.size()));
            }
        }
    }
    return results;
}

std::vector<ErrorCode> MetaAsyncRedisBackend::GetProperties(RequestContext * /*request_context*/,
                                                            const KeyTypeVec &keys,
                                                            const std::vector<std::string> &field_names,
                                                            PropertyMapVector &out_properties) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis get properties fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
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
        out_properties[i] = std::move(field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaAsyncRedisBackend::Exists(RequestContext * /*request_context*/,
                                                     const KeyTypeVec &keys,
                                                     std::vector<bool> &out_is_exist_vec) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis exists fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    return handle->Exists(full_keys, out_is_exist_vec);
}

std::vector<ErrorCode> MetaAsyncRedisBackend::ExistsLocation(RequestContext * /*request_context*/,
                                                             const KeyTypeVec &keys,
                                                             std::vector<bool> &out_exists) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis exists location fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, keys);
    return handle->ExistsFieldWithPrefix(full_keys, PROPERTY_LOCATION_PREFIX, out_exists);
}

ErrorCode MetaAsyncRedisBackend::ListKeys(RequestContext * /*request_context*/,
                                          const std::string &cursor,
                                          const int64_t limit,
                                          std::string &out_next_cursor,
                                          KeyTypeVec &out_keys) noexcept {
    out_keys.clear();
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis list keys fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
        return EC_TIMEOUT;
    }
    std::vector<std::string> full_keys;
    ErrorCode ec = handle->Scan(cache_key_prefix_, cursor, limit, out_next_cursor, full_keys);
    if (ec != EC_OK) {
        return ec;
    }
    if (!StripPrefixInKeys(cache_key_prefix_, instance_id_, full_keys, out_keys)) {
        out_keys.clear();
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode MetaAsyncRedisBackend::RandomSample(RequestContext * /*request_context*/,
                                              const int64_t count,
                                              KeyTypeVec &out_keys) noexcept {
    out_keys.clear();
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis random sample fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
        return EC_TIMEOUT;
    }
    std::vector<std::string> full_keys;
    ErrorCode ec = handle->Rand(cache_key_prefix_, count, full_keys);
    if (ec != EC_OK) {
        return ec;
    }
    if (!StripPrefixInKeys(cache_key_prefix_, instance_id_, full_keys, out_keys)) {
        out_keys.clear();
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode MetaAsyncRedisBackend::SampleReclaimKeys(RequestContext * /*request_context*/,
                                                   const int64_t count,
                                                   KeyTypeVec &out_keys) noexcept {
    return RandomSample(nullptr, count, out_keys);
}

ErrorCode MetaAsyncRedisBackend::PutMetaData(const FieldMap &field_map) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis put metadata fail, fail to acquire client, instance[%s]", instance_id_.c_str());
        return EC_TIMEOUT;
    }
    auto error_codes = handle->Set({metadata_key_}, {field_map});
    if (error_codes.empty()) {
        return EC_ERROR;
    }
    return error_codes[0];
}

ErrorCode MetaAsyncRedisBackend::GetMetaData(FieldMap &out_field_map) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis get metadata fail, fail to acquire client, instance[%s]", instance_id_.c_str());
        return EC_TIMEOUT;
    }
    FieldMapVec maps;
    auto error_codes = handle->GetAllFields({metadata_key_}, maps);
    if (error_codes.empty()) {
        return EC_ERROR;
    }
    if (error_codes[0] == EC_OK && !maps.empty()) {
        out_field_map = std::move(maps[0]);
    }
    return error_codes[0];
}

// ==================== Sync ====================

bool MetaAsyncRedisBackend::Sync(const KeyTypeVec &keys) noexcept {
    if (keys.empty()) {
        return true;
    }
    if (!is_running_.load(std::memory_order_acquire)) {
        return false;
    }

    std::unordered_set<int> touched_queues;
    for (const auto &key : keys) {
        touched_queues.insert(GetQueueIndexForKey(key));
    }

    auto barrier_ctx = std::make_shared<BarrierContext>();
    barrier_ctx->remain.store(static_cast<int>(touched_queues.size()), std::memory_order_release);

    for (int qi : touched_queues) {
        SyncBarrierItem item;
        item.barrier_ctx = barrier_ctx;
        queues_[qi]->Push(QueueItem{std::move(item)});
    }

    return barrier_ctx->Wait(std::chrono::milliseconds{sync_timeout_ms_});
}

// ==================== Metrics ====================

MetaStorageBackend::AsyncWriteStats MetaAsyncRedisBackend::GetAsyncWriteStats() noexcept {
    AsyncWriteStats stats;

    // Compute live queue sizes
    int64_t max_size = 0;
    int64_t sum_size = 0;
    for (const auto &q : queues_) {
        int64_t s = q ? q->GetKeySize() : 0;
        sum_size += s;
        if (s > max_size) {
            max_size = s;
        }
    }
    stats.max_async_queue_size = max_size;
    stats.avg_async_queue_size = queues_.empty() ? 0 : sum_size / static_cast<int64_t>(queues_.size());

    // CAS reset: read and reset accumulated counters
    stats.flush_key_count = stats_flush_key_count_.exchange(0, std::memory_order_relaxed);
    int64_t flush_count = stats_batch_flush_count_.exchange(0, std::memory_order_relaxed);
    int64_t total_flush_time_us = stats_batch_flush_time_us_.exchange(0, std::memory_order_relaxed);
    stats.batch_flush_time_us = flush_count > 0 ? total_flush_time_us / flush_count : 0;
    stats.pipeline_error_count = stats_pipeline_error_count_.exchange(0, std::memory_order_relaxed);
    return stats;
}

// ==================== Consumer Thread ====================

void MetaAsyncRedisBackend::ConsumerLoop(int queue_id) {
    KVCM_LOG_INFO("async redis consumer[%d] started, instance[%s]", queue_id, instance_id_.c_str());

    while (is_running_.load(std::memory_order_acquire)) {
        int64_t taken_keys = 0;
        auto items = queues_[queue_id]->PopBatchWait(max_batch_size_, batch_wait_timeout_us_, taken_keys);
        if (items.empty()) {
            continue;
        }
        BatchFlush(queue_id, items, taken_keys);
    }

    DrainQueue(queue_id);
    KVCM_LOG_INFO("async redis consumer[%d] stopped, instance[%s]", queue_id, instance_id_.c_str());
}

void MetaAsyncRedisBackend::CompileWriteOp(const WriteOp &op, std::vector<CmdArgs> &cmds) {
    std::vector<std::string> full_keys = AppendPrefixToKeys(cache_key_prefix_, op.keys);

    switch (op.type) {
    case WriteOpType::kPut:
        RedisClient::BuildSetCmds(full_keys, op.field_maps, cmds);
        break;
    case WriteOpType::kUpsert:
        RedisClient::BuildHashSetCmds(full_keys, op.field_maps, cmds);
        break;
    case WriteOpType::kDelete:
        RedisClient::BuildDeleteCmds(full_keys, cmds);
        break;
    case WriteOpType::kDeleteLocations:
        RedisClient::BuildHashDeleteCmds(full_keys, op.field_names_vec, cmds);
        break;
    }
}

void MetaAsyncRedisBackend::BatchFlush(int queue_id, std::vector<QueueItem> &items, int64_t total_keys) {
    const int64_t flush_begin_us = TimestampUtil::GetSteadyTimeUs();
    RedisClient *client = consumer_clients_[queue_id].get();

    std::vector<CmdArgs> all_cmds;
    all_cmds.reserve(static_cast<size_t>(total_keys) * 2);
    struct BarrierRecord {
        std::shared_ptr<BarrierContext> ctx;
        size_t cmd_begin;
        size_t cmd_end;
        int64_t key_count;
    };
    std::vector<BarrierRecord> barriers;
    size_t segment_cmd_begin = 0;
    int64_t segment_key_count = 0;

    for (auto &item : items) {
        if (std::holds_alternative<WriteOp>(item)) {
            segment_key_count += static_cast<int64_t>(std::get<WriteOp>(item).keys.size());
            CompileWriteOp(std::get<WriteOp>(item), all_cmds);
        } else {
            auto &barrier = std::get<SyncBarrierItem>(item);
            barriers.push_back({barrier.barrier_ctx, segment_cmd_begin, all_cmds.size(), segment_key_count});
            segment_cmd_begin = all_cmds.size();
            segment_key_count = 0;
        }
    }
    // trailing_key_count: keys from WriteOps after the last barrier
    const int64_t trailing_key_count = segment_key_count;
    const size_t trailing_cmd_begin = segment_cmd_begin;

    if (all_cmds.empty()) {
        for (auto &b : barriers) {
            if (b.ctx) {
                b.ctx->Fence();
            }
        }
        return;
    }

    bool all_ok = false;
    auto error_codes = client->BatchWrite(all_cmds, all_ok);

    if (!all_ok) {
        KVCM_LOG_ERROR("async redis consumer[%d] pipeline fail, cmds[%zu], instance[%s]",
                       queue_id,
                       all_cmds.size(),
                       instance_id_.c_str());
        stats_pipeline_error_count_.fetch_add(1, std::memory_order_relaxed);

        if (error_codes.size() != all_cmds.size()) {
            for (auto &b : barriers) {
                if (b.ctx) {
                    b.ctx->SetFailed();
                }
            }
        } else {
            auto segment_all_ok = [&](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    if (error_codes[i] != EC_OK) {
                        return false;
                    }
                }
                return true;
            };

            int64_t ok_key_count = 0;
            for (auto &b : barriers) {
                if (!b.ctx) {
                    continue;
                }
                if (segment_all_ok(b.cmd_begin, b.cmd_end)) {
                    b.ctx->Fence();
                    ok_key_count += b.key_count;
                } else {
                    b.ctx->SetFailed();
                }
            }
            if (segment_all_ok(trailing_cmd_begin, all_cmds.size())) {
                ok_key_count += trailing_key_count;
            }
            if (ok_key_count > 0) {
                stats_flush_key_count_.fetch_add(ok_key_count, std::memory_order_relaxed);
            }
        }
    } else {
        stats_flush_key_count_.fetch_add(total_keys, std::memory_order_relaxed);
        for (auto &b : barriers) {
            if (b.ctx) {
                b.ctx->Fence();
            }
        }
    }

    stats_batch_flush_count_.fetch_add(1, std::memory_order_relaxed);
    stats_batch_flush_time_us_.fetch_add(TimestampUtil::GetSteadyTimeUs() - flush_begin_us, std::memory_order_relaxed);
}

void MetaAsyncRedisBackend::DrainQueue(int queue_id) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(drain_timeout_ms_);

    while (std::chrono::steady_clock::now() < deadline) {
        int64_t taken_keys = 0;
        auto items = queues_[queue_id]->PopBatch(max_batch_size_, taken_keys);
        if (items.empty()) {
            break;
        }
        BatchFlush(queue_id, items, taken_keys);
    }

    size_t dropped_write_op_count = 0;
    size_t dropped_key_count = 0;
    size_t dropped_barrier_count = 0;
    while (true) {
        int64_t remaining_keys = 0;
        auto remaining = queues_[queue_id]->PopBatch(queue_max_size_, remaining_keys);
        if (remaining.empty()) {
            break;
        }
        for (auto &item : remaining) {
            if (std::holds_alternative<WriteOp>(item)) {
                ++dropped_write_op_count;
                dropped_key_count += std::get<WriteOp>(item).keys.size();
                continue;
            }
            ++dropped_barrier_count;
            auto &barrier = std::get<SyncBarrierItem>(item);
            if (barrier.barrier_ctx) {
                barrier.barrier_ctx->SetFailed();
            }
        }
    }
    if (dropped_write_op_count > 0 || dropped_barrier_count > 0) {
        KVCM_LOG_WARN("async redis consumer[%d] drain timeout, dropped write ops[%zu] keys[%zu], "
                      "sync barriers[%zu], instance[%s]",
                      queue_id,
                      dropped_write_op_count,
                      dropped_key_count,
                      dropped_barrier_count,
                      instance_id_.c_str());
    }
}

} // namespace kv_cache_manager
