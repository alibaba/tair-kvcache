#include "kv_cache_manager/meta/meta_async_redis_backend.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/utils.h"

namespace kv_cache_manager {

MetaAsyncRedisBackend::~MetaAsyncRedisBackend() { [[maybe_unused]] ErrorCode _ = Close(); }

std::string MetaAsyncRedisBackend::GetStorageType() noexcept { return META_ASYNC_REDIS_BACKEND_TYPE_STR; }

// ---------------------------------------------------------------------------
// Key prefix helpers
// ---------------------------------------------------------------------------

std::vector<std::string> MetaAsyncRedisBackend::AppendPrefixToKeys(const KeyTypeVec &keys) const {
    std::vector<std::string> keys_with_prefix;
    keys_with_prefix.reserve(keys.size());
    for (const KeyType &key : keys) {
        keys_with_prefix.emplace_back(cache_key_prefix_ + std::to_string(key));
    }
    return keys_with_prefix;
}

bool MetaAsyncRedisBackend::StripPrefixInKeys(const std::vector<std::string> &keys_with_prefix,
                                              KeyTypeVec &out_keys) const {
    out_keys.clear();
    out_keys.reserve(keys_with_prefix.size());
    for (const std::string &key_with_prefix : keys_with_prefix) {
        if (key_with_prefix.size() < cache_key_prefix_.size() ||
            key_with_prefix.compare(0, cache_key_prefix_.size(), cache_key_prefix_) != 0) {
            KVCM_LOG_ERROR("async redis strip prefix invalid key[%s], expected prefix[%s], instance[%s]",
                           key_with_prefix.c_str(),
                           cache_key_prefix_.c_str(),
                           instance_id_.c_str());
            out_keys.clear();
            return false;
        }
        const std::string keyStr = key_with_prefix.substr(cache_key_prefix_.size());
        KeyType key = 0;
        if (!StringUtil::StrToInt64(keyStr.c_str(), key)) {
            KVCM_LOG_ERROR("async redis strip prefix invalid key[%s], can not convert to int64, instance[%s]",
                           key_with_prefix.c_str(),
                           instance_id_.c_str());
            out_keys.clear();
            return false;
        }
        out_keys.emplace_back(key);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Serialization helpers (same logic as MetaRedisBackend)
// ---------------------------------------------------------------------------

FieldMap MetaAsyncRedisBackend::SerializeToFieldMap(const CacheLocationMap &locations, const PropertyMap &properties) {
    FieldMap field_map;
    for (const auto &[loc_id, loc_ptr] : locations) {
        field_map[LOCATION_PREFIX + loc_id] = loc_ptr ? loc_ptr->ToJsonString() : "";
    }
    for (const auto &[key, value] : properties) {
        field_map[key] = value;
    }
    return field_map;
}

ErrorCode MetaAsyncRedisBackend::DeserializeFieldMap(const FieldMap &field_map,
                                                     CacheLocationMap &out_locations,
                                                     PropertyMap &out_properties) {
    for (const auto &[field_name, field_value] : field_map) {
        if (field_name.rfind(LOCATION_PREFIX, 0) == 0) {
            std::string loc_id = field_name.substr(LOCATION_PREFIX.size());
            auto location = std::make_shared<CacheLocation>();
            if (field_value.empty()) {
                location->set_id(loc_id);
            } else if (!location->FromJsonString(field_value)) {
                return EC_CORRUPTION;
            }
            out_locations[loc_id] = std::move(location);
        } else {
            out_properties[field_name] = field_value;
        }
    }
    return EC_OK;
}

ErrorCode MetaAsyncRedisBackend::DeserializeLocations(const FieldMap &field_map, CacheLocationMap &out_locations) {
    for (const auto &[field_name, field_value] : field_map) {
        if (field_name.rfind(LOCATION_PREFIX, 0) != 0) {
            continue;
        }
        std::string loc_id = field_name.substr(LOCATION_PREFIX.size());
        auto location = std::make_shared<CacheLocation>();
        if (field_value.empty()) {
            location->set_id(loc_id);
        } else if (!location->FromJsonString(field_value)) {
            return EC_CORRUPTION;
        }
        out_locations[loc_id] = std::move(location);
    }
    return EC_OK;
}

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

    int64_t tmp = 0;
    storage_uri_.GetParamAs("timeout_ms", tmp);
    if (tmp > 0) {
        timeout_ms_ = tmp;
    }

    tmp = 0;
    storage_uri_.GetParamAs("async_queue_count", tmp);
    if (tmp > 0) {
        queue_count_ = static_cast<int32_t>(tmp);
    }

    tmp = 0;
    storage_uri_.GetParamAs("async_max_batch", tmp);
    if (tmp > 0) {
        max_batch_size_ = tmp;
    }

    tmp = 0;
    storage_uri_.GetParamAs("async_wait_us", tmp);
    if (tmp > 0) {
        batch_wait_timeout_us_ = tmp;
    }

    tmp = 0;
    storage_uri_.GetParamAs("async_max_size", tmp);
    if (tmp > 0) {
        queue_max_size_ = tmp;
    }

    tmp = 0;
    storage_uri_.GetParamAs("async_enqueue_retry_us", tmp);
    if (tmp > 0) {
        enqueue_retry_interval_us_ = tmp;
    }

    tmp = 0;
    storage_uri_.GetParamAs("async_enqueue_timeout_ms", tmp);
    if (tmp > 0) {
        enqueue_timeout_ms_ = tmp;
    }

    tmp = 0;
    storage_uri_.GetParamAs("async_drain_ms", tmp);
    if (tmp > 0) {
        drain_timeout_ms_ = tmp;
    }

    KVCM_LOG_INFO("meta async redis backend init ok, instance[%s], queue_count[%d], max_batch[%ld], "
                  "wait_us[%ld], max_size[%ld], drain_ms[%ld]",
                  instance_id_.c_str(),
                  queue_count_,
                  max_batch_size_,
                  batch_wait_timeout_us_,
                  queue_max_size_,
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
            return EC_ERROR;
        }
        consumer_clients_[i] = std::move(client);
    }

    constexpr int32_t kReadPoolMin = 1;
    constexpr int32_t kReadPoolMax = 16;
    read_client_pool_ = std::make_shared<DynamicClientPool<RedisClient>>(
        [this]() -> std::shared_ptr<RedisClient> {
            auto client = this->CreateRedisClient();
            if (!client || !client->Open()) {
                return nullptr;
            }
            return client;
        },
        kReadPoolMin,
        kReadPoolMax);
    if (!read_client_pool_->Initialize()) {
        KVCM_LOG_ERROR("async redis backend open failed, read_client_pool init failed, instance[%s]",
                       instance_id_.c_str());
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

    KVCM_LOG_INFO("meta async redis backend open ok, instance[%s], %d consumer threads started",
                  instance_id_.c_str(),
                  queue_count_);
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

    for (auto &client : consumer_clients_) {
        if (client) {
            client->Close();
        }
    }
    consumer_clients_.clear();

    read_client_pool_.reset();
    queues_.clear();

    KVCM_LOG_INFO("meta async redis backend closed, instance[%s]", instance_id_.c_str());
    return EC_OK;
}

int MetaAsyncRedisBackend::GetQueueIndexForKey(KeyType key) const noexcept {
    return static_cast<int>(HashKey(key) % queue_count_);
}

// ==================== Write Operations (async enqueue) ====================

bool MetaAsyncRedisBackend::WaitForQueueCapacity(int queue_id) {
    if (queues_[queue_id]->GetKeySize() < queue_max_size_) {
        return true;
    }
    int64_t waited_us = 0;
    while (waited_us < enqueue_timeout_ms_ * 1000) {
        std::this_thread::sleep_for(std::chrono::microseconds(enqueue_retry_interval_us_));
        waited_us += enqueue_retry_interval_us_;
        if (queues_[queue_id]->GetKeySize() < queue_max_size_) {
            return true;
        }
    }
    KVCM_INTERVAL_LOG_WARN(10,
                           "async redis enqueue timeout, queue[%d] key_size[%ld], instance[%s]",
                           queue_id,
                           queues_[queue_id]->GetKeySize(),
                           instance_id_.c_str());
    return false;
}

std::vector<ErrorCode> MetaAsyncRedisBackend::EnqueueWriteOp(WriteOp op) {
    if (op.keys.empty()) {
        return {};
    }

    std::unordered_map<int, std::vector<size_t>> queue_to_indices;
    for (size_t i = 0; i < op.keys.size(); ++i) {
        queue_to_indices[GetQueueIndexForKey(op.keys[i])].push_back(i);
    }

    std::vector<ErrorCode> results(op.keys.size(), EC_OK);
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

        if (!WaitForQueueCapacity(qi)) {
            for (size_t idx : indices) {
                results[idx] = EC_TIMEOUT;
            }
            continue;
        }
        queues_[qi]->Push(QueueItem{std::move(sub_op)});
    }
    return results;
}

std::vector<ErrorCode> MetaAsyncRedisBackend::Put(RequestContext * /*request_context*/,
                                                  const KeyTypeVec &keys,
                                                  const CacheLocationMapVector &locations,
                                                  const PropertyMapVector &properties) noexcept {
    WriteOp op;
    op.type = WriteOpType::kPut;
    op.keys = keys;
    op.field_maps.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        op.field_maps[i] = SerializeToFieldMap(locations[i], properties[i]);
    }
    return EnqueueWriteOp(std::move(op));
}

std::vector<ErrorCode> MetaAsyncRedisBackend::Upsert(RequestContext * /*request_context*/,
                                                     const KeyTypeVec &keys,
                                                     const CacheLocationMapVector &locations,
                                                     const PropertyMapVector &properties) noexcept {
    WriteOp op;
    op.type = WriteOpType::kUpsert;
    op.keys = keys;
    op.field_maps.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        op.field_maps[i] = SerializeToFieldMap(locations[i], properties[i]);
    }
    return EnqueueWriteOp(std::move(op));
}

std::vector<ErrorCode> MetaAsyncRedisBackend::Delete(RequestContext * /*request_context*/,
                                                     const KeyTypeVec &keys) noexcept {
    WriteOp op;
    op.type = WriteOpType::kDelete;
    op.keys = keys;
    return EnqueueWriteOp(std::move(op));
}

std::vector<ErrorCode> MetaAsyncRedisBackend::DeleteLocations(RequestContext * /*request_context*/,
                                                              const KeyTypeVec &keys,
                                                              const LocationIdsPerKey &location_ids) noexcept {
    std::vector<std::vector<std::string>> field_names_vec(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_names_vec[i].reserve(location_ids[i].size());
        for (const auto &loc_id : location_ids[i]) {
            field_names_vec[i].push_back(LOCATION_PREFIX + loc_id);
        }
    }

    WriteOp op;
    op.type = WriteOpType::kDeleteLocations;
    op.keys = keys;
    op.field_names_vec = std::move(field_names_vec);
    return EnqueueWriteOp(std::move(op));
}

// ==================== Read Operations (sync passthrough) ====================

std::vector<ErrorCode> MetaAsyncRedisBackend::Get(RequestContext * /*request_context*/,
                                                  const KeyTypeVec &keys,
                                                  CacheLocationMapVector &out_locations,
                                                  PropertyMapVector &out_properties) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis get fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    FieldMapVec field_maps;
    std::vector<ErrorCode> results = handle->GetAllFields(full_keys, field_maps);

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
    return results;
}

std::vector<ErrorCode> MetaAsyncRedisBackend::GetLocations(RequestContext * /*request_context*/,
                                                           const KeyTypeVec &keys,
                                                           CacheLocationMapVector &out_locations) noexcept {
    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis get locations fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    FieldMapVec field_maps;
    std::vector<ErrorCode> results = handle->GetAllFields(full_keys, field_maps);

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
    return results;
}

std::vector<std::vector<ErrorCode>>
MetaAsyncRedisBackend::GetLocations(RequestContext * /*request_context*/,
                                    const KeyTypeVec &keys,
                                    const LocationIdsPerKey &location_ids,
                                    LocationsPerKey &out_locations) noexcept {
    std::vector<std::vector<std::string>> field_names_vec(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_names_vec[i].reserve(location_ids[i].size());
        for (const auto &loc_id : location_ids[i]) {
            field_names_vec[i].push_back(LOCATION_PREFIX + loc_id);
        }
    }

    auto handle = read_client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "async redis get locations by id fail, fail to acquire read client, instance[%s]", instance_id_.c_str());
        std::vector<std::vector<ErrorCode>> results(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i].assign(location_ids[i].size(), EC_TIMEOUT);
        }
        return results;
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    FieldMapVec field_maps;
    std::vector<ErrorCode> key_results = handle->Get(full_keys, field_names_vec, field_maps);

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
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
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
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
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
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
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
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    return handle->ExistsFieldWithPrefix(full_keys, LOCATION_PREFIX, out_exists);
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
    if (!StripPrefixInKeys(full_keys, out_keys)) {
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
    if (!StripPrefixInKeys(full_keys, out_keys)) {
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

    return barrier_ctx->Wait();
}

// ==================== Metrics ====================

std::vector<int64_t> MetaAsyncRedisBackend::GetAsyncQueueSizes() const noexcept {
    std::vector<int64_t> sizes;
    sizes.reserve(queues_.size());
    for (const auto &q : queues_) {
        sizes.push_back(q ? q->GetKeySize() : 0);
    }
    return sizes;
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
    std::vector<std::string> full_keys = AppendPrefixToKeys(op.keys);

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
    RedisClient *client = consumer_clients_[queue_id].get();

    std::vector<CmdArgs> all_cmds;
    all_cmds.reserve(static_cast<size_t>(total_keys) * 2);
    struct BarrierRecord {
        std::shared_ptr<BarrierContext> ctx;
        size_t cmd_begin;
        size_t cmd_end;
    };
    std::vector<BarrierRecord> barriers;
    size_t segment_cmd_begin = 0;

    for (auto &item : items) {
        if (std::holds_alternative<WriteOp>(item)) {
            CompileWriteOp(std::get<WriteOp>(item), all_cmds);
        } else {
            auto &barrier = std::get<SyncBarrierItem>(item);
            barriers.push_back({barrier.barrier_ctx, segment_cmd_begin, all_cmds.size()});
            segment_cmd_begin = all_cmds.size();
        }
    }

    if (all_cmds.empty()) {
        for (auto &b : barriers) {
            if (b.ctx) {
                b.ctx->Fence();
            }
        }
        return;
    }

    auto replies = client->BatchExecute(all_cmds);

    if (replies.size() != all_cmds.size()) {
        KVCM_LOG_ERROR("async redis consumer[%d] pipeline failed, cmds[%zu] replies[%zu], instance[%s]",
                       queue_id,
                       all_cmds.size(),
                       replies.size(),
                       instance_id_.c_str());
        for (auto &b : barriers) {
            if (b.ctx) {
                b.ctx->SetFailed();
            }
        }
        return;
    }

    for (auto &b : barriers) {
        if (!b.ctx) {
            continue;
        }
        bool segment_ok = true;
        for (size_t i = b.cmd_begin; i < b.cmd_end; ++i) {
            if (replies[i] && replies[i]->type == REDIS_REPLY_ERROR) {
                segment_ok = false;
                break;
            }
        }
        if (segment_ok) {
            b.ctx->Fence();
        } else {
            b.ctx->SetFailed();
        }
    }
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

    int64_t remaining_keys = 0;
    auto remaining = queues_[queue_id]->PopBatch(queue_max_size_, remaining_keys);
    for (auto &item : remaining) {
        if (std::holds_alternative<SyncBarrierItem>(item)) {
            auto &barrier = std::get<SyncBarrierItem>(item);
            if (barrier.barrier_ctx) {
                barrier.barrier_ctx->SetFailed();
            }
        }
    }
    if (!remaining.empty()) {
        KVCM_LOG_WARN("async redis consumer[%d] drain timeout, dropped %zu items, instance[%s]",
                      queue_id,
                      remaining.size(),
                      instance_id_.c_str());
    }
}

} // namespace kv_cache_manager
