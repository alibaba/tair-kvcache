#include "kv_cache_manager/meta/meta_redis_backend.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"

namespace kv_cache_manager {

MetaRedisBackend::~MetaRedisBackend() { [[maybe_unused]] ErrorCode _ = Close(); }

std::string MetaRedisBackend::GetStorageType() noexcept { return META_REDIS_BACKEND_TYPE_STR; }

std::vector<std::string> MetaRedisBackend::AppendPrefixToKeys(const KeyTypeVec &keys) const {
    std::vector<std::string> keys_with_prefix;
    keys_with_prefix.reserve(keys.size());
    for (const KeyType &key : keys) {
        keys_with_prefix.emplace_back(cache_key_prefix_ + std::to_string(key));
    }
    return keys_with_prefix;
}

bool MetaRedisBackend::StripPrefixInKeys(const std::vector<std::string> &keys_with_prefix,
                                         std::vector<KeyType> &out_keys) const {
    out_keys.clear();
    out_keys.reserve(keys_with_prefix.size());
    for (const std::string &key_with_prefix : keys_with_prefix) {
        if (key_with_prefix.size() < cache_key_prefix_.size() ||
            key_with_prefix.compare(0, cache_key_prefix_.size(), cache_key_prefix_) != 0) {
            KVCM_LOG_ERROR("strip prefix in key encounter invalid key[%s], expected prefix[%s], instance[%s]",
                           key_with_prefix.c_str(),
                           cache_key_prefix_.c_str(),
                           instance_id_.c_str());
            out_keys.clear();
            return false;
        }
        const std::string keyStr = key_with_prefix.substr(cache_key_prefix_.size());
        KeyType key = 0;
        if (!StringUtil::StrToInt64(keyStr.c_str(), key)) {
            KVCM_LOG_ERROR("strip prefix in key encouter invalid key[%s], can not convert to int64, instance[%s]",
                           key_with_prefix.c_str(),
                           instance_id_.c_str());
            out_keys.clear();
            return false;
        }
        out_keys.emplace_back(key);
    }
    return true;
}

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

    KVCM_LOG_INFO("meta redis backend init successfully, instance[%s], storage uri[%s], timeout ms[%ld]",
                  instance_id_.c_str(),
                  config->GetStorageUri().c_str(),
                  timeout_ms_);
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

std::vector<ErrorCode> MetaRedisBackend::Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "put fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    return handle->Set(full_keys, field_maps);
}

std::vector<ErrorCode> MetaRedisBackend::UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "update fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    return handle->Update(full_keys, field_maps);
}

std::vector<ErrorCode> MetaRedisBackend::Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "upsert fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    return handle->Upsert(full_keys, field_maps);
}

std::vector<ErrorCode> MetaRedisBackend::IncrFields(const KeyTypeVec &keys,
                                                    const std::map<std::string, int64_t> &field_amounts) noexcept {
    return std::vector<ErrorCode>(keys.size(), EC_UNIMPLEMENTED);
}

std::vector<ErrorCode> MetaRedisBackend::Delete(const KeyTypeVec &keys) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "delete fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    return handle->Delete(full_keys);
}

std::vector<ErrorCode> MetaRedisBackend::Get(const KeyTypeVec &keys,
                                             const std::vector<std::string> &field_names,
                                             FieldMapVec &out_field_maps) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "get fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    return handle->Get(full_keys, field_names, out_field_maps);
}

std::vector<ErrorCode> MetaRedisBackend::GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(
            10, "get all fields fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    return handle->GetAllFields(full_keys, out_field_maps);
}

std::vector<ErrorCode> MetaRedisBackend::Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept {
    auto handle = client_pool_->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "exists fail, fail to acquire redis client, instance[%s]", instance_id_.c_str());
        return std::vector<ErrorCode>(keys.size(), EC_TIMEOUT);
    }
    std::vector<std::string> full_keys = AppendPrefixToKeys(keys);
    return handle->Exists(full_keys, out_is_exist_vec);
}

ErrorCode MetaRedisBackend::ListKeys(const std::string &cursor,
                                     const int64_t limit,
                                     std::string &out_next_cursor,
                                     std::vector<KeyType> &out_keys) noexcept {
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
    if (!StripPrefixInKeys(full_keys, out_keys)) {
        KVCM_LOG_ERROR("list keys fail, strip key prefix fail, instance[%s]", instance_id_.c_str());
        out_keys.clear();
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode MetaRedisBackend::RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept {
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
    if (!StripPrefixInKeys(full_keys, out_keys)) {
        KVCM_LOG_ERROR("random fail, strip key prefix fail, instance[%s]", instance_id_.c_str());
        out_keys.clear();
        return EC_ERROR;
    }
    return EC_OK;
}

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
