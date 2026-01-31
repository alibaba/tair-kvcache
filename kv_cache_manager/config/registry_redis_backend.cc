#include "kv_cache_manager/config/registry_redis_backend.h"

#include <cassert>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

RegistryRedisBackend::~RegistryRedisBackend() { client_.reset(); }
ErrorCode RegistryRedisBackend::Init(const StandardUri &standard_uri) noexcept {
    client_ = std::make_unique<RedisClient>(standard_uri);
    if (!client_ || !client_->Open()) {
        KVCM_LOG_ERROR("registry redis backend fail to open redis client");
        return EC_ERROR;
    }

    std::string cluster_name = standard_uri.GetParam("cluster_name");
    if (cluster_name.empty()) {
        KVCM_LOG_ERROR("registry redis backend fail to find cluster_name arg from uri");
        return EC_CONFIG_ERROR;
    }
    config_key_prefix_ = "kvcache_registry:" + cluster_name + ":";
    return EC_OK;
}

ErrorCode RegistryRedisBackend::Load(const std::string &key, std::map<std::string, std::string> &out_value) noexcept {
    std::lock_guard<std::mutex> lock(client_mutex_);
    std::vector<std::map<std::string, std::string>> out_fields;
    auto error_codes = client_->GetAllFields({config_key_prefix_ + key}, out_fields);
    assert(error_codes.size() == 1 && out_fields.size() == 1);
    out_value = std::move(out_fields[0]);
    return error_codes[0];
}

ErrorCode RegistryRedisBackend::Save(const std::string &key, const std::map<std::string, std::string> &value) noexcept {
    std::lock_guard<std::mutex> lock(client_mutex_);
    auto error_codes = client_->Set({config_key_prefix_ + key}, {value});
    assert(error_codes.size() == 1);
    return error_codes[0];
}

ErrorCode RegistryRedisBackend::Delete(const std::string &key) noexcept {
    std::lock_guard<std::mutex> lock(client_mutex_);
    auto error_codes = client_->Delete({config_key_prefix_ + key});
    assert(error_codes.size() == 1);
    return error_codes[0];
}

} // namespace kv_cache_manager
