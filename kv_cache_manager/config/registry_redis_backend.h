#pragma once

#include <mutex>
#include <string>

#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/config/registry_storage_backend.h"

namespace kv_cache_manager {

class RegistryRedisBackend : public RegistryStorageBackend {
public:
    RegistryRedisBackend() = default;
    ~RegistryRedisBackend() override;

private:
    ErrorCode Init(const StandardUri &standard_uri) noexcept override;
    ErrorCode Load(const std::string &key, std::map<std::string, std::string> &out_value) noexcept override;
    ErrorCode Save(const std::string &key, const std::map<std::string, std::string> &value) noexcept override;
    ErrorCode Delete(const std::string &key) noexcept override;
    ErrorCode UpsertField(const std::string &key, const std::string &field, const std::string &value) noexcept override;
    ErrorCode DeleteField(const std::string &key, const std::string &field) noexcept override;

private:
    std::mutex client_mutex_;
    std::unique_ptr<RedisClient> client_;
    std::string config_key_prefix_;
};

} // namespace kv_cache_manager
