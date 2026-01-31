#pragma once

#include "kv_cache_manager/common/concurrent_hash_map.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/registry_storage_backend.h"

namespace kv_cache_manager {

class RegistryLocalBackend : public RegistryStorageBackend {
public:
    RegistryLocalBackend() = default;
    ~RegistryLocalBackend() override;

public:
    ErrorCode Init(const StandardUri &standard_uri) noexcept override;
    ErrorCode Load(const std::string &key, std::map<std::string, std::string> &out_value) noexcept override;
    ErrorCode Save(const std::string &key, const std::map<std::string, std::string> &value) noexcept override;
    ErrorCode Delete(const std::string &key) noexcept override;

private:
    ErrorCode PersistToPath();
    ErrorCode Recover();

private:
    std::mutex mutex_;
    std::string path_;
    ConcurrentHashMap<std::string, std::map<std::string, std::string>> table_;
    bool enable_persistence_ = false;
};

} // namespace kv_cache_manager
