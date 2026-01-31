#pragma once
#include <memory>

namespace kv_cache_manager {
class RegistryStorageBackend;

class RegistryStorageBackendFactory {
public:
    static std::unique_ptr<RegistryStorageBackend> CreateAndInitStorageBackend(const std::string &registry_storage_uri);
};

} // namespace kv_cache_manager
