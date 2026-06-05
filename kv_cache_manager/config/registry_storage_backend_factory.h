#pragma once

#include <functional>
#include <memory>
#include <string>

namespace kv_cache_manager {
class RegistryStorageBackend;

class RegistryStorageBackendFactory {
public:
    using CreatorFunc = std::function<std::unique_ptr<RegistryStorageBackend>()>;

    static std::unique_ptr<RegistryStorageBackend> CreateAndInitStorageBackend(const std::string &registry_storage_uri);

    static void RegisterType(const std::string &protocol, CreatorFunc creator);
};

} // namespace kv_cache_manager
