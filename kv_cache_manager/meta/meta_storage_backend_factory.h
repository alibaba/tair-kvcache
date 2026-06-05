#pragma once
#include <memory>
#include <string>

namespace kv_cache_manager {
class MetaStorageBackendConfig;
class MetaStorageBackend;
class MetaCacheBaseBackend;

class MetaStorageBackendFactory {
public:
    static std::unique_ptr<MetaStorageBackend>
    CreateAndInitStorageBackend(const std::string &instance_id,
                                const std::shared_ptr<MetaStorageBackendConfig> &config);

    static std::unique_ptr<MetaStorageBackend>
    CreatePersistentBackend(const std::string &instance_id, const std::shared_ptr<MetaStorageBackendConfig> &config);

    static std::unique_ptr<MetaCacheBaseBackend>
    CreateCacheBackend(const std::string &instance_id, const std::shared_ptr<MetaStorageBackendConfig> &config);

    static void SetRaftModeEnabled(bool enabled);
};

} // namespace kv_cache_manager
