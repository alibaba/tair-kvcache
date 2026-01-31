#pragma once
#include <memory>

namespace kv_cache_manager {
class MetaStorageBackendConfig;
class MetaStorageBackend;

class MetaStorageBackendFactory {
public:
    static std::unique_ptr<MetaStorageBackend>
    CreateAndInitStorageBackend(const std::string &instance_id,
                                const std::shared_ptr<MetaStorageBackendConfig> &config);
};

} // namespace kv_cache_manager
