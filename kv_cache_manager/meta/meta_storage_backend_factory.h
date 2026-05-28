#pragma once
#include <memory>
#include <string>

namespace kv_cache_manager {
class MetaStorageBackendConfig;
class MetaStorageBackend;
class MetaCacheBaseBackend;

class MetaStorageBackendFactory {
public:
    // Generic backend factory used by callers that take any MetaStorageBackend
    // and only need a single backend instance. Recognises redis / local /
    // dummy storage types.
    static std::unique_ptr<MetaStorageBackend>
    CreateAndInitStorageBackend(const std::string &instance_id,
                                const std::shared_ptr<MetaStorageBackendConfig> &config);

    // Persistent backend factory. The persistent slot in the dual-backend
    // setup is the source-of-truth, so only durable / shareable backends are
    // accepted here: redis (production) and dummy (tests).
    static std::unique_ptr<MetaStorageBackend>
    CreatePersistentBackend(const std::string &instance_id, const std::shared_ptr<MetaStorageBackendConfig> &config);

    // Local-cache backend factory. The local slot is the in-memory hot cache
    // and must implement MetaCacheBaseBackend's conditional write API; only
    // local is accepted.
    static std::unique_ptr<MetaCacheBaseBackend>
    CreateCacheBackend(const std::string &instance_id, const std::shared_ptr<MetaStorageBackendConfig> &config);
};

} // namespace kv_cache_manager
