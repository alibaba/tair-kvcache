#pragma once
#include <memory>

namespace kv_cache_manager {
class CoordinationBackend;

class CoordinationBackendFactory {
public:
    static std::unique_ptr<CoordinationBackend>
    CreateAndInitCoordinationBackend(const std::string &coordination_backend_uri);
};

} // namespace kv_cache_manager
