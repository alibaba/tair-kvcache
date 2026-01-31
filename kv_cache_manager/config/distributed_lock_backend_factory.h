#pragma once
#include <memory>

namespace kv_cache_manager {
class DistributedLockBackend;

class DistributedLockBackendFactory {
public:
    static std::unique_ptr<DistributedLockBackend>
    CreateAndInitDistributedLockBackend(const std::string &lock_backend_uri);
};

} // namespace kv_cache_manager
