#pragma once

#include <string>

#include "kv_cache_manager/client/src/internal/sdk/caller_node_provider.h"

namespace kv_cache_manager {

// NFS-backed CallerNodeProvider: uses the local IP as node_id.
// NFS is a shared-file system where every node can reach every block,
// so the caller identity is simply the host IP — no SDK lookup needed.
class NfsCallerNodeProvider : public CallerNodeProvider {
public:
    NfsCallerNodeProvider();
    ~NfsCallerNodeProvider() override = default;
    ClientCallerNode GetCallerNode() override { return caller_; }

private:
    ClientCallerNode caller_;
};

} // namespace kv_cache_manager
