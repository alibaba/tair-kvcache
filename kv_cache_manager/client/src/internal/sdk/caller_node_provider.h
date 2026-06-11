#pragma once

#include <string>

#include "kv_cache_manager/common/affinity_types.h"

namespace kv_cache_manager {

// Resolves the caller's local node identity for a given storage type.
//
// CallerNodeProvider is the polymorphic hook on the meta-client side that
// fills the CallerNode field. Each storage backend can supply its own
// concrete provider; the factory picks one per `DataStorageType`. The
// default implementation (`NoopCallerNodeProvider`) returns an empty
// CallerNode and is fully equivalent to the pre-affinity behaviour, so
// backends that do not need affinity (or builds that compile without
// optional backends) keep working unchanged.
//
// Lifetime / threading:
//   - Constructed once per MetaClient during Connect(), after RegisterInstance
//     returns the storage_configs JSON.
//   - GetCallerNode() may be called from arbitrary client threads; the Noop
//     impl is trivially thread-safe and concrete impls should keep that
//     contract (cache result internally if the lookup is expensive).
class CallerNodeProvider {
public:
    virtual ~CallerNodeProvider() = default;

    // Returns the caller's local node identity for the storage backend.
    // An empty node_id means "unknown" and is treated as
    // "affinity not enabled for this caller" by the server.
    virtual CallerNode GetCallerNode() = 0;
};

// Default fallback: returns empty CallerNode. Used when no concrete provider
// is applicable (no tair-mempool storage configured, build without
// ENABLE_TAIR_MEMPOOL, etc.). Behaviourally identical to the old client.
class NoopCallerNodeProvider : public CallerNodeProvider {
public:
    CallerNode GetCallerNode() override { return {}; }
};

} // namespace kv_cache_manager
