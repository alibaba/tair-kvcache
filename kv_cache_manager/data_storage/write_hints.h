#pragma once

#include <string>
#include <vector>

namespace kv_cache_manager {

// Hints passed from the manager layer to a DataStorageBackend's write path.
// Backends that do not support affinity-aware placement should ignore these
// hints (default behavior of the legacy Create() overload).
//
// Note: whether the backend must honor these hints strictly is expressed by
// the separate `bool strict` parameter on DataStorageManager::Create /
// DataStorageBackend::CreateWithHints, NOT by a field on this struct. That
// keeps the "what to prefer" payload (this struct) orthogonal to the
// "must obey or may fall back" policy (the strict flag).
struct WriteHints {
    // Preferred storage node ids (e.g. hostname/IP) in priority order.
    // Empty = no affinity preference, backend uses its own placement logic.
    std::vector<std::string> preferred_node_ids;
};

inline bool HasAffinityHint(const WriteHints &hints) { return !hints.preferred_node_ids.empty(); }

} // namespace kv_cache_manager
