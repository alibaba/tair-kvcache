#pragma once

namespace kv_cache_manager {
namespace raft_meta {

bool IsRaftModeActive();
void SetRaftModeActive(bool active);

} // namespace raft_meta
} // namespace kv_cache_manager
