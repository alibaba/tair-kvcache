#include "kv_cache_manager/meta/raft/raft_mode.h"

#include <atomic>

namespace kv_cache_manager {
namespace raft_meta {

static std::atomic<bool> g_raft_mode_active{false};

bool IsRaftModeActive() { return g_raft_mode_active.load(std::memory_order_acquire); }

void SetRaftModeActive(bool active) { g_raft_mode_active.store(active, std::memory_order_release); }

} // namespace raft_meta
} // namespace kv_cache_manager
