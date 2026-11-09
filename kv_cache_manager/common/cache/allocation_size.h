#pragma once

#include <cstddef>

namespace kv_cache_manager {

// Use the active jemalloc's size class when available; otherwise keep the requested size.
size_t EstimateAllocationSize(size_t requested) noexcept;

} // namespace kv_cache_manager
