#pragma once

#include <shared_mutex>

namespace kv_cache_manager {

// coarse-grained read/write coordination between metrics
// registration / publication and instance / group / storage lifecycle
// removal.
//
// - sites that register tagged metrics (LocalMetricsReporter,
//   MetaServiceMetricsBase slow path) and the CacheManagerMetricsRecorder
//   publish span hold a shared lock around their critical region
// - lifecycle removal sites (AdminServiceImpl::RemoveInstance,
//   RemoveInstanceGroup, RemoveStorage) hold a unique lock for the
//   entire operation
//
// removals are very low frequency, so a single coarse-grained
// shared_mutex is chosen over fine-grained machinery: it eliminates
// the race interleaving of registration and tag-filter purge without
// any deferred cleanup or double-purge logic
struct MetricsLifecycle {
    mutable std::shared_mutex mut_;
};

} // namespace kv_cache_manager
