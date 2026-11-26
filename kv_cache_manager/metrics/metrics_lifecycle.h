#pragma once

#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_set>

namespace kv_cache_manager {

// coarse-grained read/write coordination between metrics
// registration / publication and instance / group / storage lifecycle
// removal.
//
// - sites that register tagged metrics (LocalMetricsReporter,
//   MetaServiceMetricsBase slow path) and the CacheManagerMetricsRecorder
//   publish span hold a shared lock around their critical region
// - CacheManager::RegisterInstance holds a shared lock so it cannot
//   overlap the exclusive phases of instance removal
// - lifecycle removal sites hold a unique lock while removing tagged
//   metrics; RemoveInstance releases it during cache and metadata cleanup
// - removal_mut_ lets different instances be removed concurrently, but
//   excludes storage removal or replacement while they are being removed
// - removing_instance_ids_ rejects concurrent removal of the same instance
struct MetricsLifecycle {
    // Acquire the instance removal gate before removal_mut_, then mut_.
    mutable std::mutex instance_removal_mut_;
    std::unordered_set<std::string> removing_instance_ids_;
    mutable std::shared_mutex removal_mut_;
    mutable std::shared_mutex mut_;
};

} // namespace kv_cache_manager
