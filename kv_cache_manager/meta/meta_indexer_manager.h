#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {

class MetaIndexerConfig;
class InstanceInfo;
class MetaIndexer;
class MetricsRegistry;
class QueryExecutor;

class MetaIndexerManager {
public:
    MetaIndexerManager() = default;

    virtual ~MetaIndexerManager() = default;

    // Set histogram configuration for revisit interval tracking.
    // Waits for in-flight creation before changing defaults for future indexers.
    void SetRevisitHistogramConfig(std::shared_ptr<MetricsRegistry> registry, const std::vector<double> &boundaries);

    // Configures the single process-level executor shared by every indexer.
    // Must be called before CreateMetaIndexer. worker_count == 1 keeps the
    // exact serial behavior while preserving the same code path.
    bool ConfigureQueryExecutor(std::size_t worker_count, std::size_t parallel_threshold, std::size_t chunk_size);

    // Initialization is serialized per instance, without holding the map lock.
    ErrorCode CreateMetaIndexer(const std::string &instance_id,
                                const std::shared_ptr<MetaIndexerConfig> &config,
                                const std::vector<double> &boundaries,
                                std::shared_ptr<const InstanceInfo> instance_info);

    std::shared_ptr<MetaIndexer> GetMetaIndexer(const std::string &instance_id) const;

    // Reuse only an indexer initialized for the same registry registration and configuration.
    // Unlike CreateMetaIndexer's EC_EXIST, this explicitly checks compatibility.
    ErrorCode GetMetaIndexerForReuse(const std::string &instance_id,
                                     const std::shared_ptr<MetaIndexerConfig> &config,
                                     const std::vector<double> &boundaries,
                                     const std::shared_ptr<const InstanceInfo> &instance_info,
                                     std::shared_ptr<MetaIndexer> &out_indexer) const;

    std::shared_ptr<MetaIndexer> ExtractMetaIndexer(const std::string &instance_id);
    bool RestoreMetaIndexer(const std::string &instance_id, std::shared_ptr<MetaIndexer> indexer);

    size_t GetIndexerSize();

    std::map<std::string, std::shared_ptr<MetaIndexer>> GetIndexers() const;
    using StorageBackendConfigs = std::map<std::string, std::pair<std::string, std::string>>;
    // Copy only configuration, without retaining indexers or waiting for recovery.
    std::optional<StorageBackendConfigs> TryGetStorageBackendConfigs() const;

    // Waits for creation/extraction/restoration, then clears the managed indexers.
    // Callers still drain traffic and coordinate cross-manager instance operations.
    void DoCleanup();

protected:
    // Allows tests to block initialization without holding the map lock.
    virtual ErrorCode InitMetaIndexer(const std::shared_ptr<MetaIndexer> &indexer,
                                      const std::string &instance_id,
                                      const std::shared_ptr<MetaIndexerConfig> &config);

private:
    std::shared_ptr<MetaIndexer> GetMetaIndexerUnsafe(const std::string &instance_id) const;
    std::shared_ptr<std::mutex> GetInstanceMutex(const std::string &instance_id) const;

private:
    // instance_id -> MetaIndexer
    std::map<std::string, std::shared_ptr<MetaIndexer>> meta_indexers_;
    mutable std::shared_mutex mutex_;
    // Lock order: lifecycle -> instance -> map. Slow initialization holds no map lock.
    // Configuration and cleanup exclude creation/extraction/restoration/reuse.
    mutable std::shared_mutex lifecycle_mutex_;
    mutable std::unordered_map<std::string, std::weak_ptr<std::mutex>> instance_mutexes_;

    // Histogram configuration (shared across all indexers)
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::vector<double> revisit_boundaries_;
    std::shared_ptr<QueryExecutor> query_executor_;
};

} // namespace kv_cache_manager
