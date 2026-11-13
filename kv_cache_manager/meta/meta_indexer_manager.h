#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {

class MetaIndexerConfig;
class MetaIndexer;
class MetricsRegistry;
class QueryExecutor;

class MetaIndexerManager {
public:
    MetaIndexerManager() = default;

    virtual ~MetaIndexerManager() = default;

    // Set histogram configuration for revisit interval tracking.
    // Must not run concurrently with CreateMetaIndexer.
    void SetRevisitHistogramConfig(std::shared_ptr<MetricsRegistry> registry, const std::vector<double> &boundaries);

    // Configures the single process-level executor shared by every indexer.
    // Must be called before CreateMetaIndexer. worker_count == 1 keeps the
    // exact serial behavior while preserving the same code path.
    bool ConfigureQueryExecutor(std::size_t worker_count, std::size_t parallel_threshold, std::size_t chunk_size);

    // Callers serialize Create/Extract/Restore for the same instance.
    ErrorCode CreateMetaIndexer(const std::string &instance_id,
                                const std::shared_ptr<MetaIndexerConfig> &config,
                                const std::vector<double> &boundaries = {});

    std::shared_ptr<MetaIndexer> GetMetaIndexer(const std::string &instance_id) const;

    std::shared_ptr<MetaIndexer> ExtractMetaIndexer(const std::string &instance_id);
    bool RestoreMetaIndexer(const std::string &instance_id, std::shared_ptr<MetaIndexer> indexer);

    size_t GetIndexerSize();

    std::map<std::string, std::shared_ptr<MetaIndexer>> GetIndexers() const;
    using StorageBackendConfigs = std::map<std::string, std::pair<std::string, std::string>>;
    // Copy only configuration, without retaining indexers or waiting for recovery.
    std::optional<StorageBackendConfigs> TryGetStorageBackendConfigs() const;

    // Callers drain creation and instance management before cleanup.
    void DoCleanup();

protected:
    // Allows tests to block initialization without holding the map lock.
    virtual ErrorCode InitMetaIndexer(const std::shared_ptr<MetaIndexer> &indexer,
                                      const std::string &instance_id,
                                      const std::shared_ptr<MetaIndexerConfig> &config);

private:
    std::shared_ptr<MetaIndexer> GetMetaIndexerUnsafe(const std::string &instance_id) const;

private:
    // instance_id -> MetaIndexer
    std::map<std::string, std::shared_ptr<MetaIndexer>> meta_indexers_;
    mutable std::shared_mutex mutex_;

    // Histogram configuration (shared across all indexers)
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::vector<double> revisit_boundaries_;
    std::shared_ptr<QueryExecutor> query_executor_;
};

} // namespace kv_cache_manager
