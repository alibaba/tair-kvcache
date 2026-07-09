#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {

class MetaIndexerConfig;
class MetaIndexer;
class MetricsRegistry;

class MetaIndexerManager {
public:
    MetaIndexerManager() = default;

    ~MetaIndexerManager() = default;

    // Set histogram configuration for revisit interval tracking.
    // Must be called before CreateMetaIndexer.
    void SetRevisitHistogramConfig(std::shared_ptr<MetricsRegistry> registry, const std::vector<double> &boundaries);

    ErrorCode CreateMetaIndexer(const std::string &instance_id,
                                const std::shared_ptr<MetaIndexerConfig> &config,
                                const std::vector<double> &boundaries = {});

    std::shared_ptr<MetaIndexer> GetMetaIndexer(const std::string &instance_id) const;

    ErrorCode DeleteMetaIndexer(const std::string &instance_id);

    size_t GetIndexerSize();

    std::map<std::string, std::shared_ptr<MetaIndexer>> GetIndexers() const;

    void DoCleanup();

private:
    std::shared_ptr<MetaIndexer> GetMetaIndexerUnsafe(const std::string &instance_id) const;

private:
    // instance_id -> MetaIndexer
    std::map<std::string, std::shared_ptr<MetaIndexer>> meta_indexers_;
    mutable std::shared_mutex mutex_;

    // Histogram configuration (shared across all indexers)
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::vector<double> revisit_boundaries_;
};

} // namespace kv_cache_manager
