#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/common_define.h"
#include "kv_cache_manager/data_storage/write_hints.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

// Result of a backend Create call. `node_id` is the storage node that
// actually served the allocation; empty string means the backend does not
// report per-node placement.
struct LocationDescriptor {
    ErrorCode ec = EC_OK;
    DataStorageUri uri;
    std::string node_id;
};

class DataStorageBackend {
public:
    DataStorageBackend() = delete;
    explicit DataStorageBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : metrics_registry_(std::move(metrics_registry)) {}

    virtual ~DataStorageBackend() = default;
    virtual DataStorageType GetType() = 0;
    virtual bool Available() = 0;
    virtual double GetStorageUsageRatio(const std::string &trace_id) const = 0;
    inline bool IsOpen() const { return is_open_.load(std::memory_order_relaxed); }
    inline void SetOpen(bool open) { is_open_.store(open, std::memory_order_relaxed); }
    inline void SetAvailable(bool available) { is_available_.store(available, std::memory_order_relaxed); }
    std::shared_ptr<DataStorageMetricsCollector> GetMetricsCollector() { return metrics_collector_; }
    virtual const StorageConfig &GetStorageConfig() { return config_; }

public:
    virtual ErrorCode Open(const StorageConfig &config, const std::string &trace_id) {
        config_ = config;
        metrics_collector_ = std::make_shared<DataStorageMetricsCollector>(
            metrics_registry_,
            MetricsTags{{"type", ToString(config.type())}, {"unique_name", config.global_unique_name()}});
        if (!metrics_collector_->Init()) {
            metrics_collector_ = nullptr;
        }
        // TODO (rui): handle metrics collector unregistration during Close()
        return DoOpen(config, trace_id);
    }
    virtual ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) = 0;
    virtual ErrorCode Close() = 0;
    virtual std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                                     size_t size_per_key,
                                                                     const std::string &trace_id,
                                                                     std::function<void()> cb) = 0;

    // Affinity-aware Create.
    //
    // `hints` carries the preferred placement; `strict` controls how the
    // backend treats those hints:
    //   strict=true  -> backend MUST allocate on hints.preferred_node_ids only
    //                   and must surface an error for keys it cannot place
    //                   there (no silent fallback to other nodes).
    //   strict=false -> hints are advisory; backend may fall back to any node
    //                   when the preferred ones are unavailable.
    // When hints.preferred_node_ids is empty, `strict` is meaningless.
    //
    // Backends that can route keys to specific storage nodes should override
    // this. The default implementation ignores both `hints` and `strict` and
    // forwards to the legacy Create(), filling node_id with "" — every
    // existing backend keeps working as a one-line fallthrough until it
    // actually wants to honor affinity.
    //
    // SupportsAffinity() lets the manager layer (and DataStorageSelector in
    // future) detect at runtime whether a backend will act on hints; defaults
    // to false.
    //
    // Returns LocationDescriptor (ec + uri + node_id) so the backend can
    // report which node actually served the allocation. Default implementation
    // forwards to the legacy Create(), which keeps every existing backend a
    // one-line fallthrough until it actually wants to honor affinity.
    virtual std::vector<LocationDescriptor> CreateWithHints(const std::vector<std::string> &keys,
                                                            size_t size_per_key,
                                                            const WriteHints &hints,
                                                            bool strict,
                                                            const std::string &trace_id,
                                                            std::function<void()> cb) {
        (void)hints;
        (void)strict;
        auto legacy = Create(keys, size_per_key, trace_id, std::move(cb));
        std::vector<LocationDescriptor> out;
        out.reserve(legacy.size());
        for (auto &p : legacy) {
            out.push_back(LocationDescriptor{p.first, std::move(p.second), /*node_id=*/""});
        }
        return out;
    }

    virtual bool SupportsAffinity() const { return false; }

    // Returns per-node capacity metrics. Default returns empty (backend does
    // not report metrics); missing metrics are treated as permissive.
    virtual std::vector<NodeMetrics> SnapshotPerNodeMetrics() const { return {}; }
    virtual std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &trace_id, std::function<void()> cb) = 0;
    virtual std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) = 0;
    virtual std::vector<bool> MightExist(const std::vector<DataStorageUri> &storage_uris) {
        // a low-latency version of Exist()
        // implementation is required to return ASAP;
        // or it should rather return false-positive result, e.g.,
        // all true if low-latency can not be guaranteed
        return std::vector<bool>(storage_uris.size(), true);
    }
    virtual std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) = 0;
    virtual std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) = 0;

protected:
    inline bool IsAvailable() const { return is_available_.load(std::memory_order_relaxed); }

protected:
    StorageConfig config_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<DataStorageMetricsCollector> metrics_collector_;

private:
    std::atomic_bool is_open_ = false;
    std::atomic_bool is_available_ = false;
};

} // namespace kv_cache_manager
