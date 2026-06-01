#pragma once

#include <optional>
#include <string>

#include "kv_cache_manager/affinity/node_metrics.h"

namespace kv_cache_manager {

// Registry of metric names that may appear in `filter` / `sort` slots of a
// CandidatePipeline.
//
// The v1 metric set is the four placeholder fields on NodeMetrics:
//
//   - "free_bytes"  -> NodeMetrics::free_bytes
//   - "load_ratio"  -> NodeMetrics::load_ratio
//   - "rx_mbps"     -> NodeMetrics::rx_mbps
//   - "tx_mbps"     -> NodeMetrics::tx_mbps
//
// Each registered metric maps to a value extractor `Extract(node)` that
// returns either the metric's value, or std::nullopt to mean "this candidate
// has no observation for this metric". Filter and sort then apply their own
// missing-metric policies on top of this:
//   - filter leaf with missing metric -> evaluates to true (permissive)
//   - sort term with missing metric -> contributes 0 to the score
//
// CandidatePipeline parsing rejects any metric name that is not registered, so an
// undefined metric in a config file fails fast at load time rather than
// silently returning all-missing.
class MetricCatalog {
public:
    // True if `name` is a registered metric.
    static bool IsKnown(const std::string &name);

    // Read metric `name` from `node`. Returns nullopt if the name is
    // unregistered (treat as missing). All v1 metrics are always-present
    // double-valued reads off NodeMetrics, so non-nullopt is the common case.
    static std::optional<double> Extract(const std::string &name, const NodeMetrics &node);
};

} // namespace kv_cache_manager
