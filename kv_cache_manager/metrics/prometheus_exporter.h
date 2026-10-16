#pragma once

#include <string>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

// Serializes all metrics held in a MetricsRegistry into the Prometheus
// text exposition format (text/plain; version=0.0.4; charset=utf-8).
//
// Naming rules applied during serialization:
//   - Metric names, label keys and the prefix are sanitized to the
//     Prometheus identifier charset [a-zA-Z0-9_].  Characters outside
//     that set are replaced with underscores; a leading digit gets a
//     '_' prefix.
//   - A configurable prefix (default "kvcm") is prepended, separated
//     by an underscore.
//   - MetricsTags become Prometheus labels (keys sanitized, values
//     escaped per the exposition format).
//   - CounterValue metrics are emitted with TYPE counter.
//   - GaugeValue   metrics are emitted with TYPE gauge.
class PrometheusExporter {
public:
    // Prepare process-level labels once at startup. Reserved or empty names are
    // omitted with a warning for each tag.
    // Normalized collisions overwrite the previous value and report the old
    // and effective values to stderr. Callers should avoid colliding names.
    static MetricsTags PrepareCommonTags(const MetricsTags &common_tags);

    // Serialize all metrics in |registry| to Prometheus text format.
    // |prefix| is prepended to every metric name (e.g. "kvcm").
    // |common_labels| must be prepared by PrepareCommonTags. Metric tags win
    // collisions after name sanitization. Common label names must not overlap
    // business labels: merging can otherwise collapse distinct registry series.
    static std::string
    Expose(MetricsRegistry &registry, const std::string &prefix = "kvcm", const MetricsTags &common_labels = {});
};

} // namespace kv_cache_manager
