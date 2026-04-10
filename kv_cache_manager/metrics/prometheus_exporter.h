#pragma once

#include <string>

namespace kv_cache_manager {

class MetricsRegistry;

// Serializes all metrics held in a MetricsRegistry into the Prometheus
// text exposition format (text/plain; version=0.0.4; charset=utf-8).
//
// Naming rules applied during serialization:
//   - Dots in the internal metric name are replaced with underscores.
//   - A configurable prefix (default "kvcm") is prepended, separated
//     by an underscore.
//   - MetricsTags become Prometheus labels.
//   - CounterValue metrics are emitted with TYPE counter.
//   - GaugeValue   metrics are emitted with TYPE gauge.
class PrometheusExporter {
public:
    // Serialize all metrics in |registry| to Prometheus text format.
    // |prefix| is prepended to every metric name (e.g. "kvcm").
    static std::string Expose(MetricsRegistry &registry, const std::string &prefix = "kvcm");
};

} // namespace kv_cache_manager
