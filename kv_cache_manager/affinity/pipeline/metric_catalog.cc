#include "kv_cache_manager/affinity/pipeline/metric_catalog.h"

namespace kv_cache_manager {

bool MetricCatalog::IsKnown(const std::string &name) {
    return name == "free_bytes" || name == "load_ratio" || name == "rx_mbps" || name == "tx_mbps";
}

std::optional<double> MetricCatalog::Extract(const std::string &name, const NodeMetrics &node) {
    if (name == "free_bytes") {
        return static_cast<double>(node.free_bytes);
    }
    if (name == "load_ratio") {
        return node.load_ratio;
    }
    if (name == "rx_mbps") {
        return node.rx_mbps;
    }
    if (name == "tx_mbps") {
        return node.tx_mbps;
    }
    return std::nullopt;
}

} // namespace kv_cache_manager
