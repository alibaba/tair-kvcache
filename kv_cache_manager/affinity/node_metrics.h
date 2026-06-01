#pragma once

#include <cstdint>
#include <string>

namespace kv_cache_manager {

// Runtime metrics describing a single node in the affinity pool.
//
// Under the v1 co-location assumption a single physical machine is both an
// inference node and a storage node, so node_id (its routing identity, e.g.
// IP/hostname used by WriteHints.preferred_node_ids) and node_name (a human
// readable label used by node_name based filter/sample primitives) describe
// the same machine.
//
// The scalar metric fields below are the v1 placeholder set referenced by the
// metric registry. They are zeroed out until a real metrics source is wired
// up; filter / sort over them currently degrades to the missing-metric
// semantics documented at MetricCatalog / FilterCond.
struct NodeMetrics {
    std::string node_id;
    std::string node_name;

    // Storage capacity / load on the machine.
    int64_t free_bytes = 0;
    double load_ratio = 0.0;

    // Network throughput observed on the machine (Mbps).
    double rx_mbps = 0.0;
    double tx_mbps = 0.0;

    // Wall-clock timestamp (microseconds) of the last update; used by callers
    // to drop stale entries before calling UpsertNodeMetrics.
    int64_t updated_at_us = 0;
};

} // namespace kv_cache_manager
