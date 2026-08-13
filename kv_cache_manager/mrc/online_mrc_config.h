#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace kv_cache_manager {

// Configuration for the observation-only online MRC profiler.
// All fields have conservative defaults; the feature is off unless
// explicitly enabled via server config.
struct OnlineMrcConfig {
    bool enable = false;
    // Largest capacity (in blocks) for which the exact online curve is
    // maintained. The LRU state is safely truncated below this boundary, so
    // memory is bounded while every reported point <= the limit stays exact.
    int64_t max_tracked_blocks = 65536;
    // Tumbling window length for the "recent window" curve.
    int64_t window_seconds = 3600;
    // Interval of the full Facts -> MRC projection. Lightweight connection and
    // queue metrics continue to use the optimizer metrics report interval.
    int64_t report_interval_seconds = 60;
    // Capacity grid (GB) at which theoretical hit rate gauges are reported.
    std::vector<double> capacity_gb_grid = {64, 128, 256, 340, 512, 1024};
    // Fallback bytes-per-block when the instance info lookup fails;
    // 0 means "skip GB-based gauges until the lookup succeeds".
    int64_t default_bytes_per_block = 0;
    // Upper bound of concurrently profiled instances. Facts within every
    // accepted instance are retained for the process lifetime without a
    // count or time bound.
    int32_t max_instances = 256;
    // Bound of the pending event queue (events, not keys).
    int64_t queue_max_size = 10000;
    // Optimizer-side ingress queue bound (RPC batches).
    int64_t receiver_queue_max_batches = 1024;
    // Release lanes with no observed request for this duration; 0 disables.
    int64_t idle_expire_seconds = 86400;

    // The optimizer discovers and connects to all KVCM nodes itself. When the
    // feature is enabled this URL is required; KVCM never needs an optimizer
    // endpoint.
    std::string kvcm_service_discovery_url;
    int64_t discovery_refresh_interval_ms = 30000;
    int64_t connect_timeout_ms = 500;
    int64_t reconnect_interval_ms = 1000;
    int64_t max_frame_bytes = 8 * 1024 * 1024;

};

struct OptimizerTraceForwarderConfig {
    bool enable = false;
    std::string endpoint;
    std::string cluster;
    int64_t queue_max_size = 10000;
    int64_t max_batch_keys = 8192;
    int64_t rpc_timeout_ms = 200;
    int64_t report_interval_seconds = 10;
    // Empty keeps the historical "all instances" behavior. Production
    // canaries should set exact instance IDs so a shared KVCM deployment does
    // not forward unrelated models into the observation stream.
    std::unordered_set<std::string> instance_allowlist;
};

} // namespace kv_cache_manager
