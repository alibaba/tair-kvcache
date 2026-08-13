#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/mrc/mrc_profiler.h"
#include "kv_cache_manager/mrc/online_mrc_config.h"

namespace kv_cache_manager {

class MetricsRegistry;

struct OnlineMrcSpan {
    std::string cluster;
    std::string instance_group;
    std::string instance_id;
    std::string source_id;
    int64_t bytes_per_block = 0;
    int64_t event_time_us = 0;
    uint64_t sequence_number = 0;
    std::vector<int64_t> keys;
};

struct OnlineMrcBatch {
    std::vector<OnlineMrcSpan> spans;
    uint64_t dropped_spans = 0;
    uint64_t dropped_keys = 0;
};

// Observation-only lane registry shared by the in-process validation path and
// the external optimizer service. It has no control-plane or quota API.
class OnlineMrcRegistry {
public:
    OnlineMrcRegistry(const OnlineMrcConfig &config, std::shared_ptr<MetricsRegistry> metrics_registry);

    bool Observe(OnlineMrcSpan span);
    void RecordDropped(uint64_t spans, uint64_t keys);
    void ReportMetrics();
    void ReportIngressMetrics(size_t queue_size, uint64_t dropped_batches);
    std::string DumpCurvesJson() const;

    size_t LaneCount() const;
    uint64_t dropped_spans() const { return dropped_spans_.load(std::memory_order_relaxed); }

private:
    struct Lane {
        Lane(const OnlineMrcConfig &config, OnlineMrcSpan span);

        mutable std::mutex mutex;
        MrcProfiler profiler;
        std::string cluster;
        std::string instance_group;
        std::string instance_id;
        std::string source_id;
        int64_t bytes_per_block = 0;
        int64_t last_observed_steady_us = 0;
        uint64_t last_sequence_number = 0;
        uint64_t sequence_gaps = 0;
        uint64_t source_switches = 0;
    };

    using LaneKey = std::pair<std::string, std::string>; // (cluster, instance_id)

    std::shared_ptr<Lane> GetOrCreateLane(OnlineMrcSpan span);
    void ExpireIdleLanes(int64_t now_steady_us);

    OnlineMrcConfig config_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;

    mutable std::mutex lanes_mutex_;
    std::map<LaneKey, std::shared_ptr<Lane>> lanes_;
    bool lane_overflow_logged_ = false;

    std::atomic<uint64_t> dropped_spans_{0};
    std::atomic<uint64_t> dropped_keys_{0};
    std::atomic<uint64_t> out_of_order_spans_{0};
};

} // namespace kv_cache_manager
