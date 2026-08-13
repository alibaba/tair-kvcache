#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/event/event_publisher.h"
#include "kv_cache_manager/mrc/mrc_profiler.h"
#include "kv_cache_manager/mrc/online_mrc_config.h"

namespace kv_cache_manager {

class LoopThread;
class MetricsRegistry;

// Observation-only online MRC pipeline.
//
// Registered on the EventManager next to the log publisher, it consumes
// CacheGetEvent (the same access stream the offline trace replay uses),
// feeds each instance's block keys into a per-instance MrcProfiler on a
// single background thread, and periodically reports theoretical hit rate
// gauges to the MetricsRegistry. Nothing reads its state on the request
// path; dropping events under pressure only degrades observation quality.
class OnlineMrcEventConsumer : public EventPublisher {
public:
    // Resolves bytes-per-block for an instance (sum of location spec sizes).
    // Returns 0 when unknown; the consumer retries on later report rounds and
    // falls back to config.default_bytes_per_block.
    using BytesPerBlockResolver = std::function<int64_t(const std::string &instance_id)>;

    OnlineMrcEventConsumer(const OnlineMrcConfig &config,
                           std::shared_ptr<MetricsRegistry> metrics_registry,
                           BytesPerBlockResolver bytes_per_block_resolver);
    ~OnlineMrcEventConsumer() override;

    bool Init(const std::string &config) override;
    bool Publish(const std::shared_ptr<BaseEvent> &event) override;
    bool Stop() override;

    // Full per-instance curves as a JSON document (debug endpoint).
    std::string DumpCurvesJson() const;

private:
    struct Lane {
        explicit Lane(const MrcProfiler::Options &options) : profiler(options) {}
        mutable std::mutex mutex;
        MrcProfiler profiler;
        int64_t bytes_per_block = 0; // resolved lazily; 0 = unknown
    };

    void WorkerThread();
    void ReportMetrics();
    std::shared_ptr<Lane> GetOrCreateLane(const std::string &instance_id);
    int64_t ResolveBytesPerBlock(const std::string &instance_id, Lane &lane);

    OnlineMrcConfig config_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    BytesPerBlockResolver bytes_per_block_resolver_;

    std::thread worker_;
    std::shared_ptr<LoopThread> report_thread_;

    mutable std::mutex lanes_mutex_;
    std::map<std::string, std::shared_ptr<Lane>> lanes_;
    bool lane_overflow_logged_ = false;
};

} // namespace kv_cache_manager
