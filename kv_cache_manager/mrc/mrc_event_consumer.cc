#include "kv_cache_manager/mrc/mrc_event_consumer.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kv_cache_manager {

namespace {

constexpr char kMetricTheoreticalHitRate[] = "online_mrc.theoretical_hit_rate";
constexpr char kMetricMaxTrackedHitRate[] = "online_mrc.max_tracked_hit_rate";
constexpr char kMetricTrackedBlocks[] = "online_mrc.tracked_blocks";
constexpr char kMetricTrackedCapacityBlocks[] = "online_mrc.tracked_capacity_blocks";
constexpr char kMetricExactCoverage[] = "online_mrc.exact_coverage";
constexpr char kMetricMemoryBytes[] = "online_mrc.memory_bytes";
constexpr char kMetricDroppedEvents[] = "online_mrc.dropped_events";
constexpr char kMetricTrackedInstances[] = "online_mrc.tracked_instances";
constexpr char kMetricQueueSize[] = "online_mrc.queue_size";

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

std::string FormatCapacityGb(double capacity_gb) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", capacity_gb);
    return buf;
}

int64_t NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace

OnlineMrcEventConsumer::OnlineMrcEventConsumer(const OnlineMrcConfig &config,
                                               std::shared_ptr<MetricsRegistry> metrics_registry,
                                               BytesPerBlockResolver bytes_per_block_resolver)
    : config_(config)
    , metrics_registry_(std::move(metrics_registry))
    , bytes_per_block_resolver_(std::move(bytes_per_block_resolver)) {}

OnlineMrcEventConsumer::~OnlineMrcEventConsumer() {
    if (running_) {
        Stop();
    }
}

bool OnlineMrcEventConsumer::Init(const std::string & /*config*/) {
    InitBasicQueue(static_cast<size_t>(config_.queue_max_size));
    running_ = true;
    worker_ = std::thread(&OnlineMrcEventConsumer::WorkerThread, this);
    report_thread_ = LoopThread::CreateLoopThread([this]() { ReportMetrics(); },
                                                  config_.report_interval_seconds * 1000000,
                                                  "online_mrc_report");
    if (!report_thread_) {
        KVCM_LOG_ERROR("online mrc: create report loop thread failed");
        Stop();
        return false;
    }
    KVCM_LOG_INFO("online mrc consumer started: max_tracked_blocks=%" PRId64 " window_seconds=%" PRId64
                  " report_interval_seconds=%" PRId64,
                  config_.max_tracked_blocks,
                  config_.window_seconds,
                  config_.report_interval_seconds);
    return true;
}

bool OnlineMrcEventConsumer::Publish(const std::shared_ptr<BaseEvent> &event) {
    if (!running_ || !event) {
        return false;
    }
    // Only the read-path access stream matters for the theoretical hit rate.
    if (std::dynamic_pointer_cast<CacheGetEvent>(event) == nullptr) {
        return true;
    }
    // BasicEnqueue drops (and counts) when the queue is full: observation
    // quality degrades but the request path is never blocked.
    BasicEnqueue(event);
    return true;
}

bool OnlineMrcEventConsumer::Stop() {
    if (!running_) {
        return true;
    }
    running_ = false;
    if (report_thread_) {
        report_thread_->Stop();
        report_thread_.reset();
    }
    ClearBasicQueue();
    if (basic_queue_) {
        basic_queue_->queue_cv.notify_all();
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    return true;
}

void OnlineMrcEventConsumer::WorkerThread() {
    while (running_) {
        BasicWait();

        std::shared_ptr<BaseEvent> event;
        while (BasicDequeue(event)) {
            auto cache_get_event = std::dynamic_pointer_cast<CacheGetEvent>(event);
            if (!cache_get_event) {
                continue;
            }
            const auto &keys = cache_get_event->get_keys();
            if (keys.empty()) {
                continue;
            }
            auto lane = GetOrCreateLane(cache_get_event->event_source());
            if (!lane) {
                continue;
            }
            std::lock_guard<std::mutex> guard(lane->mutex);
            lane->profiler.Observe(keys, NowUs());
        }
    }
}

std::shared_ptr<OnlineMrcEventConsumer::Lane> OnlineMrcEventConsumer::GetOrCreateLane(const std::string &instance_id) {
    if (instance_id.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(lanes_mutex_);
    auto it = lanes_.find(instance_id);
    if (it != lanes_.end()) {
        return it->second;
    }
    if (static_cast<int32_t>(lanes_.size()) >= config_.max_instances) {
        if (!lane_overflow_logged_) {
            lane_overflow_logged_ = true;
            KVCM_LOG_WARN("online mrc: instance lane limit[%d] reached, instance[%s] and later newcomers are ignored",
                          config_.max_instances,
                          instance_id.c_str());
        }
        return nullptr;
    }
    MrcProfiler::Options options;
    options.max_tracked_blocks = config_.max_tracked_blocks;
    options.window_seconds = config_.window_seconds;
    auto lane = std::make_shared<Lane>(options);
    lanes_[instance_id] = lane;
    KVCM_LOG_INFO("online mrc: start profiling instance[%s]", instance_id.c_str());
    return lane;
}

int64_t OnlineMrcEventConsumer::ResolveBytesPerBlock(const std::string &instance_id, Lane &lane) {
    if (lane.bytes_per_block > 0) {
        return lane.bytes_per_block;
    }
    if (bytes_per_block_resolver_) {
        const int64_t resolved = bytes_per_block_resolver_(instance_id);
        if (resolved > 0) {
            lane.bytes_per_block = resolved;
            return resolved;
        }
    }
    return config_.default_bytes_per_block;
}

void OnlineMrcEventConsumer::ReportMetrics() {
    if (!metrics_registry_) {
        return;
    }

    std::map<std::string, std::shared_ptr<Lane>> lanes_copy;
    {
        std::lock_guard<std::mutex> guard(lanes_mutex_);
        lanes_copy = lanes_;
    }

    const MetricsTags empty_tags;
    REPORT_DYNAMIC_GAUGE_(
        metrics_registry_, kMetricDroppedEvents, empty_tags, static_cast<double>(BasicDroppedCount()));
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricQueueSize, empty_tags, static_cast<double>(BasicQueueSize()));
    REPORT_DYNAMIC_GAUGE_(
        metrics_registry_, kMetricTrackedInstances, empty_tags, static_cast<double>(lanes_copy.size()));

    for (const auto &[instance_id, lane] : lanes_copy) {
        std::lock_guard<std::mutex> guard(lane->mutex);

        const int64_t bytes_per_block = ResolveBytesPerBlock(instance_id, *lane);

        MetricsTags instance_tags{{"instance_id", instance_id}};
        REPORT_DYNAMIC_GAUGE_(
            metrics_registry_, kMetricTrackedBlocks, instance_tags, static_cast<double>(lane->profiler.tracked_blocks()));
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              kMetricTrackedCapacityBlocks,
                              instance_tags,
                              static_cast<double>(lane->profiler.tracked_capacity_blocks()));
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              kMetricMemoryBytes,
                              instance_tags,
                              static_cast<double>(lane->profiler.memory_usage_bytes()));

        if (bytes_per_block <= 0) {
            // Cannot translate the GB grid into blocks yet; retried next round.
            continue;
        }
        std::vector<double> capacity_blocks;
        capacity_blocks.reserve(config_.capacity_gb_grid.size());
        for (const double capacity_gb : config_.capacity_gb_grid) {
            capacity_blocks.push_back(capacity_gb * kGiB / static_cast<double>(bytes_per_block));
        }

        const auto report_scope = [&](const char *scope, const MrcProfiler::Snapshot &snapshot) {
            MetricsTags scope_tags = instance_tags;
            scope_tags["scope"] = scope;
            REPORT_DYNAMIC_GAUGE_(
                metrics_registry_, kMetricMaxTrackedHitRate, scope_tags, snapshot.max_tracked_hit_rate);
            for (size_t i = 0; i < config_.capacity_gb_grid.size(); ++i) {
                MetricsTags point_tags = scope_tags;
                point_tags["capacity_gb"] = FormatCapacityGb(config_.capacity_gb_grid[i]);
                const bool covered = snapshot.hit_rates[i] >= 0.0;
                REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricExactCoverage, point_tags, covered ? 1.0 : 0.0);
                if (covered) {
                    REPORT_DYNAMIC_GAUGE_(
                        metrics_registry_, kMetricTheoreticalHitRate, point_tags, snapshot.hit_rates[i]);
                }
            }
        };

        MrcProfiler::Snapshot snapshot;
        lane->profiler.QueryCumulative(capacity_blocks, snapshot);
        report_scope("cumulative", snapshot);
        if (lane->profiler.QueryWindow(capacity_blocks, snapshot)) {
            report_scope("window", snapshot);
        }
    }
}

std::string OnlineMrcEventConsumer::DumpCurvesJson() const {
    std::map<std::string, std::shared_ptr<Lane>> lanes_copy;
    {
        std::lock_guard<std::mutex> guard(lanes_mutex_);
        lanes_copy = lanes_;
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    const auto write_curve = [&writer](const std::vector<MrcProfiler::CurvePoint> &points, int64_t bytes_per_block) {
        writer.StartArray();
        for (const auto &point : points) {
            writer.StartObject();
            writer.Key("capacity_blocks");
            writer.Double(point.capacity_blocks);
            if (bytes_per_block > 0) {
                writer.Key("capacity_gb");
                writer.Double(point.capacity_blocks * static_cast<double>(bytes_per_block) / kGiB);
            }
            writer.Key("hit_rate");
            writer.Double(point.hit_rate);
            writer.EndObject();
        }
        writer.EndArray();
    };

    writer.StartObject();
    writer.Key("instances");
    writer.StartArray();
    for (const auto &[instance_id, lane] : lanes_copy) {
        std::lock_guard<std::mutex> guard(lane->mutex);
        writer.StartObject();
        writer.Key("instance_id");
        writer.String(instance_id.c_str());
        writer.Key("bytes_per_block");
        writer.Int64(lane->bytes_per_block);
        writer.Key("engine");
        writer.String("lite_hit_exact_bounded");
        writer.Key("tracked_blocks");
        writer.Int64(lane->profiler.tracked_blocks());
        writer.Key("tracked_capacity_blocks");
        writer.Int64(lane->profiler.tracked_capacity_blocks());
        writer.Key("memory_bytes");
        writer.Int64(lane->profiler.memory_usage_bytes());

        MrcProfiler::Snapshot snapshot;
        lane->profiler.QueryCumulative({}, snapshot);
        writer.Key("cumulative");
        writer.StartObject();
        writer.Key("total_accesses");
        writer.Double(snapshot.total_accesses);
        writer.Key("max_tracked_hit_rate");
        writer.Double(snapshot.max_tracked_hit_rate);
        writer.Key("curve");
        write_curve(lane->profiler.DumpCurve(/*cumulative=*/true), lane->bytes_per_block);
        writer.EndObject();

        if (lane->profiler.QueryWindow({}, snapshot)) {
            writer.Key("window");
            writer.StartObject();
            writer.Key("total_accesses");
            writer.Double(snapshot.total_accesses);
            writer.Key("max_tracked_hit_rate");
            writer.Double(snapshot.max_tracked_hit_rate);
            writer.Key("curve");
            write_curve(lane->profiler.DumpCurve(/*cumulative=*/false), lane->bytes_per_block);
            writer.EndObject();
        }
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
    return buffer.GetString();
}

} // namespace kv_cache_manager
