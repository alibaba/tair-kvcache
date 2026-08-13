#include "kv_cache_manager/mrc/online_mrc_registry.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kv_cache_manager {
namespace {

constexpr char kMetricTheoreticalHitRate[] = "online_mrc.theoretical_hit_rate";
constexpr char kMetricMaxTrackedHitRate[] = "online_mrc.max_tracked_hit_rate";
constexpr char kMetricTrackedBlocks[] = "online_mrc.tracked_blocks";
constexpr char kMetricTrackedCapacityBlocks[] = "online_mrc.tracked_capacity_blocks";
constexpr char kMetricMemoryBytes[] = "online_mrc.memory_bytes";
constexpr char kMetricExactCoverage[] = "online_mrc.exact_coverage";
constexpr char kMetricStreamComplete[] = "online_mrc.stream_complete";
constexpr char kMetricSequenceGaps[] = "online_mrc.sequence_gaps";
constexpr char kMetricSourceSwitches[] = "online_mrc.source_switches";
constexpr char kMetricDroppedSpans[] = "online_mrc.dropped_spans";
constexpr char kMetricDroppedKeys[] = "online_mrc.dropped_keys";
constexpr char kMetricOutOfOrderSpans[] = "online_mrc.out_of_order_spans";
constexpr char kMetricTrackedInstances[] = "online_mrc.tracked_instances";
constexpr char kMetricReceiverQueueSize[] = "online_mrc.receiver_queue_size";
constexpr char kMetricReceiverDroppedBatches[] = "online_mrc.receiver_dropped_batches";

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

int64_t SteadyNowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string FormatCapacityGb(double capacity_gb) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", capacity_gb);
    return buf;
}

} // namespace

OnlineMrcRegistry::Lane::Lane(const OnlineMrcConfig &config, OnlineMrcSpan span)
    : profiler(MrcProfiler::Options{config.max_tracked_blocks, config.window_seconds})
    , cluster(std::move(span.cluster))
    , instance_group(std::move(span.instance_group))
    , instance_id(std::move(span.instance_id))
    , source_id(std::move(span.source_id))
    , bytes_per_block(span.bytes_per_block)
    , last_observed_steady_us(SteadyNowUs()) {}

OnlineMrcRegistry::OnlineMrcRegistry(const OnlineMrcConfig &config,
                                     std::shared_ptr<MetricsRegistry> metrics_registry)
    : config_(config), metrics_registry_(std::move(metrics_registry)) {}

std::shared_ptr<OnlineMrcRegistry::Lane> OnlineMrcRegistry::GetOrCreateLane(OnlineMrcSpan span) {
    if (span.instance_id.empty()) {
        return nullptr;
    }
    const LaneKey key{span.cluster, span.instance_id};
    std::lock_guard<std::mutex> guard(lanes_mutex_);
    auto it = lanes_.find(key);
    if (it != lanes_.end()) {
        return it->second;
    }
    if (static_cast<int32_t>(lanes_.size()) >= config_.max_instances) {
        if (!lane_overflow_logged_) {
            lane_overflow_logged_ = true;
            KVCM_LOG_WARN("online mrc: lane limit[%d] reached; new instance[%s/%s] ignored",
                          config_.max_instances,
                          span.cluster.c_str(),
                          span.instance_id.c_str());
        }
        return nullptr;
    }
    auto lane = std::make_shared<Lane>(config_, std::move(span));
    lanes_.emplace(key, lane);
    return lane;
}

bool OnlineMrcRegistry::Observe(OnlineMrcSpan span) {
    auto lane = GetOrCreateLane(span);
    if (!lane) {
        RecordDropped(1, span.keys.size());
        return false;
    }

    std::lock_guard<std::mutex> guard(lane->mutex);
    if (span.bytes_per_block > 0) {
        if (lane->bytes_per_block == 0) {
            lane->bytes_per_block = span.bytes_per_block;
        } else if (lane->bytes_per_block != span.bytes_per_block) {
            // A capacity-axis change makes old and new GB points
            // incomparable. Keep the original axis and mark the stream.
            ++lane->source_switches;
        }
    }
    if (!span.instance_group.empty()) {
        lane->instance_group = span.instance_group;
    }

    if (span.sequence_number > 0) {
        if (!span.source_id.empty() && !lane->source_id.empty() && span.source_id != lane->source_id) {
            ++lane->source_switches;
            lane->source_id = span.source_id;
            lane->last_sequence_number = 0;
        } else if (lane->source_id.empty()) {
            lane->source_id = span.source_id;
        }
        if (lane->last_sequence_number > 0 && span.sequence_number <= lane->last_sequence_number) {
            out_of_order_spans_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (lane->last_sequence_number > 0 && span.sequence_number != lane->last_sequence_number + 1) {
            lane->sequence_gaps += span.sequence_number - lane->last_sequence_number - 1;
        }
        lane->last_sequence_number = span.sequence_number;
    }

    lane->last_observed_steady_us = SteadyNowUs();
    lane->profiler.Observe(span.keys, span.event_time_us > 0 ? span.event_time_us : lane->last_observed_steady_us);
    return true;
}

void OnlineMrcRegistry::RecordDropped(uint64_t spans, uint64_t keys) {
    dropped_spans_.fetch_add(spans, std::memory_order_relaxed);
    dropped_keys_.fetch_add(keys, std::memory_order_relaxed);
}

void OnlineMrcRegistry::ReportIngressMetrics(size_t queue_size, uint64_t dropped_batches) {
    if (!metrics_registry_) {
        return;
    }
    const MetricsTags tags;
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricReceiverQueueSize, tags, queue_size);
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricReceiverDroppedBatches, tags, dropped_batches);
}

size_t OnlineMrcRegistry::LaneCount() const {
    std::lock_guard<std::mutex> guard(lanes_mutex_);
    return lanes_.size();
}

void OnlineMrcRegistry::ExpireIdleLanes(int64_t now_steady_us) {
    if (config_.idle_expire_seconds <= 0) {
        return;
    }
    const int64_t expire_before = now_steady_us - config_.idle_expire_seconds * 1000000;
    std::vector<LaneKey> expired;
    {
        std::lock_guard<std::mutex> guard(lanes_mutex_);
        for (const auto &[key, lane] : lanes_) {
            std::lock_guard<std::mutex> lane_guard(lane->mutex);
            if (lane->last_observed_steady_us < expire_before) {
                expired.push_back(key);
            }
        }
        for (const auto &key : expired) {
            lanes_.erase(key);
        }
    }
    if (metrics_registry_) {
        for (const auto &[cluster, instance_id] : expired) {
            MetricsTags tags{{"cluster", cluster}, {"instance_id", instance_id}};
            metrics_registry_->RemoveByTagFilter(tags);
        }
    }
}

void OnlineMrcRegistry::ReportMetrics() {
    if (!metrics_registry_) {
        return;
    }
    ExpireIdleLanes(SteadyNowUs());

    std::map<LaneKey, std::shared_ptr<Lane>> lanes;
    {
        std::lock_guard<std::mutex> guard(lanes_mutex_);
        lanes = lanes_;
    }

    const MetricsTags empty_tags;
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricTrackedInstances, empty_tags, lanes.size());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricDroppedSpans, empty_tags, dropped_spans_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricDroppedKeys, empty_tags, dropped_keys_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricOutOfOrderSpans, empty_tags, out_of_order_spans_.load());

    for (const auto &entry : lanes) {
        const auto &lane = entry.second;
        std::lock_guard<std::mutex> guard(lane->mutex);
        MetricsTags instance_tags{{"cluster", lane->cluster},
                                  {"instance_group", lane->instance_group},
                                  {"instance_id", lane->instance_id}};
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricTrackedBlocks, instance_tags, lane->profiler.tracked_blocks());
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              kMetricTrackedCapacityBlocks,
                              instance_tags,
                              lane->profiler.tracked_capacity_blocks());
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricMemoryBytes, instance_tags, lane->profiler.memory_usage_bytes());
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricSequenceGaps, instance_tags, lane->sequence_gaps);
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricSourceSwitches, instance_tags, lane->source_switches);
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              kMetricStreamComplete,
                              instance_tags,
                              (lane->sequence_gaps == 0 && lane->source_switches == 0 &&
                               dropped_spans_.load(std::memory_order_relaxed) == 0)
                                  ? 1.0
                                  : 0.0);

        if (lane->bytes_per_block <= 0) {
            continue;
        }
        std::vector<double> capacity_blocks;
        for (const double capacity_gb : config_.capacity_gb_grid) {
            capacity_blocks.push_back(capacity_gb * kGiB / lane->bytes_per_block);
        }
        const auto report = [&](const char *scope, const MrcProfiler::Snapshot &snapshot) {
            MetricsTags scope_tags = instance_tags;
            scope_tags["scope"] = scope;
            REPORT_DYNAMIC_GAUGE_(
                metrics_registry_, kMetricMaxTrackedHitRate, scope_tags, snapshot.max_tracked_hit_rate);
            for (size_t i = 0; i < capacity_blocks.size(); ++i) {
                MetricsTags tags = scope_tags;
                tags["capacity_gb"] = FormatCapacityGb(config_.capacity_gb_grid[i]);
                const bool covered = snapshot.hit_rates[i] >= 0;
                REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricExactCoverage, tags, covered ? 1.0 : 0.0);
                if (covered) {
                    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricTheoreticalHitRate, tags, snapshot.hit_rates[i]);
                }
            }
        };
        MrcProfiler::Snapshot snapshot;
        lane->profiler.QueryCumulative(capacity_blocks, snapshot);
        report("cumulative", snapshot);
        if (lane->profiler.QueryWindow(capacity_blocks, snapshot)) {
            report("window", snapshot);
        }
    }
}

std::string OnlineMrcRegistry::DumpCurvesJson() const {
    std::map<LaneKey, std::shared_ptr<Lane>> lanes;
    {
        std::lock_guard<std::mutex> guard(lanes_mutex_);
        lanes = lanes_;
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("engine");
    writer.String("lite_hit_exact_bounded");
    writer.Key("dropped_spans");
    writer.Uint64(dropped_spans_.load());
    writer.Key("dropped_keys");
    writer.Uint64(dropped_keys_.load());
    writer.Key("instances");
    writer.StartArray();
    for (const auto &entry : lanes) {
        const auto &lane = entry.second;
        std::lock_guard<std::mutex> guard(lane->mutex);
        writer.StartObject();
        writer.Key("cluster");
        writer.String(lane->cluster.c_str());
        writer.Key("instance_group");
        writer.String(lane->instance_group.c_str());
        writer.Key("instance_id");
        writer.String(lane->instance_id.c_str());
        writer.Key("bytes_per_block");
        writer.Int64(lane->bytes_per_block);
        writer.Key("tracked_capacity_blocks");
        writer.Int64(lane->profiler.tracked_capacity_blocks());
        writer.Key("tracked_blocks");
        writer.Int64(lane->profiler.tracked_blocks());
        writer.Key("memory_bytes");
        writer.Int64(lane->profiler.memory_usage_bytes());
        writer.Key("stream_complete");
        writer.Bool(lane->sequence_gaps == 0 && lane->source_switches == 0 &&
                    dropped_spans_.load(std::memory_order_relaxed) == 0);
        writer.Key("sequence_gaps");
        writer.Uint64(lane->sequence_gaps);

        const auto write_scope = [&](const char *name, bool cumulative) {
            MrcProfiler::Snapshot snapshot;
            if (!cumulative && !lane->profiler.QueryWindow({}, snapshot)) {
                return;
            }
            if (cumulative) {
                lane->profiler.QueryCumulative({}, snapshot);
            }
            writer.Key(name);
            writer.StartObject();
            writer.Key("total_accesses");
            writer.Double(snapshot.total_accesses);
            writer.Key("max_tracked_hit_rate");
            writer.Double(snapshot.max_tracked_hit_rate);
            writer.Key("curve");
            writer.StartArray();
            for (const auto &point : lane->profiler.DumpCurve(cumulative)) {
                writer.StartObject();
                writer.Key("capacity_blocks");
                writer.Double(point.capacity_blocks);
                if (lane->bytes_per_block > 0) {
                    writer.Key("capacity_gb");
                    writer.Double(point.capacity_blocks * lane->bytes_per_block / kGiB);
                }
                writer.Key("hit_rate");
                writer.Double(point.hit_rate);
                writer.EndObject();
            }
            writer.EndArray();
            writer.EndObject();
        };
        write_scope("cumulative", true);
        write_scope("window", false);
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
    return buffer.GetString();
}

} // namespace kv_cache_manager
