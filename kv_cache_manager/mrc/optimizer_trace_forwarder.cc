#include "kv_cache_manager/mrc/optimizer_trace_forwarder.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {
namespace {

constexpr char kMetricDroppedEvents[] = "optimizer_forwarder.dropped_events";
constexpr char kMetricDroppedKeys[] = "optimizer_forwarder.dropped_keys";
constexpr char kMetricSendFailures[] = "optimizer_forwarder.send_failures";
constexpr char kMetricQueueSize[] = "optimizer_forwarder.queue_size";
constexpr char kMetricSentSpans[] = "optimizer_forwarder.sent_spans_total";
constexpr char kMetricSentKeys[] = "optimizer_forwarder.sent_keys_total";
constexpr char kMetricFilteredSpans[] = "optimizer_forwarder.filtered_spans_total";
constexpr char kMetricFilteredKeys[] = "optimizer_forwarder.filtered_keys_total";

} // namespace

OptimizerTraceForwarder::OptimizerTraceForwarder(const OptimizerTraceForwarderConfig &config,
                                                 std::shared_ptr<MetricsRegistry> metrics_registry,
                                                 MetadataResolver metadata_resolver)
    : config_(config)
    , metrics_registry_(std::move(metrics_registry))
    , metadata_resolver_(std::move(metadata_resolver))
    , source_id_(StringUtil::GenerateRandomString(32)) {}

OptimizerTraceForwarder::~OptimizerTraceForwarder() {
    if (running_) {
        Stop();
    }
}

bool OptimizerTraceForwarder::Init(const std::string &) {
    if (config_.endpoint.empty()) {
        KVCM_LOG_ERROR("optimizer forwarder: endpoint is empty");
        return false;
    }
    InitBasicQueue(static_cast<size_t>(std::max<int64_t>(config_.queue_max_size, 1)));
    stub_ = proto::optimizer::OptimizerService::NewStub(
        grpc::CreateChannel(config_.endpoint, grpc::InsecureChannelCredentials()));
    if (!stub_) {
        return false;
    }
    running_ = true;
    worker_ = std::thread(&OptimizerTraceForwarder::WorkerLoop, this);
    report_thread_ = LoopThread::CreateLoopThread([this]() { ReportMetrics(); },
                                                  std::max<int64_t>(config_.report_interval_seconds, 1) * 1000000,
                                                  "optimizer_forwarder_report");
    if (!report_thread_) {
        Stop();
        return false;
    }
    KVCM_LOG_INFO("optimizer trace forwarder started: endpoint[%s] queue[%" PRId64 "] max_batch_keys[%" PRId64 "]",
                  config_.endpoint.c_str(),
                  config_.queue_max_size,
                  config_.max_batch_keys);
    return true;
}

bool OptimizerTraceForwarder::Publish(const std::shared_ptr<BaseEvent> &event) {
    if (!running_ || !event) {
        return false;
    }
    auto cache_get = std::dynamic_pointer_cast<CacheGetEvent>(event);
    if (!cache_get || cache_get->get_keys().empty()) {
        return true;
    }
    if (!config_.instance_allowlist.empty() &&
        config_.instance_allowlist.count(cache_get->event_source()) == 0) {
        filtered_spans_.fetch_add(1, std::memory_order_relaxed);
        filtered_keys_.fetch_add(cache_get->get_keys().size(), std::memory_order_relaxed);
        return true;
    }
    if (!BasicEnqueue(event)) {
        dropped_spans_.fetch_add(1, std::memory_order_relaxed);
        dropped_keys_.fetch_add(cache_get->get_keys().size(), std::memory_order_relaxed);
    }
    // A drop is an observation-quality event, not an EventManager delivery
    // failure. Returning true avoids a warning on every saturated request.
    return true;
}

bool OptimizerTraceForwarder::Stop() {
    if (!running_.exchange(false)) {
        return true;
    }
    if (report_thread_) {
        report_thread_->Stop();
        report_thread_.reset();
    }
    if (basic_queue_) {
        basic_queue_->queue_cv.notify_all();
    }
    backoff_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    ClearBasicQueue();
    return true;
}

void OptimizerTraceForwarder::WorkerLoop() {
    while (running_) {
        BasicWait();
        proto::optimizer::ReportAccessTraceRequest request;
        request.set_cluster(config_.cluster);
        request.set_trace_id(source_id_);

        int64_t batch_keys = 0;
        std::shared_ptr<BaseEvent> event;
        while (BasicDequeue(event)) {
            auto cache_get = std::dynamic_pointer_cast<CacheGetEvent>(event);
            if (!cache_get || cache_get->get_keys().empty()) {
                continue;
            }
            auto metadata_it = metadata_by_instance_.find(cache_get->event_source());
            if (metadata_it == metadata_by_instance_.end() || metadata_it->second.bytes_per_block <= 0) {
                const auto metadata = metadata_resolver_ ? metadata_resolver_(cache_get->event_source())
                                                         : OptimizerTraceInstanceMetadata{};
                metadata_it = metadata_by_instance_.insert_or_assign(cache_get->event_source(), metadata).first;
            }
            const auto &metadata = metadata_it->second;
            if (metadata.bytes_per_block <= 0) {
                dropped_spans_.fetch_add(1, std::memory_order_relaxed);
                dropped_keys_.fetch_add(cache_get->get_keys().size(), std::memory_order_relaxed);
                continue;
            }
            auto *span = request.add_spans();
            span->set_instance_group(metadata.instance_group);
            span->set_instance_id(cache_get->event_source());
            span->set_block_size(metadata.block_size);
            span->set_bytes_per_block(metadata.bytes_per_block);
            span->set_event_time_us(cache_get->event_trigger_time_us());
            span->set_source_id(source_id_);
            span->set_sequence_number(++next_sequence_by_instance_[cache_get->event_source()]);
            for (const int64_t key : cache_get->get_keys()) {
                span->add_keys(key);
            }
            batch_keys += cache_get->get_keys().size();
            if (batch_keys >= std::max<int64_t>(config_.max_batch_keys, 1)) {
                break;
            }
        }
        if (request.spans().empty()) {
            continue;
        }
        const uint64_t dropped_spans = dropped_spans_.load(std::memory_order_relaxed);
        const uint64_t dropped_keys = dropped_keys_.load(std::memory_order_relaxed);
        request.set_dropped_spans(dropped_spans - acknowledged_dropped_spans_);
        request.set_dropped_keys(dropped_keys - acknowledged_dropped_keys_);
        if (Send(request)) {
            acknowledged_dropped_spans_ = dropped_spans;
            acknowledged_dropped_keys_ = dropped_keys;
            consecutive_send_failures_ = 0;
        } else {
            ++consecutive_send_failures_;
            const int64_t shift = std::min<int64_t>(consecutive_send_failures_ - 1, 5);
            const auto delay = std::chrono::seconds(std::min<int64_t>(int64_t{1} << shift, 30));
            std::unique_lock<std::mutex> lock(backoff_mutex_);
            backoff_cv_.wait_for(lock, delay, [this]() { return !running_; });
        }
    }
}

bool OptimizerTraceForwarder::Send(proto::optimizer::ReportAccessTraceRequest &request) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(std::max<int64_t>(config_.rpc_timeout_ms, 1)));
    proto::optimizer::CommonResponse response;
    const grpc::Status status = stub_->ReportAccessTrace(&context, request, &response);
    if (!status.ok() || response.header().status().code() != proto::optimizer::OK) {
        send_failures_.fetch_add(1, std::memory_order_relaxed);
        uint64_t keys = 0;
        for (const auto &span : request.spans()) {
            keys += span.keys_size();
        }
        dropped_spans_.fetch_add(request.spans_size(), std::memory_order_relaxed);
        dropped_keys_.fetch_add(keys, std::memory_order_relaxed);
        return false;
    }
    uint64_t keys = 0;
    for (const auto &span : request.spans()) {
        keys += span.keys_size();
    }
    sent_spans_.fetch_add(request.spans_size(), std::memory_order_relaxed);
    sent_keys_.fetch_add(keys, std::memory_order_relaxed);
    return true;
}

void OptimizerTraceForwarder::ReportMetrics() {
    if (!metrics_registry_) {
        return;
    }
    const MetricsTags tags;
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricDroppedEvents, tags, dropped_spans_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricDroppedKeys, tags, dropped_keys_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricSendFailures, tags, send_failures_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricQueueSize, tags, BasicQueueSize());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricSentSpans, tags, sent_spans_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricSentKeys, tags, sent_keys_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricFilteredSpans, tags, filtered_spans_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricFilteredKeys, tags, filtered_keys_.load());
}

} // namespace kv_cache_manager
