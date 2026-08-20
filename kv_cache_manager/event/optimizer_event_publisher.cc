#include "kv_cache_manager/event/optimizer_event_publisher.h"

#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/event/base_event.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

OptimizerEventPublisher::OptimizerEventPublisher(std::shared_ptr<EventSink> sink,
                                                 const OptimizerEventPublisherConfig &config)
    : sink_(std::move(sink)), config_(config) {}

OptimizerEventPublisher::~OptimizerEventPublisher() {
    if (running_) {
        Stop();
    }
}

bool OptimizerEventPublisher::Init(const std::string & /*config*/) {
    if (!sink_) {
        KVCM_LOG_ERROR("OptimizerEventPublisher: no sink provided");
        return false;
    }
    InitBasicQueue(config_.queue_size());
    running_ = true;
    worker_ = std::thread(&OptimizerEventPublisher::WorkerThread, this);
    KVCM_LOG_INFO("OptimizerEventPublisher: initialized, queue_size=%zu", config_.queue_size());
    return true;
}

bool OptimizerEventPublisher::Publish(const std::shared_ptr<BaseEvent> &event) {
    if (!event || !running_) {
        return false;
    }
    // Runs on a serving thread: enqueue and return, nothing else. A full queue
    // drops the event (counted by the base class) rather than blocking.
    return BasicEnqueue(event);
}

bool OptimizerEventPublisher::Stop() {
    if (!running_) {
        return true;
    }
    running_ = false;
    ClearBasicQueue();
    // Required, not belt-and-braces: BasicWait() blocks in
    // condition_variable::wait(), whose predicate is only re-evaluated when it
    // is notified. Clearing running_ alone leaves the worker asleep and join()
    // below would never return.
    if (basic_queue_) {
        basic_queue_->queue_cv.notify_all();
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    if (sink_) {
        sink_->Stop();
    }
    KVCM_LOG_INFO("OptimizerEventPublisher: stopped, forwarded=%zu skipped=%zu queue_dropped=%zu",
                  forwarded_.load(),
                  skipped_.load(),
                  BasicDroppedCount());
    return true;
}

void OptimizerEventPublisher::WorkerThread() {
    // A single worker keeps conversion and delivery off serving threads
    // without adding synchronization between multiple consumers of the queue.
    while (running_) {
        BasicWait();

        std::shared_ptr<BaseEvent> event;
        while (BasicDequeue(event)) {
            proto::optimizer::TraceQueryRequest request;
            if (!Convert(event, &request)) {
                skipped_.fetch_add(1);
                continue;
            }
            if (sink_->Send(request)) {
                forwarded_.fetch_add(1);
            }
        }
    }
}

bool OptimizerEventPublisher::Convert(const std::shared_ptr<BaseEvent> &event,
                                      proto::optimizer::TraceQueryRequest *out) {
    // Every publisher registered with EventManager sees every event, so write
    // and reclaim events arrive here too. Only cache reads can be replayed.
    const auto *get_event = dynamic_cast<const CacheGetEvent *>(event.get());
    if (get_event == nullptr) {
        return false;
    }

    out->set_trace_id(get_event->trace_id());
    // CacheGetEvent's source is the instance the request was served for.
    out->set_instance_id(get_event->event_source());
    for (const auto key : get_event->get_keys()) {
        out->add_block_keys(key);
    }
    // The event carries microseconds; the replay works in nanoseconds.
    out->set_timestamp_ns(get_event->event_trigger_time_us() * 1000);

    // Only the token count is needed, never the ids. An empty tokens vector
    // means the caller passed pre-computed keys instead, and then the exact
    // input length is simply not known here: 0 says "unknown" and the
    // consumer infers it from block count times block size. That inference
    // loses the trailing partial block, which biases the hit rate upwards -
    // the opposite direction from dropped events, so the two do not cancel.
    out->set_input_token_len(static_cast<std::int64_t>(get_event->get_tokens().size()));

    // An empty block_keys list is legitimate, not junk: a prompt shorter than
    // one block has no complete block. Its input_token_len still belongs in
    // the hit-rate denominator, so the event is forwarded as-is.
    return true;
}

} // namespace kv_cache_manager
