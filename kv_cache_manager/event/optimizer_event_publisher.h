#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>

#include "kv_cache_manager/event/event_publisher.h"
#include "kv_cache_manager/event/event_publishers_config.h"
#include "kv_cache_manager/event/optimizer_stream/event_sink.h"

namespace kv_cache_manager {

class BaseEvent;

namespace proto {
namespace optimizer {
class TraceQueryRequest;
} // namespace optimizer
} // namespace proto

// Forwards cache read events to a consumer (the optimizer) for replay.
//
// Publish() runs on a serving thread - it is called on the return path of
// GetCacheLocation - so it does nothing but enqueue. Conversion and
// subscription delivery happen on the worker thread. When the queue is full
// the event is dropped rather than the caller blocked: these are analysis
// samples, and a slow or absent consumer must never slow down serving.
//
// Dropping samples may slightly affect replay accuracy, which is accepted for
// this best-effort analysis path.
class OptimizerEventPublisher : public EventPublisher {
public:
    OptimizerEventPublisher(std::shared_ptr<EventSink> sink, const OptimizerEventPublisherConfig &config);
    ~OptimizerEventPublisher() override;

    OptimizerEventPublisher(const OptimizerEventPublisher &) = delete;
    OptimizerEventPublisher &operator=(const OptimizerEventPublisher &) = delete;

    // The argument is unused: this publisher is configured through its
    // constructor. It exists because EventPublisher requires it.
    bool Init(const std::string &config) override;
    bool Publish(const std::shared_ptr<BaseEvent> &event) override;
    bool Stop() override;

    // Events refused because the queue was full. Read from the base queue, so
    // no extra counter is maintained for it.
    std::size_t DroppedCount() const { return BasicDroppedCount(); }
    // Events handed to the sink. Says nothing about the consumer receiving
    // them - the sink contract cannot promise that.
    std::size_t ForwardedCount() const { return forwarded_.load(); }
    // Events the worker deliberately did not forward, e.g. because they were
    // not cache reads.
    std::size_t SkippedCount() const { return skipped_.load(); }

private:
    void WorkerThread();
    // Fills |out| from a cache read event. False means the event is not
    // something the optimizer can replay and should be skipped.
    static bool Convert(const std::shared_ptr<BaseEvent> &event, proto::optimizer::TraceQueryRequest *out);

    std::shared_ptr<EventSink> sink_;
    OptimizerEventPublisherConfig config_;
    std::thread worker_;
    std::atomic<std::size_t> forwarded_{0};
    std::atomic<std::size_t> skipped_{0};
};

} // namespace kv_cache_manager
