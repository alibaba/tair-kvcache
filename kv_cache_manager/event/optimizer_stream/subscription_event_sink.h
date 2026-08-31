#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/event/event_publishers_config.h"
#include "kv_cache_manager/event/optimizer_stream/event_sink.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

// Fan-out queue for long-lived gRPC event subscriptions. Each subscriber owns
// a bounded queue, so one slow consumer only loses its own samples and never
// blocks the publisher or another consumer.
class SubscriptionEventSink final : public EventSink {
public:
    class Subscription {
    public:
        enum class WaitResult {
            kEvent,
            kTimeout,
            kClosed,
        };

        WaitResult WaitNext(proto::optimizer::TraceQueryRequest *event, std::chrono::milliseconds timeout);
        const std::string &consumer_id() const { return consumer_id_; }

    private:
        friend class SubscriptionEventSink;

        Subscription(std::string consumer_id, std::size_t queue_size);
        bool Enqueue(const proto::optimizer::TraceQueryRequest &event);
        void Close();

        const std::string consumer_id_;
        const std::size_t queue_size_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<proto::optimizer::TraceQueryRequest> queue_;
        bool closed_ = false;
    };

    explicit SubscriptionEventSink(const OptimizerEventPublisherConfig &config);
    ~SubscriptionEventSink() override;

    SubscriptionEventSink(const SubscriptionEventSink &) = delete;
    SubscriptionEventSink &operator=(const SubscriptionEventSink &) = delete;

    std::shared_ptr<Subscription> Subscribe(const std::string &consumer_id);
    void Unsubscribe(const std::shared_ptr<Subscription> &subscription);

    void EnableSubscriptions();
    void DisableSubscriptions();
    bool accepting_subscriptions() const { return accepting_subscriptions_.load(); }

    bool Send(const proto::optimizer::TraceQueryRequest &event) override;
    void Stop() override;
    std::size_t DroppedCount() const;

    std::size_t SubscriberCount() const;
    bool stopped() const { return stopped_.load(); }

private:
    OptimizerEventPublisherConfig config_;
    mutable std::mutex subscriptions_mutex_;
    std::vector<std::shared_ptr<Subscription>> subscriptions_;
    std::atomic<bool> accepting_subscriptions_{true};
    std::atomic<bool> stopped_{false};
    std::atomic<std::size_t> dropped_{0};
};

} // namespace kv_cache_manager
