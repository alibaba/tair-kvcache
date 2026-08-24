#include "kv_cache_manager/event/optimizer_stream/subscription_event_sink.h"

#include <algorithm>
#include <utility>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

SubscriptionEventSink::Subscription::Subscription(std::string consumer_id, std::size_t queue_size)
    : consumer_id_(std::move(consumer_id)), queue_size_(queue_size) {}

SubscriptionEventSink::Subscription::WaitResult
SubscriptionEventSink::Subscription::WaitNext(proto::optimizer::TraceQueryRequest *event,
                                              std::chrono::milliseconds timeout) {
    if (event == nullptr) {
        return WaitResult::kClosed;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] { return closed_ || !queue_.empty(); })) {
        return WaitResult::kTimeout;
    }
    if (queue_.empty()) {
        return WaitResult::kClosed;
    }
    *event = std::move(queue_.front());
    queue_.pop_front();
    return WaitResult::kEvent;
}

bool SubscriptionEventSink::Subscription::EnqueueBatch(const std::vector<proto::optimizer::TraceQueryRequest> &events) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() + events.size() > queue_size_) {
            return false;
        }
        queue_.insert(queue_.end(), events.begin(), events.end());
    }
    cv_.notify_all();
    return true;
}

std::size_t SubscriptionEventSink::Subscription::QueueSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool SubscriptionEventSink::Subscription::Enqueue(const proto::optimizer::TraceQueryRequest &event) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() >= queue_size_) {
            return false;
        }
        queue_.push_back(event);
    }
    cv_.notify_one();
    return true;
}

void SubscriptionEventSink::Subscription::Close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        closed_ = true;
        queue_.clear();
    }
    cv_.notify_all();
}

SubscriptionEventSink::SubscriptionEventSink(const OptimizerEventPublisherConfig &config) : config_(config) {}

SubscriptionEventSink::~SubscriptionEventSink() { Stop(); }

std::shared_ptr<SubscriptionEventSink::Subscription> SubscriptionEventSink::Subscribe(const std::string &consumer_id) {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    if (stopped_ || !accepting_subscriptions_ || subscriptions_.size() >= config_.max_subscribers()) {
        return nullptr;
    }
    auto subscription = std::shared_ptr<Subscription>(
        new Subscription(consumer_id.empty() ? "anonymous" : consumer_id, config_.subscriber_queue_size()));
    subscriptions_.push_back(subscription);
    KVCM_LOG_INFO("SubscriptionEventSink: subscriber ready, consumer_id=%s, subscribers=%zu",
                  subscription->consumer_id().c_str(),
                  subscriptions_.size());
    return subscription;
}

void SubscriptionEventSink::Unsubscribe(const std::shared_ptr<Subscription> &subscription) {
    if (!subscription) {
        return;
    }
    std::size_t subscribers = 0;
    {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        subscriptions_.erase(std::remove(subscriptions_.begin(), subscriptions_.end(), subscription),
                             subscriptions_.end());
        subscribers = subscriptions_.size();
    }
    subscription->Close();
    KVCM_LOG_INFO("SubscriptionEventSink: subscriber left, consumer_id=%s, subscribers=%zu",
                  subscription->consumer_id().c_str(),
                  subscribers);
}

void SubscriptionEventSink::EnableSubscriptions() {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    if (!stopped_) {
        accepting_subscriptions_ = true;
    }
}

void SubscriptionEventSink::DisableSubscriptions() {
    std::vector<std::shared_ptr<Subscription>> subscriptions;
    {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        accepting_subscriptions_ = false;
        subscriptions.swap(subscriptions_);
    }
    for (const auto &subscription : subscriptions) {
        subscription->Close();
    }
}

bool SubscriptionEventSink::Send(const proto::optimizer::TraceQueryRequest &event) { return SendBatch({event}); }

bool SubscriptionEventSink::SendBatch(const std::vector<proto::optimizer::TraceQueryRequest> &events) {
    if (events.empty()) return true;
    std::vector<std::shared_ptr<Subscription>> subscriptions;
    {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        if (stopped_ || !accepting_subscriptions_) {
            dropped_.fetch_add(events.size());
            return false;
        }
        subscriptions = subscriptions_;
    }
    if (subscriptions.empty()) {
        dropped_.fetch_add(events.size());
        return false;
    }

    bool delivered = false;
    for (const auto &subscription : subscriptions) {
        if (subscription->EnqueueBatch(events)) {
            delivered = true;
        } else {
            dropped_.fetch_add(events.size());
        }
    }
    return delivered;
}

std::size_t SubscriptionEventSink::QueuedCount() const {
    std::vector<std::shared_ptr<Subscription>> subscriptions;
    { std::lock_guard<std::mutex> lock(subscriptions_mutex_); subscriptions = subscriptions_; }
    std::size_t queued = 0;
    for (const auto &subscription : subscriptions) queued += subscription->QueueSize();
    return queued;
}

void SubscriptionEventSink::Stop() {
    if (stopped_.exchange(true)) {
        return;
    }
    accepting_subscriptions_ = false;
    std::vector<std::shared_ptr<Subscription>> subscriptions;
    {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        subscriptions.swap(subscriptions_);
    }
    for (const auto &subscription : subscriptions) {
        subscription->Close();
    }
    KVCM_LOG_INFO("SubscriptionEventSink: stopped, dropped=%zu", dropped_.load());
}

std::size_t SubscriptionEventSink::DroppedCount() const { return dropped_.load(); }

std::size_t SubscriptionEventSink::SubscriberCount() const {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    return subscriptions_.size();
}

} // namespace kv_cache_manager
