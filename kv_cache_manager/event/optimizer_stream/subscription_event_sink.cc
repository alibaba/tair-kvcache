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

SubscriptionEventSink::Subscription::EnqueueResult
SubscriptionEventSink::Subscription::Enqueue(const proto::optimizer::TraceQueryRequest &event) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return EnqueueResult::CLOSED;
        }
        if (queue_.size() >= queue_size_) {
            return EnqueueResult::FULL;
        }
        queue_.push_back(event);
    }
    cv_.notify_one();
    return EnqueueResult::ACCEPTED;
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

bool SubscriptionEventSink::Send(const proto::optimizer::TraceQueryRequest &event) {
    return TrySend(event).accepted != 0;
}

SubscriptionEventSink::DeliveryResult SubscriptionEventSink::TrySend(const proto::optimizer::TraceQueryRequest &event) {
    DeliveryResult result;
    std::vector<std::shared_ptr<Subscription>> subscriptions;
    {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        if (stopped_ || !accepting_subscriptions_) {
            dropped_.fetch_add(1);
            result.failures.push_back({DropReason::UNAVAILABLE, {}});
            return result;
        }
        subscriptions = subscriptions_;
    }
    if (subscriptions.empty()) {
        dropped_.fetch_add(1);
        result.failures.push_back({DropReason::NO_SUBSCRIBER, {}});
        return result;
    }

    for (const auto &subscription : subscriptions) {
        const auto outcome = subscription->Enqueue(event);
        if (outcome == Subscription::EnqueueResult::ACCEPTED) {
            ++result.accepted;
        } else {
            dropped_.fetch_add(1);
            result.failures.push_back({outcome == Subscription::EnqueueResult::FULL ? DropReason::QUEUE_FULL
                                                                                    : DropReason::SUBSCRIPTION_CLOSED,
                                       subscription->consumer_id()});
        }
    }
    return result;
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
