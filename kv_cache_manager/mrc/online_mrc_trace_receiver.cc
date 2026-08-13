#include "kv_cache_manager/mrc/online_mrc_trace_receiver.h"

#include <algorithm>

namespace kv_cache_manager {

OnlineMrcTraceReceiver::OnlineMrcTraceReceiver(std::shared_ptr<OnlineMrcRegistry> registry,
                                               int64_t queue_max_batches)
    : registry_(std::move(registry))
    , queue_max_batches_(static_cast<size_t>(std::max<int64_t>(queue_max_batches, 1))) {}

OnlineMrcTraceReceiver::~OnlineMrcTraceReceiver() { Stop(); }

bool OnlineMrcTraceReceiver::Start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return true;
    }
    worker_ = std::thread(&OnlineMrcTraceReceiver::WorkerLoop, this);
    return true;
}

void OnlineMrcTraceReceiver::Stop() {
    if (!running_.exchange(false)) {
        return;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool OnlineMrcTraceReceiver::Enqueue(OnlineMrcBatch batch) {
    if (!running_) {
        return false;
    }
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (queue_.size() >= queue_max_batches_) {
            dropped_batches_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        queue_.push_back(std::move(batch));
    }
    cv_.notify_one();
    return true;
}

size_t OnlineMrcTraceReceiver::queue_size() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return queue_.size();
}

void OnlineMrcTraceReceiver::WorkerLoop() {
    while (true) {
        OnlineMrcBatch batch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return !running_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (!running_) {
                    break;
                }
                continue;
            }
            batch = std::move(queue_.front());
            queue_.pop_front();
        }
        if (!registry_) {
            continue;
        }
        registry_->RecordDropped(batch.dropped_spans, batch.dropped_keys);
        for (auto &span : batch.spans) {
            registry_->Observe(std::move(span));
        }
    }
}

} // namespace kv_cache_manager
