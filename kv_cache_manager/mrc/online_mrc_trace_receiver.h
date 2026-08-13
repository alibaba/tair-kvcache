#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "kv_cache_manager/mrc/online_mrc_registry.h"

namespace kv_cache_manager {

// Bounded optimizer-side ingress queue. RPC handlers only validate/copy and
// enqueue; exact Fenwick work runs on this dedicated observation thread.
class OnlineMrcTraceReceiver {
public:
    OnlineMrcTraceReceiver(std::shared_ptr<OnlineMrcRegistry> registry, int64_t queue_max_batches);
    ~OnlineMrcTraceReceiver();

    bool Start();
    void Stop();
    bool Enqueue(OnlineMrcBatch batch);

    size_t queue_size() const;
    uint64_t dropped_batches() const { return dropped_batches_.load(std::memory_order_relaxed); }

private:
    void WorkerLoop();

    std::shared_ptr<OnlineMrcRegistry> registry_;
    size_t queue_max_batches_ = 1;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<OnlineMrcBatch> queue_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> dropped_batches_{0};
};

} // namespace kv_cache_manager
