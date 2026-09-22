#include "kv_cache_manager/meta/mpsc_write_queue.h"

namespace kv_cache_manager {

MpscWriteQueue::~MpscWriteQueue() {
    Node *node = consumer_head_;
    while (node) {
        Node *next = node->next;
        delete node;
        node = next;
    }
    node = head_.load(std::memory_order_acquire);
    while (node) {
        Node *next = node->next;
        delete node;
        node = next;
    }
}

void MpscWriteQueue::PushBarrier(SyncBarrierItem item) { Publish(new Node(QueueItem{std::move(item)}, 0)); }

bool MpscWriteQueue::TryReserve(int64_t key_count, int64_t capacity) noexcept {
    return TryReserve(key_count, capacity, false);
}

bool MpscWriteQueue::TryReserve(int64_t key_count, int64_t capacity, bool allow_oversized_item) noexcept {
    int64_t current = key_size_.load(std::memory_order_relaxed);
    do {
        if (key_count > capacity) {
            if (!allow_oversized_item || current != 0) {
                return false;
            }
        } else if (current > capacity - key_count) {
            return false;
        }
    } while (!key_size_.compare_exchange_weak(current, current + key_count, std::memory_order_acq_rel));
    return true;
}

bool MpscWriteQueue::HasCapacity(int64_t key_count, int64_t capacity, bool allow_oversized_item) const noexcept {
    const int64_t current = key_size_.load(std::memory_order_acquire);
    if (key_count > capacity) {
        return allow_oversized_item && current == 0;
    }
    return current <= capacity - key_count;
}

bool MpscWriteQueue::WaitAndReserve(int64_t key_count, int64_t capacity, int64_t timeout_us) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
    while (!TryReserve(key_count, capacity, true)) {
        std::unique_lock<std::mutex> lock(capacity_mutex_);
        if (HasCapacity(key_count, capacity, true)) {
            continue;
        }
        const bool has_capacity =
            capacity_cv_.wait_until(lock, deadline, [&] { return HasCapacity(key_count, capacity, true); });
        if (!has_capacity) {
            return false;
        }
    }
    return true;
}

void MpscWriteQueue::PushReserved(QueueItem item, int64_t key_count) { Publish(new Node(std::move(item), key_count)); }

void MpscWriteQueue::PushUnbounded(QueueItem item, int64_t key_count) {
    auto *node = new Node(std::move(item), key_count);
    key_size_.fetch_add(key_count, std::memory_order_acq_rel);
    Publish(node);
}

void MpscWriteQueue::NotifyCapacityWaiters() noexcept {
    std::lock_guard<std::mutex> lock(capacity_mutex_);
    capacity_cv_.notify_all();
}

void MpscWriteQueue::Publish(Node *new_node) {
    Node *old_head = head_.load(std::memory_order_relaxed);
    do {
        new_node->next = old_head;
    } while (!head_.compare_exchange_weak(old_head, new_node, std::memory_order_acq_rel, std::memory_order_relaxed));
    if (old_head == nullptr) {
        // Pair the empty-to-nonempty transition with the consumer's wait lock,
        // so publication cannot land between its predicate check and sleep.
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wait_cv_.notify_one();
    }
}

std::vector<QueueItem> MpscWriteQueue::PopBatch(int64_t max_batch_size, int64_t &out_taken_keys) {
    std::vector<QueueItem> result;
    out_taken_keys = 0;

    // 1. Serve from consumer-local leftover chain (FIFO order preserved)
    int64_t leftover_consumed = 0;
    while (consumer_head_ && out_taken_keys < max_batch_size) {
        out_taken_keys += consumer_head_->key_count;
        leftover_consumed += consumer_head_->key_count;
        result.emplace_back(std::move(consumer_head_->item));
        Node *to_delete = consumer_head_;
        consumer_head_ = consumer_head_->next;
        delete to_delete;
    }
    if (leftover_consumed > 0) {
        key_size_.fetch_sub(leftover_consumed, std::memory_order_release);
        NotifyCapacityWaiters();
    }
    if (out_taken_keys >= max_batch_size) {
        return result;
    }

    // 2. Drain from lock-free list
    Node *chain = head_.exchange(nullptr, std::memory_order_acq_rel);
    if (!chain) {
        return result;
    }

    // Reverse the chain to restore FIFO order (MPSC push builds LIFO)
    Node *prev = nullptr;
    Node *current = chain;
    while (current) {
        Node *next = current->next;
        current->next = prev;
        prev = current;
        current = next;
    }

    // 3. Take from reversed chain
    int64_t chain_consumed = 0;
    Node *node = prev;
    while (node && out_taken_keys < max_batch_size) {
        out_taken_keys += node->key_count;
        chain_consumed += node->key_count;
        result.emplace_back(std::move(node->item));
        Node *to_delete = node;
        node = node->next;
        delete to_delete;
    }

    // 4. Remaining stays in consumer-local chain for next PopBatch
    consumer_head_ = node;

    if (chain_consumed > 0) {
        key_size_.fetch_sub(chain_consumed, std::memory_order_release);
        NotifyCapacityWaiters();
    }

    return result;
}

std::vector<QueueItem>
MpscWriteQueue::PopBatchWait(int64_t max_batch_size, int64_t wait_timeout_us, int64_t &out_taken_keys) {
    std::vector<QueueItem> result = PopBatch(max_batch_size, out_taken_keys);
    if (!result.empty()) {
        return result;
    }

    // Wait for data to arrive
    {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait_for(lock, std::chrono::microseconds(wait_timeout_us), [this] {
            return wake_requested_ || head_.load(std::memory_order_acquire) != nullptr;
        });
        wake_requested_ = false;
    }

    return PopBatch(max_batch_size, out_taken_keys);
}

void MpscWriteQueue::NotifyConsumer() {
    {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wake_requested_ = true;
    }
    wait_cv_.notify_one();
}

} // namespace kv_cache_manager
