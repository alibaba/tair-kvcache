#include "kv_cache_manager/meta/mpsc_write_queue.h"

#include <algorithm>

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

void MpscWriteQueue::Push(QueueItem item) {
    int64_t kc = 0;
    if (auto *op = std::get_if<WriteOp>(&item)) {
        kc = static_cast<int64_t>(op->keys.size());
    }
    Node *new_node = new Node(std::move(item), kc);
    Node *old_head = head_.load(std::memory_order_relaxed);
    do {
        new_node->next = old_head;
    } while (!head_.compare_exchange_weak(old_head, new_node, std::memory_order_acq_rel, std::memory_order_relaxed));
    key_size_.fetch_add(kc, std::memory_order_relaxed);
    wait_cv_.notify_one();
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
        capacity_cv_.notify_all();
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
        capacity_cv_.notify_all();
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
            return head_.load(std::memory_order_acquire) != nullptr;
        });
    }

    return PopBatch(max_batch_size, out_taken_keys);
}

void MpscWriteQueue::NotifyConsumer() { wait_cv_.notify_one(); }

bool MpscWriteQueue::WaitForCapacity(int64_t capacity_threshold, int64_t incoming_key_count, int64_t timeout_us) {
    auto has_capacity = [&] {
        const int64_t current_key_count = key_size_.load(std::memory_order_acquire);
        if (incoming_key_count >= capacity_threshold) {
            return current_key_count == 0;
        }
        return current_key_count + incoming_key_count <= capacity_threshold;
    };

    if (has_capacity()) {
        return true;
    }
    std::unique_lock<std::mutex> lock(capacity_mutex_);
    return capacity_cv_.wait_for(lock, std::chrono::microseconds(timeout_us), has_capacity);
}

} // namespace kv_cache_manager
