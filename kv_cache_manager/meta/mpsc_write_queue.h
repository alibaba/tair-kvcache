#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

enum class WriteOpType {
    kPut,
    kUpsert,
    kDelete,
    kDeleteLocations,
};

struct WriteOp {
    WriteOpType type;
    KeyTypeVec keys;
    FieldMapVec field_maps;
    std::vector<std::vector<std::string>> field_names_vec;

    WriteOp() = default;
    WriteOp(WriteOp &&) = default;
    WriteOp &operator=(WriteOp &&) = default;
    WriteOp(const WriteOp &) = delete;
    WriteOp &operator=(const WriteOp &) = delete;
};

struct BarrierContext {
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<int> remain{0};
    std::atomic<bool> failed{false};

    void Fence() {
        if (remain.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(mu);
            cv.notify_all();
        }
    }

    void SetFailed() {
        failed.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lock(mu);
        cv.notify_all();
    }

    bool Wait(std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) {
        std::unique_lock<std::mutex> lock(mu);
        bool completed = cv.wait_for(lock, timeout, [this] {
            return remain.load(std::memory_order_acquire) == 0 || failed.load(std::memory_order_acquire);
        });
        if (!completed) {
            return false;
        }
        return !failed.load(std::memory_order_acquire);
    }
};

struct SyncBarrierItem {
    std::shared_ptr<BarrierContext> barrier_ctx;
};

using QueueItem = std::variant<WriteOp, SyncBarrierItem>;

class MpscWriteQueue {
public:
    MpscWriteQueue() = default;
    ~MpscWriteQueue();

    MpscWriteQueue(const MpscWriteQueue &) = delete;
    MpscWriteQueue &operator=(const MpscWriteQueue &) = delete;

    void Push(QueueItem item);
    std::vector<QueueItem> PopBatch(int64_t max_batch_size, int64_t &out_taken_keys);
    std::vector<QueueItem> PopBatchWait(int64_t max_batch_size, int64_t wait_timeout_us, int64_t &out_taken_keys);
    int64_t GetKeySize() const { return key_size_.load(std::memory_order_relaxed); }
    void NotifyConsumer();

private:
    struct Node {
        QueueItem item;
        int64_t key_count = 0;
        Node *next = nullptr;
        Node(QueueItem &&i, int64_t kc) : item(std::move(i)), key_count(kc) {}
    };

    std::atomic<Node *> head_{nullptr};
    std::atomic<int64_t> key_size_{0};

    // Consumer-local leftover chain (only accessed by single consumer thread)
    Node *consumer_head_ = nullptr;

    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
};

} // namespace kv_cache_manager
