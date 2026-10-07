#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/client/include/common.h"

namespace kv_cache_manager {

class MetaClient;
class TransferClient;

class ReleaseGuard {
public:
    ReleaseGuard() = default;
    explicit ReleaseGuard(std::function<void()> fn) : fn_(std::move(fn)) {}
    ~ReleaseGuard() { Release(); }

    ReleaseGuard(ReleaseGuard &&other) noexcept : fn_(std::move(other.fn_)) { other.fn_ = nullptr; }
    ReleaseGuard &operator=(ReleaseGuard &&other) noexcept {
        if (this != &other) {
            Release();
            fn_ = std::move(other.fn_);
            other.fn_ = nullptr;
        }
        return *this;
    }

    ReleaseGuard(const ReleaseGuard &) = delete;
    ReleaseGuard &operator=(const ReleaseGuard &) = delete;

private:
    void Release() noexcept {
        // Cleanup may run while a transfer exception is already unwinding.
        // A failing user callback or best-effort abort must not terminate the
        // worker or prevent the remaining ownership guards from running.
        auto fn = std::move(fn_);
        fn_ = nullptr;
        try {
            if (fn) fn();
        } catch (...) {
        }
    }
    std::function<void()> fn_;
};

struct ReplicationOptions {
    size_t max_buffer_bytes = 256 * 1024 * 1024;
    size_t max_pending_bytes = 256 * 1024 * 1024;
    uint64_t node_bytes_per_second = 0; // 0 = unlimited
    uint32_t max_age_ms = 30000;
    std::string instance_id;
    std::function<void(const ReplicationStats &)> metrics_callback;
};

// Shared by all SDK executors in a process. Fair admission rotates among
// waiting instances; per-target pacing is shared across those instances.
class ReplicationResources {
public:
    using Clock = std::chrono::steady_clock;
    static ReplicationResources &Global();
    bool TryRetain(size_t bytes, size_t limit);
    void Release(size_t bytes);
    bool Acquire(const std::string &instance, const std::string &node, size_t bytes, size_t limit,
                 uint64_t transfer_bytes, uint64_t rate, Clock::time_point deadline,
                 const std::atomic<bool> &stopped);
private:
    struct Waiter { std::string instance; };
    std::mutex mu_;
    std::condition_variable cv_;
    size_t used_bytes_{0};
    std::deque<Waiter *> waiters_;
    std::string last_instance_;
    std::map<std::string, Clock::time_point> next_transfer_;
};

struct ReplicationTask {
    ClientReplicationHint hint;
    const void *data = nullptr;
    size_t data_size = 0;
    ReplicationOutcome outcome = ReplicationOutcome::SERVER_COPY_FAILED;
    ClientErrorCode error_code = ER_SERVICE_INTERNAL_ERROR;
    uint64_t copied_bytes = 0;
    ReplicationResultCallback result_callback;
    ReleaseGuard retained_memory;
    ReleaseGuard guard;
    size_t pending_bytes = 0;
    ReplicationResources::Clock::time_point submitted_at = ReplicationResources::Clock::now();
    std::vector<ClientReplicationBuffer> named_buffers;

};

class ReplicationExecutor {
public:
    ReplicationExecutor(MetaClient *meta_client, TransferClient *transfer_client, int num_workers = 2,
                        size_t max_pending_tasks = 1024, ReplicationOptions options = {});
    ~ReplicationExecutor();

    void Submit(const std::vector<ClientReplicationHint> &hints);
    bool SubmitWithData(ClientReplicationHint hint,
                        const void *data,
                        size_t size,
                        std::function<void()> release_fn,
                        ReplicationResultCallback result_callback = {});
    bool SubmitWithBuffers(ClientReplicationHint hint,
                           std::vector<ClientReplicationBuffer> buffers,
                           ReplicationResultCallback result_callback = {});
    void Shutdown();
    ReplicationStats GetStats() const;

private:
    void WorkerLoop();
    void ExecuteTask(ReplicationTask &task, bool try_server_copy = true);
    void ExecuteServerCopyBatch(std::vector<ReplicationTask *> tasks);
    void CompleteTask(ReplicationTask &task,
                      ReplicationResources::Clock::time_point started_at);
    static void Notify(const ClientReplicationHint &hint,
                       ReplicationResultCallback &callback,
                       ReplicationOutcome outcome,
                       ClientErrorCode error_code,
                       uint64_t copied_bytes = 0,
                       uint64_t latency_us = 0) noexcept;
    std::string MakeKey(int64_t block_key, const std::string &target_node_id) const;

private:
    MetaClient *meta_client_;
    TransferClient *transfer_client_;
    int max_piggyback_queue_;
    size_t max_pending_tasks_;
    ReplicationOptions options_;
    size_t pending_bytes_{0};

    mutable std::mutex mu_;
    struct Counters {
        std::atomic<uint64_t> submitted{0};
        std::atomic<uint64_t> admitted{0};
        std::atomic<uint64_t> succeeded{0};
        std::atomic<uint64_t> failed{0};
        std::atomic<uint64_t> skipped{0};
        std::atomic<uint64_t> expired{0};
        std::atomic<uint64_t> dropped_queue{0};
        std::atomic<uint64_t> dropped_budget{0};
        std::atomic<uint64_t> dropped_invalid{0};
        std::atomic<uint64_t> duplicates{0};
        std::atomic<uint64_t> copied_bytes{0};
        std::atomic<uint64_t> latency_us{0};
        std::atomic<uint64_t> queue_wait_us{0};
        std::atomic<uint64_t> active{0};
        std::atomic<uint64_t> server_copy_succeeded{0};
        std::atomic<uint64_t> server_copy_failed{0};
        std::atomic<uint64_t> client_fallback{0};
        std::atomic<uint64_t> allocation_failed{0};
        std::atomic<uint64_t> transfer_failed{0};
        std::atomic<uint64_t> publish_failed{0};
    } counters_;
    std::condition_variable cv_;
    std::deque<ReplicationTask> queue_;
    std::set<std::string> inflight_;
    int piggyback_queue_size_{0};
    std::atomic<bool> stopped_{false};
    std::vector<std::thread> workers_;
};

} // namespace kv_cache_manager
