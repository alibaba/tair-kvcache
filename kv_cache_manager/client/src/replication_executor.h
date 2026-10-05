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
    ~ReleaseGuard() {
        if (fn_)
            fn_();
    }

    ReleaseGuard(ReleaseGuard &&other) noexcept : fn_(std::move(other.fn_)) { other.fn_ = nullptr; }
    ReleaseGuard &operator=(ReleaseGuard &&other) noexcept {
        if (this != &other) {
            if (fn_)
                fn_();
            fn_ = std::move(other.fn_);
            other.fn_ = nullptr;
        }
        return *this;
    }

    ReleaseGuard(const ReleaseGuard &) = delete;
    ReleaseGuard &operator=(const ReleaseGuard &) = delete;

private:
    std::function<void()> fn_;
};

struct ReplicationOptions {
    size_t max_buffer_bytes = 256 * 1024 * 1024;
    size_t max_pending_bytes = 256 * 1024 * 1024;
    uint64_t node_bytes_per_second = 0; // 0 = unlimited
    uint32_t max_age_ms = 30000;
    std::string instance_id;
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
    ReleaseGuard retained_memory;
    ReleaseGuard guard;
    size_t pending_bytes = 0;
    ReplicationResources::Clock::time_point submitted_at = ReplicationResources::Clock::now();

};

class ReplicationExecutor {
public:
    ReplicationExecutor(MetaClient *meta_client, TransferClient *transfer_client, int num_workers = 2,
                        size_t max_pending_tasks = 1024, ReplicationOptions options = {});
    ~ReplicationExecutor();

    void Submit(const std::vector<ClientReplicationHint> &hints);
    void SubmitWithData(ClientReplicationHint hint, const void *data, size_t size, std::function<void()> release_fn);
    void Shutdown();

private:
    void WorkerLoop();
    void ExecuteTask(ReplicationTask &task);
    std::string MakeKey(int64_t block_key, const std::string &target_node_id) const;

private:
    MetaClient *meta_client_;
    TransferClient *transfer_client_;
    int max_piggyback_queue_;
    size_t max_pending_tasks_;
    ReplicationOptions options_;
    size_t pending_bytes_{0};

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<ReplicationTask> queue_;
    std::set<std::string> inflight_;
    int piggyback_queue_size_{0};
    std::atomic<bool> stopped_{false};
    std::vector<std::thread> workers_;
};

} // namespace kv_cache_manager
