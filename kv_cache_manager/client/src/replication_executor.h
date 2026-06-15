#pragma once

#include <atomic>
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

struct ReplicationTask {
    ClientReplicationHint hint;
    const void *data = nullptr;
    size_t data_size = 0;
    ReleaseGuard guard;
};

class ReplicationExecutor {
public:
    ReplicationExecutor(MetaClient *meta_client, TransferClient *transfer_client, int num_workers = 2);
    ~ReplicationExecutor();

    void Submit(const std::vector<ClientReplicationHint> &hints);
    void SubmitWithData(ClientReplicationHint hint, const void *data, size_t size, std::function<void()> release_fn);
    void Shutdown();

private:
    void WorkerLoop();
    void ExecuteTask(ReplicationTask &task);
    void
    ExecuteWrite(const std::string &trace_id, const ClientReplicationHint &hint, const void *data, size_t data_size);
    void ExecuteHintAsync(const std::string &trace_id, const ClientReplicationHint &hint);
    std::string MakeKey(int64_t block_key, const std::string &target_node_id) const;

private:
    MetaClient *meta_client_;
    TransferClient *transfer_client_;
    int max_piggyback_queue_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<ReplicationTask> queue_;
    std::set<std::string> inflight_;
    int piggyback_queue_size_{0};
    std::atomic<bool> stopped_{false};
    std::vector<std::thread> workers_;
};

} // namespace kv_cache_manager
