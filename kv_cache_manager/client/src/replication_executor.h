#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/affinity_types.h"

namespace kv_cache_manager {

class MetaClient;
class TransferClient;

class ReplicationExecutor {
public:
    ReplicationExecutor(MetaClient *meta_client, TransferClient *transfer_client);
    ~ReplicationExecutor();

    void Submit(const std::vector<ReplicationHint> &hints);
    void Shutdown();

private:
    void WorkerLoop();
    void ExecuteHint(const ReplicationHint &hint);
    std::string MakeKey(int64_t block_key, const std::string &target_node_id) const;

private:
    MetaClient *meta_client_;
    TransferClient *transfer_client_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<ReplicationHint> queue_;
    std::set<std::string> inflight_;
    std::atomic<bool> stopped_{false};
    std::thread worker_;
};

} // namespace kv_cache_manager
