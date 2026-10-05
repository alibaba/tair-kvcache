#include "kv_cache_manager/client/src/replication_executor.h"

#include <algorithm>
#include <exception>
#include <unordered_map>
#include <unordered_set>

#include "kv_cache_manager/client/include/meta_client.h"
#include "kv_cache_manager/client/include/transfer_client.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {

ReplicationExecutor::ReplicationExecutor(MetaClient *meta_client, TransferClient *transfer_client, int num_workers,
                                         size_t max_pending_tasks)
    : meta_client_(meta_client), transfer_client_(transfer_client),
      max_piggyback_queue_(std::max(1, num_workers) * 2), max_pending_tasks_(max_pending_tasks) {
    for (int i = 0; i < std::max(1, num_workers); ++i) {
        workers_.emplace_back(&ReplicationExecutor::WorkerLoop, this);
    }
}

ReplicationExecutor::~ReplicationExecutor() { Shutdown(); }

void ReplicationExecutor::Submit(const std::vector<ClientReplicationHint> &hints) {
    if (hints.empty() || stopped_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (stopped_.load(std::memory_order_relaxed)) {
        return;
    }
    for (const auto &hint : hints) {
        if (queue_.size() >= max_pending_tasks_) {
            KVCM_LOG_WARN("[replication] async queue full (%zu/%zu), remaining hints dropped",
                          queue_.size(), max_pending_tasks_);
            break;
        }
        std::string key = MakeKey(hint.block_key, hint.target_node_id);
        if (inflight_.count(key)) {
            KVCM_LOG_DEBUG("[replication] Submit: block_key [%ld] target [%s] already inflight, skipped",
                           hint.block_key,
                           hint.target_node_id.c_str());
            continue;
        }
        inflight_.insert(key);
        queue_.push_back(ReplicationTask{hint});
        KVCM_LOG_INFO("[replication] Submit: block_key [%ld] target [%s] enqueued (queue_size=%zu)",
                      hint.block_key,
                      hint.target_node_id.c_str(),
                      queue_.size());
    }
    if (!queue_.empty()) {
        cv_.notify_one();
    }
}

void ReplicationExecutor::SubmitWithData(ClientReplicationHint hint,
                                         const void *data,
                                         size_t size,
                                         std::function<void()> release_fn) {
    ReleaseGuard guard(std::move(release_fn));
    if (data == nullptr || size == 0) {
        Submit({hint});
        return;
    }
    if (stopped_.load(std::memory_order_relaxed)) {
        KVCM_LOG_WARN("[replication] SubmitWithData: executor stopped, block_key [%ld] dropped", hint.block_key);
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (stopped_.load(std::memory_order_relaxed) || queue_.size() >= max_pending_tasks_) {
        return;
    }
    std::string key = MakeKey(hint.block_key, hint.target_node_id);
    if (inflight_.count(key)) {
        KVCM_LOG_INFO("[replication] SubmitWithData: block_key [%ld] target [%s] already inflight "
                      "(likely auto-submitted by MatchLocation), piggyback data dropped. "
                      "Async replication via ExecuteHintAsync is in progress.",
                      hint.block_key,
                      hint.target_node_id.c_str());
        return;
    }
    if (piggyback_queue_size_ >= max_piggyback_queue_) {
        KVCM_LOG_WARN("[replication] SubmitWithData: block_key [%ld] target [%s] dropped, "
                      "piggyback queue full (%d/%d)",
                      hint.block_key,
                      hint.target_node_id.c_str(),
                      piggyback_queue_size_,
                      max_piggyback_queue_);
        return;
    }
    inflight_.insert(key);
    ++piggyback_queue_size_;
    queue_.push_front(ReplicationTask{std::move(hint), data, size, std::move(guard)});
    KVCM_LOG_INFO("[replication] SubmitWithData: block_key [%ld] target [%s] enqueued (piggyback_queue=%d/%d)",
                  hint.block_key,
                  hint.target_node_id.c_str(),
                  piggyback_queue_size_,
                  max_piggyback_queue_);
    cv_.notify_one();
}

void ReplicationExecutor::Shutdown() {
    if (stopped_.exchange(true)) {
        return;
    }
    cv_.notify_all();
    for (auto &w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    std::lock_guard<std::mutex> lk(mu_);
    queue_.clear();
    piggyback_queue_size_ = 0;
}

void ReplicationExecutor::WorkerLoop() {
    while (true) {
        ReplicationTask task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stopped_.load(std::memory_order_relaxed) || !queue_.empty(); });
            if (stopped_.load(std::memory_order_relaxed) && queue_.empty()) {
                return;
            }
            if (queue_.empty()) {
                continue;
            }
            task = std::move(queue_.front());
            queue_.pop_front();
            if (task.data) {
                --piggyback_queue_size_;
            }
        }
        try {
            ExecuteTask(task);
        } catch (const std::exception &e) {
            KVCM_LOG_WARN("[replication] block_key [%ld] failed: %s", task.hint.block_key, e.what());
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            inflight_.erase(MakeKey(task.hint.block_key, task.hint.target_node_id));
        }
    }
}

void ReplicationExecutor::ExecuteTask(ReplicationTask &task) {
    const auto &hint = task.hint;
    const std::string trace_id = "repl_" + StringUtil::GenerateRandomString(16);
    const auto caller = meta_client_->GetCallerNode();
    if (!caller.empty() && caller != hint.target_node_id) {
        KVCM_LOG_WARN("[replication] caller changed; dropping hint for node [%s]", hint.target_node_id.c_str());
        return;
    }
    auto [start_ec, write_loc] = meta_client_->StartWrite(
        trace_id, {hint.block_key}, {}, {}, /*write_timeout_seconds=*/60, /*is_replication=*/true);
    if (start_ec != ER_OK || write_loc.locations.empty()) {
        return; // Failed allocation, or the replica is already local.
    }

    // Release allocated destinations on every failure before publication. Once
    // FinishWrite has been attempted its result may be ambiguous, so never send
    // a contradictory abort after a failed success acknowledgement.
    bool finish_attempted = false;
    ReleaseGuard abort_session([&] {
        if (!finish_attempted && !write_loc.write_session_id.empty()) {
            const auto ec = meta_client_->FinishWrite(
                trace_id, write_loc.write_session_id, BlockMaskVector{false}, {});
            if (ec != ER_OK) {
                KVCM_LOG_WARN("[replication] failed to abort session [%s], ec [%d]",
                              write_loc.write_session_id.c_str(), ec);
            }
        }
    });
    if (write_loc.locations.size() != 1 || write_loc.locations.front().empty()) {
        return;
    }
    const auto &destinations = write_loc.locations.front();
    std::unordered_map<std::string, std::string> sources;
    for (const auto &source : hint.source_specs) {
        if (source.uri.empty() || !sources.emplace(source.spec_name, source.uri).second) {
            return;
        }
    }
    if (sources.empty() && destinations.size() == 1) {
        sources.emplace(destinations.front().spec_name, hint.source_uri);
    }

    UriStrVec dest_uris;
    BlockBuffers buffers;
    std::vector<std::vector<char>> owned_buffers(destinations.size());
    // The existing piggyback API owns one unnamed buffer. It is safe only for
    // a single spec; multi-spec hints use their complete named source set.
    const bool piggyback = task.data != nullptr && task.data_size > 0 && destinations.size() == 1 &&
                           hint.source_specs.size() <= 1;
    std::unordered_set<std::string> destination_names;
    for (size_t i = 0; i < destinations.size(); ++i) {
        const auto &dest = destinations[i];
        if (dest.uri.empty() || !destination_names.insert(dest.spec_name).second) {
            return;
        }
        auto source = sources.find(dest.spec_name);
        if (source == sources.end()) {
            return; // Never publish a location whose other specs were not copied.
        }
        size_t size = task.data_size;
        void *data = const_cast<void *>(task.data);
        if (!piggyback) {
            size = 0;
            StandardUri(source->second).GetParamAs<size_t>("size", size);
            if (size == 0) {
                return;
            }
        }
        size_t dest_size = 0;
        StandardUri(dest.uri).GetParamAs<size_t>("size", dest_size);
        if (dest_size != 0 && dest_size != size) {
            return;
        }
        if (!piggyback) {
            owned_buffers[i].resize(size);
            data = owned_buffers[i].data();
        }
        BlockBuffer buffer;
        buffer.iovs.push_back(Iov{MemoryType::CPU, data, size, false});
        if (!piggyback && transfer_client_->LoadKvCaches({source->second}, {buffer}) != ER_OK) {
            return;
        }
        dest_uris.push_back(dest.uri);
        buffers.push_back(std::move(buffer));
    }

    auto [save_ec, actual_uris] = transfer_client_->SaveKvCaches(dest_uris, buffers);
    if (save_ec != ER_OK || (!actual_uris.empty() && actual_uris.size() != destinations.size())) {
        return;
    }
    Location finished = destinations;
    for (size_t i = 0; i < actual_uris.size(); ++i) {
        if (actual_uris[i].empty()) {
            return;
        }
        finished[i].uri = actual_uris[i];
    }
    finish_attempted = true;
    const auto finish_ec = meta_client_->FinishWrite(
        trace_id, write_loc.write_session_id, BlockMaskOffset(1), {finished});
    if (finish_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] FinishWrite failed for block_key [%ld], ec [%d]", hint.block_key, finish_ec);
    }
}

std::string ReplicationExecutor::MakeKey(int64_t block_key, const std::string &target_node_id) const {
    return std::to_string(block_key) + ":" + target_node_id;
}

} // namespace kv_cache_manager
