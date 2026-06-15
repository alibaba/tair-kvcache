#include "kv_cache_manager/client/src/replication_executor.h"

#include "kv_cache_manager/client/include/meta_client.h"
#include "kv_cache_manager/client/include/transfer_client.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {

ReplicationExecutor::ReplicationExecutor(MetaClient *meta_client, TransferClient *transfer_client, int num_workers)
    : meta_client_(meta_client), transfer_client_(transfer_client), max_piggyback_queue_(num_workers * 2) {
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&ReplicationExecutor::WorkerLoop, this);
    }
}

ReplicationExecutor::~ReplicationExecutor() { Shutdown(); }

void ReplicationExecutor::Submit(const std::vector<ClientReplicationHint> &hints) {
    if (hints.empty() || stopped_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto &hint : hints) {
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
    if (stopped_.load(std::memory_order_relaxed)) {
        KVCM_LOG_WARN("[replication] SubmitWithData: executor stopped, block_key [%ld] dropped", hint.block_key);
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
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
        ExecuteTask(task);
        {
            std::lock_guard<std::mutex> lk(mu_);
            inflight_.erase(MakeKey(task.hint.block_key, task.hint.target_node_id));
        }
    }
}

void ReplicationExecutor::ExecuteTask(ReplicationTask &task) {
    std::string trace_id = "repl_" + StringUtil::GenerateRandomString(16);
    if (task.data && task.data_size > 0) {
        ExecuteWrite(trace_id, task.hint, task.data, task.data_size);
    } else {
        ExecuteHintAsync(trace_id, task.hint);
    }
}

void ReplicationExecutor::ExecuteWrite(const std::string &trace_id,
                                       const ClientReplicationHint &hint,
                                       const void *data,
                                       size_t data_size) {
    KVCM_LOG_DEBUG("[replication] ExecuteWrite BEGIN trace_id [%s] block_key [%ld] target [%s] data_size [%zu]",
                   trace_id.c_str(),
                   hint.block_key,
                   hint.target_node_id.c_str(),
                   data_size);

    auto [start_ec, write_loc] = meta_client_->StartWrite(
        trace_id, {hint.block_key}, {}, {}, /*write_timeout_seconds=*/60, /*is_replication=*/true);
    if (start_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] trace_id [%s] StartWrite failed for block_key [%ld], ec [%d]",
                      trace_id.c_str(),
                      hint.block_key,
                      start_ec);
        return;
    }
    KVCM_LOG_DEBUG("[replication] ExecuteWrite trace_id [%s] StartWrite OK: session_id [%s] locations=%zu",
                   trace_id.c_str(),
                   write_loc.write_session_id.c_str(),
                   write_loc.locations.size());

    if (write_loc.locations.empty()) {
        KVCM_LOG_INFO(
            "[replication] trace_id [%s] block_key [%ld] already replicated (no locations returned by server)",
            trace_id.c_str(),
            hint.block_key);
        return;
    }

    const auto &location = write_loc.locations[0];
    if (location.empty()) {
        KVCM_LOG_WARN("[replication] trace_id [%s] write location has no spec URIs", trace_id.c_str());
        return;
    }
    std::string dest_uri = location[0].uri;
    KVCM_LOG_DEBUG("[replication] ExecuteWrite trace_id [%s] dest_uri [%s] spec_name [%s]",
                   trace_id.c_str(),
                   dest_uri.c_str(),
                   location[0].spec_name.c_str());

    BlockBuffer block_buf;
    block_buf.iovs.push_back(Iov{MemoryType::CPU, const_cast<void *>(data), data_size, false});
    BlockBuffers bufs = {block_buf};

    auto [save_ec, actual_uris] = transfer_client_->SaveKvCaches({dest_uri}, bufs);
    if (save_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] trace_id [%s] SaveKvCaches to [%s] failed, ec [%d]",
                      trace_id.c_str(),
                      dest_uri.c_str(),
                      save_ec);
        return;
    }
    std::string actual_uri = actual_uris.empty() ? "(empty)" : actual_uris[0];
    KVCM_LOG_DEBUG("[replication] ExecuteWrite trace_id [%s] SaveKvCaches OK: actual_uri [%s]",
                   trace_id.c_str(),
                   actual_uri.c_str());

    Locations finish_locations;
    if (!actual_uris.empty()) {
        Location loc;
        loc.push_back({location[0].spec_name, actual_uris[0]});
        finish_locations.push_back(std::move(loc));
    } else {
        finish_locations.push_back(location);
    }

    BlockMask success_mask = BlockMaskOffset(1);
    auto finish_ec = meta_client_->FinishWrite(trace_id, write_loc.write_session_id, success_mask, finish_locations);
    if (finish_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] trace_id [%s] FinishWrite failed, ec [%d]", trace_id.c_str(), finish_ec);
        return;
    }

    KVCM_LOG_INFO("[replication] ExecuteWrite DONE trace_id [%s] block_key [%ld] replicated to node [%s] "
                  "dest_uri [%s] (piggyback)",
                  trace_id.c_str(),
                  hint.block_key,
                  hint.target_node_id.c_str(),
                  dest_uri.c_str());
}

void ReplicationExecutor::ExecuteHintAsync(const std::string &trace_id, const ClientReplicationHint &hint) {
    auto [start_ec, write_loc] = meta_client_->StartWrite(
        trace_id, {hint.block_key}, {}, {}, /*write_timeout_seconds=*/60, /*is_replication=*/true);
    if (start_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] trace_id [%s] StartWrite failed for block_key [%ld], ec [%d]",
                      trace_id.c_str(),
                      hint.block_key,
                      start_ec);
        return;
    }

    if (write_loc.locations.empty()) {
        KVCM_LOG_DEBUG("[replication] trace_id [%s] block_key [%ld] already replicated (no locations returned)",
                       trace_id.c_str(),
                       hint.block_key);
        return;
    }

    StandardUri source_uri(hint.source_uri);
    size_t block_size = 0;
    source_uri.GetParamAs<size_t>("size", block_size);
    if (block_size == 0) {
        KVCM_LOG_WARN("[replication] trace_id [%s] cannot determine block size from source_uri [%s]",
                      trace_id.c_str(),
                      hint.source_uri.c_str());
        return;
    }

    std::vector<char> buffer(block_size);
    BlockBuffer block_buf;
    block_buf.iovs.push_back(Iov{MemoryType::CPU, buffer.data(), block_size, false});
    BlockBuffers bufs = {block_buf};

    auto load_ec = transfer_client_->LoadKvCaches({hint.source_uri}, bufs);
    if (load_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] trace_id [%s] LoadKvCaches from [%s] failed, ec [%d]",
                      trace_id.c_str(),
                      hint.source_uri.c_str(),
                      load_ec);
        return;
    }

    const auto &location = write_loc.locations[0];
    if (location.empty()) {
        KVCM_LOG_WARN("[replication] trace_id [%s] write location has no spec URIs", trace_id.c_str());
        return;
    }
    std::string dest_uri = location[0].uri;

    auto [save_ec, actual_uris] = transfer_client_->SaveKvCaches({dest_uri}, bufs);
    if (save_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] trace_id [%s] SaveKvCaches to [%s] failed, ec [%d]",
                      trace_id.c_str(),
                      dest_uri.c_str(),
                      save_ec);
        return;
    }

    Locations finish_locations;
    if (!actual_uris.empty()) {
        Location loc;
        loc.push_back({location[0].spec_name, actual_uris[0]});
        finish_locations.push_back(std::move(loc));
    } else {
        finish_locations.push_back(location);
    }

    BlockMask success_mask = BlockMaskOffset(1);
    auto finish_ec = meta_client_->FinishWrite(trace_id, write_loc.write_session_id, success_mask, finish_locations);
    if (finish_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] trace_id [%s] FinishWrite failed, ec [%d]", trace_id.c_str(), finish_ec);
        return;
    }

    KVCM_LOG_INFO("[replication] trace_id [%s] block_key [%ld] replicated to node [%s]",
                  trace_id.c_str(),
                  hint.block_key,
                  hint.target_node_id.c_str());
}

std::string ReplicationExecutor::MakeKey(int64_t block_key, const std::string &target_node_id) const {
    return std::to_string(block_key) + ":" + target_node_id;
}

} // namespace kv_cache_manager
