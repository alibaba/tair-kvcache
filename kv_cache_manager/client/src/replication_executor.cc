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

ReplicationResources &ReplicationResources::Global() {
    static ReplicationResources resources;
    return resources;
}

bool ReplicationResources::TryRetain(size_t bytes, size_t limit) {
    std::lock_guard<std::mutex> lock(mu_);
    if (bytes > limit || used_bytes_ > limit - bytes) return false;
    used_bytes_ += bytes;
    return true;
}

void ReplicationResources::Release(size_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    used_bytes_ -= bytes;
    cv_.notify_all();
}

bool ReplicationResources::Acquire(const std::string &instance, const std::string &node,
    size_t bytes, size_t limit, uint64_t transfer_bytes, uint64_t rate, Clock::time_point deadline,
    const std::atomic<bool> &stopped) {
    if (bytes > limit) return false;
    std::unique_lock<std::mutex> lock(mu_);
    Waiter waiter{instance};
    waiters_.push_back(&waiter);
    auto remove = [&] {
        waiters_.erase(std::find(waiters_.begin(), waiters_.end(), &waiter));
        cv_.notify_all();
    };
    while (Clock::now() < deadline) {
        // Preserve FIFO within an instance and rotate when another instance waits.
        auto selected = std::find_if(waiters_.begin(), waiters_.end(), [&](const auto *w) {
            return w->instance != last_instance_;
        });
        if (selected == waiters_.end()) selected = waiters_.begin();
        const auto now = Clock::now();
        if (*selected == &waiter && used_bytes_ <= limit - bytes &&
            (rate == 0 || next_transfer_[node] <= now)) {
            used_bytes_ += bytes;
            last_instance_ = instance;
            if (rate > 0) {
                // One transfer burst, then reserve its wire time. Bound arithmetic
                // before conversion; an oversized transfer expires, never wraps.
                const long double seconds = static_cast<long double>(transfer_bytes) / rate;
                if (seconds > 86400) { used_bytes_ -= bytes; remove(); return false; }
                next_transfer_[node] = now + std::chrono::microseconds(static_cast<int64_t>(seconds * 1000000));
            }
            for (auto it = next_transfer_.begin(); it != next_transfer_.end();) {
                if (it->second <= now) it = next_transfer_.erase(it); else ++it;
            }
            remove();
            return true;
        }
        if (stopped.load()) break;
        cv_.wait_until(lock, std::min(deadline, now + std::chrono::milliseconds(10)));
    }
    remove();
    return false;
}

namespace {
size_t HintBytes(const ClientReplicationHint &hint) {
    size_t bytes = 0;
    auto add = [&](const std::string &uri) {
        size_t size = 0;
        StandardUri(uri).GetParamAs<size_t>("size", size);
        if (size > SIZE_MAX - bytes) bytes = SIZE_MAX; else bytes += size;
    };
    if (hint.source_specs.empty()) add(hint.source_uri);
    else for (const auto &spec : hint.source_specs) add(spec.uri);
    return bytes;
}
} // namespace


ReplicationExecutor::ReplicationExecutor(MetaClient *meta_client, TransferClient *transfer_client, int num_workers,
                                         size_t max_pending_tasks, ReplicationOptions options)
    : meta_client_(meta_client), transfer_client_(transfer_client),
      max_piggyback_queue_(std::max(1, num_workers) * 2), max_pending_tasks_(max_pending_tasks), options_(std::move(options)) {
    for (int i = 0; i < std::max(1, num_workers); ++i) {
        workers_.emplace_back(&ReplicationExecutor::WorkerLoop, this);
    }
}

ReplicationExecutor::~ReplicationExecutor() { Shutdown(); }

void ReplicationExecutor::Submit(const std::vector<ClientReplicationHint> &hints) {
    counters_.submitted += hints.size();
    if (hints.empty() || stopped_.load(std::memory_order_relaxed)) {
        counters_.dropped_queue += hints.size();
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (stopped_.load(std::memory_order_relaxed)) {
        counters_.dropped_queue += hints.size();
        return;
    }
    for (size_t hint_index = 0; hint_index < hints.size(); ++hint_index) {
        const auto &hint = hints[hint_index];
        if (queue_.size() >= max_pending_tasks_) {
            counters_.dropped_queue += hints.size() - hint_index;
            KVCM_LOG_WARN("[replication] async queue full (%zu/%zu), remaining hints dropped",
                          queue_.size(), max_pending_tasks_);
            break;
        }
        const size_t bytes = HintBytes(hint);
        if (bytes > options_.max_buffer_bytes || bytes > options_.max_pending_bytes ||
            pending_bytes_ > options_.max_pending_bytes - bytes) { ++counters_.dropped_budget; continue; }
        std::string key = MakeKey(hint.block_key, hint.target_node_id);
        if (inflight_.count(key)) {
            ++counters_.duplicates;
            KVCM_LOG_DEBUG("[replication] Submit: block_key [%ld] target [%s] already inflight, skipped",
                           hint.block_key,
                           hint.target_node_id.c_str());
            continue;
        }
        ++counters_.admitted;
        inflight_.insert(key);
        queue_.push_back(ReplicationTask{hint});
        queue_.back().pending_bytes = bytes;
        pending_bytes_ += bytes;
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
    ++counters_.submitted;
    if (stopped_.load(std::memory_order_relaxed)) {
        ++counters_.dropped_queue;
        KVCM_LOG_WARN("[replication] SubmitWithData: executor stopped, block_key [%ld] dropped", hint.block_key);
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (stopped_.load(std::memory_order_relaxed)) {
        ++counters_.dropped_queue;
        return;
    }
    std::string key = MakeKey(hint.block_key, hint.target_node_id);
    if (inflight_.count(key)) {
        auto queued = std::find_if(queue_.begin(), queue_.end(), [&](const auto &task) {
            return MakeKey(task.hint.block_key, task.hint.target_node_id) == key;
        });
        if (queued != queue_.end() && queued->data == nullptr && queued->named_buffers.empty() &&
            piggyback_queue_size_ < max_piggyback_queue_) {
            const size_t bytes = std::max(size, HintBytes(hint));
            const size_t without_old = pending_bytes_ - queued->pending_bytes;
            if (bytes <= options_.max_pending_bytes && without_old <= options_.max_pending_bytes - bytes &&
                ReplicationResources::Global().TryRetain(size, options_.max_buffer_bytes)) {
                queued->hint = std::move(hint);
                queued->data = data;
                queued->data_size = size;
                queued->retained_memory = ReleaseGuard([size] { ReplicationResources::Global().Release(size); });
                queued->guard = std::move(guard);
                pending_bytes_ = without_old + bytes;
                queued->pending_bytes = bytes;
                ++piggyback_queue_size_;
                KVCM_LOG_INFO("[replication] upgraded queued automatic copy to piggyback for key [%s]", key.c_str());
                return;
            }
        }
        ++counters_.duplicates;
        KVCM_LOG_INFO("[replication] SubmitWithData: block_key [%ld] target [%s] already inflight "
                      "(likely auto-submitted by MatchLocation), piggyback data dropped. "
                      "Async replication via ExecuteHintAsync is in progress.",
                      hint.block_key,
                      hint.target_node_id.c_str());
        return;
    }
    if (queue_.size() >= max_pending_tasks_) {
        ++counters_.dropped_queue;
        return;
    }
    if (piggyback_queue_size_ >= max_piggyback_queue_) {
        ++counters_.dropped_queue;
        KVCM_LOG_WARN("[replication] SubmitWithData: block_key [%ld] target [%s] dropped, "
                      "piggyback queue full (%d/%d)",
                      hint.block_key,
                      hint.target_node_id.c_str(),
                      piggyback_queue_size_,
                      max_piggyback_queue_);
        return;
    }
    const auto source_bytes = HintBytes(hint);
    const size_t bytes = std::max(size, source_bytes);
    if (bytes > options_.max_pending_bytes || pending_bytes_ > options_.max_pending_bytes - bytes ||
        !ReplicationResources::Global().TryRetain(size, options_.max_buffer_bytes)) { ++counters_.dropped_budget; return; }
    ReleaseGuard retained([size] { ReplicationResources::Global().Release(size); });
    ++counters_.admitted;
    inflight_.insert(key);
    ++piggyback_queue_size_;
    // FIFO prevents a stream of piggyback tasks from starving ordinary hints.
    ReplicationTask task;
    task.hint = std::move(hint);
    task.data = data;
    task.data_size = size;
    task.retained_memory = std::move(retained);
    task.guard = std::move(guard);
    queue_.push_back(std::move(task));
    queue_.back().pending_bytes = bytes;
    pending_bytes_ += bytes;
    KVCM_LOG_INFO("[replication] SubmitWithData: block_key [%ld] target [%s] enqueued (piggyback_queue=%d/%d)",
                  hint.block_key,
                  hint.target_node_id.c_str(),
                  piggyback_queue_size_,
                  max_piggyback_queue_);
    cv_.notify_one();
}

bool ReplicationExecutor::SubmitWithBuffers(ClientReplicationHint hint,
                                             std::vector<ClientReplicationBuffer> buffers) {
    ++counters_.submitted;
    size_t retained_bytes = 0;
    std::set<std::string> names;
    std::map<std::string, std::string> sources;
    for (const auto &source : hint.source_specs) {
        if (source.spec_name.empty() || source.uri.empty() || !sources.emplace(source.spec_name, source.uri).second) {
            ++counters_.dropped_invalid; return false;
        }
    }
    for (const auto &buffer : buffers) {
        const auto source = sources.find(buffer.spec_name);
        size_t source_size = 0;
        if (source != sources.end()) StandardUri(source->second).GetParamAs<size_t>("size", source_size);
        if (!buffer.data || !buffer.owner || buffer.size == 0 || buffer.spec_name.empty() ||
            (buffer.memory_type != MemoryType::CPU && buffer.memory_type != MemoryType::GPU) ||
            !names.insert(buffer.spec_name).second || source == sources.end() || source_size != buffer.size ||
            buffer.size > SIZE_MAX - retained_bytes) {
            ++counters_.dropped_invalid; return false;
        }
        retained_bytes += buffer.size;
    }
    if (buffers.empty()) { ++counters_.dropped_invalid; return false; }
    const auto bytes = std::max(retained_bytes, HintBytes(hint));
    std::lock_guard<std::mutex> lock(mu_);
    if (stopped_.load()) { ++counters_.dropped_queue; return false; }
    const auto key = MakeKey(hint.block_key, hint.target_node_id);
    if (inflight_.count(key)) {
        auto queued = std::find_if(queue_.begin(), queue_.end(), [&](const auto &task) {
            return MakeKey(task.hint.block_key, task.hint.target_node_id) == key;
        });
        if (queued != queue_.end() && queued->data == nullptr && queued->named_buffers.empty() &&
            piggyback_queue_size_ < max_piggyback_queue_) {
            const size_t without_old = pending_bytes_ - queued->pending_bytes;
            if (bytes <= options_.max_pending_bytes && without_old <= options_.max_pending_bytes - bytes &&
                bytes <= options_.max_buffer_bytes &&
                ReplicationResources::Global().TryRetain(retained_bytes, options_.max_buffer_bytes)) {
                queued->retained_memory =
                    ReleaseGuard([retained_bytes] { ReplicationResources::Global().Release(retained_bytes); });
                queued->hint = std::move(hint);
                queued->named_buffers = std::move(buffers);
                pending_bytes_ = without_old + bytes;
                queued->pending_bytes = bytes;
                ++piggyback_queue_size_;
                return true;
            }
        }
        ++counters_.duplicates;
        return false;
    }
    if (queue_.size() >= max_pending_tasks_ || piggyback_queue_size_ >= max_piggyback_queue_) {
        ++counters_.dropped_queue; return false;
    }
    if (bytes > options_.max_pending_bytes || bytes > options_.max_buffer_bytes ||
        pending_bytes_ > options_.max_pending_bytes - bytes ||
        !ReplicationResources::Global().TryRetain(retained_bytes, options_.max_buffer_bytes)) {
        ++counters_.dropped_budget; return false;
    }
    ReplicationTask task;
    task.retained_memory = ReleaseGuard([retained_bytes] { ReplicationResources::Global().Release(retained_bytes); });
    task.hint = std::move(hint);
    task.named_buffers = std::move(buffers);
    task.pending_bytes = bytes;
    queue_.push_back(std::move(task));
    inflight_.insert(key);
    pending_bytes_ += bytes;
    ++piggyback_queue_size_;
    ++counters_.admitted;
    cv_.notify_one();
    return true;
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
            pending_bytes_ -= task.pending_bytes;
            if (task.data || !task.named_buffers.empty()) {
                --piggyback_queue_size_;
            }
        }
        const auto begin = ReplicationResources::Clock::now();
        counters_.queue_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(begin - task.submitted_at).count();
        ++counters_.active;
        try {
            ExecuteTask(task);
        } catch (const std::exception &e) {
            KVCM_LOG_WARN("[replication] block_key [%ld] failed: %s", task.hint.block_key, e.what());
        } catch (...) {
            KVCM_LOG_WARN("[replication] unknown worker exception");
        }
        --counters_.active;
        counters_.latency_us += std::chrono::duration_cast<std::chrono::microseconds>(
            ReplicationResources::Clock::now() - begin).count();
        switch (task.outcome) {
        case ReplicationTask::Outcome::Succeeded: ++counters_.succeeded; counters_.copied_bytes += task.copied_bytes; break;
        case ReplicationTask::Outcome::Skipped: ++counters_.skipped; break;
        case ReplicationTask::Outcome::Expired: ++counters_.expired; break;
        case ReplicationTask::Outcome::Failed: ++counters_.failed; break;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            inflight_.erase(MakeKey(task.hint.block_key, task.hint.target_node_id));
        }
        if (options_.metrics_callback) {
            try { options_.metrics_callback(GetStats()); }
            catch (...) { KVCM_LOG_WARN("[replication] metrics callback failed"); }
        }
    }
}

ReplicationStats ReplicationExecutor::GetStats() const {
    ReplicationStats result;
    result.submitted = counters_.submitted.load(std::memory_order_relaxed);
    result.admitted = counters_.admitted.load(std::memory_order_relaxed);
    result.succeeded = counters_.succeeded.load(std::memory_order_relaxed);
    result.failed = counters_.failed.load(std::memory_order_relaxed);
    result.skipped = counters_.skipped.load(std::memory_order_relaxed);
    result.expired = counters_.expired.load(std::memory_order_relaxed);
    result.dropped_queue = counters_.dropped_queue.load(std::memory_order_relaxed);
    result.dropped_budget = counters_.dropped_budget.load(std::memory_order_relaxed);
    result.dropped_invalid = counters_.dropped_invalid.load(std::memory_order_relaxed);
    result.duplicates = counters_.duplicates.load(std::memory_order_relaxed);
    result.copied_bytes = counters_.copied_bytes.load(std::memory_order_relaxed);
    result.latency_us = counters_.latency_us.load(std::memory_order_relaxed);
    result.queue_wait_us = counters_.queue_wait_us.load(std::memory_order_relaxed);
    result.active = counters_.active.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    result.queued = queue_.size();
    result.pending_bytes = pending_bytes_;
    return result;
}

void ReplicationExecutor::ExecuteTask(ReplicationTask &task) {
    if (ReplicationResources::Clock::now() >= task.submitted_at + std::chrono::milliseconds(options_.max_age_ms)) {
        task.outcome = ReplicationTask::Outcome::Expired;
        return;
    }
    const auto &hint = task.hint;
    const std::string trace_id = "repl_" + StringUtil::GenerateRandomString(16);
    const auto caller = meta_client_->GetCallerNode();
    if (!caller.empty() && caller != hint.target_node_id) {
        task.outcome = ReplicationTask::Outcome::Skipped;
        KVCM_LOG_WARN("[replication] caller changed; dropping hint for node [%s]", hint.target_node_id.c_str());
        return;
    }
    const bool has_client_buffer = (task.data != nullptr && task.data_size > 0) || !task.named_buffers.empty();
    if (!has_client_buffer) {
        const auto server_ec = meta_client_->ReplicateCache(trace_id, hint, /*write_timeout_seconds=*/60);
        if (server_ec == ER_OK) {
            task.outcome = ReplicationTask::Outcome::Succeeded;
            task.copied_bytes = HintBytes(hint);
            return;
        }
        if (server_ec != ER_SERVICE_UNSUPPORTED) {
            KVCM_LOG_WARN("[replication] server-side copy failed for block_key [%ld], ec [%d]",
                          hint.block_key, server_ec);
            return;
        }
    }
    auto [start_ec, write_loc] = meta_client_->StartReplicationWrite(
        trace_id, {hint.block_key}, {}, /*write_timeout_seconds=*/60, hint.target_node_id);
    if (start_ec != ER_OK) return;

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
    if (write_loc.locations.empty()) {
        finish_attempted = true;
        const auto ec = meta_client_->FinishWrite(
            trace_id, write_loc.write_session_id, BlockMaskOffset(0), {});
        if (ec == ER_OK) task.outcome = ReplicationTask::Outcome::Skipped;
        return; // The complete replica already exists on the target.
    }
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
    // The existing piggyback API owns one unnamed buffer. It is safe only for
    // a single spec; multi-spec hints use their complete named source set.
    const bool piggyback = task.data != nullptr && task.data_size > 0 && destinations.size() == 1 &&
                           hint.source_specs.size() <= 1;
    std::unordered_map<std::string, const ClientReplicationBuffer *> reused;
    for (const auto &buffer : task.named_buffers) reused.emplace(buffer.spec_name, &buffer);
    for (const auto &entry : reused) {
        if (std::none_of(destinations.begin(), destinations.end(), [&](const auto &dest) {
                return dest.spec_name == entry.first;
            })) return;
    }
    size_t buffer_bytes = 0, allocate_bytes = 0;
    for (const auto &dest : destinations) {
        const auto source = sources.find(dest.spec_name);
        if (source == sources.end()) return;
        size_t bytes = task.data_size;
        if (!piggyback) {
            bytes = 0;
            StandardUri(source->second).GetParamAs<size_t>("size", bytes);
        }
        if (bytes == 0 || bytes > SIZE_MAX - buffer_bytes) return;
        const auto reuse = reused.find(dest.spec_name);
        if (reuse != reused.end() && reuse->second->size != bytes) return;
        buffer_bytes += bytes;
        if (!piggyback && reuse == reused.end()) allocate_bytes += bytes;
    }
    if (!ReplicationResources::Global().Acquire(options_.instance_id, hint.target_node_id,
            allocate_bytes, options_.max_buffer_bytes, buffer_bytes, options_.node_bytes_per_second,
            task.submitted_at + std::chrono::milliseconds(options_.max_age_ms), stopped_)) {
        if (ReplicationResources::Clock::now() >= task.submitted_at + std::chrono::milliseconds(options_.max_age_ms))
            task.outcome = ReplicationTask::Outcome::Expired;
        else ++counters_.dropped_budget;
        return;
    }
    ReleaseGuard memory([allocate_bytes] { ReplicationResources::Global().Release(allocate_bytes); });
    std::vector<std::vector<char>> owned_buffers(destinations.size());
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
        const auto reuse = reused.find(dest.spec_name);
        MemoryType memory_type = MemoryType::CPU;
        if (reuse != reused.end()) {
            if (reuse->second->size != size) return;
            data = const_cast<void *>(reuse->second->data);
            memory_type = reuse->second->memory_type;
        } else if (!piggyback) {
            owned_buffers[i].resize(size);
            data = owned_buffers[i].data();
        }
        BlockBuffer buffer;
        buffer.iovs.push_back(Iov{memory_type, data, size, false});
        if (!piggyback && reuse == reused.end() && transfer_client_->LoadKvCaches({source->second}, {buffer}) != ER_OK) {
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
    if (finish_ec == ER_OK) {
        task.outcome = ReplicationTask::Outcome::Succeeded;
        task.copied_bytes = buffer_bytes;
    }
    if (finish_ec != ER_OK) {
        KVCM_LOG_WARN("[replication] FinishWrite failed for block_key [%ld], ec [%d]", hint.block_key, finish_ec);
    }
}

std::string ReplicationExecutor::MakeKey(int64_t block_key, const std::string &target_node_id) const {
    return std::to_string(block_key) + ":" + target_node_id;
}

} // namespace kv_cache_manager
