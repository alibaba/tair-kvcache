#include "kv_cache_manager/service/service_request_executor.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <system_error>
#include <utility>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

struct ServiceRequestExecutor::State {
    struct Entry {
        Request task;
        CompletionToken completion;
        std::chrono::steady_clock::time_point enqueued_at;
    };

    explicit State(std::size_t capacity) : queue_capacity(capacity) {}

    std::optional<SubmitResult> RejectIfUnavailable(bool over_protocol_limit = false) {
        if (closed) {
            ++rejected;
            return SubmitResult::kClosed;
        }
        if (over_protocol_limit || (queue_capacity != 0 && queue.size() >= queue_capacity)) {
            ++rejected;
            return SubmitResult::kFull;
        }
        return std::nullopt;
    }

    using Clock = std::chrono::steady_clock;
    void AccumulateQueueDepth(Clock::time_point now) {
        queue_area_us += static_cast<long double>(queue.size()) *
                         std::chrono::duration<double, std::micro>(now - depth_updated_at).count();
        depth_updated_at = now;
    }

    Snapshot SnapshotLocked(Clock::time_point now) {
        AccumulateQueueDepth(now);
        Snapshot snapshot;
        snapshot.queued = queue.size();
        snapshot.running = running;
        snapshot.discarding = discarding;
        snapshot.completion_tokens = completion_tokens;
        snapshot.rejected = rejected;
        snapshot.discarded = discarded;
        snapshot.queue_wait = queue_wait;
        snapshot.queue_length_max = queue_length_max;
        const auto elapsed_us = std::chrono::duration<double, std::micro>(now - window_started_at).count();
        snapshot.queue_length_avg = elapsed_us > 0 ? queue_area_us / elapsed_us : queue.size();
        snapshot.wait_time = wait_time;
        snapshot.execution_time = execution_time;
        return snapshot;
    }

    std::mutex mutex;
    std::condition_variable work_available;
    std::condition_variable changed;
    std::deque<Entry> queue;
    const std::size_t queue_capacity;
    bool closed = false;
    std::size_t running = 0;
    std::size_t discarding = 0;
    std::size_t completion_tokens = 0;
    std::size_t rejected = 0;
    std::size_t discarded = 0;
    QueueWaitStats queue_wait;
    Snapshot::TimingStats wait_time;
    Snapshot::TimingStats execution_time;
    std::size_t queue_length_max = 0;
    long double queue_area_us = 0;
    Clock::time_point window_started_at = Clock::now();
    Clock::time_point depth_updated_at = window_started_at;
};

struct ServiceRequestExecutor::CompletionLease {
    explicit CompletionLease(std::shared_ptr<State> state) : state(std::move(state)) {}

    ~CompletionLease() {
        if (accepted) {
            // Use the same mutex as Drain to avoid losing the final wakeup.
            std::lock_guard<std::mutex> lock(state->mutex);
            --state->completion_tokens;
            state->changed.notify_all();
        }
    }

    std::shared_ptr<State> state;
    bool accepted = false;
};

bool ServiceRequestExecutor::Init(std::size_t worker_count, std::size_t queue_capacity) {
    if (worker_count == 0 || state_) {
        return false;
    }
    state_ = std::make_shared<State>(queue_capacity);
    workers_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
        try {
            workers_.emplace_back([state = state_] { WorkerLoop(state); });
        } catch (const std::system_error &error) {
            KVCM_LOG_ERROR("create service request worker failed: %s", error.what());
            CloseAdmission();
            Join();
            return false;
        }
    }
    return true;
}

ServiceRequestExecutor::~ServiceRequestExecutor() {
    CloseAdmission();
    Drain();
    Join();
}

ServiceRequestExecutor::SubmitResult ServiceRequestExecutor::TrySubmit(Request task) {
    if (!task.run || !task.on_discard) {
        return SubmitResult::kInvalid;
    }

    if (!state_) {
        return SubmitResult::kClosed;
    }

    // Keep a local lease until after the lock is released. If queue allocation
    // throws, destroying a temporary Entry cannot acquire this mutex again.
    std::shared_ptr<CompletionLease> lease;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (auto rejection = state_->RejectIfUnavailable()) {
            return *rejection;
        }
        lease = std::make_shared<CompletionLease>(state_);
        state_->AccumulateQueueDepth(State::Clock::now());
        state_->queue.push_back({std::move(task), CompletionToken(lease), {}});
        state_->queue_length_max = std::max(state_->queue_length_max, state_->queue.size());
        ++state_->completion_tokens;
        lease->accepted = true;
        // Timestamp only successful admission, after any deque allocation and
        // before a worker can claim the task. Wall-clock changes do not matter.
        state_->queue.back().enqueued_at = std::chrono::steady_clock::now();
    }
    state_->work_available.notify_one();
    return SubmitResult::kAccepted;
}

std::optional<ServiceRequestExecutor::SubmitResult>
ServiceRequestExecutor::RejectIfUnavailable(bool over_protocol_limit) {
    if (!state_) {
        return SubmitResult::kClosed;
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->RejectIfUnavailable(over_protocol_limit);
}

void ServiceRequestExecutor::CloseAdmission(ShutdownMode mode) {
    if (!state_) {
        return;
    }
    std::deque<State::Entry> discarded;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->closed) {
            return;
        }
        state_->closed = true;
        if (mode == ShutdownMode::kDiscardPending) {
            state_->AccumulateQueueDepth(State::Clock::now());
            discarded.swap(state_->queue);
            state_->discarding += discarded.size();
            state_->discarded += discarded.size();
        }
    }
    state_->work_available.notify_all();
    state_->changed.notify_all();

    while (!discarded.empty()) {
        {
            // Destroy callback captures and the local token before announcing
            // this discard as complete. Adapters may retain their own token.
            auto entry = std::move(discarded.front());
            discarded.pop_front();
            entry.task.on_discard(std::move(entry.completion));
        }
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            --state_->discarding;
        }
        state_->changed.notify_all();
    }
}

void ServiceRequestExecutor::Drain() {
    if (!state_) {
        return;
    }
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->changed.wait(lock, [&] {
        return state_->queue.empty() && state_->running == 0 && state_->discarding == 0 &&
               state_->completion_tokens == 0;
    });
}

void ServiceRequestExecutor::Join() {
    CloseAdmission();
    std::lock_guard<std::mutex> lock(join_mutex_);
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

ServiceRequestExecutor::Snapshot ServiceRequestExecutor::GetSnapshot() const {
    if (!state_) {
        return {};
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->SnapshotLocked(State::Clock::now());
}

ServiceRequestExecutor::Snapshot ServiceRequestExecutor::TakeMetricsSnapshot() {
    if (!state_) {
        return {};
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto now = State::Clock::now();
    auto snapshot = state_->SnapshotLocked(now);
    state_->wait_time = {};
    state_->execution_time = {};
    state_->queue_length_max = state_->queue.size();
    state_->queue_area_us = 0;
    state_->window_started_at = now;
    return snapshot;
}

void ServiceRequestExecutor::WorkerLoop(const std::shared_ptr<State> &state) {
    for (;;) {
        State::Clock::time_point started_at;
        {
            State::Entry entry;
            {
                std::unique_lock<std::mutex> lock(state->mutex);
                state->work_available.wait(lock, [&] { return state->closed || !state->queue.empty(); });
                if (state->queue.empty()) {
                    return;
                }
                started_at = State::Clock::now();
                state->AccumulateQueueDepth(started_at);
                entry = std::move(state->queue.front());
                state->queue.pop_front();
                ++state->running;
                const auto wait_us = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(started_at - entry.enqueued_at).count());
                const auto bucket = std::lower_bound(kQueueWaitBoundsUs.begin(), kQueueWaitBoundsUs.end(), wait_us) -
                                    kQueueWaitBoundsUs.begin();
                // Reuse the dequeue lock for three integer updates. No metric
                // lookup, allocation or per-request reporting on this path.
                ++state->queue_wait.count;
                state->queue_wait.sum_us += wait_us;
                ++state->queue_wait.buckets[bucket];
                ++state->wait_time.count;
                state->wait_time.sum_us += wait_us;
                state->wait_time.max_us = std::max(state->wait_time.max_us, wait_us);
            }
            entry.task.run(std::move(entry.completion));
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            --state->running;
            const auto execution_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(State::Clock::now() - started_at).count());
            ++state->execution_time.count;
            state->execution_time.sum_us += execution_us;
            state->execution_time.max_us = std::max(state->execution_time.max_us, execution_us);
        }
        state->changed.notify_all();
    }
}

} // namespace kv_cache_manager
