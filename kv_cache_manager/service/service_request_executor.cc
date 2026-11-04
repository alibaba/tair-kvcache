#include "kv_cache_manager/service/service_request_executor.h"

#include <condition_variable>
#include <deque>
#include <stdexcept>
#include <utility>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

struct ServiceRequestExecutor::State {
    struct Entry {
        Task task;
        CompletionToken completion;
    };

    explicit State(std::size_t capacity) : queue_capacity(capacity) {}

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

ServiceRequestExecutor::ServiceRequestExecutor(std::size_t worker_count, std::size_t queue_capacity)
    : state_(std::make_shared<State>(queue_capacity)) {
    if (worker_count == 0) {
        throw std::invalid_argument("ServiceRequestExecutor requires at least one worker");
    }
    workers_.reserve(worker_count);
    try {
        for (std::size_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([state = state_] { WorkerLoop(state); });
        }
    } catch (...) {
        // A partially constructed executor has no submitted work. Wake and
        // join the threads that were created before propagating the failure.
        CloseAdmission();
        Join();
        throw;
    }
}

ServiceRequestExecutor::~ServiceRequestExecutor() {
    CloseAdmission();
    Drain();
    Join();
}

ServiceRequestExecutor::SubmitResult ServiceRequestExecutor::TrySubmit(Task task) {
    if (!task.run || !task.on_discard) {
        throw std::invalid_argument("ServiceRequestExecutor requires run and on_discard callbacks");
    }

    // Keep a local lease until after the lock is released. If queue allocation
    // throws, destroying a temporary Entry cannot acquire this mutex again.
    auto lease = std::make_shared<CompletionLease>(state_);
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->closed) {
            ++state_->rejected;
            return SubmitResult::kClosed;
        }
        if (state_->queue_capacity != 0 && state_->queue.size() >= state_->queue_capacity) {
            ++state_->rejected;
            return SubmitResult::kFull;
        }
        state_->queue.push_back({std::move(task), CompletionToken(lease)});
        ++state_->completion_tokens;
        lease->accepted = true;
    }
    state_->work_available.notify_one();
    return SubmitResult::kAccepted;
}

void ServiceRequestExecutor::CloseAdmission(ShutdownMode mode) {
    std::deque<State::Entry> discarded;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->closed) {
            return;
        }
        state_->closed = true;
        if (mode == ShutdownMode::kDiscardPending) {
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
            try {
                entry.task.on_discard(std::move(entry.completion));
            } catch (const std::exception &e) {
                KVCM_LOG_ERROR("service request discard callback threw: %s", e.what());
            } catch (...) {
                KVCM_LOG_ERROR("service request discard callback threw an unknown exception");
            }
        }
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            --state_->discarding;
        }
        state_->changed.notify_all();
    }
}

void ServiceRequestExecutor::Drain() {
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
    std::lock_guard<std::mutex> lock(state_->mutex);
    return {state_->queue.size(),
            state_->running,
            state_->discarding,
            state_->completion_tokens,
            state_->rejected,
            state_->discarded};
}

void ServiceRequestExecutor::WorkerLoop(const std::shared_ptr<State> &state) {
    for (;;) {
        {
            State::Entry entry;
            {
                std::unique_lock<std::mutex> lock(state->mutex);
                state->work_available.wait(lock, [&] { return state->closed || !state->queue.empty(); });
                if (state->queue.empty()) {
                    return;
                }
                entry = std::move(state->queue.front());
                state->queue.pop_front();
                ++state->running;
            }
            try {
                entry.task.run(std::move(entry.completion));
            } catch (const std::exception &e) {
                KVCM_LOG_ERROR("service request callback threw: %s", e.what());
            } catch (...) {
                KVCM_LOG_ERROR("service request callback threw an unknown exception");
            }
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            --state->running;
        }
        state->changed.notify_all();
    }
}

} // namespace kv_cache_manager
