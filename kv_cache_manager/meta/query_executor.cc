#include "kv_cache_manager/meta/query_executor.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {

thread_local const QueryExecutor *current_query_executor = nullptr;

struct ParallelState {
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex mutex;
    std::condition_variable condition;
    bool accepting_workers = true;
    std::size_t active_workers = 0;

    bool TryStartWorker() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!accepting_workers) {
            return false;
        }
        ++active_workers;
        return true;
    }

    void CompleteWorker() {
        std::lock_guard<std::mutex> lock(mutex);
        if (--active_workers == 0) {
            condition.notify_all();
        }
    }

    void StopWorkersAndWait() {
        std::unique_lock<std::mutex> lock(mutex);
        accepting_workers = false;
        condition.wait(lock, [this] { return active_workers == 0; });
    }
};

} // namespace

QueryExecutor::QueryExecutor(std::size_t worker_count,
                             std::size_t parallel_threshold,
                             std::size_t chunk_size,
                             std::size_t queue_capacity)
    : worker_count_(std::max<std::size_t>(1, worker_count))
    , parallel_threshold_(std::max<std::size_t>(1, parallel_threshold))
    , chunk_size_(std::max<std::size_t>(1, chunk_size))
    , queue_capacity_(std::max<std::size_t>(1, queue_capacity)) {
    workers_.reserve(worker_count_ - 1);
    for (std::size_t i = 1; i < worker_count_; ++i) {
        workers_.emplace_back([this] { WorkerLoop(); });
    }
}

QueryExecutor::~QueryExecutor() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

bool QueryExecutor::TrySubmit(std::function<void()> task) const {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || tasks_.size() >= queue_capacity_) {
            return false;
        }
        tasks_.push_back(std::move(task));
    }
    condition_.notify_one();
    return true;
}

void QueryExecutor::WorkerLoop() {
    current_query_executor = this;
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) {
                break;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
    }
    current_query_executor = nullptr;
}

bool QueryExecutor::ParallelFor(std::size_t count, const RangeFunction &fn) const noexcept {
    if (count == 0) {
        return true;
    }
    if (worker_count_ <= 1 || count < parallel_threshold_ || current_query_executor == this) {
        try {
            fn(0, count);
            return true;
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("query executor serial callback threw exception: %s", e.what());
        } catch (...) { KVCM_LOG_ERROR("query executor serial callback threw unknown exception"); }
        return false;
    }

    const std::size_t chunk_count = 1 + (count - 1) / chunk_size_;
    const std::size_t parallelism = std::min(worker_count_, chunk_count);
    if (parallelism <= 1) {
        try {
            fn(0, count);
            return true;
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("query executor callback threw exception: %s", e.what());
        } catch (...) { KVCM_LOG_ERROR("query executor callback threw unknown exception"); }
        return false;
    }

    auto state = std::make_shared<ParallelState>();
    auto fn_holder = std::make_shared<RangeFunction>(fn);
    auto run_ranges = [state, fn_holder, count, chunk_size = chunk_size_]() noexcept {
        while (true) {
            const std::size_t begin = state->next.fetch_add(chunk_size, std::memory_order_relaxed);
            if (begin >= count) {
                return;
            }
            const std::size_t end = std::min(count, begin + chunk_size);
            try {
                (*fn_holder)(begin, end);
            } catch (const std::exception &e) {
                state->failed.store(true, std::memory_order_relaxed);
                KVCM_LOG_ERROR("query executor parallel callback threw exception: %s", e.what());
            } catch (...) {
                state->failed.store(true, std::memory_order_relaxed);
                KVCM_LOG_ERROR("query executor parallel callback threw unknown exception");
            }
        }
    };

    for (std::size_t i = 1; i < parallelism; ++i) {
        if (!TrySubmit([state, run_ranges] {
                // The caller may have consumed every range while this task was
                // waiting behind another request. In that case it cancels the
                // queued helper and returns without waiting for a no-op task to
                // reach the head of the global queue.
                if (!state->TryStartWorker()) {
                    return;
                }
                run_ranges();
                state->CompleteWorker();
            })) {
            // The caller and any admitted workers consume every range via the
            // shared atomic cursor when the bounded queue is full.
            break;
        }
    }

    run_ranges();
    state->StopWorkersAndWait();
    return !state->failed.load(std::memory_order_relaxed);
}

} // namespace kv_cache_manager
