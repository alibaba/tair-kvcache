// Executor: a fixed pool of worker threads that only runs short computations,
// state transitions and coroutine continuations.
//
// Network waiting happens on the transport reactor / gRPC completion-queue
// threads, never here: a coroutine that issues an RPC suspends and releases
// its worker immediately. The number of logical clients, sessions and
// in-flight RPCs is therefore independent of `workers`.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "async_simple/Executor.h"
#include "async_simple/coro/Lazy.h"
#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/runtime/stop_token.h"

namespace kvcm_swarm {

template <typename T = void>
using Task = async_simple::coro::Lazy<T>;

class SwarmExecutor : public async_simple::Executor {
public:
    explicit SwarmExecutor(uint32_t worker_count);
    ~SwarmExecutor() override;

    bool schedule(Func func) override;
    bool currentThreadInExecutor() const override;
    async_simple::ExecutorStat stat() const override;

    // Runs `func` on a worker thread once `when` has passed. The timer thread
    // never runs user code, so timer accuracy is independent of worker load.
    void ScheduleAt(TimePoint when, Func func);

    // Stops accepting work and joins every thread. Safe to call twice.
    void Shutdown();

    uint32_t worker_count() const { return worker_count_; }
    uint64_t peak_queue_depth() const { return peak_queue_depth_.load(std::memory_order_relaxed); }
    uint64_t scheduled_total() const { return scheduled_total_.load(std::memory_order_relaxed); }
    // Sum of the time tasks spent queued before a worker picked them up.
    double queue_delay_sum_ms() const;
    uint64_t queue_delay_samples() const { return queue_delay_samples_.load(std::memory_order_relaxed); }
    uint64_t timer_count() const { return timer_count_.load(std::memory_order_relaxed); }

protected:
    // The base class default spawns one thread per timer; override both
    // overloads so delayed scheduling uses the single timer thread instead.
    void schedule(Func func, async_simple::Executor::Duration dur) override {
        ScheduleAt(Now() + std::chrono::duration_cast<::kvcm_swarm::Duration>(dur), std::move(func));
    }
    void schedule(Func func,
                  async_simple::Executor::Duration dur,
                  uint64_t /*schedule_info*/,
                  async_simple::Slot * /*slot*/) override {
        ScheduleAt(Now() + std::chrono::duration_cast<::kvcm_swarm::Duration>(dur), std::move(func));
    }

private:
    struct QueuedTask {
        Func func;
        TimePoint enqueued_at;
    };

    struct TimerEntry {
        TimePoint when;
        uint64_t sequence;
        Func func;
        bool operator>(const TimerEntry &other) const {
            if (when != other.when) {
                return when > other.when;
            }
            return sequence > other.sequence;
        }
    };

    void WorkerLoop();
    void TimerLoop();

    const uint32_t worker_count_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<QueuedTask> queue_;
    std::atomic<bool> stopped_{false};

    std::mutex timer_mutex_;
    std::condition_variable timer_cv_;
    std::priority_queue<TimerEntry, std::vector<TimerEntry>, std::greater<TimerEntry>> timers_;
    uint64_t timer_sequence_ = 0;

    std::vector<std::thread> workers_;
    std::thread timer_thread_;

    std::atomic<uint64_t> peak_queue_depth_{0};
    std::atomic<uint64_t> scheduled_total_{0};
    std::atomic<uint64_t> queue_delay_ns_{0};
    std::atomic<uint64_t> queue_delay_samples_{0};
    std::atomic<uint64_t> timer_count_{0};
};

// One-shot asynchronous slot: a coroutine awaits it and some other thread
// (timer, completion queue, permit release) resumes the coroutine on the
// Executor. This is the single primitive used for every non-network wait.
template <typename T>
class AsyncSlot {
public:
    AsyncSlot(SwarmExecutor &executor) : executor_(executor) {}

    AsyncSlot(const AsyncSlot &) = delete;
    AsyncSlot &operator=(const AsyncSlot &) = delete;

    // Returns false when the slot had already been completed.
    bool Complete(T value) {
        std::coroutine_handle<> handle;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ready_) {
                return false;
            }
            ready_ = true;
            value_ = std::move(value);
            handle = handle_;
            handle_ = nullptr;
        }
        if (handle) {
            executor_.schedule([handle]() mutable { handle.resume(); });
        }
        return true;
    }

    bool ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ready_;
    }

    struct Awaiter {
        AsyncSlot *slot;

        bool await_ready() const noexcept { return slot->ready(); }

        bool await_suspend(std::coroutine_handle<> handle) noexcept {
            std::lock_guard<std::mutex> lock(slot->mutex_);
            if (slot->ready_) {
                return false;
            }
            slot->handle_ = handle;
            return true;
        }

        T await_resume() noexcept {
            std::lock_guard<std::mutex> lock(slot->mutex_);
            return std::move(slot->value_);
        }

        // Opt out of async_simple's ViaCoroutine wrapping: `Complete` already
        // reschedules the continuation on the Executor.
        Awaiter coAwait(async_simple::Executor *) { return *this; }
    };

    Awaiter operator co_await() { return Awaiter{this}; }

private:
    SwarmExecutor &executor_;
    mutable std::mutex mutex_;
    bool ready_ = false;
    T value_{};
    std::coroutine_handle<> handle_ = nullptr;
};

// FIFO asynchronous weighted budget. An operation reserves an explicit amount
// of one logical resource without occupying an Executor worker while it waits.
class AsyncCapacityBudget {
public:
    class Guard {
    public:
        Guard() = default;
        Guard(AsyncCapacityBudget *owner, uint64_t amount) : owner_(owner), amount_(amount) {}
        Guard(Guard &&other) noexcept : owner_(other.owner_), amount_(other.amount_) {
            other.owner_ = nullptr;
            other.amount_ = 0;
        }
        Guard &operator=(Guard &&other) noexcept {
            if (this != &other) {
                Release();
                owner_ = other.owner_;
                amount_ = other.amount_;
                other.owner_ = nullptr;
                other.amount_ = 0;
            }
            return *this;
        }
        Guard(const Guard &) = delete;
        Guard &operator=(const Guard &) = delete;
        ~Guard() { Release(); }

        bool valid() const { return owner_ != nullptr; }
        uint64_t amount() const { return amount_; }
        void Release();

    private:
        AsyncCapacityBudget *owner_ = nullptr;
        uint64_t amount_ = 0;
    };

    AsyncCapacityBudget(SwarmExecutor &executor, uint64_t capacity) : executor_(executor), capacity_(capacity) {}

    // An invalid Guard means the deadline passed or the run is stopping.
    Task<Guard> Acquire(uint64_t amount, TimePoint deadline, StopToken stop);

    uint64_t capacity() const { return capacity_; }
    uint64_t in_use() const;
    uint64_t peak_in_use() const;
    uint64_t waits() const;
    uint64_t timeouts() const;
    uint64_t wait_ns_total() const;
    uint64_t wait_ns_max() const;

private:
    friend class Guard;
    struct Waiter {
        uint64_t amount = 0;
        std::shared_ptr<AsyncSlot<bool>> slot;
    };

    void Release(uint64_t amount);
    void DispatchWaiters();

    SwarmExecutor &executor_;
    const uint64_t capacity_;
    mutable std::mutex mutex_;
    uint64_t in_use_ = 0;
    uint64_t peak_in_use_ = 0;
    uint64_t waits_ = 0;
    uint64_t timeouts_ = 0;
    uint64_t wait_ns_total_ = 0;
    uint64_t wait_ns_max_ = 0;
    std::deque<Waiter> waiters_;
};

// Suspends until `until`, or until `stop` is requested. Returns true when the
// deadline elapsed, false when the wait was cancelled.
Task<bool> SleepUntil(SwarmExecutor &executor, TimePoint until, StopToken stop);

Task<bool> SleepFor(SwarmExecutor &executor, Duration duration, StopToken stop);

// Hands the current coroutine back to the Executor queue.
Task<> Reschedule(SwarmExecutor &executor);

} // namespace kvcm_swarm
