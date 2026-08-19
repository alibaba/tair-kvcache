#include "tools/kvcm_swarm/runtime/executor.h"

#include <utility>

namespace kvcm_swarm {
namespace {
thread_local bool g_in_executor = false;
} // namespace

SwarmExecutor::SwarmExecutor(uint32_t worker_count)
    : async_simple::Executor("kvcm_swarm"), worker_count_(worker_count == 0 ? 1u : worker_count) {
    workers_.reserve(worker_count_);
    for (uint32_t i = 0; i < worker_count_; ++i) {
        workers_.emplace_back([this]() { WorkerLoop(); });
    }
    timer_thread_ = std::thread([this]() { TimerLoop(); });
}

SwarmExecutor::~SwarmExecutor() { Shutdown(); }

bool SwarmExecutor::schedule(Func func) {
    const TimePoint now = Now();
    size_t depth = 0;
    if (stopped_.load(std::memory_order_acquire)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_.load(std::memory_order_relaxed)) {
            return false;
        }
        queue_.push_back(QueuedTask{std::move(func), now});
        depth = queue_.size();
    }
    scheduled_total_.fetch_add(1, std::memory_order_relaxed);
    uint64_t peak = peak_queue_depth_.load(std::memory_order_relaxed);
    while (depth > peak && !peak_queue_depth_.compare_exchange_weak(peak, depth, std::memory_order_relaxed)) {}
    cv_.notify_one();
    return true;
}

bool SwarmExecutor::currentThreadInExecutor() const { return g_in_executor; }

async_simple::ExecutorStat SwarmExecutor::stat() const {
    async_simple::ExecutorStat s;
    std::lock_guard<std::mutex> lock(mutex_);
    s.pendingTaskCount = queue_.size();
    return s;
}

double SwarmExecutor::queue_delay_sum_ms() const {
    return static_cast<double>(queue_delay_ns_.load(std::memory_order_relaxed)) / 1e6;
}

void SwarmExecutor::ScheduleAt(TimePoint when, Func func) {
    {
        std::lock_guard<std::mutex> lock(timer_mutex_);
        if (stopped_.load(std::memory_order_relaxed)) {
            return;
        }
        timers_.push(TimerEntry{when, timer_sequence_++, std::move(func)});
    }
    timer_count_.fetch_add(1, std::memory_order_relaxed);
    timer_cv_.notify_one();
}

void SwarmExecutor::Shutdown() {
    if (stopped_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    { std::lock_guard<std::mutex> lock(mutex_); }
    { std::lock_guard<std::mutex> lock(timer_mutex_); }
    cv_.notify_all();
    timer_cv_.notify_all();
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }
    // Drop any coroutine continuations that never got a chance to run so their
    // frames are destroyed instead of leaked.
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
    std::lock_guard<std::mutex> timer_lock(timer_mutex_);
    while (!timers_.empty()) {
        timers_.pop();
    }
}

void SwarmExecutor::WorkerLoop() {
    g_in_executor = true;
    while (true) {
        QueuedTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stopped_.load(std::memory_order_relaxed) || !queue_.empty(); });
            if (queue_.empty()) {
                if (stopped_.load(std::memory_order_relaxed)) {
                    return;
                }
                continue;
            }
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        const ::kvcm_swarm::Duration delay = Now() - task.enqueued_at;
        queue_delay_ns_.fetch_add(static_cast<uint64_t>(delay.count() < 0 ? 0 : delay.count()),
                                  std::memory_order_relaxed);
        queue_delay_samples_.fetch_add(1, std::memory_order_relaxed);
        task.func();
    }
}

void SwarmExecutor::TimerLoop() {
    while (true) {
        Func due;
        {
            std::unique_lock<std::mutex> lock(timer_mutex_);
            if (stopped_.load(std::memory_order_relaxed)) {
                return;
            }
            if (timers_.empty()) {
                timer_cv_.wait(lock, [this]() { return stopped_.load(std::memory_order_relaxed) || !timers_.empty(); });
                if (stopped_.load(std::memory_order_relaxed)) {
                    return;
                }
                continue;
            }
            const TimePoint next = timers_.top().when;
            if (next > Now()) {
                timer_cv_.wait_until(lock, next);
                continue;
            }
            due = std::move(const_cast<TimerEntry &>(timers_.top()).func);
            timers_.pop();
        }
        // Timer callbacks run on a worker so this thread stays a pure clock.
        schedule(std::move(due));
    }
}

void AsyncCapacityBudget::Guard::Release() {
    if (owner_ != nullptr) {
        owner_->Release(amount_);
        owner_ = nullptr;
        amount_ = 0;
    }
}

Task<AsyncCapacityBudget::Guard> AsyncCapacityBudget::Acquire(uint64_t amount, TimePoint deadline, StopToken stop) {
    std::shared_ptr<AsyncSlot<bool>> slot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop.StopRequested() || deadline <= Now() || amount > capacity_) {
            ++timeouts_;
            co_return Guard();
        }
        if (amount <= capacity_ - in_use_ && waiters_.empty()) {
            in_use_ += amount;
            peak_in_use_ = std::max(peak_in_use_, in_use_);
            co_return Guard(this, amount);
        }
        slot = std::make_shared<AsyncSlot<bool>>(executor_);
        waiters_.push_back(Waiter{amount, slot});
        ++waits_;
    }
    const TimePoint start = Now();
    executor_.ScheduleAt(deadline, [slot]() { slot->Complete(false); });
    StopCallbackGuard guard(stop, [slot]() { slot->Complete(false); });
    const bool granted = co_await *slot;
    const uint64_t waited = static_cast<uint64_t>(std::max<int64_t>(0, (Now() - start).count()));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        wait_ns_total_ += waited;
        wait_ns_max_ = std::max(wait_ns_max_, waited);
        if (!granted) {
            for (auto it = waiters_.begin(); it != waiters_.end(); ++it) {
                if (it->slot == slot) {
                    waiters_.erase(it);
                    break;
                }
            }
            ++timeouts_;
        } else {
            peak_in_use_ = std::max(peak_in_use_, in_use_);
        }
    }
    if (!granted) {
        DispatchWaiters();
        co_return Guard();
    }
    co_return Guard(this, amount);
}

void AsyncCapacityBudget::Release(uint64_t amount) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        in_use_ = amount > in_use_ ? 0 : in_use_ - amount;
    }
    DispatchWaiters();
}

void AsyncCapacityBudget::DispatchWaiters() {
    while (true) {
        Waiter next;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (waiters_.empty() || waiters_.front().amount > capacity_ - in_use_) {
                return;
            }
            next = waiters_.front();
            waiters_.pop_front();
            in_use_ += next.amount;
            peak_in_use_ = std::max(peak_in_use_, in_use_);
        }
        if (!next.slot->Complete(true)) {
            std::lock_guard<std::mutex> lock(mutex_);
            in_use_ -= next.amount;
        }
    }
}

uint64_t AsyncCapacityBudget::in_use() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return in_use_;
}
uint64_t AsyncCapacityBudget::peak_in_use() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_in_use_;
}
uint64_t AsyncCapacityBudget::waits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return waits_;
}
uint64_t AsyncCapacityBudget::timeouts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timeouts_;
}
uint64_t AsyncCapacityBudget::wait_ns_total() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return wait_ns_total_;
}
uint64_t AsyncCapacityBudget::wait_ns_max() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return wait_ns_max_;
}

Task<bool> SleepUntil(SwarmExecutor &executor, TimePoint until, StopToken stop) {
    if (stop.StopRequested()) {
        co_return false;
    }
    if (until <= Now()) {
        co_return true;
    }
    auto slot = std::make_shared<AsyncSlot<bool>>(executor);
    executor.ScheduleAt(until, [slot]() { slot->Complete(true); });
    StopCallbackGuard guard(stop, [slot]() { slot->Complete(false); });
    const bool elapsed = co_await *slot;
    co_return elapsed;
}

Task<bool> SleepFor(SwarmExecutor &executor, Duration duration, StopToken stop) {
    return SleepUntil(executor, Now() + duration, std::move(stop));
}

Task<> Reschedule(SwarmExecutor &executor) {
    auto slot = std::make_shared<AsyncSlot<bool>>(executor);
    executor.schedule([slot]() { slot->Complete(true); });
    co_await *slot;
    co_return;
}

} // namespace kvcm_swarm
