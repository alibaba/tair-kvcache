#include "kv_cache_manager/common/lock_free_thread_pool.h"

#include <autil/LambdaWorkItem.h>
#include <autil/LockFreeThreadPool.h>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {

struct ThreadCreationFailed {};

class CheckedLockFreeThreadPool : public autil::LockFreeThreadPool {
public:
    using autil::LockFreeThreadPool::LockFreeThreadPool;

protected:
    bool createThreads(const std::string &name) override {
        if (!autil::LockFreeThreadPool::createThreads(name)) {
            // autil::start holds _cond and calls stop on false, which locks
            // _cond again. Unwind that lock before cleaning up partial startup.
            throw ThreadCreationFailed{};
        }
        return true;
    }
};

} // namespace

LockFreeThreadPool::LockFreeThreadPool(size_t thread_num,
                                       size_t queue_size,
                                       const std::string &thread_name,
                                       bool stop_if_has_exception)
    : pool_(std::make_unique<CheckedLockFreeThreadPool>(
          thread_num, queue_size, nullptr, thread_name, stop_if_has_exception))
    , queue_size_(pool_->getQueueSize()) {}

LockFreeThreadPool::~LockFreeThreadPool() { stop(); }

bool LockFreeThreadPool::start(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::New) {
        return false;
    }
    state_ = State::Stopped;
    try {
        if (pool_->start(name)) {
            state_ = State::Running;
            return true;
        }
    } catch (const ThreadCreationFailed &) {
        pool_->stop();
    }
    KVCM_LOG_WARN("start lock free thread pool %s failed", name.c_str());
    return false;
}

void LockFreeThreadPool::stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ == State::Stopping) {
        finished_.wait(lock, [this]() { return state_ == State::Stopped; });
        return;
    }
    if (state_ == State::Stopped) {
        return;
    }
    state_ = State::Stopping;
    space_available_.notify_all();
    finished_.wait(lock, [this]() { return pending_tasks_ == 0; });
    lock.unlock();
    // autil::LockFreeThreadPool::stop ignores STOP_AFTER_QUEUE_EMPTY. Drain
    // accepted tasks ourselves before allowing it to stop the workers.
    pool_->stop();
    lock.lock();
    state_ = State::Stopped;
    finished_.notify_all();
}

void LockFreeThreadPool::waitFinish() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_.wait(lock, [this]() { return pending_tasks_ == 0; });
}

bool LockFreeThreadPool::isFull() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ != State::Running || queued_tasks_ >= queue_size_;
}

bool LockFreeThreadPool::enqueue(std::function<void()> task) { return schedule(std::move(task), true); }

bool LockFreeThreadPool::trySchedule(std::function<void()> task) { return schedule(std::move(task), false); }

bool LockFreeThreadPool::schedule(std::function<void()> task, bool blocked, bool sdk_compatible_admission) {
    if (!task) {
        return false;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    const auto has_capacity = [this, sdk_compatible_admission]() {
        // autil's async submission historically accepts Q + 1 queued tasks.
        // Keep that SDK boundary without relaxing enqueue/trySchedule admission.
        return sdk_compatible_admission ? queued_tasks_ <= queue_size_ : queued_tasks_ < queue_size_;
    };
    if (blocked) {
        space_available_.wait(lock, [this, &has_capacity]() { return state_ != State::Running || has_capacity(); });
    }
    if (state_ != State::Running || !has_capacity()) {
        return false;
    }
    auto tracked_task = [this, task = std::move(task)]() mutable {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --queued_tasks_;
            // Strict and SDK waiters have different capacity predicates: waking
            // only a strict waiter could leave an eligible SDK waiter asleep.
            space_available_.notify_all();
        }
        struct Completion {
            LockFreeThreadPool *owner;
            ~Completion() { owner->completeTask(); }
        } completion{this};
        try {
            task();
        } catch (const std::exception &e) {
            KVCM_LOG_WARN("lock free thread pool task failed: %s", e.what());
        } catch (...) {
            KVCM_LOG_WARN("lock free thread pool task failed with an unknown exception");
        }
        task = nullptr;
    };
    auto work_item = std::make_unique<autil::LambdaWorkItem>(std::move(tracked_task));
    ++queued_tasks_;
    ++pending_tasks_;
    try {
        // Admission is serialized; the worker returns capacity before running
        // the task. Never block inside autil while holding the accounting lock.
        if (pool_->pushWorkItem(work_item.get(), false) == autil::ThreadPool::ERROR_NONE) {
            work_item.release();
            return true;
        }
    } catch (...) {
        --queued_tasks_;
        --pending_tasks_;
        space_available_.notify_all();
        finished_.notify_all();
        throw;
    }
    --queued_tasks_;
    --pending_tasks_;
    space_available_.notify_all();
    finished_.notify_all();
    return false;
}

void LockFreeThreadPool::completeTask() {
    std::lock_guard<std::mutex> lock(mutex_);
    --pending_tasks_;
    finished_.notify_all();
}

} // namespace kv_cache_manager
