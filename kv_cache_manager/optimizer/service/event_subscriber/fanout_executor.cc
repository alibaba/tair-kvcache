#include "kv_cache_manager/optimizer/service/event_subscriber/fanout_executor.h"

#include <exception>
#include <stdexcept>
#include <utility>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

struct FanoutExecutor::Completion {
    explicit Completion(std::size_t task_count) : remaining(task_count) {}

    void Finish(bool task_succeeded) {
        std::lock_guard<std::mutex> lock(mutex);
        succeeded = succeeded && task_succeeded;
        if (--remaining == 0) {
            cv.notify_one();
        }
    }

    bool Wait() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return remaining == 0; });
        return succeeded;
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::size_t remaining;
    bool succeeded = true;
};

struct FanoutExecutor::WorkItem {
    Task task;
    std::shared_ptr<Completion> completion;
};

FanoutExecutor::FanoutExecutor(std::size_t parallelism) : parallelism_(parallelism) {
    if (parallelism_ == 0) {
        throw std::invalid_argument("fanout parallelism must be greater than zero");
    }

    workers_.reserve(parallelism_);
    try {
        for (std::size_t i = 0; i < parallelism_; ++i) {
            workers_.emplace_back(&FanoutExecutor::WorkerLoop, this);
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        throw;
    }
}

FanoutExecutor::~FanoutExecutor() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stopping_ = true;
    }
    queue_cv_.notify_all();
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

bool FanoutExecutor::Run(std::vector<Task> tasks) noexcept {
    if (tasks.empty()) {
        return true;
    }
    if (tasks.size() > parallelism_) {
        KVCM_LOG_ERROR("FanoutExecutor: task count[%zu] exceeds parallelism[%zu]", tasks.size(), parallelism_);
        return false;
    }

    std::lock_guard<std::mutex> run_lock(run_mutex_);
    try {
        auto completion = std::make_shared<Completion>(tasks.size());
        std::list<WorkItem> pending;
        for (auto &task : tasks) {
            pending.push_back({std::move(task), completion});
        }

        {
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            if (stopping_) {
                return false;
            }
            queue_.splice(queue_.end(), pending);
        }
        queue_cv_.notify_all();
        return completion->Wait();
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("FanoutExecutor: failed to submit batch: %s", e.what());
    } catch (...) { KVCM_LOG_ERROR("FanoutExecutor: failed to submit batch with unknown exception"); }
    return false;
}

void FanoutExecutor::WorkerLoop() {
    while (true) {
        WorkItem item;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) {
                return;
            }
            item = std::move(queue_.front());
            queue_.pop_front();
        }

        bool succeeded = true;
        try {
            item.task();
        } catch (const std::exception &e) {
            succeeded = false;
            KVCM_LOG_ERROR("FanoutExecutor: task failed: %s", e.what());
        } catch (...) {
            succeeded = false;
            KVCM_LOG_ERROR("FanoutExecutor: task failed with unknown exception");
        }
        item.completion->Finish(succeeded);
    }
}

} // namespace kv_cache_manager
