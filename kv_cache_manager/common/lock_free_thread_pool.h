#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>

namespace autil {
class LockFreeThreadPool;
}

namespace kv_cache_manager {

// Reusable autil workers with bounded admission and draining shutdown. Tasks
// must not stop/wait for this pool or submit blocking work to the same pool.
class LockFreeThreadPool {
public:
    LockFreeThreadPool(size_t thread_num,
                       size_t queue_size,
                       const std::string &thread_name,
                       bool stop_if_has_exception = false);
    ~LockFreeThreadPool();

    bool start(const std::string &name = "");
    // Reject new work, finish every accepted task, then join the workers.
    // A stopped pool cannot restart.
    void stop();
    void waitFinish();
    bool isFull() const;

    // Wait for queue capacity. False means no task was accepted or executed.
    // Stopping the pool wakes blocked producers. Task exceptions are contained.
    bool enqueue(std::function<void()> task);
    // Reject immediately if no queue capacity is available.
    bool trySchedule(std::function<void()> task);

    // Preserve SDK semantics: wait for queue capacity, then execute on the
    // caller if admission is closed. Exceptions are delivered by the future.
    template <typename Function>
    auto async(Function &&function) -> std::future<std::invoke_result_t<std::decay_t<Function> &>> {
        using Result = std::invoke_result_t<std::decay_t<Function> &>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Function>(function));
        auto future = task->get_future();
        if (!enqueue([task]() { (*task)(); })) {
            (*task)();
        }
        return future;
    }

private:
    enum class State {
        New,
        Running,
        Stopping,
        Stopped
    };

    bool schedule(std::function<void()> task, bool blocked);
    void completeTask();

    std::unique_ptr<autil::LockFreeThreadPool> pool_;
    const size_t queue_size_;
    mutable std::mutex mutex_;
    std::condition_variable space_available_;
    std::condition_variable finished_;
    State state_ = State::New;
    size_t queued_tasks_ = 0;
    size_t pending_tasks_ = 0;
};

} // namespace kv_cache_manager
