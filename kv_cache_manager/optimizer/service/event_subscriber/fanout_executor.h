#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace kv_cache_manager {

// A reusable worker pool for one synchronous fanout batch at a time. Run()
// returns only after every task in the batch has finished, so callers can
// preserve ordering between consecutive source events while processing the
// targets of one event in parallel.
class FanoutExecutor {
public:
    using Task = std::function<void()>;

    explicit FanoutExecutor(std::size_t parallelism);
    ~FanoutExecutor();

    FanoutExecutor(const FanoutExecutor &) = delete;
    FanoutExecutor &operator=(const FanoutExecutor &) = delete;

    bool Run(std::vector<Task> tasks) noexcept;
    std::size_t parallelism() const noexcept { return parallelism_; }

private:
    struct Completion;
    struct WorkItem;

    void WorkerLoop();

    const std::size_t parallelism_;
    std::vector<std::thread> workers_;
    std::mutex run_mutex_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::list<WorkItem> queue_;
    bool stopping_ = false;
};

} // namespace kv_cache_manager
