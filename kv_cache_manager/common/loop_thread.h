#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace kv_cache_manager {

class LoopThread {
public:
    using LoopFunction = std::function<void()>;
    static std::shared_ptr<LoopThread> CreateLoopThread(const LoopFunction &loop_function,
                                                        int64_t loop_interval_us,
                                                        const std::string &name = "",
                                                        bool strict_mode = false);

    ~LoopThread();

    void Stop();
    void RunOnce();

private:
    LoopThread();

    void Loop();
    static int64_t GetMonotonicTimeUs();

    bool strict_mode_ = false;
    std::atomic<bool> is_running_{false};
    int64_t loop_interval_us_ = 0;
    int64_t next_run_time_us_ = 0;
    std::string name_;
    LoopFunction loop_function_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable condition_variable_;
};

} // namespace kv_cache_manager