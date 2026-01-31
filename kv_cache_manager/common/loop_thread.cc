#include "kv_cache_manager/common/loop_thread.h"

#include <chrono>

namespace kv_cache_manager {

LoopThread::LoopThread() {}

LoopThread::~LoopThread() { Stop(); }

std::shared_ptr<LoopThread> LoopThread::CreateLoopThread(const LoopFunction &loop_function,
                                                         int64_t loop_interval_us,
                                                         const std::string &name,
                                                         bool strict_mode) {
    if (loop_interval_us < 0) {
        return nullptr;
    }

    std::shared_ptr<LoopThread> loop_thread(new LoopThread());
    loop_thread->name_ = name;
    loop_thread->strict_mode_ = strict_mode;
    loop_thread->loop_function_ = loop_function;
    loop_thread->loop_interval_us_ = loop_interval_us;
    loop_thread->is_running_ = true;

    loop_thread->thread_ = std::thread(&LoopThread::Loop, loop_thread.get());

    return loop_thread;
}

void LoopThread::Stop() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!is_running_) {
            return;
        }
        is_running_ = false;
        condition_variable_.notify_all();
    }

    if (thread_.joinable()) {
        thread_.join();
    }
}

void LoopThread::RunOnce() {
    std::unique_lock<std::mutex> lock(mutex_);
    next_run_time_us_ = -1;
    condition_variable_.notify_all();
}

void LoopThread::Loop() {
    while (true) {
        int64_t now_us;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!is_running_) {
                break;
            }

            now_us = GetMonotonicTimeUs();
            if (now_us < next_run_time_us_) {
                condition_variable_.wait_for(lock, std::chrono::microseconds(next_run_time_us_ - now_us));
                if (!is_running_) {
                    break;
                }
            }

            if (next_run_time_us_ == -1) {
                next_run_time_us_ = 0;
            }

            now_us = GetMonotonicTimeUs();
            if (now_us < next_run_time_us_) {
                continue;
            }
        }

        loop_function_();

        {
            if (strict_mode_) {
                now_us = GetMonotonicTimeUs();
            }

            std::unique_lock<std::mutex> lock(mutex_);
            if (next_run_time_us_ != -1) {
                next_run_time_us_ = now_us + loop_interval_us_;
            }
        }
    }
}

int64_t LoopThread::GetMonotonicTimeUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace kv_cache_manager