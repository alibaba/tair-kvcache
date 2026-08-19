#pragma once

#include <coroutine>
#include <mutex>
#include <utility>

#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"

namespace kv_cache_manager::async_rpc {

template <typename T>
class AsyncSlot {
public:
    explicit AsyncSlot(const ContinuationScheduler &scheduler) : scheduler_(scheduler) {}

    AsyncSlot(const AsyncSlot &) = delete;
    AsyncSlot &operator=(const AsyncSlot &) = delete;

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
            scheduler_.schedule([handle]() mutable { handle.resume(); });
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

        Awaiter coAwait(async_simple::Executor *) { return *this; }
    };

    Awaiter operator co_await() { return Awaiter{this}; }

private:
    const ContinuationScheduler &scheduler_;
    mutable std::mutex mutex_;
    bool ready_ = false;
    T value_{};
    std::coroutine_handle<> handle_ = nullptr;
};

} // namespace kv_cache_manager::async_rpc
