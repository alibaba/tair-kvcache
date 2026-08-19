// Cooperative cancellation shared by the asynchronous RPC client and its
// callers.
//
// A CancellationToken never blocks: waiters register a callback that is invoked once,
// on the thread that requests the stop, so a pending timer or permit wait can
// be resumed promptly instead of being polled.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace kv_cache_manager::async_rpc {

class CancellationState {
public:
    using Callback = std::function<void()>;

    bool StopRequested() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopped_;
    }

    void RequestStop() {
        std::map<uint64_t, Callback> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) {
                return;
            }
            stopped_ = true;
            pending.swap(callbacks_);
        }
        for (auto &entry : pending) {
            entry.second();
        }
    }

    // Returns 0 when the stop was already requested; the callback then runs
    // inline and there is nothing to unregister.
    uint64_t Register(Callback callback) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!stopped_) {
                const uint64_t id = next_id_++;
                callbacks_.emplace(id, std::move(callback));
                return id;
            }
        }
        callback();
        return 0;
    }

    void Unregister(uint64_t id) {
        if (id == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_.erase(id);
    }

private:
    mutable std::mutex mutex_;
    bool stopped_ = false;
    uint64_t next_id_ = 1;
    std::map<uint64_t, Callback> callbacks_;
};

class CancellationToken {
public:
    CancellationToken() = default;
    explicit CancellationToken(std::shared_ptr<CancellationState> state) : state_(std::move(state)) {}

    bool StopPossible() const { return state_ != nullptr; }
    bool StopRequested() const { return state_ != nullptr && state_->StopRequested(); }

    uint64_t Register(CancellationState::Callback callback) const {
        if (state_ == nullptr) {
            return 0;
        }
        return state_->Register(std::move(callback));
    }

    void Unregister(uint64_t id) const {
        if (state_ != nullptr) {
            state_->Unregister(id);
        }
    }

private:
    std::shared_ptr<CancellationState> state_;
};

class CancellationSource {
public:
    CancellationSource() : state_(std::make_shared<CancellationState>()) {}

    CancellationToken Token() const { return CancellationToken(state_); }
    void RequestStop() { state_->RequestStop(); }
    bool StopRequested() const { return state_->StopRequested(); }

private:
    std::shared_ptr<CancellationState> state_;
};

// RAII guard so a completed wait always detaches its stop callback.
class CancellationCallbackGuard {
public:
    CancellationCallbackGuard(CancellationToken token, CancellationState::Callback callback)
        : token_(std::move(token)), id_(token_.Register(std::move(callback))) {}
    ~CancellationCallbackGuard() { token_.Unregister(id_); }

    CancellationCallbackGuard(const CancellationCallbackGuard &) = delete;
    CancellationCallbackGuard &operator=(const CancellationCallbackGuard &) = delete;

private:
    CancellationToken token_;
    uint64_t id_ = 0;
};

} // namespace kv_cache_manager::async_rpc
