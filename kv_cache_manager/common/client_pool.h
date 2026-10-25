#pragma once

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <stdio.h>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

template <typename ClientType>
class ClientPool {
public:
    using CreateClientCallback = std::function<std::shared_ptr<ClientType>()>;

    explicit ClientPool(CreateClientCallback cb) : cb_(cb) {}
    virtual ~ClientPool() = default;

    class PoolState {
    public:
        std::shared_ptr<ClientType> AcquireClient(int64_t lock_timeout_ms) {
            std::chrono::milliseconds timeout(lock_timeout_ms);
            {
                std::unique_lock lock(mtx_);
                if (!cv_.wait_for(lock, timeout, [this] { return !client_pool_.empty(); })) {
                    return nullptr;
                }
                std::shared_ptr<ClientType> client = std::move(client_pool_.front());
                client_pool_.pop();
                return client;
            }
            return nullptr;
        }

        void ReleaseClient(std::shared_ptr<ClientType> client, bool is_new = false) {
            if (client) {
                std::unique_lock lock(mtx_);
                if (is_new) {
                    client_ref_.push_back(client);
                }
                client_pool_.push(client);
                cv_.notify_one();
            }
        }

        // Track a new client without making it available to other borrowers.
        void RegisterClient(const std::shared_ptr<ClientType> &client) {
            std::unique_lock lock(mtx_);
            client_ref_.push_back(client);
        }

        size_t AllClientSize() const {
            std::shared_lock lock(mtx_);
            return client_ref_.size();
        }

        size_t FreeClientSize() const {
            std::shared_lock lock(mtx_);
            return client_pool_.size();
        }

    private:
        mutable std::shared_mutex mtx_;
        std::condition_variable_any cv_;
        std::queue<std::shared_ptr<ClientType>> client_pool_;
        std::vector<std::shared_ptr<ClientType>> client_ref_;
    };

    class ClientHandle {
    public:
        ClientHandle(std::shared_ptr<PoolState> pool_state, std::shared_ptr<ClientType> client)
            : pool_state_(pool_state), client_(std::move(client)) {}
        ClientHandle(const ClientHandle &other) = delete;
        ClientHandle(ClientHandle &&other)
            : pool_state_(std::move(other.pool_state_)), client_(std::move(other.client_)) {}

        ~ClientHandle() {
            if (client_ && pool_state_) {
                pool_state_->ReleaseClient(client_);
            }
        }

        ClientType *operator->() { return client_.get(); }
        ClientType &operator*() { return *client_; }
        explicit operator bool() const { return client_ != nullptr; }

    private:
        std::shared_ptr<PoolState> pool_state_;
        std::shared_ptr<ClientType> client_;
    };

    virtual bool Initialize() = 0;
    virtual ClientHandle AcquireClient(int64_t timeout_ms = 1000) = 0;

protected:
    bool InitializePoolStateWithSize(size_t pool_size) {
        pool_state_ = std::make_shared<PoolState>();
        for (size_t i = 0; i < pool_size; ++i) {
            auto client = cb_();
            if (client == nullptr) {
                return false;
            }
            pool_state_->ReleaseClient(client, true);
        }
        return true;
    }

    std::shared_ptr<PoolState> pool_state_;
    CreateClientCallback cb_;
};

template <typename ClientType>
class StaticClientPool : public ClientPool<ClientType> {
    using Base = ClientPool<ClientType>;
    static constexpr size_t kDefaultPoolSize = 4;

public:
    explicit StaticClientPool(typename Base::CreateClientCallback cb, size_t pool_size = kDefaultPoolSize)
        : ClientPool<ClientType>(cb), pool_size_(pool_size) {}

    bool Initialize() override { return this->InitializePoolStateWithSize(pool_size_); }
    typename Base::ClientHandle AcquireClient(int64_t timeout_ms = 1000) override {
        if (!this->pool_state_) {
            return typename Base::ClientHandle(nullptr, nullptr);
        }
        auto client = this->pool_state_->AcquireClient(timeout_ms);
        return typename Base::ClientHandle(this->pool_state_, std::move(client));
    }

private:
    size_t pool_size_;
};

template <typename ClientType>
class DynamicClientPool : public ClientPool<ClientType> {
    using Base = ClientPool<ClientType>;

public:
    explicit DynamicClientPool(typename Base::CreateClientCallback cb, int32_t min_pool_size, int32_t max_pool_size)
        : ClientPool<ClientType>(cb), min_pool_size_(min_pool_size), max_pool_size_(max_pool_size) {}

    bool Initialize() override { return this->InitializePoolStateWithSize(min_pool_size_); }
    typename Base::ClientHandle AcquireClient(int64_t timeout_ms = 1000) override {
        if (!this->pool_state_) {
            return typename Base::ClientHandle(nullptr, nullptr);
        }
        // Checking the free count before acquiring races with other borrowers:
        // the last idle client may disappear while capacity remains to grow.
        auto client = this->pool_state_->AcquireClient(0);
        if (client == nullptr) {
            if (static_cast<int32_t>(this->pool_state_->AllClientSize()) < max_pool_size_) {
                std::unique_lock lock(acq_mux_);
                // Reuse clients returned while waiting to grow the pool.
                client = this->pool_state_->AcquireClient(0);
                if (client == nullptr && static_cast<int32_t>(this->pool_state_->AllClientSize()) < max_pool_size_) {
                    client = this->cb_();
                    if (client != nullptr) {
                        // Keep ownership; publishing then reacquiring lets another
                        // borrower steal the new client and makes this caller wait.
                        this->pool_state_->RegisterClient(client);
                    } else {
                        KVCM_LOG_WARN("DynamicClientPool: failed to create new client, current pool size: %zu",
                                      this->pool_state_->AllClientSize());
                    }
                }
            }
            if (client == nullptr) {
                client = this->pool_state_->AcquireClient(timeout_ms);
            }
        }

        return typename Base::ClientHandle(this->pool_state_, std::move(client));
    }

private:
    int32_t min_pool_size_;
    int32_t max_pool_size_;
    std::mutex acq_mux_;
};

} // namespace kv_cache_manager
