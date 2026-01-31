#pragma once

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <queue>

#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {

/*
[instance_id]:cache_key_set ---> [key1, key2, key3]

[instance_id]:cache_key:[key1] --- > {{f1, v1-1}, {f2, v1-2}}
[instance_id]:cache_key:[key2] --- > {{f1, v2-1}, {f2, v2-2}}
 */
class MetaRedisBackend : public MetaStorageBackend {
public:
    MetaRedisBackend() = default;
    ~MetaRedisBackend() override;

    std::string GetStorageType() noexcept override;

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;

    // write
    std::vector<ErrorCode> Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> IncrFields(const KeyTypeVec &keys,
                                      const std::map<std::string, int64_t> &field_amounts) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept override;

    // read
    std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                               const std::vector<std::string> &field_names,
                               FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept override;
    ErrorCode ListKeys(const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept override;

    // meta data
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

private:
    class PoolState {
    public:
        std::unique_ptr<RedisClient> AcquireClient(int64_t lock_timeout_ms) {
            std::chrono::milliseconds timeout(lock_timeout_ms);
            {
                std::unique_lock<std::mutex> lock(mtx_);
                if (!cv_.wait_for(lock, timeout, [this] { return !client_pool_.empty(); })) {
                    return nullptr;
                }
                assert(!client_pool_.empty());
                std::unique_ptr<RedisClient> client = std::move(client_pool_.front());
                client_pool_.pop();
                return client;
            }
            return nullptr;
        }
        void ReleaseClient(std::unique_ptr<RedisClient> client) {
            if (client) {
                std::lock_guard<std::mutex> lock(mtx_);
                client_pool_.push(std::move(client));
                cv_.notify_one();
            }
        }

    private:
        std::mutex mtx_;
        std::condition_variable cv_;
        std::queue<std::unique_ptr<RedisClient>> client_pool_;
    };

    class ClientHandle {
    public:
        ClientHandle(std::weak_ptr<PoolState> pool_state, std::unique_ptr<RedisClient> client)
            : pool_state_(std::move(pool_state)), client_(std::move(client)) {}
        ~ClientHandle() {
            if (client_) {
                if (auto pool_state = pool_state_.lock()) {
                    pool_state->ReleaseClient(std::move(client_));
                } else {
                    client_.reset();
                }
            }
        }
        RedisClient *operator->() { return client_.get(); }
        RedisClient &operator*() { return *client_; }
        explicit operator bool() const { return client_ != nullptr; }

    private:
        std::weak_ptr<PoolState> pool_state_;
        std::unique_ptr<RedisClient> client_;
    };

private:
    std::shared_ptr<PoolState> GetPoolState() const;
    void SetPoolState(const std::shared_ptr<PoolState> &pool_state);
    ClientHandle AcquireClientFromPool();
    std::vector<std::string> AppendPrefixToKeys(const KeyTypeVec &keys) const;
    bool StripPrefixInKeys(const std::vector<std::string> &keys_with_prefix, std::vector<KeyType> &out_keys) const;

    // virtual for test
    virtual std::unique_ptr<RedisClient> CreateRedisClient() const;

private:
    mutable std::mutex pool_state_mtx_;
    std::shared_ptr<PoolState> pool_state_;
    StandardUri storage_uri_;
    std::string instance_id_;
    std::string cache_key_prefix_;
    std::string metadata_key_;
    int64_t timeout_ms_ = 1000;
};
} // namespace kv_cache_manager
