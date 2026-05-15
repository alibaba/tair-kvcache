#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/client_pool.h"
#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/meta/mpsc_write_queue.h"

namespace kv_cache_manager {

class MetaAsyncRedisBackend : public MetaStorageBackend {
public:
    MetaAsyncRedisBackend() = default;
    ~MetaAsyncRedisBackend() override;

    std::string GetStorageType() noexcept override;

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;

    // ----- Write (async enqueue) -----
    std::vector<ErrorCode> Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept override;
    std::vector<ErrorCode> DeleteFields(const KeyTypeVec &keys,
                                        const std::vector<std::vector<std::string>> &field_names_vec) noexcept override;

    // ----- Read (sync passthrough) -----
    std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                               const std::vector<std::string> &field_names,
                               FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                               const std::vector<std::vector<std::string>> &field_names_vec,
                               FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept override;
    std::vector<ErrorCode> ExistsFieldWithPrefix(const KeyTypeVec &keys,
                                                 const std::string &field_prefix,
                                                 std::vector<bool> &out_exists_vec) noexcept override;
    std::vector<ErrorCode>
    GetFieldNamesWithPrefix(const KeyTypeVec &keys,
                            const std::string &field_prefix,
                            std::vector<std::vector<std::string>> &out_field_names_vec) noexcept override;
    ErrorCode ListKeys(const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       KeyTypeVec &out_keys) noexcept override;
    ErrorCode RandomSample(const int64_t count, KeyTypeVec &out_keys) noexcept override;
    ErrorCode SampleReclaimKeys(const int64_t count, KeyTypeVec &out_keys) noexcept override;

    // ----- MetaData (sync passthrough) -----
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

    // ----- Sync -----
    bool Sync(const KeyTypeVec &keys) noexcept override;

    // ----- Metrics -----
    std::vector<int64_t> GetAsyncQueueSizes() const noexcept override;

private:
    std::vector<ErrorCode> EnqueueWriteOp(WriteOp op);
    bool WaitForQueueCapacity(int queue_id);
    void ConsumerLoop(int queue_id);
    void BatchFlush(int queue_id, std::vector<QueueItem> &items, int64_t total_keys);
    void DrainQueue(int queue_id);
    int GetQueueIndexForKey(KeyType key) const noexcept;

    using CmdArgs = std::vector<std::string>;
    void CompileWriteOp(const WriteOp &op, std::vector<CmdArgs> &cmds);

    std::vector<std::string> AppendPrefixToKeys(const KeyTypeVec &keys) const;
    bool StripPrefixInKeys(const std::vector<std::string> &keys_with_prefix, KeyTypeVec &out_keys) const;

    // virtual for test
    virtual std::shared_ptr<RedisClient> CreateRedisClient() const;

private:
    StandardUri storage_uri_;
    std::string instance_id_;
    std::string cache_key_prefix_;
    std::string metadata_key_;
    int64_t timeout_ms_ = 2000;

    // Async config
    int32_t queue_count_ = 8;
    int64_t max_batch_size_ = 51200;
    int64_t batch_wait_timeout_us_ = 1000;
    int64_t queue_max_size_ = 102400;
    int64_t enqueue_retry_interval_us_ = 100;
    int64_t enqueue_timeout_ms_ = 100;
    int64_t drain_timeout_ms_ = 5000;

    // Runtime state
    std::atomic<bool> is_running_{false};
    std::vector<std::unique_ptr<MpscWriteQueue>> queues_;
    std::vector<std::thread> consumer_threads_;

    // Per-consumer RedisClient (方案A: each consumer owns one)
    std::vector<std::shared_ptr<RedisClient>> consumer_clients_;

    // Read client pool (for concurrent read operations)
    std::shared_ptr<DynamicClientPool<RedisClient>> read_client_pool_;
};

} // namespace kv_cache_manager
