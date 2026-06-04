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
    std::vector<ErrorCode> Put(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               const CacheLocationMapVector &locations,
                               const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept override;
    std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                           const KeyTypeVec &keys,
                                           const LocationIdsPerKey &location_ids) noexcept override;

    // ----- Read (sync passthrough) -----
    std::vector<ErrorCode> Get(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               CacheLocationMapVector &out_locations,
                               PropertyMapVector &out_properties) noexcept override;
    std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                        const KeyTypeVec &keys,
                                        CacheLocationMapVector &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept override;
    std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          LocationIdsPerKey &out_location_ids) noexcept override;
    std::vector<ErrorCode> GetProperties(RequestContext *request_context,
                                         const KeyTypeVec &keys,
                                         const std::vector<std::string> &field_names,
                                         PropertyMapVector &out_properties) noexcept override;
    std::vector<ErrorCode> Exists(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  std::vector<bool> &out_is_exist_vec) noexcept override;
    std::vector<ErrorCode> ExistsLocation(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          std::vector<bool> &out_exists) noexcept override;
    ErrorCode ListKeys(RequestContext *request_context,
                       const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       KeyTypeVec &out_keys) noexcept override;
    ErrorCode
    RandomSample(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept override;
    ErrorCode
    SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept override;

    // ----- MetaData (sync passthrough) -----
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

    // ----- Sync -----
    bool Sync(const KeyTypeVec &keys) noexcept override;

    // ----- Metrics -----
    AsyncWriteStats GetAsyncWriteStats() noexcept override;

private:
    std::vector<ErrorCode> EnqueueWriteOp(RequestContext *request_context, WriteOp op);
    bool WaitForQueueCapacity(int queue_id, int64_t incoming_key_count);
    void ConsumerLoop(int queue_id);
    void CleanupResources() noexcept;
    void BatchFlush(int queue_id, std::vector<QueueItem> &items, int64_t total_keys);
    void DrainQueue(int queue_id);
    int GetQueueIndexForKey(KeyType key) const noexcept;

    using CmdArgs = std::vector<std::string>;
    void CompileWriteOp(const WriteOp &op, std::vector<CmdArgs> &cmds);

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
    int64_t enqueue_timeout_ms_ = 100;
    int64_t sync_timeout_ms_ = 1000;
    int64_t drain_timeout_ms_ = 30000;

    // Runtime state
    std::atomic<bool> is_running_{false};
    std::vector<std::unique_ptr<MpscWriteQueue>> queues_;
    std::vector<std::thread> consumer_threads_;

    // Per-consumer RedisClient (each consumer owns one)
    std::vector<std::shared_ptr<RedisClient>> consumer_clients_;

    // Read client pool (for concurrent read operations)
    std::shared_ptr<DynamicClientPool<RedisClient>> read_client_pool_;

    // Consumer-thread write stats (atomics, reset on read via CAS)
    std::atomic<int64_t> stats_flush_key_count_{0};
    std::atomic<int64_t> stats_batch_flush_count_{0};
    std::atomic<int64_t> stats_batch_flush_time_us_{0};
    std::atomic<int64_t> stats_pipeline_error_count_{0};
};

} // namespace kv_cache_manager
