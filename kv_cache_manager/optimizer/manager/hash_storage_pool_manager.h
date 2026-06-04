#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_cache_manager/optimizer/analysis/stats_collector.h"
#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"
#include "kv_cache_manager/optimizer/config/insight_simulator_types.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/index/hash_table_index.h"

namespace kv_cache_manager {

struct HashStoragePoolReadResult {
    size_t hit_blocks = 0;
    std::vector<size_t> hit_indices;
};

struct HashStoragePoolReadRequest {
    HashStoragePoolReadRequest(std::string instance_id,
                               std::string trace_id,
                               int64_t timestamp,
                               const std::vector<int64_t> &block_ids,
                               std::vector<size_t> local_hit_indices,
                               int64_t input_tokens,
                               std::string query_type,
                               bool touch_local_hits)
        : instance_id(std::move(instance_id))
        , trace_id(std::move(trace_id))
        , timestamp(timestamp)
        , block_ids(&block_ids)
        , local_hit_indices(std::move(local_hit_indices))
        , input_tokens(input_tokens)
        , query_type(std::move(query_type))
        , touch_local_hits(touch_local_hits) {}

    std::string instance_id;
    std::string trace_id;
    int64_t timestamp = 0;
    const std::vector<int64_t> *block_ids = nullptr;
    std::vector<size_t> local_hit_indices;
    int64_t input_tokens = 0;
    std::string query_type;
    bool touch_local_hits = false;
};

class HashStoragePoolManager {
public:
    explicit HashStoragePoolManager(const HierarchicalStoragePoolConfig &config,
                                    bool enable_lifecycle_tracking = false);

    bool Init();

    HashStoragePoolReadResult Read(const HashStoragePoolReadRequest &request);

    WriteCacheRes WriteKeys(const std::string &instance_id,
                            const std::string &trace_id,
                            int64_t timestamp,
                            const std::vector<int64_t> &keys,
                            int64_t ttl_us,
                            bool touch_existing);

    void AnalyzeResults();

private:
    struct PoolInstance {
        std::string pool_id;
        std::string group_name;
        int32_t block_size = 0;
        int64_t bytes_per_token = 0;
        EvictionPolicyType eviction_policy_type = EvictionPolicyType::POLICY_UNSPECIFIED;
        EvictionPolicyParam eviction_policy_param;
        std::unique_ptr<HashTableIndex> index;
        int64_t default_ttl_ns = 0;
        bool ttl_disabled = true;
        bool ttl_refresh_on_read = true;
        std::shared_ptr<EvictionPolicy> eviction_policy;
    };

    struct PoolGroup {
        std::string group_name;
        size_t quota_bytes = 0;
        std::vector<std::string> instance_ids;
    };

    PoolInstance &GetInstanceOrThrow(const std::string &instance_id);
    const PoolInstance &GetInstanceOrThrow(const std::string &instance_id) const;
    int64_t ResolveTtlNs(const PoolInstance &instance, int64_t ttl_us) const;
    BlockEntry *FindLiveBlock(PoolInstance &instance, int64_t key);
    void InsertNewBlock(PoolInstance &instance, int64_t key, int64_t timestamp, int64_t ttl_ns);
    void TouchBlock(PoolInstance &instance,
                    BlockEntry *block,
                    int64_t timestamp,
                    bool count_read,
                    bool refresh_ttl,
                    bool count_write_touch);
    void
    RemoveBlock(PoolInstance &instance, BlockEntry *block, int64_t timestamp, bool use_logical_expire_time = false);
    void EvictExpiredForGroup(const std::string &group_name, int64_t timestamp);
    void EvictExpiredForInstance(PoolInstance &instance, int64_t timestamp);
    void CheckAndEvict(const std::string &instance_id, int64_t timestamp);
    void EvictInstance(PoolInstance &instance, size_t bytes_to_evict, bool precise, int64_t timestamp);
    void EvictGroupRough(PoolGroup &group, size_t bytes_to_evict, int64_t timestamp);
    size_t GroupUsageBytes(const PoolGroup &group) const;
    size_t GroupQuotaBytes(const PoolGroup &group) const;
    size_t BytesPerBlock(const PoolInstance &instance) const;
    ReadRecord BuildReadRecord(const PoolInstance &instance,
                               const std::string &trace_id,
                               int64_t timestamp,
                               size_t remote_read_blocks,
                               size_t remote_hit_blocks,
                               size_t input_tokens) const;
    WriteRecord BuildWriteRecord(const std::string &trace_id,
                                 int64_t timestamp,
                                 size_t write_blocks,
                                 size_t newly_inserted_blocks) const;

    const EvictionConfig &eviction_config() const { return config_.eviction_config(); }

    HierarchicalStoragePoolConfig config_;
    OptimizerConfig export_config_;
    bool enable_lifecycle_tracking_ = false;
    std::shared_ptr<StatsCollector> stats_collector_;
    std::unordered_map<std::string, PoolGroup> groups_;
    std::unordered_map<std::string, PoolInstance> instances_;
};

} // namespace kv_cache_manager
