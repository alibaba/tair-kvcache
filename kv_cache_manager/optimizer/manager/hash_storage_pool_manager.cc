#include "kv_cache_manager/optimizer/manager/hash_storage_pool_manager.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/analysis/tracker/block_lifecycle_tracker.h"
#include "kv_cache_manager/optimizer/analysis/tracker/hit_rate_tracker.h"
#include "kv_cache_manager/optimizer/eviction_policy/policy_factory.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {
namespace {

constexpr double kBytesPerGb = static_cast<double>(1LL << 30);
constexpr char kStoragePoolGroupName[] = "storage_pool";

int64_t TtlUsToNs(int64_t ttl_us) { return ttl_us > 0 ? ttl_us * 1000 : ttl_us; }

size_t GbToBytes(double gb) { return static_cast<size_t>(gb * kBytesPerGb); }

} // namespace

HashStoragePoolManager::HashStoragePoolManager(const HierarchicalStoragePoolConfig &config,
                                               bool enable_lifecycle_tracking)
    : config_(config), enable_lifecycle_tracking_(enable_lifecycle_tracking) {}

bool HashStoragePoolManager::Init() {
    if (config_.output_result_path().empty() || config_.storage_name().empty() || config_.capacity() <= 0.0) {
        KVCM_LOG_ERROR("Hash storage pool requires output_result_path, storage_name and positive capacity.");
        return false;
    }
    if (config_.eviction_config().eviction_mode() == EvictionMode::EVICTION_MODE_UNSPECIFIED) {
        KVCM_LOG_ERROR("Hash storage pool requires a valid eviction_mode.");
        return false;
    }
    if ((config_.eviction_config().eviction_mode() == EvictionMode::EVICTION_MODE_GROUP_ROUGH ||
         config_.eviction_config().eviction_mode() == EvictionMode::EVICTION_MODE_INSTANCE_ROUGH) &&
        config_.eviction_config().eviction_batch_size_per_instance() <= 0) {
        KVCM_LOG_ERROR("Hash storage pool rough eviction requires positive eviction_batch_size_per_instance.");
        return false;
    }
    if (config_.pools().empty()) {
        KVCM_LOG_ERROR("Hash storage pool has no pools.");
        return false;
    }
    export_config_.set_output_result_path(config_.output_result_path());

    stats_collector_ = std::make_shared<StatsCollector>();
    stats_collector_->EmplaceTracker<HitRateTracker>();
    if (enable_lifecycle_tracking_) {
        stats_collector_->EmplaceTracker<BlockLifecycleTracker>();
    }

    PoolGroup group;
    group.group_name = kStoragePoolGroupName;
    group.quota_bytes = GbToBytes(config_.capacity());
    for (const auto &pool_config : config_.pools()) {
        const std::string &pool_id = pool_config.pool_id();
        const auto &model = pool_config.model();
        if (pool_id.empty()) {
            KVCM_LOG_ERROR("Hash storage pool has an empty pool_id.");
            return false;
        }
        if (instances_.find(pool_id) != instances_.end()) {
            KVCM_LOG_ERROR("Duplicate hash storage pool id: %s", pool_id.c_str());
            return false;
        }
        if (model.block_size() <= 0 || model.bytes_per_token() <= 0) {
            KVCM_LOG_ERROR("Hash storage pool %s requires positive block_size and bytes_per_token.", pool_id.c_str());
            return false;
        }
        if (model.eviction_policy_type() == EvictionPolicyType::POLICY_LEAF_AWARE_LRU) {
            KVCM_LOG_ERROR("Hash storage pool does not support leaf-aware LRU because it has no radix leaves.");
            return false;
        }

        auto policy = EvictionPolicyFactory::CreatePolicy(model.eviction_policy_type(),
                                                          config_.storage_name(),
                                                          config_.eviction_config().eviction_batch_size_per_instance(),
                                                          model.eviction_policy_param());
        if (!policy) {
            KVCM_LOG_ERROR("Failed to create hash storage pool eviction policy for pool %s.", pool_id.c_str());
            return false;
        }

        PoolInstance instance;
        instance.pool_id = pool_id;
        instance.group_name = group.group_name;
        instance.block_size = model.block_size();
        instance.bytes_per_token = model.bytes_per_token();
        instance.eviction_policy_type = model.eviction_policy_type();
        instance.eviction_policy_param = model.eviction_policy_param();
        instance.index = std::make_unique<HashTableIndex>(config_.storage_name());
        instance.default_ttl_ns = config_.ttl_config().default_block_ttl_seconds() * 1000000000;
        instance.ttl_disabled = config_.ttl_config().default_block_ttl_seconds() == 0;
        instance.ttl_refresh_on_read = model.eviction_policy_type() == EvictionPolicyType::POLICY_TTL
                                           ? config_.ttl_config().refresh_on_read()
                                           : true;
        instance.eviction_policy = std::move(policy);
        group.instance_ids.push_back(pool_id);
        instances_.emplace(pool_id, std::move(instance));
    }
    groups_.emplace(group.group_name, std::move(group));

    return !instances_.empty();
}

HashStoragePoolReadResult HashStoragePoolManager::Read(const HashStoragePoolReadRequest &request) {
    if (request.input_tokens <= 0) {
        throw std::runtime_error("HashStoragePoolManager::Read requires positive input_tokens");
    }
    if (request.block_ids == nullptr) {
        throw std::runtime_error("HashStoragePoolManager::Read requires block_ids");
    }

    HashStoragePoolReadResult result;
    auto &instance = GetInstanceOrThrow(request.instance_id);
    stats_collector_->UpdateTimestamp(request.instance_id, request.timestamp);
    const auto &block_ids = *request.block_ids;
    if (block_ids.empty()) {
        stats_collector_->OnReadComplete(
            request.instance_id,
            BuildReadRecord(
                instance, request.trace_id, request.timestamp, 0, 0, static_cast<size_t>(request.input_tokens)));
        return result;
    }

    EvictExpiredForGroup(instance.group_name, request.timestamp);
    std::unordered_set<size_t> local_hit_set;
    for (const size_t idx : request.local_hit_indices) {
        if (idx < block_ids.size()) {
            local_hit_set.insert(idx);
        }
    }

    if (request.touch_local_hits) {
        for (const size_t idx : local_hit_set) {
            BlockEntry *block = FindLiveBlock(instance, block_ids[idx]);
            if (block != nullptr) {
                TouchBlock(instance, block, request.timestamp, false, false, false);
            }
        }
    }

    size_t remote_read_blocks = 0;
    if (IsPrefixMatchQueryType(request.query_type)) {
        for (size_t idx = 0; idx < block_ids.size(); ++idx) {
            if (local_hit_set.count(idx) > 0) {
                continue;
            }
            ++remote_read_blocks;
            BlockEntry *block = FindLiveBlock(instance, block_ids[idx]);
            if (block == nullptr) {
                break;
            }
            TouchBlock(instance, block, request.timestamp, true, instance.ttl_refresh_on_read, false);
            ++result.hit_blocks;
            result.hit_indices.push_back(idx);
        }
    } else if (IsBatchGetQueryType(request.query_type)) {
        remote_read_blocks = block_ids.size() - local_hit_set.size();
        for (size_t idx = 0; idx < block_ids.size(); ++idx) {
            if (local_hit_set.count(idx) > 0) {
                continue;
            }
            BlockEntry *block = FindLiveBlock(instance, block_ids[idx]);
            if (block == nullptr) {
                continue;
            }
            TouchBlock(instance, block, request.timestamp, true, instance.ttl_refresh_on_read, false);
            ++result.hit_blocks;
            result.hit_indices.push_back(idx);
        }
    } else {
        throw std::runtime_error("Unsupported hash storage pool query_type: " + request.query_type);
    }

    stats_collector_->OnReadComplete(request.instance_id,
                                     BuildReadRecord(instance,
                                                     request.trace_id,
                                                     request.timestamp,
                                                     remote_read_blocks,
                                                     result.hit_blocks,
                                                     static_cast<size_t>(request.input_tokens)));
    return result;
}

WriteCacheRes HashStoragePoolManager::WriteKeys(const std::string &instance_id,
                                                const std::string &trace_id,
                                                int64_t timestamp,
                                                const std::vector<int64_t> &keys,
                                                int64_t ttl_us,
                                                bool touch_existing) {
    auto &instance = GetInstanceOrThrow(instance_id);
    stats_collector_->UpdateTimestamp(instance_id, timestamp);
    EvictExpiredForGroup(instance.group_name, timestamp);

    const int64_t ttl_ns = ResolveTtlNs(instance, ttl_us);
    size_t newly_inserted = 0;
    for (const int64_t key : keys) {
        BlockEntry *block = FindLiveBlock(instance, key);
        if (block != nullptr) {
            if (touch_existing) {
                TouchBlock(instance, block, timestamp, false, false, true);
            }
            continue;
        }

        InsertNewBlock(instance, key, timestamp, ttl_ns);
        ++newly_inserted;
    }

    CheckAndEvict(instance_id, timestamp);
    stats_collector_->OnWriteComplete(instance_id, BuildWriteRecord(trace_id, timestamp, keys.size(), newly_inserted));

    WriteCacheRes res;
    res.trace_id = trace_id;
    res.kvcm_write_length = newly_inserted;
    res.kvcm_write_hit_length = keys.size() - newly_inserted;
    return res;
}

void HashStoragePoolManager::AnalyzeResults() {
    for (const auto &[instance_id, _] : instances_) {
        const int64_t final_timestamp = stats_collector_->GetLastTimestamp(instance_id);
        stats_collector_->FinalizeAll(instance_id, final_timestamp);
        stats_collector_->ExportAll(instance_id, export_config_);
        stats_collector_->ResetAll(instance_id);
    }
}

HashStoragePoolManager::PoolInstance &HashStoragePoolManager::GetInstanceOrThrow(const std::string &instance_id) {
    auto it = instances_.find(instance_id);
    if (it == instances_.end()) {
        throw std::runtime_error("Unknown hash storage pool instance: " + instance_id);
    }
    return it->second;
}

const HashStoragePoolManager::PoolInstance &
HashStoragePoolManager::GetInstanceOrThrow(const std::string &instance_id) const {
    auto it = instances_.find(instance_id);
    if (it == instances_.end()) {
        throw std::runtime_error("Unknown hash storage pool instance: " + instance_id);
    }
    return it->second;
}

int64_t HashStoragePoolManager::ResolveTtlNs(const PoolInstance &instance, int64_t ttl_us) const {
    if (instance.ttl_disabled) {
        return 0;
    }
    const int64_t explicit_ttl_ns = TtlUsToNs(ttl_us);
    if (explicit_ttl_ns > 0) {
        return explicit_ttl_ns;
    }
    if (explicit_ttl_ns < 0) {
        return 0;
    }
    return instance.default_ttl_ns > 0 ? instance.default_ttl_ns : 0;
}

BlockEntry *HashStoragePoolManager::FindLiveBlock(PoolInstance &instance, int64_t key) {
    if (!instance.index) {
        return nullptr;
    }
    return instance.index->Find(key);
}

void HashStoragePoolManager::InsertNewBlock(PoolInstance &instance, int64_t key, int64_t timestamp, int64_t ttl_ns) {
    BlockEntry *ptr = instance.index->Insert(key, timestamp, ttl_ns);
    instance.eviction_policy->OnBlockWritten(ptr);
    if (stats_collector_) {
        stats_collector_->OnBlockBirth(instance.pool_id, ptr, timestamp);
    }
}

void HashStoragePoolManager::TouchBlock(PoolInstance &instance,
                                        BlockEntry *block,
                                        int64_t timestamp,
                                        bool count_read,
                                        bool refresh_ttl,
                                        bool count_write_touch) {
    if (block == nullptr) {
        return;
    }
    instance.index->Touch(block, timestamp, count_read, count_write_touch);
    instance.eviction_policy->OnBlockAccessedWithOptions(block, timestamp, refresh_ttl);
}

void HashStoragePoolManager::RemoveBlock(PoolInstance &instance,
                                         BlockEntry *block,
                                         int64_t timestamp,
                                         bool use_logical_expire_time) {
    if (block == nullptr) {
        return;
    }
    int64_t eviction_timestamp = timestamp;
    if (use_logical_expire_time && block->ttl_ns > 0 && block->ttl_anchor_time >= 0) {
        eviction_timestamp = std::min(timestamp, block->ttl_anchor_time + block->ttl_ns);
    }
    if (stats_collector_) {
        stats_collector_->OnBlockEviction(instance.pool_id, block, eviction_timestamp);
    }
    instance.index->Remove(block);
}

void HashStoragePoolManager::EvictExpiredForGroup(const std::string &group_name, int64_t timestamp) {
    auto group_it = groups_.find(group_name);
    if (group_it == groups_.end()) {
        throw std::runtime_error("Unknown hash storage pool group: " + group_name);
    }
    for (const auto &instance_id : group_it->second.instance_ids) {
        EvictExpiredForInstance(GetInstanceOrThrow(instance_id), timestamp);
    }
}

void HashStoragePoolManager::EvictExpiredForInstance(PoolInstance &instance, int64_t timestamp) {
    instance.eviction_policy->AdvanceClock(timestamp);
    auto evicted = instance.eviction_policy->EvictExpired();
    for (auto *block : evicted) {
        RemoveBlock(instance, block, timestamp, true);
    }
}

void HashStoragePoolManager::CheckAndEvict(const std::string &instance_id, int64_t timestamp) {
    auto &instance = GetInstanceOrThrow(instance_id);
    auto group_it = groups_.find(instance.group_name);
    if (group_it == groups_.end()) {
        throw std::runtime_error("Unknown hash storage pool group: " + instance.group_name);
    }
    PoolGroup &group = group_it->second;
    const size_t quota = GroupQuotaBytes(group);
    if (quota == std::numeric_limits<size_t>::max()) {
        return;
    }
    const size_t usage = GroupUsageBytes(group);
    if (usage <= quota) {
        return;
    }

    const size_t excess = usage - quota;
    switch (config_.eviction_config().eviction_mode()) {
    case EvictionMode::EVICTION_MODE_GROUP_ROUGH:
        EvictGroupRough(group, excess, timestamp);
        break;
    case EvictionMode::EVICTION_MODE_INSTANCE_ROUGH:
        EvictInstance(instance, excess, false, timestamp);
        break;
    case EvictionMode::EVICTION_MODE_INSTANCE_PRECISE:
        EvictInstance(instance, excess, true, timestamp);
        break;
    default:
        break;
    }
}

void HashStoragePoolManager::EvictInstance(PoolInstance &instance,
                                           size_t bytes_to_evict,
                                           bool precise,
                                           int64_t timestamp) {
    const size_t bytes_per_block = BytesPerBlock(instance);
    size_t evicted_bytes = 0;
    while (evicted_bytes < bytes_to_evict) {
        size_t evict_count = static_cast<size_t>(config_.eviction_config().eviction_batch_size_per_instance());
        if (precise) {
            const size_t remaining = bytes_to_evict - evicted_bytes;
            evict_count = (remaining + bytes_per_block - 1) / bytes_per_block;
        }
        if (evict_count == 0 || !instance.eviction_policy->NeedCapacityEviction()) {
            return;
        }
        auto evicted = instance.eviction_policy->EvictBlocks(evict_count);
        if (evicted.empty()) {
            return;
        }
        evicted_bytes += evicted.size() * bytes_per_block;
        for (auto *block : evicted) {
            RemoveBlock(instance, block, timestamp, false);
        }
    }
}

void HashStoragePoolManager::EvictGroupRough(PoolGroup &group, size_t bytes_to_evict, int64_t timestamp) {
    size_t evicted_bytes = 0;
    while (evicted_bytes < bytes_to_evict) {
        bool evicted_any = false;
        for (const auto &instance_id : group.instance_ids) {
            auto &instance = GetInstanceOrThrow(instance_id);
            if (!instance.eviction_policy->NeedCapacityEviction()) {
                continue;
            }
            auto evicted =
                instance.eviction_policy->EvictBlocks(config_.eviction_config().eviction_batch_size_per_instance());
            if (evicted.empty()) {
                continue;
            }
            evicted_any = true;
            evicted_bytes += evicted.size() * BytesPerBlock(instance);
            for (auto *block : evicted) {
                RemoveBlock(instance, block, timestamp, false);
            }
            if (evicted_bytes >= bytes_to_evict) {
                break;
            }
        }
        if (!evicted_any) {
            return;
        }
    }
}

size_t HashStoragePoolManager::GroupUsageBytes(const PoolGroup &group) const {
    size_t usage = 0;
    for (const auto &instance_id : group.instance_ids) {
        const auto &instance = GetInstanceOrThrow(instance_id);
        usage += instance.index->Size() * BytesPerBlock(instance);
    }
    return usage;
}

size_t HashStoragePoolManager::GroupQuotaBytes(const PoolGroup &group) const { return group.quota_bytes; }

size_t HashStoragePoolManager::BytesPerBlock(const PoolInstance &instance) const {
    return static_cast<size_t>(instance.block_size) * static_cast<size_t>(instance.bytes_per_token);
}

ReadRecord HashStoragePoolManager::BuildReadRecord(const PoolInstance &instance,
                                                   const std::string &trace_id,
                                                   int64_t timestamp,
                                                   size_t remote_read_blocks,
                                                   size_t remote_hit_blocks,
                                                   size_t input_tokens) const {
    ReadRecord record{};
    record.timestamp_ns = timestamp;
    record.trace_id = trace_id;
    record.current_cache_blocks = instance.index->Size();
    record.remote_read_blocks = remote_read_blocks;
    record.remote_hit_blocks = remote_hit_blocks;
    record.local_read_blocks = 0;
    record.local_hit_blocks = 0;
    record.input_tokens = input_tokens;
    record.block_size_tokens = static_cast<size_t>(instance.block_size);
    record.tier_names = {instance.index->tier_name()};
    record.per_tier_hit_blocks = {remote_hit_blocks};
    record.per_tier_blocks = {instance.index->Size()};
    record.blocks_per_instance.reserve(instances_.size());
    for (const auto &[_, pool_instance] : instances_) {
        record.blocks_per_instance.push_back(pool_instance.index->Size());
    }
    return record;
}

WriteRecord HashStoragePoolManager::BuildWriteRecord(const std::string &trace_id,
                                                     int64_t timestamp,
                                                     size_t write_blocks,
                                                     size_t newly_inserted_blocks) const {
    WriteRecord record;
    record.timestamp_ns = timestamp;
    record.trace_id = trace_id;
    record.write_blocks = write_blocks;
    record.newly_inserted_blocks = newly_inserted_blocks;
    return record;
}

} // namespace kv_cache_manager
