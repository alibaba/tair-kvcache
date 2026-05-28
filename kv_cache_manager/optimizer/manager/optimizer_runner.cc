#include "kv_cache_manager/optimizer/manager/optimizer_runner.h"

#include <algorithm>
#include <chrono>
#include <variant>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/manager/optimizer_loader.h"

namespace kv_cache_manager {
namespace {
int64_t TtlUsToNs(int64_t ttl_us) { return ttl_us > 0 ? ttl_us * 1000 : ttl_us; }
} // namespace

void OptimizerRunner::Run(OptimizerConfig &config) {
    auto starting_time = std::chrono::high_resolution_clock::now();
    auto traces = OptimizerLoader::LoadTrace(config);
    auto ending_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ending_time - starting_time).count();
    KVCM_LOG_INFO(
        "Loaded %zu traces from file: %s in %ld ms", traces.size(), config.trace_file_path().c_str(), duration);

    starting_time = std::chrono::high_resolution_clock::now();
    RunTraces(traces);
    ending_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(ending_time - starting_time).count();
    KVCM_LOG_INFO("Playback traces in %ld ms", duration);
}

void OptimizerRunner::RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    for (const auto &trace : traces) {
        RunTrace(trace);
    }
}

void OptimizerRunner::RunTrace(std::shared_ptr<OptimizerSchemaTrace> trace) {
    if (!trace) {
        return;
    }

    if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
        if (get_trace->query_type() != "prefix_match") {
            KVCM_LOG_WARN("Unsupported query type: %s", get_trace->query_type().c_str());
            return;
        }
        HandleGetLocation(*get_trace);
        stats_collector_->UpdateTimestamp(get_trace->instance_id(), get_trace->timestamp_ns());
    } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
        HandleWriteCache(*write_trace);
        stats_collector_->UpdateTimestamp(write_trace->instance_id(), write_trace->timestamp_ns());
    } else {
        KVCM_LOG_WARN("Unknown trace type, skipping");
    }
}

std::shared_ptr<RadixTreeIndex> OptimizerRunner::GetIndexer(const std::string &instance_id) {
    auto indexer = indexer_manager_->GetOptIndexer(instance_id);
    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
    }
    return indexer;
}

void OptimizerRunner::SubmitReadRecord(const std::string &instance_id,
                                       const std::string &trace_id,
                                       const std::vector<int64_t> &keys,
                                       int64_t timestamp_ns,
                                       const QueryHit &query_hit,
                                       const std::shared_ptr<RadixTreeIndex> &indexer,
                                       size_t local_read_block_num,
                                       size_t remote_read_block_num) {
    ReadRecord record{};
    record.timestamp_ns = timestamp_ns;
    record.trace_id = trace_id;
    record.keys_ptr = &keys;
    record.current_cache_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);

    auto indexer_map = indexer_manager_->GetAllOptIndexers();
    record.blocks_per_instance.resize(indexer_map.size(), 0);
    size_t idx = 0;
    for (const auto &pair : indexer_map) {
        record.blocks_per_instance[idx] = eviction_manager_->GetCurrentInstanceUsage(pair.first);
        idx++;
    }

    record.remote_hit_blocks = query_hit.remote_hit_block_num;
    record.local_hit_blocks = query_hit.local_hit_block_num;
    record.per_tier_hit_blocks = query_hit.per_tier_hit_block_num;
    record.tier_names = indexer->GetTierNames();
    record.per_tier_blocks = eviction_manager_->GetCurrentInstanceUsagePerTier(instance_id);
    record.local_read_blocks = local_read_block_num;
    record.remote_read_blocks = remote_read_block_num;

    stats_collector_->OnReadComplete(instance_id, record);
}

void OptimizerRunner::HandleGetLocation(const GetLocationSchemaTrace &trace) {
    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return;
    }

    // 读请求前统一清理过期 block，并做节点清理（TTL 使用逻辑过期时刻记录）
    auto expired_evicted_blocks = indexer_manager_->EvictExpiredBeforeAccess(instance_id, trace.timestamp_ns());
    indexer_manager_->CleanEvictedBlocks(expired_evicted_blocks, trace.timestamp_ns(), true);

    bool refresh_ttl_on_read = true;
    auto it = instance_ttl_refresh_on_read_.find(instance_id);
    if (it != instance_ttl_refresh_on_read_.end()) {
        refresh_ttl_on_read = it->second;
    }

    QueryHit query_hit;
    indexer->PrefixQuery(trace.keys(), trace.block_mask(), trace.timestamp_ns(), &query_hit, refresh_ttl_on_read);

    size_t local_read_block_num = 0;
    if (std::holds_alternative<BlockMaskVector>(trace.block_mask())) {
        const auto &mask_vector = std::get<BlockMaskVector>(trace.block_mask());
        local_read_block_num = std::count(mask_vector.begin(), mask_vector.end(), true);
    } else if (std::holds_alternative<BlockMaskOffset>(trace.block_mask())) {
        local_read_block_num = std::get<BlockMaskOffset>(trace.block_mask());
    }
    size_t remote_read_block_num = trace.keys().size() - local_read_block_num;

    SubmitReadRecord(instance_id,
                     trace.trace_id(),
                     trace.keys(),
                     trace.timestamp_ns(),
                     query_hit,
                     indexer,
                     local_read_block_num,
                     remote_read_block_num);
}

void OptimizerRunner::HandleWriteCache(const WriteCacheSchemaTrace &trace) {
    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return;
    }

    // 写请求前统一清理过期 block，并做节点清理（TTL 使用逻辑过期时刻记录）
    auto expired_evicted_blocks = indexer_manager_->EvictExpiredBeforeAccess(instance_id, trace.timestamp_ns());
    indexer_manager_->CleanEvictedBlocks(expired_evicted_blocks, trace.timestamp_ns(), true);

    int64_t effective_ttl_ns = TtlUsToNs(trace.ttl_us());
    auto ttl_disabled_it = instance_group_ttl_disabled_.find(instance_id);
    if (ttl_disabled_it != instance_group_ttl_disabled_.end() && ttl_disabled_it->second) {
        effective_ttl_ns = -1;
    }

    auto result = indexer->InsertOnly(trace.keys(), trace.timestamp_ns(), effective_ttl_ns);
    auto capacity_evicted_blocks = indexer_manager_->CheckAndEvict(instance_id, trace.timestamp_ns());
    indexer_manager_->CleanEvictedBlocks(capacity_evicted_blocks, trace.timestamp_ns());
    bool evicted = !capacity_evicted_blocks.empty();
    if (evicted) {
        KVCM_LOG_DEBUG("Eviction at ts=%lld for instance_id: %s",
                       static_cast<long long>(trace.timestamp_ns()),
                       instance_id.c_str());
    }

    WriteRecord record;
    record.timestamp_ns = trace.timestamp_ns();
    record.write_blocks = trace.keys().size();
    record.newly_inserted_blocks = result.inserted_keys.size();
    record.trace_id = trace.trace_id();
    stats_collector_->OnWriteComplete(instance_id, record);
}
} // namespace kv_cache_manager
