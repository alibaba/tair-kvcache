#include "kv_cache_manager/optimizer/manager/optimizer_runner.h"

#include <algorithm>
#include <chrono>
#include <variant>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/manager/optimizer_loader.h"

namespace kv_cache_manager {

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

    if (auto turn_trace = std::dynamic_pointer_cast<DialogTurnSchemaTrace>(trace)) {
        if (turn_trace->query_type() != "prefix_match") {
            KVCM_LOG_WARN("Unsupported query type: %s", turn_trace->query_type().c_str());
            return;
        }
        HandleDialogTurn(*turn_trace);
        stats_collector_->UpdateTimestamp(turn_trace->instance_id(), turn_trace->timestamp_us());
    } else if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
        if (get_trace->query_type() != "prefix_match") {
            KVCM_LOG_WARN("Unsupported query type: %s", get_trace->query_type().c_str());
            return;
        }
        HandleGetLocation(*get_trace);
        stats_collector_->UpdateTimestamp(get_trace->instance_id(), get_trace->timestamp_us());
    } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
        HandleWriteCache(*write_trace);
        stats_collector_->UpdateTimestamp(write_trace->instance_id(), write_trace->timestamp_us());
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
                                       int64_t timestamp_us,
                                       const QueryHit &query_hit,
                                       const std::shared_ptr<RadixTreeIndex> &indexer,
                                       size_t local_read_block_num,
                                       size_t remote_read_block_num) {
    ReadRecord record{};
    record.timestamp_us = timestamp_us;
    record.current_cache_block_num = eviction_manager_->GetCurrentInstanceUsage(instance_id);

    auto indexer_map = indexer_manager_->GetAllOptIndexers();
    record.block_num_per_instance.resize(indexer_map.size(), 0);
    size_t idx = 0;
    for (const auto &pair : indexer_map) {
        record.block_num_per_instance[idx] = eviction_manager_->GetCurrentInstanceUsage(pair.first);
        idx++;
    }

    record.remote_hit_block_num = query_hit.remote_hit_block_num;
    record.local_hit_block_num = query_hit.local_hit_block_num;
    record.per_tier_hit_block_num = query_hit.per_tier_hit_block_num;
    record.tier_names = indexer->GetTierNames();
    record.per_tier_block_num = eviction_manager_->GetCurrentInstanceUsagePerTier(instance_id);
    record.local_read_block_num = local_read_block_num;
    record.remote_read_block_num = remote_read_block_num;

    stats_collector_->OnReadComplete(instance_id, record);
}

void OptimizerRunner::HandleGetLocation(const GetLocationSchemaTrace &trace) {
    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return;
    }

    QueryHit query_hit;
    indexer->PrefixQuery(trace.keys(), trace.block_mask(), trace.timestamp_us(), &query_hit);

    size_t local_read_block_num = 0;
    if (std::holds_alternative<BlockMaskVector>(trace.block_mask())) {
        const auto &mask_vector = std::get<BlockMaskVector>(trace.block_mask());
        local_read_block_num = std::count(mask_vector.begin(), mask_vector.end(), true);
    } else if (std::holds_alternative<BlockMaskOffset>(trace.block_mask())) {
        local_read_block_num = std::get<BlockMaskOffset>(trace.block_mask());
    }
    size_t remote_read_block_num = trace.keys().size() - local_read_block_num;

    SubmitReadRecord(
        instance_id, trace.timestamp_us(), query_hit, indexer, local_read_block_num, remote_read_block_num);
}

void OptimizerRunner::HandleWriteCache(const WriteCacheSchemaTrace &trace) {
    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return;
    }

    auto result = indexer->InsertOnly(trace.keys(), trace.timestamp_us());
    bool evicted = indexer_manager_->CheckAndEvict(instance_id, trace.timestamp_us());
    if (evicted) {
        KVCM_LOG_DEBUG("Eviction in %zu to instance_id: %s", trace.timestamp_us(), instance_id.c_str());
    }

    WriteRecord record;
    record.timestamp_us = trace.timestamp_us();
    record.write_block_num = trace.keys().size();
    record.newly_inserted_block_num = result.inserted_keys.size();
    record.trace_id = trace.trace_id();
    stats_collector_->OnWriteComplete(instance_id, record);
}

void OptimizerRunner::HandleDialogTurn(const DialogTurnSchemaTrace &trace) {
    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return;
    }

    QueryHit query_hit;
    indexer->InsertWithQuery(trace.total_keys(), trace.timestamp_us(), &query_hit);
    indexer_manager_->CheckAndEvict(instance_id, trace.timestamp_us());

    SubmitReadRecord(instance_id, trace.timestamp_us(), query_hit, indexer, 0, trace.keys().size());

    size_t decode_block_num = trace.total_keys().size() - trace.keys().size();
    WriteRecord write_record{trace.timestamp_us(), decode_block_num, decode_block_num};
    stats_collector_->OnWriteComplete(instance_id, write_record);
}
} // namespace kv_cache_manager