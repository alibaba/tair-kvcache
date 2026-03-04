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

ReadRecord OptimizerRunner::BuildReadRecord(const std::string &instance_id, int64_t timestamp_us) {
    ReadRecord record{};
    record.timestamp_us = timestamp_us;
    record.current_cache_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);

    auto indexer_map = indexer_manager_->GetAllOptIndexers();
    record.blocks_per_instance.resize(indexer_map.size(), 0);
    size_t idx = 0;
    for (const auto &pair : indexer_map) {
        record.blocks_per_instance[idx] = eviction_manager_->GetCurrentInstanceUsage(pair.first);
        idx++;
    }
    return record;
}

void OptimizerRunner::HandleGetLocation(const GetLocationSchemaTrace &trace) {
    std::string instance_id = trace.instance_id();
    auto indexer = indexer_manager_->GetOptIndexer(instance_id);
    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
        return;
    }

    std::vector<std::vector<int64_t>> external_hits;
    std::vector<std::vector<int64_t>> internal_hits;
    indexer->PrefixQuery(trace.keys(), trace.block_mask(), trace.timestamp_us(), external_hits, internal_hits);

    // ---- 构造 ReadRecord 并委托给 StatsCollector ----
    ReadRecord record = BuildReadRecord(instance_id, trace.timestamp_us());

    if (std::holds_alternative<BlockMaskVector>(trace.block_mask())) {
        const auto &mask_vector = std::get<BlockMaskVector>(trace.block_mask());
        record.internal_read_blocks = std::count(mask_vector.begin(), mask_vector.end(), true);
    } else if (std::holds_alternative<BlockMaskOffset>(trace.block_mask())) {
        record.internal_read_blocks = std::get<BlockMaskOffset>(trace.block_mask());
    }
    record.external_read_blocks = trace.keys().size() - record.internal_read_blocks;

    for (const auto &hit : external_hits) {
        record.external_hit_blocks += hit.size();
    }
    for (const auto &hit : internal_hits) {
        record.internal_hit_blocks += hit.size();
    }

    stats_collector_->OnReadComplete(instance_id, record);
}

void OptimizerRunner::HandleWriteCache(const WriteCacheSchemaTrace &trace) {
    std::string instance_id = trace.instance_id();
    auto indexer = indexer_manager_->GetOptIndexer(instance_id);
    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
        return;
    }

    auto inserted_keys = indexer->InsertOnly(trace.keys(), trace.timestamp_us());
    bool evicted = indexer_manager_->CheckAndEvict(instance_id, trace.timestamp_us());
    if (evicted) {
        KVCM_LOG_DEBUG("Eviction in %zu to instance_id: %s", trace.timestamp_us(), instance_id.c_str());
    }

    WriteRecord record{trace.timestamp_us(), trace.keys().size(), inserted_keys.size()};
    stats_collector_->OnWriteComplete(instance_id, record);
}

void OptimizerRunner::HandleDialogTurn(const DialogTurnSchemaTrace &trace) {
    std::string instance_id = trace.instance_id();
    auto indexer = indexer_manager_->GetOptIndexer(instance_id);
    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
        return;
    }

    std::vector<std::vector<int64_t>> hits;
    indexer->InsertWithQuery(trace.total_keys(), trace.timestamp_us(), hits);
    indexer_manager_->CheckAndEvict(instance_id, trace.timestamp_us());

    // ---- ReadRecord ----
    ReadRecord read_record = BuildReadRecord(instance_id, trace.timestamp_us());
    read_record.external_read_blocks = trace.keys().size();
    read_record.internal_read_blocks = 0;
    for (const auto &hit : hits) {
        read_record.external_hit_blocks += hit.size();
    }
    read_record.internal_hit_blocks = 0;

    stats_collector_->OnReadComplete(instance_id, read_record);

    // ---- WriteRecord ----
    // DialogTurn 的写入部分是 decode 新增的 block，全部视为新插入
    size_t decode_blocks = trace.total_keys().size() - trace.keys().size();
    WriteRecord write_record{trace.timestamp_us(), decode_blocks, decode_blocks};
    stats_collector_->OnWriteComplete(instance_id, write_record);
}
} // namespace kv_cache_manager
