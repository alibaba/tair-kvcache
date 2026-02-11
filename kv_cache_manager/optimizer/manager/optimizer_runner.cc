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

    // 自动识别trace类型并处理
    // 注意: 必须先检查DialogTurnSchemaTrace(最具体的子类),再检查GetLocationSchemaTrace(父类)
    if (auto turn_trace = std::dynamic_pointer_cast<DialogTurnSchemaTrace>(trace)) {
        // DialogTurn: 用于推理引擎模拟器,Optimizer内部会拆分为读写操作
        if (turn_trace->query_type() != "prefix_match") {
            KVCM_LOG_WARN("Unsupported query type: %s", turn_trace->query_type().c_str());
            return;
        }
        HandleDialogTurn(*turn_trace);
        result_map_[turn_trace->instance_id()]->counters.total_requests += 1;
    } else if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
        // Get: 读操作(prefill阶段)
        if (get_trace->query_type() != "prefix_match") {
            KVCM_LOG_WARN("Unsupported query type: %s", get_trace->query_type().c_str());
            return;
        }
        HandleGetLocation(*get_trace);
        result_map_[get_trace->instance_id()]->counters.total_requests += 1;
    } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
        // Write: 写操作(decode阶段)
        HandleWriteCache(*write_trace);
        result_map_[write_trace->instance_id()]->counters.total_requests += 1;
    } else {
        KVCM_LOG_WARN("Unknown trace type, skipping");
    }
}

void OptimizerRunner::HandleGetLocation(const GetLocationSchemaTrace &trace) {
    // 模拟从缓存中获取位置
    std::string instance_id = trace.instance_id();
    auto result_ptr = result_map_.find(instance_id);
    if (result_ptr == result_map_.end()) {
        KVCM_LOG_ERROR("Instance ID not found in result map: %s", instance_id.c_str());
        return;
    }
    auto &result = result_ptr->second;

    auto indexer = indexer_manager_->GetOptIndexer(instance_id);

    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
        return;
    }
    std::vector<std::vector<int64_t>> external_hits;
    std::vector<std::vector<int64_t>> internal_hits;
    indexer->PrefixQuery(trace.keys(),
                         trace.block_mask(),
                         trace.timestamp_us(),
                         /*external_hits=*/external_hits,
                         /*internal_hits=*/internal_hits);

    // TODO 详细统计信息获取
    ReadRecord read_record;
    read_record.timestamp_us = trace.timestamp_us();

    if (std::holds_alternative<BlockMaskVector>(trace.block_mask())) {
        const auto &mask_vector = std::get<BlockMaskVector>(trace.block_mask());
        read_record.internal_read_blocks = std::count(mask_vector.begin(), mask_vector.end(), true);
    } else if (std::holds_alternative<BlockMaskOffset>(trace.block_mask())) {
        const auto &mask_offset = std::get<BlockMaskOffset>(trace.block_mask());
        read_record.internal_read_blocks = mask_offset;
    }
    read_record.external_read_blocks = trace.keys().size() - read_record.internal_read_blocks;

    read_record.external_hit_blocks = 0;
    for (const auto &external_hit : external_hits) {
        read_record.external_hit_blocks += external_hit.size();
    }
    read_record.internal_hit_blocks = 0;
    for (const auto &internal_hit : internal_hits) {
        read_record.internal_hit_blocks += internal_hit.size();
    }

    read_record.current_cache_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);

    auto indexer_map = indexer_manager_->GetAllOptIndexers();
    read_record.blocks_per_instance.resize(indexer_map.size(), 0);
    size_t instance_idx = 0;
    for (const auto &pair : indexer_map) {
        read_record.blocks_per_instance[instance_idx] = eviction_manager_->GetCurrentInstanceUsage(pair.first);
        instance_idx++;
    }

    result->read_results.push_back(read_record);

    result->counters.total_read_blocks += trace.keys().size();
    result->counters.total_hit_blocks += read_record.external_hit_blocks + read_record.internal_hit_blocks;

    result->counters.total_read_requests += 1;
}

void OptimizerRunner::HandleWriteCache(const WriteCacheSchemaTrace &trace) {
    // 模拟将新的blocks写入缓存
    std::string instance_id = trace.instance_id();
    auto result_ptr = result_map_.find(instance_id);
    if (result_ptr == result_map_.end()) {
        KVCM_LOG_ERROR("Instance ID not found in result map: %s", instance_id.c_str());
        return;
    }
    auto &result = result_ptr->second;

    auto indexer = indexer_manager_->GetOptIndexer(instance_id);

    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
        return;
    }
    auto insert_block_keys = indexer->InsertOnly(trace.keys(), trace.timestamp_us());

    // 写入后进行驱逐
    bool evicted = indexer_manager_->CheckAndEvict(instance_id);
    if (evicted) {
        KVCM_LOG_DEBUG("Eviction in %zu to instance_id: %s", trace.timestamp_us(), instance_id.c_str());
    }
    // TODO 更详细统计信息获取
    WriteRecord write_record;
    write_record.timestamp_us = trace.timestamp_us();
    // 注意：这里统计trace中的block数（含重复），而非实际插入数（insert_block_keys.size()）
    // 用于分析trace的数据规模，而非缓存的实际变化
    write_record.write_blocks = trace.keys().size();
    result->write_results.push_back(write_record);
    // 总的写入：减去目前的缓存块数，可以得到驱逐的块数，用于检查前缀树的正确性
    result->counters.total_write_blocks += write_record.write_blocks;
    result->counters.total_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);
}

void OptimizerRunner::HandleDialogTurn(const DialogTurnSchemaTrace &trace) {
    // 模拟对话轮次的读写操作
    std::string instance_id = trace.instance_id();
    auto result_ptr = result_map_.find(instance_id);
    if (result_ptr == result_map_.end()) {
        KVCM_LOG_ERROR("Instance ID not found in result map: %s", instance_id.c_str());
        return;
    }
    auto &result = result_ptr->second;

    auto indexer = indexer_manager_->GetOptIndexer(instance_id);

    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
        return;
    }
    std::vector<std::vector<int64_t>> hits;
    auto insert_block_keys = indexer->InsertWithQuery(trace.total_keys(), trace.timestamp_us(), hits);
    indexer_manager_->CheckAndEvict(instance_id);

    ReadRecord read_record;
    read_record.timestamp_us = trace.timestamp_us();
    read_record.external_read_blocks = trace.keys().size();
    read_record.internal_read_blocks = 0;
    read_record.external_hit_blocks = 0;
    for (const auto &hit : hits) {
        read_record.external_hit_blocks += hit.size();
    }
    read_record.internal_hit_blocks = 0;
    read_record.current_cache_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);
    auto indexer_map = indexer_manager_->GetAllOptIndexers();
    read_record.blocks_per_instance.resize(indexer_map.size(), 0);
    size_t instance_idx = 0;
    for (const auto &pair : indexer_map) {
        read_record.blocks_per_instance[instance_idx] = eviction_manager_->GetCurrentInstanceUsage(pair.first);
        instance_idx++;
    }
    result->read_results.push_back(read_record);

    result->counters.total_read_blocks += trace.keys().size();
    result->counters.total_hit_blocks += read_record.external_hit_blocks + read_record.internal_hit_blocks;
    result->counters.total_read_requests += 1;
    WriteRecord write_record;
    write_record.timestamp_us = trace.timestamp_us();
    // 注意：统计trace中的block数，而非实际插入数
    write_record.write_blocks = trace.total_keys().size() - trace.keys().size();
    result->write_results.push_back(write_record);
    result->counters.total_write_blocks += write_record.write_blocks;
    result->counters.total_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);
}
} // namespace kv_cache_manager