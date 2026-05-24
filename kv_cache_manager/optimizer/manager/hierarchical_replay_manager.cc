#include "kv_cache_manager/optimizer/manager/hierarchical_replay_manager.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"

namespace kv_cache_manager {
namespace {

std::unordered_map<std::string, OptInstanceConfig> CollectInstances(const OptimizerConfig &config) {
    std::unordered_map<std::string, OptInstanceConfig> instances;
    for (const auto &group : config.instance_groups()) {
        for (auto instance : group.instances()) {
            instance.set_instance_group_name(group.group_name());
            instances.emplace(instance.instance_id(), instance);
        }
    }
    return instances;
}

size_t PositiveBlockSizeOrZero(const OptInstanceConfig &instance) {
    return instance.block_size() > 0 ? static_cast<size_t>(instance.block_size()) : 0;
}

} // namespace

HierarchicalReplayManager::HierarchicalReplayManager(const HierarchicalReplayConfig &config) : config_(config) {}

bool HierarchicalReplayManager::Init() {
    if (!ValidateAndBuildMappings()) {
        return false;
    }

    engine_manager_ = std::make_unique<OptimizerManager>(config_.engine_config());
    if (!engine_manager_->Init()) {
        KVCM_LOG_ERROR("Hierarchical replay failed to initialize engine manager.");
        return false;
    }

    pool_manager_ = std::make_unique<OptimizerManager>(config_.pool_config());
    if (!pool_manager_->Init()) {
        KVCM_LOG_ERROR("Hierarchical replay failed to initialize pool manager.");
        return false;
    }
    return true;
}

bool HierarchicalReplayManager::ValidateAndBuildMappings() {
    engine_to_pool_.clear();
    engine_block_size_.clear();
    sorted_engine_instance_ids_.clear();

    const auto engine_instances = CollectInstances(config_.engine_config());
    const auto pool_instances = CollectInstances(config_.pool_config());
    if (engine_instances.empty()) {
        KVCM_LOG_ERROR("Hierarchical replay engine_config has no instances.");
        return false;
    }
    if (pool_instances.empty()) {
        KVCM_LOG_ERROR("Hierarchical replay pool_config has no instances.");
        return false;
    }

    for (const auto &mapping : config_.engine_to_pool()) {
        const auto &engine_instance_id = mapping.engine_instance_id();
        const auto &pool_instance_id = mapping.pool_instance_id();
        auto engine_it = engine_instances.find(engine_instance_id);
        if (engine_it == engine_instances.end()) {
            KVCM_LOG_ERROR("engine_to_pool references unknown engine instance: %s", engine_instance_id.c_str());
            return false;
        }
        auto pool_it = pool_instances.find(pool_instance_id);
        if (pool_it == pool_instances.end()) {
            KVCM_LOG_ERROR("engine_to_pool references unknown pool instance: %s", pool_instance_id.c_str());
            return false;
        }
        if (engine_to_pool_.find(engine_instance_id) != engine_to_pool_.end()) {
            KVCM_LOG_ERROR("engine instance is mapped more than once: %s", engine_instance_id.c_str());
            return false;
        }

        const size_t engine_block_size = PositiveBlockSizeOrZero(engine_it->second);
        const size_t pool_block_size = PositiveBlockSizeOrZero(pool_it->second);
        if (engine_block_size == 0 || pool_block_size == 0 || engine_block_size != pool_block_size) {
            KVCM_LOG_ERROR("engine/pool block_size mismatch for engine=%s pool=%s",
                           engine_instance_id.c_str(),
                           pool_instance_id.c_str());
            return false;
        }

        engine_to_pool_[engine_instance_id] = pool_instance_id;
        engine_block_size_[engine_instance_id] = engine_block_size;
        sorted_engine_instance_ids_.push_back(engine_instance_id);
    }

    if (engine_to_pool_.size() != engine_instances.size()) {
        KVCM_LOG_ERROR("Every engine instance must have exactly one engine_to_pool mapping.");
        return false;
    }
    std::sort(sorted_engine_instance_ids_.begin(), sorted_engine_instance_ids_.end());
    return true;
}

const std::string &HierarchicalReplayManager::PoolInstanceForEngine(const std::string &engine_instance_id) const {
    auto it = engine_to_pool_.find(engine_instance_id);
    if (it == engine_to_pool_.end()) {
        throw std::runtime_error("No pool instance mapping for engine instance: " + engine_instance_id);
    }
    return it->second;
}

void HierarchicalReplayManager::DirectRun() {
    auto traces = StandardTraceLoader::LoadFromFile(config_.trace_file_path());
    ScheduleTraces(traces);
    RunTraces(traces);
}

void HierarchicalReplayManager::ScheduleTraces(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) const {
    if (config_.engine_scheduling_strategy() == "preserve_trace") {
        return;
    }
    if (config_.engine_scheduling_strategy() != "round_robin") {
        throw std::runtime_error("Unknown engine_scheduling_strategy: " + config_.engine_scheduling_strategy());
    }
    if (sorted_engine_instance_ids_.empty()) {
        throw std::runtime_error("round_robin scheduling requires at least one engine instance");
    }

    size_t request_idx = 0;
    std::string current_engine_instance_id = sorted_engine_instance_ids_.front();
    for (auto &trace : traces) {
        if (!trace) {
            continue;
        }
        if (std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
            current_engine_instance_id = sorted_engine_instance_ids_[request_idx % sorted_engine_instance_ids_.size()];
            request_idx++;
        } else if (request_idx == 0) {
            current_engine_instance_id = sorted_engine_instance_ids_.front();
        }
        trace->set_instance_id(current_engine_instance_id);
    }
}

void HierarchicalReplayManager::RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    for (const auto &trace : traces) {
        RunTrace(trace);
    }
}

void HierarchicalReplayManager::RunTrace(const std::shared_ptr<OptimizerSchemaTrace> &trace) {
    if (!trace) {
        return;
    }
    if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
        GetCacheLocation(get_trace->instance_id(),
                         get_trace->trace_id(),
                         get_trace->timestamp_ns(),
                         get_trace->keys(),
                         get_trace->input_len());
    } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
        WriteCacheWithTtlUs(write_trace->instance_id(),
                            write_trace->trace_id(),
                            write_trace->timestamp_ns(),
                            write_trace->keys(),
                            write_trace->ttl_us());
    } else {
        KVCM_LOG_WARN("Hierarchical replay skips unknown trace type.");
    }
}

HierarchicalGetCacheLocationRes HierarchicalReplayManager::GetCacheLocation(const std::string &engine_instance_id,
                                                                            const std::string &trace_id,
                                                                            int64_t timestamp,
                                                                            const std::vector<int64_t> &block_ids,
                                                                            int64_t input_len) {
    if (!engine_manager_ || !pool_manager_) {
        throw std::runtime_error("HierarchicalReplayManager is not initialized");
    }
    auto block_size_it = engine_block_size_.find(engine_instance_id);
    if (block_size_it == engine_block_size_.end()) {
        throw std::runtime_error("Unknown engine instance: " + engine_instance_id);
    }

    const std::string &pool_instance_id = PoolInstanceForEngine(engine_instance_id);
    const BlockMask empty_mask = BlockMaskVector{};
    const auto engine_res =
        engine_manager_->GetCacheLocation(engine_instance_id, trace_id, timestamp, block_ids, empty_mask, input_len);
    const size_t engine_hit_blocks =
        std::min(static_cast<size_t>(std::max<int64_t>(engine_res.kvcm_hit_length, 0)), block_ids.size());

    size_t pool_hit_blocks = 0;
    if (engine_hit_blocks < block_ids.size()) {
        const auto pool_res = pool_manager_->GetCacheLocationAfterPrefix(
            pool_instance_id, trace_id, timestamp, block_ids, engine_hit_blocks, input_len);
        pool_hit_blocks = std::min(static_cast<size_t>(std::max<int64_t>(pool_res.kvcm_hit_length, 0)),
                                   block_ids.size() - engine_hit_blocks);
    }

    CombinedReadRecord record;
    record.trace_id = trace_id;
    record.engine_instance_id = engine_instance_id;
    record.pool_instance_id = pool_instance_id;
    record.timestamp_ns = timestamp;
    record.read_blocks = block_ids.size();
    record.engine_hit_blocks = engine_hit_blocks;
    record.pool_hit_blocks = pool_hit_blocks;
    record.input_tokens = static_cast<size_t>(input_len);
    record.block_size_tokens = block_size_it->second;
    combined_read_records_.push_back(record);

    HierarchicalGetCacheLocationRes res;
    res.trace_id = trace_id;
    res.engine_hit_length = static_cast<int64_t>(engine_hit_blocks);
    res.pool_hit_length = static_cast<int64_t>(pool_hit_blocks);
    res.total_hit_length = static_cast<int64_t>(engine_hit_blocks + pool_hit_blocks);
    return res;
}

WriteCacheRes HierarchicalReplayManager::WriteCache(const std::string &engine_instance_id,
                                                    const std::string &trace_id,
                                                    int64_t timestamp,
                                                    const std::vector<int64_t> &block_ids,
                                                    int64_t ttl_seconds) {
    const int64_t ttl_us = ttl_seconds > 0 ? ttl_seconds * 1000000 : ttl_seconds;
    return WriteCacheWithTtlUs(engine_instance_id, trace_id, timestamp, block_ids, ttl_us);
}

WriteCacheRes HierarchicalReplayManager::WriteCacheWithTtlUs(const std::string &engine_instance_id,
                                                             const std::string &trace_id,
                                                             int64_t timestamp,
                                                             const std::vector<int64_t> &block_ids,
                                                             int64_t ttl_us) {
    if (!engine_manager_ || !pool_manager_) {
        throw std::runtime_error("HierarchicalReplayManager is not initialized");
    }
    const std::string &pool_instance_id = PoolInstanceForEngine(engine_instance_id);

    auto engine_res = engine_manager_->WriteCacheWithTtlUs(engine_instance_id, trace_id, timestamp, block_ids, ttl_us);
    pool_manager_->WriteCacheWithTtlUs(pool_instance_id, trace_id, timestamp, block_ids, ttl_us);

    CombinedWriteRecord record;
    record.timestamp_ns = timestamp;
    record.write_blocks = block_ids.size();
    combined_write_records_.push_back(record);
    return engine_res;
}

void HierarchicalReplayManager::AnalyzeResults() {
    ExportCombinedHitRates();
    if (engine_manager_) {
        engine_manager_->AnalyzeResults();
    }
    if (pool_manager_) {
        pool_manager_->AnalyzeResults();
    }
}

void HierarchicalReplayManager::ExportCombinedHitRates() const {
    std::filesystem::create_directories(config_.output_result_path());
    const std::string filename = config_.output_result_path() + "/hierarchical_hit_rates.csv";
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open hierarchical hit-rate CSV: " + filename);
    }

    file << "TimestampNs,TraceId,EngineInstanceId,PoolInstanceId,ReadBlocks,EngineHitBlocks,PoolHitBlocks,HitBlocks,"
            "InputTokens,EngineHitTokens,PoolHitTokens,HitTokens,EngineHitRate,PoolHitRate,HitRate,"
            "AccReadBlocks,AccHitBlocks,AccInputTokens,AccHitTokens,AccHitRate,AccWriteBlocks\n";

    size_t acc_read_blocks = 0;
    size_t acc_hit_blocks = 0;
    size_t acc_input_tokens = 0;
    size_t acc_hit_tokens = 0;
    size_t acc_write_blocks = 0;
    size_t write_index = 0;

    for (const auto &record : combined_read_records_) {
        while (write_index < combined_write_records_.size() &&
               combined_write_records_[write_index].timestamp_ns <= record.timestamp_ns) {
            acc_write_blocks += combined_write_records_[write_index].write_blocks;
            write_index++;
        }

        const size_t hit_blocks = record.engine_hit_blocks + record.pool_hit_blocks;
        const size_t engine_hit_tokens = record.engine_hit_blocks * record.block_size_tokens;
        const size_t pool_hit_tokens = record.pool_hit_blocks * record.block_size_tokens;
        const size_t hit_tokens = hit_blocks * record.block_size_tokens;

        acc_read_blocks += record.read_blocks;
        acc_hit_blocks += hit_blocks;
        acc_input_tokens += record.input_tokens;
        acc_hit_tokens += hit_tokens;

        file << record.timestamp_ns << "," << record.trace_id << "," << record.engine_instance_id << ","
             << record.pool_instance_id << "," << record.read_blocks << "," << record.engine_hit_blocks << ","
             << record.pool_hit_blocks << "," << hit_blocks << "," << record.input_tokens << "," << engine_hit_tokens
             << "," << pool_hit_tokens << "," << hit_tokens << ","
             << (record.input_tokens > 0 ? static_cast<double>(engine_hit_tokens) / record.input_tokens : 0.0) << ","
             << (record.input_tokens > 0 ? static_cast<double>(pool_hit_tokens) / record.input_tokens : 0.0) << ","
             << (record.input_tokens > 0 ? static_cast<double>(hit_tokens) / record.input_tokens : 0.0) << ","
             << acc_read_blocks << "," << acc_hit_blocks << "," << acc_input_tokens << "," << acc_hit_tokens << ","
             << (acc_input_tokens > 0 ? static_cast<double>(acc_hit_tokens) / acc_input_tokens : 0.0) << ","
             << acc_write_blocks << "\n";
    }

    KVCM_LOG_INFO("Hierarchical hit rates exported to: %s", filename.c_str());
}

} // namespace kv_cache_manager
