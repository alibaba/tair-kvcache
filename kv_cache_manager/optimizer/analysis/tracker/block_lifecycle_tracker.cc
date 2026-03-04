#include "kv_cache_manager/optimizer/analysis/tracker/block_lifecycle_tracker.h"

#include <filesystem>
#include <fstream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

BlockLifecycleTracker::BlockLifecycleTracker() : StatsTracker("BlockLifecycleTracker") {}

// ============================================================================
// Block级事件
// ============================================================================
void BlockLifecycleTracker::OnBlockBirth(const std::string &instance_id, BlockEntry *block, int64_t timestamp) {
    if (!block) {
        KVCM_LOG_ERROR("Null block pointer in OnBlockBirth");
        return;
    }

    auto &data = instance_data_[instance_id];

    auto it = data.alive_blocks.find(block->key);
    if (it != data.alive_blocks.end()) {
        // 同一 key 重新 birth，说明上一轮生命周期未被正确 evict，强制关闭
        KVCM_LOG_ERROR("CRITICAL: Block %ld already has active lifecycle (birth=%ld), force closing old record",
                       block->key,
                       it->second->birth_time_us);
        OnBlockEviction(instance_id, block, timestamp > 0 ? timestamp - 1 : timestamp);
    }

    auto record = std::make_shared<BlockLifecycleRecord>(BlockLifecycleRecord{
        .block_key = block->key,
        .birth_time_us = timestamp,
        .death_time_us = -1,
        .lifespan_us = 0,
        .access_count = 0,
        .last_access_time_us = timestamp,
        .is_alive = true,
    });

    data.alive_blocks[block->key] = record;
    data.records.push_back(std::move(record));
}

void BlockLifecycleTracker::OnBlockEviction(const std::string &instance_id, BlockEntry *block, int64_t timestamp) {
    if (!block) {
        return;
    }

    auto inst_it = instance_data_.find(instance_id);
    if (inst_it == instance_data_.end()) {
        KVCM_LOG_DEBUG("Evicting untracked block key %ld (instance: %s)", block->key, instance_id.c_str());
        return;
    }

    auto &data = inst_it->second;
    auto it = data.alive_blocks.find(block->key);
    if (it == data.alive_blocks.end()) {
        KVCM_LOG_DEBUG("Evicting untracked block key %ld (instance: %s)", block->key, instance_id.c_str());
        return;
    }

    auto &record = it->second;
    record->death_time_us = timestamp;
    record->lifespan_us = timestamp - record->birth_time_us;
    // block 此时仍存活，可以安全读取其字段
    record->access_count = block->access_count;
    record->last_access_time_us = block->last_access_time;
    record->is_alive = false;

    data.alive_blocks.erase(it);
}

// ============================================================================
// 生命周期管理
// ============================================================================
void BlockLifecycleTracker::Finalize(const std::string &instance_id, int64_t final_timestamp) {
    auto inst_it = instance_data_.find(instance_id);
    if (inst_it == instance_data_.end()) {
        return;
    }

    auto &data = inst_it->second;
    KVCM_LOG_INFO("Finalizing %zu alive blocks at timestamp %ld (instance: %s)",
                  data.alive_blocks.size(),
                  final_timestamp,
                  instance_id.c_str());

    // access_count/last_access_time 在 block 存活期间无法获取（block 可能已销毁），
    // 保留 birth 时记录的初始值，标记为 is_alive=true 表示 trace 结束时仍存活
    for (auto &[block_key, record] : data.alive_blocks) {
        record->death_time_us = final_timestamp;
        record->lifespan_us = final_timestamp - record->birth_time_us;
    }

    data.alive_blocks.clear();

    KVCM_LOG_INFO(
        "Lifecycle tracking complete: %zu total records (instance: %s)", data.records.size(), instance_id.c_str());
}

void BlockLifecycleTracker::Export(const std::string &instance_id, const OptimizerConfig &config) {
    auto inst_it = instance_data_.find(instance_id);
    if (inst_it == instance_data_.end()) {
        return;
    }

    const auto &records = inst_it->second.records;
    if (records.empty()) {
        KVCM_LOG_WARN("No lifecycle records found for instance: %s", instance_id.c_str());
        return;
    }

    std::string file_dir = config.output_result_path();
    std::filesystem::create_directories(file_dir);

    std::string filename = file_dir + "/" + instance_id + "_lifecycle.csv";
    std::ofstream file(filename);
    if (!file.is_open()) {
        KVCM_LOG_ERROR("Failed to open file for writing lifecycle stats: %s", filename.c_str());
        return;
    }

    file << "BlockKey,BirthTimeUs,DeathTimeUs,"
         << "LifespanUs,AccessCount,LastAccessTimeUs,IsAlive\n";

    for (const auto &record : records) {
        file << record->block_key << "," << record->birth_time_us << "," << record->death_time_us << ","
             << record->lifespan_us << "," << record->access_count << "," << record->last_access_time_us << ","
             << (record->is_alive ? "true" : "false") << "\n";
    }

    file.close();
    KVCM_LOG_INFO("Lifecycle stats exported to: %s (total records: %zu)", filename.c_str(), records.size());
}

void BlockLifecycleTracker::Reset(const std::string &instance_id) {
    auto it = instance_data_.find(instance_id);
    if (it != instance_data_.end()) {
        it->second = InstanceData{};
    }
}

} // namespace kv_cache_manager
