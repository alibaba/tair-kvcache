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
                       it->second->birth_time_ns);
        OnBlockEviction(instance_id, block, timestamp > 0 ? timestamp - 1 : timestamp);
    }

    auto record = std::make_shared<BlockLifecycleRecord>(BlockLifecycleRecord{
        .block_key = block->key,
        .birth_time_ns = timestamp,
        .death_time_ns = -1,
        .lifespan_ns = 0,
        .access_count = 0,
        .last_access_time_ns = timestamp,
        .is_alive = true,
        .block_ptr = block,
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
    record->death_time_ns = timestamp;
    record->lifespan_ns = timestamp - record->birth_time_ns;
    record->access_count = block->access_count;
    record->last_access_time_ns = block->last_access_time;
    record->is_alive = false;
    record->block_ptr = nullptr; // block 即将销毁，清除指针

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

    // trace 结束时仍存活的 block：从持有的指针直接读取最终统计值
    for (auto &[block_key, record] : data.alive_blocks) {
        record->death_time_ns = final_timestamp;
        record->lifespan_ns = final_timestamp - record->birth_time_ns;
        if (record->block_ptr) {
            record->access_count = record->block_ptr->access_count;
            record->last_access_time_ns = record->block_ptr->last_access_time;
            record->block_ptr = nullptr;
        } else {
            KVCM_LOG_WARN("Block %ld still alive at Finalize but block_ptr is null, "
                          "access_count may be stale (instance: %s)",
                          block_key,
                          instance_id.c_str());
        }
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

    file << "BlockKey,BirthTimeNs,DeathTimeNs,"
         << "LifespanNs,AccessCount,LastAccessTimeNs,IsAlive\n";

    for (const auto &record : records) {
        file << record->block_key << "," << record->birth_time_ns << "," << record->death_time_ns << ","
             << record->lifespan_ns << "," << record->access_count << "," << record->last_access_time_ns << ","
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
