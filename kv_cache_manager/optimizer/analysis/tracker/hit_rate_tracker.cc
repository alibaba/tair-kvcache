#include "kv_cache_manager/optimizer/analysis/tracker/hit_rate_tracker.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace {
[[noreturn]] void LogAndThrowHitRateExportError(const std::string &message) {
    KVCM_LOG_ERROR("%s", message.c_str());
    throw std::runtime_error(message);
}
} // namespace

HitRateTracker::HitRateTracker() : StatsTracker("HitRateTracker") {}

// ============================================================================
// 事件处理
// ============================================================================
void HitRateTracker::OnReadComplete(const std::string &instance_id, const ReadRecord &record) {
    instance_data_[instance_id].read_records.push_back(record);
}

void HitRateTracker::OnWriteComplete(const std::string &instance_id, const WriteRecord &record) {
    instance_data_[instance_id].write_records.push_back(record);
}

// ============================================================================
// 查询接口 — 供 OptimizerManager 的在线接口使用
// ============================================================================
const ReadRecord *HitRateTracker::LastReadRecord(const std::string &instance_id) const {
    auto it = instance_data_.find(instance_id);
    if (it == instance_data_.end() || it->second.read_records.empty()) {
        return nullptr;
    }
    return &it->second.read_records.back();
}

const WriteRecord *HitRateTracker::LastWriteRecord(const std::string &instance_id) const {
    auto it = instance_data_.find(instance_id);
    if (it == instance_data_.end() || it->second.write_records.empty()) {
        return nullptr;
    }
    return &it->second.write_records.back();
}

// ============================================================================
// 导出 — 计算命中率并写入 CSV
// ============================================================================
void HitRateTracker::Export(const std::string &instance_id, const OptimizerConfig &config) {
    auto it = instance_data_.find(instance_id);
    if (it == instance_data_.end() || it->second.read_records.empty()) {
        KVCM_LOG_WARN("No read requests for instance: %s, skipping hit rate export", instance_id.c_str());
        return;
    }
    ExportHitRates(instance_id, it->second, config);
}

void HitRateTracker::ExportHitRates(const std::string &instance_id,
                                    const InstanceData &data,
                                    const OptimizerConfig &config) {
    for (const auto &record : data.read_records) {
        if (record.block_size_tokens == 0) {
            LogAndThrowHitRateExportError("Cannot export hit rates for instance " + instance_id +
                                          ": block_size must be set to a positive value");
        }
    }

    std::string file_dir = config.output_result_path();
    std::filesystem::create_directories(file_dir);

    std::string filename = file_dir + "/" + instance_id + "_hit_rates.csv";
    std::ofstream file(filename);
    if (!file.is_open()) {
        LogAndThrowHitRateExportError("Failed to open file for writing hit rates: " + filename);
    }

    auto SumVecSizeT = [](const std::vector<size_t> &v) -> size_t {
        size_t s = 0;
        for (auto x : v)
            s += x;
        return s;
    };

    // per-tier 动态命中率
    size_t num_tiers = 0;
    std::vector<std::string> tier_names;
    for (const auto &r : data.read_records) {
        if (!r.tier_names.empty()) {
            num_tiers = r.tier_names.size();
            tier_names = r.tier_names;
            break;
        }
    }
    bool has_tiered_data = (num_tiers > 0);

    // ---- 写入 CSV ----
    file << "TimestampNs,CachedBlocks,CachedBlocksAllInstances,ReadBlocks,LocalHitBlocks,RemoteHitBlocks,HitBlocks,"
            "InputTokens,LocalHitTokens,RemoteHitTokens,HitTokens,LocalHitRate,RemoteHitRate,HitRate,"
            "AccReadBlocks,AccHitBlocks,AccInputTokens,AccLocalHitTokens,AccRemoteHitTokens,AccHitTokens,"
            "AccLocalHitRate,AccRemoteHitRate,AccHitRate,AccWriteBlocks";
    if (has_tiered_data) {
        for (size_t t = 0; t < num_tiers; ++t) {
            const auto &name = tier_names[t];
            file << ",Tier" << t << "(" << name << ")_HitTokens";
        }
        for (size_t t = 0; t < num_tiers; ++t) {
            const auto &name = tier_names[t];
            file << ",Tier" << t << "(" << name << ")_HitRate";
        }
        for (size_t t = 0; t < num_tiers; ++t) {
            const auto &name = tier_names[t];
            file << ",AccTier" << t << "(" << name << ")_HitRate";
        }
        for (size_t t = 0; t < num_tiers; ++t) {
            const auto &name = tier_names[t];
            file << ",Tier" << t << "(" << name << ")_BlockNum";
        }
    }
    file << "\n";

    size_t acc_write_blocks = 0;
    size_t acc_read_blocks = 0;
    size_t acc_hit_blocks = 0;
    size_t acc_input_tokens = 0;
    size_t acc_local_hit_tokens = 0;
    size_t acc_remote_hit_tokens = 0;
    size_t acc_hit_tokens = 0;
    size_t write_index = 0;
    std::vector<size_t> acc_tier_hit_tokens(num_tiers, 0);

    for (size_t i = 0; i < data.read_records.size(); ++i) {
        const auto &r = data.read_records[i];
        size_t read_blocks = r.local_read_blocks + r.remote_read_blocks;
        size_t current_hit = r.local_hit_blocks + r.remote_hit_blocks;
        size_t block_size_tokens = r.block_size_tokens;
        size_t input_tokens = r.input_tokens;
        size_t local_hit_tokens = r.local_hit_blocks * block_size_tokens;
        size_t remote_hit_tokens = r.remote_hit_blocks * block_size_tokens;
        size_t hit_tokens = current_hit * block_size_tokens;
        acc_read_blocks += read_blocks;
        acc_hit_blocks += current_hit;
        acc_input_tokens += input_tokens;
        acc_local_hit_tokens += local_hit_tokens;
        acc_remote_hit_tokens += remote_hit_tokens;
        acc_hit_tokens += hit_tokens;

        while (write_index < data.write_records.size() &&
               data.write_records[write_index].timestamp_ns <= r.timestamp_ns) {
            acc_write_blocks += data.write_records[write_index].write_blocks;
            write_index++;
        }

        file << r.timestamp_ns << "," << r.current_cache_blocks << "," << SumVecSizeT(r.blocks_per_instance) << ","
             << read_blocks << "," << r.local_hit_blocks << "," << r.remote_hit_blocks << "," << current_hit << ","
             << input_tokens << "," << local_hit_tokens << "," << remote_hit_tokens << "," << hit_tokens << ","
             << (input_tokens > 0 ? static_cast<double>(local_hit_tokens) / input_tokens : 0.0) << ","
             << (input_tokens > 0 ? static_cast<double>(remote_hit_tokens) / input_tokens : 0.0) << ","
             << (input_tokens > 0 ? static_cast<double>(hit_tokens) / input_tokens : 0.0) << "," << acc_read_blocks
             << "," << acc_hit_blocks << "," << acc_input_tokens << "," << acc_local_hit_tokens << ","
             << acc_remote_hit_tokens << "," << acc_hit_tokens << ","
             << (acc_input_tokens > 0 ? static_cast<double>(acc_local_hit_tokens) / acc_input_tokens : 0.0) << ","
             << (acc_input_tokens > 0 ? static_cast<double>(acc_remote_hit_tokens) / acc_input_tokens : 0.0) << ","
             << (acc_input_tokens > 0 ? static_cast<double>(acc_hit_tokens) / acc_input_tokens : 0.0) << ","
             << acc_write_blocks;
        if (has_tiered_data) {
            for (size_t t = 0; t < num_tiers; ++t) {
                size_t hits = (t < r.per_tier_hit_blocks.size()) ? r.per_tier_hit_blocks[t] : 0;
                file << "," << (hits * block_size_tokens);
            }
            for (size_t t = 0; t < num_tiers; ++t) {
                size_t hits = (t < r.per_tier_hit_blocks.size()) ? r.per_tier_hit_blocks[t] : 0;
                size_t tier_hit_tokens = hits * block_size_tokens;
                acc_tier_hit_tokens[t] += tier_hit_tokens;
                file << "," << (input_tokens > 0 ? static_cast<double>(tier_hit_tokens) / input_tokens : 0.0);
            }
            for (size_t t = 0; t < num_tiers; ++t) {
                file << ","
                     << (acc_input_tokens > 0 ? static_cast<double>(acc_tier_hit_tokens[t]) / acc_input_tokens : 0.0);
            }
            for (size_t t = 0; t < num_tiers; ++t) {
                size_t blocks = (t < r.per_tier_blocks.size()) ? r.per_tier_blocks[t] : 0;
                file << "," << blocks;
            }
        }
        file << "\n";
    }

    file.close();
    KVCM_LOG_INFO("Hit rates exported to: %s", filename.c_str());
}

void HitRateTracker::Reset(const std::string &instance_id) {
    auto it = instance_data_.find(instance_id);
    if (it != instance_data_.end()) {
        it->second = InstanceData{};
    }
}

} // namespace kv_cache_manager
