#include "kv_cache_manager/optimizer/analysis/tracker/hit_rate_tracker.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

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
    std::string file_dir = config.output_result_path();
    std::filesystem::create_directories(file_dir);

    std::string filename = file_dir + "/" + instance_id + "_hit_rates.csv";
    std::ofstream file(filename);
    if (!file.is_open()) {
        KVCM_LOG_ERROR("Failed to open file for writing hit rates: %s", filename.c_str());
        return;
    }

    auto JoinVecSizeT = [](const std::vector<size_t> &v) {
        std::ostringstream oss;
        for (size_t k = 0; k < v.size(); ++k) {
            if (k)
                oss << ';';
            oss << v[k];
        }
        return oss.str();
    };

    auto SumVecSizeT = [](const std::vector<size_t> &v) -> size_t {
        size_t s = 0;
        for (auto x : v)
            s += x;
        return s;
    };

    // ---- 计算逐请求命中率和累计命中率 ----
    std::vector<double> remote_hit_rates;
    std::vector<double> local_hit_rates;
    std::vector<double> acc_remote_hit_rates;
    std::vector<double> acc_local_hit_rates;

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

    // per-tier: tier_hit_rates[tier_idx][record_idx], acc 同理
    std::vector<std::vector<double>> tier_hit_rates(num_tiers);
    std::vector<std::vector<double>> acc_tier_hit_rates(num_tiers);
    std::vector<size_t> acc_tier_hits(num_tiers, 0);

    size_t acc_total_read = 0;
    size_t acc_remote_hit = 0;
    size_t acc_local_hit = 0;

    for (const auto &record : data.read_records) {
        size_t total = record.remote_read_blocks + record.local_read_blocks;
        double remote_rate = total > 0 ? static_cast<double>(record.remote_hit_blocks) / total : 0.0;
        double local_rate = total > 0 ? static_cast<double>(record.local_hit_blocks) / total : 0.0;
        remote_hit_rates.push_back(remote_rate);
        local_hit_rates.push_back(local_rate);

        // per-tier 命中率
        for (size_t t = 0; t < num_tiers; ++t) {
            size_t tier_hits = (t < record.per_tier_hit_blocks.size()) ? record.per_tier_hit_blocks[t] : 0;
            double t_rate = total > 0 ? static_cast<double>(tier_hits) / total : 0.0;
            tier_hit_rates[t].push_back(t_rate);
            if (tier_hits > 0) {
                has_tiered_data = true;
            }
            acc_tier_hits[t] += tier_hits;
        }

        acc_total_read += total;
        acc_remote_hit += record.remote_hit_blocks;
        acc_local_hit += record.local_hit_blocks;
        acc_remote_hit_rates.push_back(acc_total_read > 0 ? static_cast<double>(acc_remote_hit) / acc_total_read : 0.0);
        acc_local_hit_rates.push_back(acc_total_read > 0 ? static_cast<double>(acc_local_hit) / acc_total_read : 0.0);
        for (size_t t = 0; t < num_tiers; ++t) {
            acc_tier_hit_rates[t].push_back(acc_total_read > 0 ? static_cast<double>(acc_tier_hits[t]) / acc_total_read
                                                               : 0.0);
        }
    }

    // ---- 写入 CSV ----
    file << "TimestampNs,CachedBlocksCurrentInstance,CachedBlocksPerInstance,CachedBlocksAllInstance,"
            "LocalReadBlocks,RemoteReadBlocks,TotalReadBlocks,LocalHitBlocks,"
            "LocalHitRate,RemoteHitBlocks,RemoteHitRate,HitRate,AccLocalHitRate,AccRemoteHitRate,"
            "AccHitRate,AccReadBlocks,AccWriteBlocks";
    if (has_tiered_data) {
        for (size_t t = 0; t < num_tiers; ++t) {
            const auto &name = tier_names[t];
            file << ",Tier" << t << "(" << name << ")_HitBlocks";
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

    size_t acc_read_blocks = 0;
    size_t acc_write_blocks = 0;
    size_t write_index = 0;

    for (size_t i = 0; i < data.read_records.size(); ++i) {
        const auto &r = data.read_records[i];
        size_t current_read = r.local_read_blocks + r.remote_read_blocks;
        acc_read_blocks += current_read;

        while (write_index < data.write_records.size() &&
               data.write_records[write_index].timestamp_ns <= r.timestamp_ns) {
            acc_write_blocks += data.write_records[write_index].write_blocks;
            write_index++;
        }

        file << r.timestamp_ns << "," << r.current_cache_blocks << "," << JoinVecSizeT(r.blocks_per_instance) << ","
             << SumVecSizeT(r.blocks_per_instance) << "," << r.local_read_blocks << "," << r.remote_read_blocks << ","
             << current_read << "," << r.local_hit_blocks << "," << local_hit_rates[i] << "," << r.remote_hit_blocks
             << "," << remote_hit_rates[i] << "," << (local_hit_rates[i] + remote_hit_rates[i]) << ","
             << acc_local_hit_rates[i] << "," << acc_remote_hit_rates[i] << ","
             << (acc_local_hit_rates[i] + acc_remote_hit_rates[i]) << "," << acc_read_blocks << "," << acc_write_blocks;
        if (has_tiered_data) {
            for (size_t t = 0; t < num_tiers; ++t) {
                size_t hits = (t < r.per_tier_hit_blocks.size()) ? r.per_tier_hit_blocks[t] : 0;
                file << "," << hits;
            }
            for (size_t t = 0; t < num_tiers; ++t) {
                file << "," << tier_hit_rates[t][i];
            }
            for (size_t t = 0; t < num_tiers; ++t) {
                file << "," << acc_tier_hit_rates[t][i];
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
