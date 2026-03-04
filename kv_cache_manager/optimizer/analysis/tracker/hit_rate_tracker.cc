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
    std::vector<double> external_hit_rates;
    std::vector<double> internal_hit_rates;
    std::vector<double> acc_external_hit_rates;
    std::vector<double> acc_internal_hit_rates;

    size_t acc_total_read = 0;
    size_t acc_ext_hit = 0;
    size_t acc_int_hit = 0;

    for (const auto &record : data.read_records) {
        size_t total = record.external_read_blocks + record.internal_read_blocks;
        double ext_rate = total > 0 ? static_cast<double>(record.external_hit_blocks) / total : 0.0;
        double int_rate = total > 0 ? static_cast<double>(record.internal_hit_blocks) / total : 0.0;
        external_hit_rates.push_back(ext_rate);
        internal_hit_rates.push_back(int_rate);

        acc_total_read += total;
        acc_ext_hit += record.external_hit_blocks;
        acc_int_hit += record.internal_hit_blocks;
        acc_external_hit_rates.push_back(acc_total_read > 0 ? static_cast<double>(acc_ext_hit) / acc_total_read : 0.0);
        acc_internal_hit_rates.push_back(acc_total_read > 0 ? static_cast<double>(acc_int_hit) / acc_total_read : 0.0);
    }

    // ---- 写入 CSV ----
    file << "TimestampUs,CachedBlocksCurrentInstance,CachedBlocksPerInstance,CachedBlocksAllInstance,"
            "InternalReadBlocks,ExternalReadBlocks,TotalReadBlocks,InternalHitBlocks,"
            "InternalHitRate,ExternalHitBlocks,ExternalHitRate,HitRate,AccInternalHitRate,AccExternalHitRate,"
            "AccHitRate,AccReadBlocks,AccWriteBlocks\n";

    size_t acc_read_blocks = 0;
    size_t acc_write_blocks = 0;
    size_t write_index = 0;

    for (size_t i = 0; i < data.read_records.size(); ++i) {
        const auto &r = data.read_records[i];
        size_t current_read = r.internal_read_blocks + r.external_read_blocks;
        acc_read_blocks += current_read;

        while (write_index < data.write_records.size() &&
               data.write_records[write_index].timestamp_us <= r.timestamp_us) {
            acc_write_blocks += data.write_records[write_index].write_blocks;
            write_index++;
        }

        file << r.timestamp_us << "," << r.current_cache_blocks << "," << JoinVecSizeT(r.blocks_per_instance) << ","
             << SumVecSizeT(r.blocks_per_instance) << "," << r.internal_read_blocks << "," << r.external_read_blocks
             << "," << current_read << "," << r.internal_hit_blocks << "," << internal_hit_rates[i] << ","
             << r.external_hit_blocks << "," << external_hit_rates[i] << ","
             << (internal_hit_rates[i] + external_hit_rates[i]) << "," << acc_internal_hit_rates[i] << ","
             << acc_external_hit_rates[i] << "," << (acc_internal_hit_rates[i] + acc_external_hit_rates[i]) << ","
             << acc_read_blocks << "," << acc_write_blocks << "\n";
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
