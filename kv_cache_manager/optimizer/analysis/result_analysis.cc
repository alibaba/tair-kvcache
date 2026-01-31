#include "kv_cache_manager/optimizer/analysis/result_analysis.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
void HitAnalysis::Analyze(const std::unordered_map<std::string, std::shared_ptr<Result>> &result_map,
                          const OptimizerConfig &config) {
    for (const auto &pair : result_map) {
        const std::string &instance_id = pair.first;
        const std::shared_ptr<Result> &result = pair.second;
        KVCM_LOG_INFO("Analyzing hit rates for instance: %s", instance_id.c_str());
        if (result->counters.total_read_requests == 0) {
            KVCM_LOG_WARN("No read requests for instance: %s, skipping analysis", instance_id.c_str());
            continue;
        }
        std::vector<double> external_hit_rates;
        std::vector<double> internal_hit_rates;

        size_t acc_total_read_blocks = 0;
        size_t acc_external_hit_blocks = 0;
        size_t acc_internal_hit_blocks = 0;
        std::vector<double> acc_external_hit_rates;
        std::vector<double> acc_internal_hit_rates;

        for (const auto &record : result->read_results) {
            double external_hit_rate = 0.0;
            double internal_hit_rate = 0.0;
            size_t total_read_blocks = record.external_read_blocks + record.internal_read_blocks;

            if (total_read_blocks > 0) {
                external_hit_rate = static_cast<double>(record.external_hit_blocks) / total_read_blocks;
                internal_hit_rate = static_cast<double>(record.internal_hit_blocks) / total_read_blocks;
            }
            external_hit_rates.push_back(external_hit_rate);
            internal_hit_rates.push_back(internal_hit_rate);

            double acc_external_hit_rate = 0.0;
            double acc_internal_hit_rate = 0.0;
            acc_total_read_blocks += total_read_blocks;
            acc_external_hit_blocks += record.external_hit_blocks;
            acc_internal_hit_blocks += record.internal_hit_blocks;
            if (acc_total_read_blocks > 0) {
                acc_external_hit_rate = static_cast<double>(acc_external_hit_blocks) / acc_total_read_blocks;
                acc_internal_hit_rate = static_cast<double>(acc_internal_hit_blocks) / acc_total_read_blocks;
            }
            acc_external_hit_rates.push_back(acc_external_hit_rate);
            acc_internal_hit_rates.push_back(acc_internal_hit_rate);
        }
        ExportHitRates(instance_id,
                       result,
                       external_hit_rates,
                       internal_hit_rates,
                       acc_external_hit_rates,
                       acc_internal_hit_rates,
                       config);
    }
}

void HitAnalysis::ExportHitRates(const std::string &instance_id,
                                 const std::shared_ptr<Result> &result,
                                 const std::vector<double> &external_hit_rates,
                                 const std::vector<double> &internal_hit_rates,
                                 const std::vector<double> &acc_external_hit_rates,
                                 const std::vector<double> &acc_internal_hit_rates,
                                 const OptimizerConfig &config) {
    std::string file_dir = config.output_result_path();
    std::filesystem::create_directories(file_dir); // 目录已存在则不创建

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
                oss << ';'; // 用 ; 分隔，避免与 CSV 的逗号冲突
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
    file << "TimestampUs,CachedBlocksCurrentInstance,CachedBlocksPerInstance,CachedBlocksAllInstance,"
            "InternalReadBlocks,ExternalReadBlocks,TotalReadBlocks,InternalHitBlocks,"
            "InternalHitRate,ExternalHitBlocks,ExternalHitRate,HitRate,AccInternalHitRate,AccExternalHitRate,"
            "AccHitRate\n";
    for (size_t i = 0; i < result->read_results.size(); ++i) {
        const auto &blocks_per_instance = result->read_results[i].blocks_per_instance;
        file << result->read_results[i].timestamp_us << ", " << result->read_results[i].current_cache_blocks << ", "
             << JoinVecSizeT(blocks_per_instance) << ", " << SumVecSizeT(blocks_per_instance) << ", "
             << result->read_results[i].internal_read_blocks << ", " << result->read_results[i].external_read_blocks
             << ", " << (result->read_results[i].internal_read_blocks + result->read_results[i].external_read_blocks)
             << ", " << result->read_results[i].internal_hit_blocks << ", " << internal_hit_rates[i] << ", "
             << result->read_results[i].external_hit_blocks << ", " << external_hit_rates[i] << ", "
             << (internal_hit_rates[i] + external_hit_rates[i]) << ", " << acc_internal_hit_rates[i] << ", "
             << acc_external_hit_rates[i] << ", " << (acc_internal_hit_rates[i] + acc_external_hit_rates[i]) << "\n";
    }
    file.close();
    KVCM_LOG_INFO("Hit rates exported to: %s", filename.c_str());
}
} // namespace kv_cache_manager