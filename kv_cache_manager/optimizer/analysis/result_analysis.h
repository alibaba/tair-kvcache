#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/analysis/result_structure.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
namespace kv_cache_manager {
class HitAnalysis {
public:
    HitAnalysis() = default;
    ~HitAnalysis() = default;
    void Analyze(const std::unordered_map<std::string, std::shared_ptr<Result>> &result_map,
                 const OptimizerConfig &config);

private:
    void ExportHitRates(const std::string &instance_id,
                        const std::shared_ptr<Result> &result,
                        const std::vector<double> &external_hit_rates,
                        const std::vector<double> &internal_hit_rates,
                        const std::vector<double> &acc_external_hit_rates,
                        const std::vector<double> &acc_internal_hit_rates,
                        const OptimizerConfig &config);
};
} // namespace kv_cache_manager