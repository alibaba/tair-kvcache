#pragma once
#include <memory>
#include <vector>

#include "kv_cache_manager/optimizer/analysis/result_analysis.h"
#include "kv_cache_manager/optimizer/analysis/result_structure.h"
#include "kv_cache_manager/optimizer/config/insight_simulator_types.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_config_loader.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/index/radix_tree_index.h"
#include "kv_cache_manager/optimizer/manager/eviction_manager.h"
#include "kv_cache_manager/optimizer/manager/indexer_manager.h"
#include "kv_cache_manager/optimizer/manager/optimizer_runner.h"
#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_trace.h"

namespace kv_cache_manager {

class OptimizerManager {
public:
    OptimizerManager(const OptimizerConfig &config);
    ~OptimizerManager() = default;
    bool Init();

public:
    void DirectRun();

    WriteCacheRes WriteCache(const std::string &instance_id,
                             const std::string &trace_id,
                             const int64_t timestamp,
                             const std::vector<int64_t> &block_ids,
                             const std::vector<int64_t> &token_ids);
    GetCacheLocationRes GetCacheLocation(const std::string &instance_id,
                                         const std::string &trace_id,
                                         const int64_t timestamp,
                                         const std::vector<int64_t> &block_ids,
                                         const std::vector<int64_t> &token_ids,
                                         const BlockMask &block_mask);
    void AnalyzeResults();

    // 导出前缀树用于可视化
    std::unordered_map<std::string, RadixTreeIndex::RadixTreeExport> ExportRadixTrees() const;

    // 清空指定实例的缓存（不重置统计结果）
    bool ClearCache(const std::string &instance_id);

    // 清空所有实例的缓存（不重置统计结果）
    void ClearAllCaches();

    // 清空指定实例的缓存并重置统计结果
    bool ClearCacheAndResetStats(const std::string &instance_id);

    // 清空所有实例的缓存并重置统计结果
    void ClearAllCachesAndResetStats();

private:
    bool CreateRadixTreeIndex(const OptInstanceConfig &instance_config,
                              const std::vector<OptTierConfig> &storage_configs);

private:
    OptimizerConfig config_;
    std::unordered_map<std::string, std::shared_ptr<Result>> result_map_;
    std::unordered_map<std::string, OptInstanceGroupConfig> instance_group_configs_;
    std::unordered_map<std::string, OptInstanceConfig> instance_configs_;

    std::shared_ptr<OptEvictionManager> eviction_manager_;
    std::shared_ptr<OptIndexerManager> indexer_manager_;
    std::shared_ptr<OptimizerRunner> optimizer_runner_;
    std::shared_ptr<HitAnalysis> analyzer_;
};
} // namespace kv_cache_manager