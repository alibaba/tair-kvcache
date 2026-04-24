#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {
// 前置声明
class StatsCollector;

void AppendBlockLocation(BlockEntry *block, const std::string &unique_name, int64_t timestamp);
class RadixTreeIndex {
public:
    // 新构造函数 (多 tier)
    RadixTreeIndex(const std::string &instance_id, std::vector<std::shared_ptr<EvictionPolicy>> tier_policies);
    // 兼容构造函数 (单 policy)
    RadixTreeIndex(const std::string &instance_id, const std::shared_ptr<EvictionPolicy> &eviction_policy);
    RadixTreeIndex();
    ~RadixTreeIndex() = default;

    struct InsertResult {
        std::vector<int64_t> inserted_keys;
    };

    InsertResult InsertOnly(const std::vector<int64_t> &block_keys, const int64_t timestamp);
    void PrefixQuery(const std::vector<int64_t> &block_keys,
                     const BlockMask &block_mask,
                     const int64_t timestamp,
                     QueryHit *query_hit = nullptr);

    std::vector<int64_t>
    InsertWithQuery(const std::vector<int64_t> &block_keys, const int64_t timestamp, QueryHit *query_hit = nullptr);
    void CleanEmptyBlocks(const std::vector<BlockEntry *> &blocks, int64_t eviction_timestamp);

    // 清空整个RadixTree的缓存
    void Clear();

    void SetStatsCollector(std::shared_ptr<StatsCollector> collector) { stats_collector_ = collector; }

    const std::vector<std::string> &GetTierNames() const { return tier_names_; }

    // 导出前缀树用于可视化
    struct RadixTreeExportNode {
        std::string node_id;
        size_t access_count;
        int64_t last_access_time;
        std::vector<int64_t> total_blocks;
        std::vector<int64_t> cached_blocks;
        bool is_leaf;
        std::string parent_id;
    };

    struct RadixTreeExport {
        std::string instance_id;
        std::vector<RadixTreeExportNode> nodes;
        std::vector<std::pair<std::string, std::string>> edges;
    };

    RadixTreeExport ExportForVisualization() const;

    const RadixTreeNode *GetRoot() const { return root_.get(); }

private:
    std::unique_ptr<RadixTreeNode> root_;
    std::vector<std::shared_ptr<EvictionPolicy>> tier_policies_; // 按 tier priority 排序
    std::vector<std::string> tier_names_;                        // 缓存 policy name
    std::string instance_id_;
    std::shared_ptr<StatsCollector> stats_collector_;

private:
    std::vector<BlockEntry *>
    AppendNewBlocks(RadixTreeNode *node, const std::vector<int64_t> &block_keys, const int64_t timestamp);

    InsertResult InsertNode(RadixTreeNode *node, const std::vector<int64_t> &block_keys, const int64_t timestamp);
    void SplitNode(RadixTreeNode *existing_node,
                   size_t split_pos,
                   const std::vector<int64_t> &remaining_keys,
                   int64_t timestamp);
    std::vector<int64_t> InsertQuery(RadixTreeNode *node,
                                     const std::vector<int64_t> &block_keys,
                                     const int64_t timestamp,
                                     bool is_prefix_hit,
                                     QueryHit *query_hit = nullptr);

    using WriteModify = std::function<std::vector<BlockEntry *>(const std::vector<int64_t> &, int64_t)>;
    WriteModify AppendEvictBlocks(std::unordered_map<int64_t, BlockEntry *> blocks_map);

    void
    WriteToTier(RadixTreeNode *node, const std::vector<int64_t> &block_keys, const int64_t timestamp, WriteModify cb);

    void OnBlockAccessed(BlockEntry *block, int64_t timestamp);
    bool IsBlockEvict(BlockEntry *block) const;

    // per-tier 命中检测辅助方法
    void RecordTieredHit(BlockEntry *block, bool is_remote, QueryHit *query_hit) const;
};
} // namespace kv_cache_manager
