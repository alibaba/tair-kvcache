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
    RadixTreeIndex(const std::string &instance_id,
                   std::vector<std::shared_ptr<EvictionPolicy>> tier_policies,
                   TierWriteMode write_mode = TierWriteMode::WRITE_THROUGH,
                   int64_t default_ttl_ns = 0);
    // 兼容构造函数 (单 policy)
    RadixTreeIndex(const std::string &instance_id,
                   const std::shared_ptr<EvictionPolicy> &eviction_policy,
                   int64_t default_ttl_ns = 0);
    RadixTreeIndex();
    ~RadixTreeIndex() = default;

    struct InsertResult {
        std::vector<int64_t> inserted_keys;
    };

    // ttl_ns: 0 = 使用 default_ttl_ns_，-1 = 禁用 TTL，>0 = 自定义纳秒
    InsertResult InsertOnly(const std::vector<int64_t> &block_keys, int64_t timestamp, int64_t ttl_ns = 0);
    void PrefixQuery(const std::vector<int64_t> &block_keys,
                     const BlockMask &block_mask,
                     const int64_t timestamp,
                     QueryHit *query_hit = nullptr,
                     bool refresh_ttl_on_read = true);

    void CleanEmptyBlocks(const std::vector<BlockEntry *> &blocks,
                          int64_t eviction_timestamp,
                          bool use_logical_expire_time = false);

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
    // 写入模式：WRITE_THROUGH = 所有 tier 都写，CASCADING = 仅写 tier 0，超出的通过 EvictionManager 级联降级
    TierWriteMode write_mode_ = TierWriteMode::WRITE_THROUGH;
    // 写入流量应落地的 tier 数，构造时结合 write_mode_ 与 tier 数一次性确定
    // WRITE_THROUGH=全部层，CASCADING=仅 tier 0（单层退化为全部）
    size_t write_tier_count_ = 0;
    std::string instance_id_;
    int64_t default_ttl_ns_ = 0;
    std::shared_ptr<StatsCollector> stats_collector_;

private:
    std::vector<BlockEntry *>
    AppendNewBlocks(RadixTreeNode *node, const std::vector<int64_t> &block_keys, int64_t timestamp, int64_t ttl_ns);

    InsertResult
    InsertNode(RadixTreeNode *node, const std::vector<int64_t> &block_keys, int64_t timestamp, int64_t ttl_ns);
    void SplitNode(RadixTreeNode *existing_node,
                   size_t split_pos,
                   const std::vector<int64_t> &remaining_keys,
                   int64_t timestamp,
                   int64_t ttl_ns = 0);

    using WriteModify = std::function<std::vector<BlockEntry *>(const std::vector<int64_t> &, int64_t)>;
    WriteModify AppendEvictBlocks(std::unordered_map<int64_t, BlockEntry *> blocks_map, int64_t ttl_ns);

    void WriteToTier(
        RadixTreeNode *node, const std::vector<int64_t> &block_keys, int64_t timestamp, int64_t ttl_ns, WriteModify cb);

    void OnBlockAccessed(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read = true);
    bool IsBlockEvict(BlockEntry *block, int64_t timestamp) const;

    // per-tier 命中检测辅助方法
    void RecordTieredHit(BlockEntry *block, bool is_remote, QueryHit *query_hit) const;
};
} // namespace kv_cache_manager
