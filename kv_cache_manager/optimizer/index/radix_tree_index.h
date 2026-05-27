#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/tier_flow/tier_flow_recorder.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {
// 前置声明
class StatsCollector;

void AppendBlockLocation(BlockEntry *block,
                         const std::string &unique_name,
                         int64_t timestamp,
                         size_t write_touch_count = 1);
void CopyBlockLocation(BlockEntry *block,
                       const std::string &unique_name,
                       int64_t timestamp,
                       size_t write_touch_count = 1);
class RadixTreeIndex {
public:
    // 新构造函数 (多 tier)
    RadixTreeIndex(const std::string &instance_id,
                   std::vector<std::shared_ptr<EvictionPolicy>> tier_policies,
                   TierWriteMode write_mode = TierWriteMode::WRITE_THROUGH,
                   int64_t default_ttl_ns = 0,
                   size_t selective_write_threshold = 2,
                   bool tier_access_propagation_enabled = true,
                   std::vector<TierFlowStrategy> tier_flow_strategies = {});
    // 兼容构造函数 (单 policy)
    RadixTreeIndex(const std::string &instance_id,
                   const std::shared_ptr<EvictionPolicy> &eviction_policy,
                   int64_t default_ttl_ns = 0);
    RadixTreeIndex();
    ~RadixTreeIndex() = default;

    struct InsertResult {
        std::vector<int64_t> inserted_keys;
        TierFlowRecorder tier_flow;
    };

    // ttl_ns: 0 = 使用 default_ttl_ns_，-1 = 禁用 TTL，>0 = 自定义纳秒
    InsertResult InsertOnly(const std::vector<int64_t> &block_keys, int64_t timestamp, int64_t ttl_ns = 0);
    InsertResult FillPathOnly(const std::vector<int64_t> &block_keys,
                              const std::vector<size_t> &materialized_indices,
                              int64_t timestamp,
                              int64_t ttl_ns = 0);
    void PrefixQuery(const std::vector<int64_t> &block_keys,
                     const BlockMask &block_mask,
                     const int64_t timestamp,
                     QueryHit *query_hit = nullptr,
                     bool refresh_ttl_on_read = true,
                     bool touch_local_hits = true,
                     bool local_hits_are_reads = true);
    void BatchQuery(const std::vector<int64_t> &block_keys,
                    const BlockMask &block_mask,
                    const int64_t timestamp,
                    QueryHit *query_hit = nullptr,
                    bool refresh_ttl_on_read = true,
                    bool touch_local_hits = true,
                    bool local_hits_are_reads = true);
    void TouchKeysAtTier(const std::vector<int64_t> &block_keys,
                         const std::string &tier_name,
                         int64_t timestamp,
                         bool refresh_ttl_on_read);
    size_t PrefixMatchCount(const std::vector<int64_t> &block_keys, int64_t timestamp) const;
    std::vector<int64_t>
    PoolSourceWriteTouchKeysAtLeast(const std::vector<int64_t> &block_keys, size_t threshold, int64_t timestamp) const;
    size_t PoolSourceTierIndex() const { return tier_names_.empty() ? 0 : tier_names_.size() - 1; }
    const std::string &PoolSourceTierName() const { return tier_names_.at(PoolSourceTierIndex()); }

    void CleanEmptyBlocks(const std::vector<BlockEntry *> &blocks,
                          int64_t eviction_timestamp,
                          bool use_logical_expire_time = false);

    // 清空整个RadixTree的缓存
    void Clear();

    void SetStatsCollector(std::shared_ptr<StatsCollector> collector) { stats_collector_ = collector; }

    const std::vector<std::string> &GetTierNames() const { return tier_names_; }
    std::vector<int64_t> PrefixPathForBlock(const BlockEntry *block) const;

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
    bool ConsumeReadTriggeredTierWrite() {
        bool triggered = read_triggered_tier_write_;
        read_triggered_tier_write_ = false;
        return triggered;
    }
    TierFlowRecorder ConsumeTierFlow() {
        TierFlowRecorder recorder = std::move(current_tier_flow_);
        current_tier_flow_.Clear();
        return recorder;
    }

private:
    std::unique_ptr<RadixTreeNode> root_;
    std::vector<std::shared_ptr<EvictionPolicy>> tier_policies_; // 按 tier 顺序排序
    std::vector<std::string> tier_names_;                        // 缓存 policy name
    std::vector<TierFlowStrategy> tier_flow_strategies_;
    std::unordered_map<int64_t, BlockEntry *> block_index_;
    // 写入流量应落地的 tier 数，构造时结合相邻 tier flow 一次性确定
    // WRITE_THROUGH=全部层，CASCADING/WRITE_THROUGH_SELECTIVE=仅 tier 0（单层退化为全部）
    size_t write_tier_count_ = 0;
    std::string instance_id_;
    int64_t default_ttl_ns_ = 0;
    std::shared_ptr<StatsCollector> stats_collector_;
    bool read_triggered_tier_write_ = false;
    TierFlowRecorder current_tier_flow_;

private:
    std::vector<BlockEntry *> AppendPathBlocks(RadixTreeNode *node,
                                               const std::vector<int64_t> &block_keys,
                                               int64_t timestamp,
                                               int64_t ttl_ns,
                                               bool count_new_tier_write_touch,
                                               const std::vector<bool> *materialized_blocks,
                                               size_t materialized_offset);

    InsertResult InsertNode(RadixTreeNode *node,
                            const std::vector<int64_t> &block_keys,
                            int64_t timestamp,
                            int64_t ttl_ns,
                            bool touch_existing,
                            bool count_new_tier_write_touch,
                            const std::vector<bool> *materialized_blocks = nullptr,
                            size_t materialized_offset = 0);
    void SplitNode(RadixTreeNode *existing_node,
                   size_t split_pos,
                   const std::vector<int64_t> &remaining_keys,
                   int64_t timestamp,
                   int64_t ttl_ns,
                   bool count_new_tier_write_touch,
                   const std::vector<bool> *materialized_blocks = nullptr,
                   size_t materialized_offset = 0);

    void WriteToTier(RadixTreeNode *node,
                     const std::vector<int64_t> &block_keys,
                     int64_t timestamp,
                     int64_t ttl_ns,
                     bool count_new_tier_write_touch,
                     const std::vector<bool> *materialized_blocks = nullptr,
                     size_t materialized_offset = 0);

    bool MaterializeExistingBlockOnWrite(BlockEntry *block,
                                         int64_t timestamp,
                                         int64_t ttl_ns,
                                         bool count_new_tier_write_touch);
    void RefreshExistingBlockOnWrite(BlockEntry *block, int64_t timestamp);
    void TouchExistingTierOnWrite(BlockEntry *block, size_t tier_idx, int64_t timestamp, bool count_write_touch);
    void PlaceBlockOnWriteTiers(BlockEntry *block, int64_t timestamp, bool count_write_touch);
    void RecordTierEnter(BlockEntry *block,
                         size_t tier_idx,
                         const std::string &from_tier,
                         TierFlowEventReason reason,
                         int64_t timestamp);
    void RegisterBlockToWriteTier(BlockEntry *block, size_t tier_idx);
    void RegisterBlocksToWriteTiers(const std::vector<BlockEntry *> &blocks);
    void TouchTierLocation(BlockEntry *block,
                           size_t tier_idx,
                           int64_t timestamp,
                           bool refresh_ttl_on_read,
                           bool update_writing_time,
                           bool increase_access_count);
    bool ShouldPropagateReadAcrossEdge(size_t edge_idx) const;
    bool ShouldPropagateWriteAcrossEdge(size_t edge_idx) const;
    bool IsWriteThroughEdge(size_t edge_idx) const;
    void OnBlockAccessed(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read = true);
    void TouchBlock(BlockEntry *block, int64_t timestamp);
    void TouchBlockLocations(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read, bool count_read);
    bool IsBlockEvict(const BlockEntry *block, int64_t timestamp) const;

    // per-tier 命中检测辅助方法
    void RecordTieredHit(BlockEntry *block, size_t block_idx, bool is_remote, QueryHit *query_hit) const;
    void PromoteToHigherTiers(BlockEntry *block, int64_t timestamp);
    void MaybeSelectiveWriteToNextTier(BlockEntry *block, size_t tier_idx, int64_t timestamp);
    void SelectiveWriteToNextTier(BlockEntry *block, size_t hit_tier_idx, int64_t timestamp);
    bool AppendBlockToTierAndWriteThrough(BlockEntry *block, size_t tier_idx, int64_t timestamp);
    void InitTierFlowStrategies(TierWriteMode write_mode,
                                size_t selective_write_threshold,
                                bool tier_access_propagation_enabled,
                                std::vector<TierFlowStrategy> tier_flow_strategies);
};
} // namespace kv_cache_manager
