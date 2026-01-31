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
#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_trace.h"

namespace kv_cache_manager {
void AppendBlockLocation(BlockEntry *block, const std::string &unique_name, int64_t timestamp);
class RadixTreeIndex {
public:
    RadixTreeIndex(const std::string &instance_id, const std::shared_ptr<EvictionPolicy> &eviction_policy);
    RadixTreeIndex();
    ~RadixTreeIndex() = default;

    std::vector<int64_t> InsertOnly(const std::vector<int64_t> &block_keys, const int64_t timestamp);
    void PrefixQuery(const std::vector<int64_t> &block_keys,
                     const BlockMask &block_mask,
                     const int64_t timestamp,
                     std::vector<std::vector<int64_t>> &external_hits,
                     std::vector<std::vector<int64_t>> &internal_hits);

    std::vector<int64_t> InsertWithQuery(const std::vector<int64_t> &block_keys,
                                         const int64_t timestamp,
                                         std::vector<std::vector<int64_t>> &hits);
    void CleanEmptyBlocks(const std::vector<BlockEntry *> &blocks);

    // 清空整个RadixTree的缓存
    void Clear();

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

private:
    std::unique_ptr<RadixTreeNode> root_;
    std::shared_ptr<EvictionPolicy> eviction_policy_;
    std::string instance_id_; // 添加实例ID记录

private:
    std::vector<BlockEntry *>
    AppendNewBlocks(RadixTreeNode *node, const std::vector<int64_t> &block_keys, const int64_t timestamp);

    std::vector<int64_t>
    InsertNode(RadixTreeNode *node, const std::vector<int64_t> &block_keys, const int64_t timestamp);
    void SplitNode(RadixTreeNode *existing_node,
                   size_t split_pos,
                   const std::vector<int64_t> &remaining_keys,
                   int64_t timestamp);
    std::vector<int64_t> InsertQuery(RadixTreeNode *node,
                                     const std::vector<int64_t> &block_keys,
                                     const int64_t timestamp,
                                     bool is_prefix_hit,
                                     std::vector<std::vector<int64_t>> &hits);

    using WriteModify = std::function<std::vector<BlockEntry *>(const std::vector<int64_t> &, int64_t)>;
    WriteModify AppendEvictBlocks(std::unordered_map<int64_t, BlockEntry *> blocks_map);

    void
    WriteToTier(RadixTreeNode *node, const std::vector<int64_t> &block_keys, const int64_t timestamp, WriteModify cb);

    void OnBlockAccessed(BlockEntry *block, int64_t timestamp);
    bool IsBlockEvict(BlockEntry *block) const;
};
} // namespace kv_cache_manager