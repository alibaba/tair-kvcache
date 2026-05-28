#pragma once
#include <unordered_map>
#include <unordered_set>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/eviction_policy/common_structure.h"
namespace kv_cache_manager {

class LeafAwareLruEvictionPolicy : public EvictionPolicy {
private:
    LruParams params_;
    struct LeafLRUListNode : public LinkedListNode {
        BlockEntry *payload_;
    };
    LinkedList leaf_lru_list_;
    std::unordered_map<BlockEntry *, LeafLRUListNode *> node_map_;
    std::unordered_set<BlockEntry *> leaf_blocks_; // 跟踪在 leaf_lru_list_ 中的 block

public:
    explicit LeafAwareLruEvictionPolicy(const std::string &name, const LruParams &params);
    ~LeafAwareLruEvictionPolicy() override;
    void OnBlockWritten(BlockEntry *block) override;
    void OnNodeWritten(std::vector<BlockEntry *> &blocks) override;
    std::vector<BlockEntry *> EvictBlocks(size_t count) override;
    void Clear() override;
    size_t size() const override { return node_map_.size(); }

private:
    // NVI hook：外部统一通过基类 OnBlockAccessedWithOptions 入口调用。
    void OnBlockAccessed(BlockEntry *block, int64_t timestamp) override;
    void insert_sorted_by_priority(LeafLRUListNode *node);
    // 返回 true 表示真正处理了（是 data leaf 且插入了 blocks）
    bool UpdateNodeState(RadixTreeNode *node);
};
} // namespace kv_cache_manager