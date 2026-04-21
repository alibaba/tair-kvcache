#pragma once
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/eviction_policy/common_structure.h"

namespace kv_cache_manager {

// ============================================================
//  TtlEvictionPolicy — 纯时间维度驱逐
//  只回收已过期的 block，未过期的绝不触碰
// ============================================================
class TtlEvictionPolicy : public EvictionPolicy {
public:
    TtlEvictionPolicy(const std::string &name, bool fallback_on_pressure = true);
    ~TtlEvictionPolicy() override;

    std::string name() const override { return name_; }
    void set_name(const std::string &name) override { name_ = name; }
    size_t size() const override { return node_map_.size(); }

    void OnBlockWritten(BlockEntry *block) override;
    void OnNodeWritten(std::vector<BlockEntry *> &blocks) override;
    void OnBlockAccessed(BlockEntry *block, int64_t timestamp) override;
    std::vector<BlockEntry *> EvictBlocks(size_t count) override;
    void Clear() override;

private:
    void EvictOne(BlockEntry *block);

    struct ListNode : public LinkedListNode {
        BlockEntry *payload_ = nullptr;
    };

    std::string name_;
    bool fallback_on_pressure_;
    LinkedList list_;
    std::unordered_map<BlockEntry *, ListNode *> node_map_;
    int64_t last_known_timestamp_ = 0;
};

} // namespace kv_cache_manager
