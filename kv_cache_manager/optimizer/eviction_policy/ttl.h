#pragma once
#include <cstdint>
#include <queue>
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

    size_t size() const override { return node_map_.size(); }

    void OnBlockWritten(BlockEntry *block) override;
    void OnNodeWritten(std::vector<BlockEntry *> &blocks) override;
    void OnBlockAccessedWithOptions(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read) override;
    std::vector<BlockEntry *> EvictBlocks(size_t count) override;
    std::vector<BlockEntry *> EvictExpired() override;
    void Clear() override;
    bool NeedCapacityEviction() const override { return fallback_on_pressure_; }
    void AdvanceClock(int64_t timestamp) override;

private:
    // NVI hook：外部只能通过 OnBlockAccessedWithOptions 调用，此处仅供基类默认实现 dispatch。
    void OnBlockAccessed(BlockEntry *block, int64_t timestamp) override;
    void EvictOne(BlockEntry *block);
    void PushExpireEvent(BlockEntry *block);
    bool TryPopOneExpired(BlockEntry *&expired_block);
    std::vector<BlockEntry *> HarvestExpiredBlocks();
    void MaybeCompactExpireHeap();
    void RebuildExpireHeap();

    struct ExpireEvent {
        int64_t expire_ts = 0;
        uint64_t version = 0;
        BlockEntry *block = nullptr;
    };

    struct ExpireEventCompare {
        bool operator()(const ExpireEvent &a, const ExpireEvent &b) const { return a.expire_ts > b.expire_ts; }
    };

    struct ListNode : public LinkedListNode {
        BlockEntry *payload_ = nullptr;
    };

    bool fallback_on_pressure_;
    LinkedList list_;
    std::unordered_map<BlockEntry *, ListNode *> node_map_;
    std::priority_queue<ExpireEvent, std::vector<ExpireEvent>, ExpireEventCompare> expire_min_heap_;
    std::unordered_map<BlockEntry *, uint64_t> expire_event_version_;
    int64_t last_known_timestamp_ = 0;
};

} // namespace kv_cache_manager
