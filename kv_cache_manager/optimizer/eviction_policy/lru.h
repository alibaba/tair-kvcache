#pragma once
#include <climits>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/eviction_policy/common_structure.h"

namespace kv_cache_manager {

class LruEvictionPolicy : public EvictionPolicy {
private:
    int32_t shard_count_ = 1;
    int32_t sample_times_ = 1;
    double amplification_factor_ = 1.0;

    struct LRUListNode : public LinkedListNode {
        BlockEntry *payload_;
    };

    std::vector<LinkedList> shard_lists_;
    std::unordered_map<BlockEntry *, LRUListNode *> node_map_;

    int32_t GetShardIndex(BlockEntry *block) const;
    int64_t GetShardTailTime(int32_t shard_index) const;

    struct CandidateEntry {
        BlockEntry *block;
        LRUListNode *node;
        int32_t shard_index;
    };

    void SampleFromShard(int32_t shard_index, size_t count, std::vector<CandidateEntry> &candidates);
    void ReturnCandidates(const std::vector<CandidateEntry> &candidates);
    void CommitEviction(const std::vector<CandidateEntry> &candidates, std::vector<BlockEntry *> &evicted_blocks);

    // NVI hook：外部统一通过基类 OnBlockAccessedWithOptions 入口调用。
    void OnBlockAccessed(BlockEntry *block, int64_t timestamp) override;

public:
    explicit LruEvictionPolicy(const std::string &name, const LruParams &params);
    ~LruEvictionPolicy() override;
    void OnBlockWritten(BlockEntry *block) override;
    void OnNodeWritten(std::vector<BlockEntry *> &blocks) override;
    std::vector<BlockEntry *> EvictBlocks(size_t count) override;
    void Clear() override;
    size_t size() const override { return node_map_.size(); }
};
} // namespace kv_cache_manager
