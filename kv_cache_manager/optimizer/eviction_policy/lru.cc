#include "kv_cache_manager/optimizer/eviction_policy/lru.h"

#include <algorithm>

namespace kv_cache_manager {

LruEvictionPolicy::LruEvictionPolicy(const std::string &name, const LruParams &params)
    : EvictionPolicy(name)
    , shard_count_(params.shard_count > 0 ? params.shard_count : 1)
    , sample_times_(params.sample_times > 0 ? params.sample_times : 1)
    , amplification_factor_(params.eviction_amplification_factor > 1.0 ? params.eviction_amplification_factor : 1.0)
    , shard_lists_(shard_count_) {}

LruEvictionPolicy::~LruEvictionPolicy() {
    for (auto &shard_list : shard_lists_) {
        shard_list.clear();
    }
    node_map_.clear();
}

int32_t LruEvictionPolicy::GetShardIndex(BlockEntry *block) const {
    return static_cast<int32_t>(static_cast<uint64_t>(block->key) % shard_count_);
}

int64_t LruEvictionPolicy::GetShardTailTime(int32_t shard_index) const {
    LinkedListNode *tail = shard_lists_[shard_index].getTail();
    if (tail == nullptr) {
        return INT64_MAX;
    }
    auto *lru_node = static_cast<const LRUListNode *>(tail);
    return lru_node->payload_ ? GetTierAccessTime(lru_node->payload_) : INT64_MAX;
}

void LruEvictionPolicy::OnBlockWritten(BlockEntry *block) {
    if (block == nullptr) {
        return;
    }
    int32_t shard_index = GetShardIndex(block);
    auto *node = new LRUListNode();
    node->payload_ = block;
    shard_lists_[shard_index].push_front(node);
    node_map_[block] = node;
}

void LruEvictionPolicy::OnNodeWritten(std::vector<BlockEntry *> &blocks) {
    for (auto *block : blocks) {
        OnBlockWritten(block);
    }
}

void LruEvictionPolicy::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    auto it = node_map_.find(block);
    if (it == node_map_.end()) {
        return;
    }
    // block 级 / tier 级 last_access_time 由 RadixTreeIndex::OnBlockAccessed 统一写入；
    // 本函数仅负责将 block 在 LRU 链表中的物理位置刷新到头部
    LRUListNode *node = it->second;
    shard_lists_[GetShardIndex(block)].move_to_front(node);
}

void LruEvictionPolicy::SampleFromShard(int32_t shard_index, size_t count, std::vector<CandidateEntry> &candidates) {
    LinkedList &shard_list = shard_lists_[shard_index];
    for (size_t i = 0; i < count; ++i) {
        if (shard_list.empty()) {
            break;
        }
        LinkedListNode *tail_node = shard_list.getTail();
        if (tail_node == nullptr) {
            break;
        }
        auto *lru_node = static_cast<LRUListNode *>(tail_node);
        if (lru_node->payload_ == nullptr) {
            shard_list.remove(tail_node);
            continue;
        }
        shard_list.unlink(tail_node);
        candidates.push_back({lru_node->payload_, lru_node, shard_index});
    }
}

void LruEvictionPolicy::ReturnCandidates(const std::vector<CandidateEntry> &candidates) {
    for (const auto &entry : candidates) {
        shard_lists_[entry.shard_index].push_back(entry.node);
    }
}

void LruEvictionPolicy::CommitEviction(const std::vector<CandidateEntry> &candidates,
                                       std::vector<BlockEntry *> &evicted_blocks) {
    for (const auto &entry : candidates) {
        BlockEntry *block = entry.block;
        evicted_blocks.push_back(block);
        node_map_.erase(block);
        ClearBlockLocation(block);
        delete entry.node;
    }
}

std::vector<BlockEntry *> LruEvictionPolicy::EvictBlocks(size_t count) {
    std::vector<BlockEntry *> evicted_blocks;
    if (node_map_.empty() || count == 0) {
        return evicted_blocks;
    }

    size_t amplified_count = static_cast<size_t>(amplification_factor_ * count);
    amplified_count = std::max(amplified_count, count);

    int32_t num_rounds = std::min(sample_times_, shard_count_);
    size_t per_shard_count = (amplified_count + num_rounds - 1) / num_rounds;

    std::vector<CandidateEntry> candidates;
    for (int32_t round = 0; round < num_rounds; ++round) {
        int32_t oldest_shard = -1;
        int64_t oldest_time = INT64_MAX;
        for (int32_t s = 0; s < shard_count_; ++s) {
            if (shard_lists_[s].empty()) {
                continue;
            }
            int64_t tail_time = GetShardTailTime(s);
            if (tail_time < oldest_time) {
                oldest_time = tail_time;
                oldest_shard = s;
            }
        }
        if (oldest_shard < 0) {
            break;
        }
        SampleFromShard(oldest_shard, per_shard_count, candidates);
        if (candidates.size() >= amplified_count) {
            break;
        }
    }

    if (candidates.empty()) {
        return evicted_blocks;
    }

    if (candidates.size() <= count) {
        CommitEviction(candidates, evicted_blocks);
    } else {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + count,
                          candidates.end(),
                          [this](const CandidateEntry &a, const CandidateEntry &b) {
                              return GetTierAccessTime(a.block) < GetTierAccessTime(b.block);
                          });

        std::vector<CandidateEntry> to_evict(candidates.begin(), candidates.begin() + count);
        std::vector<CandidateEntry> to_return(candidates.begin() + count, candidates.end());

        CommitEviction(to_evict, evicted_blocks);
        ReturnCandidates(to_return);
    }

    return evicted_blocks;
}

void LruEvictionPolicy::Clear() {
    for (auto &[block, node] : node_map_) {
        ClearBlockLocation(block);
    }
    // 清空LRU链表和映射
    for (auto &shard_list : shard_lists_) {
        shard_list.clear();
    }
    node_map_.clear();
}

} // namespace kv_cache_manager
