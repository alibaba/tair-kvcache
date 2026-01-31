#include "kv_cache_manager/optimizer/eviction_policy/random_lru.h"

namespace kv_cache_manager {

RandomLruEvictionPolicy::RandomLruEvictionPolicy(const std::string &name,
                                                 const RandomLruParams &params,
                                                 const int32_t batch_size)
    : name_(name) {
    auto sample_rate = params.sample_rate;
    if (sample_rate <= 0.0) {
        sample_rate = 0.1;
    }
    sampling_size_ = static_cast<size_t>(batch_size / sample_rate);
}
RandomLruEvictionPolicy::~RandomLruEvictionPolicy() { block_to_index_.clear(); }

void RandomLruEvictionPolicy::OnBlockWritten(BlockEntry *block) {
    blocks_.push_back(block);
    timestamps_.push_back(block->last_access_time);
    block_to_index_[block] = blocks_.size() - 1;
}
void RandomLruEvictionPolicy::OnNodeWritten(std::vector<BlockEntry *> &blocks) {
    for (auto *block : blocks) {
        OnBlockWritten(block);
    }
}
void RandomLruEvictionPolicy::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    auto it = block_to_index_.find(block);
    if (it != block_to_index_.end()) {

        block->last_access_time = timestamp;
        block->access_count += 1;
        timestamps_[it->second] = timestamp;
    }
}
// 驱逐 count 个块（分批，每批按 RandomLRU）
std::vector<BlockEntry *> RandomLruEvictionPolicy::EvictBlocks(size_t count) {
    std::vector<BlockEntry *> evicted;
    if (blocks_.empty())
        return evicted;
    size_t total_to_evict = std::min(count, blocks_.size());

    auto victims = SampleAndPickLru(sampling_size_, total_to_evict);
    for (auto *victim : victims) {
        if (block_to_index_.count(victim)) {
            evicted.push_back(victim);
            RemoveBlock(victim);
            if (evicted.size() == total_to_evict) {
                return evicted;
            }
        }
    }
    return evicted;
}

std::vector<BlockEntry *> RandomLruEvictionPolicy::SampleAndPickLru(size_t sample_size, size_t k) {
    size_t n = blocks_.size();
    if (n == 0)
        return {};

    sample_size = std::min(sample_size, n);
    k = std::min(k, sample_size);

    // 采样
    std::vector<size_t> indices;
    indices.reserve(sample_size);
    for (size_t i = 0; i < sample_size; ++i) {
        indices.push_back(FastRand(n));
    }

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [this](size_t a, size_t b) {
        return timestamps_[a] < timestamps_[b];
    });

    // 转换成指针
    std::vector<BlockEntry *> result;
    result.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        result.push_back(blocks_[indices[i]]);
    }

    return result;
}
void RandomLruEvictionPolicy::RemoveBlock(BlockEntry *block) {
    auto it = block_to_index_.find(block);
    if (it != block_to_index_.end()) {
        size_t index = it->second;
        BlockEntry *last_block = blocks_.back();
        blocks_[index] = last_block;
        timestamps_[index] = timestamps_.back();
        block_to_index_[last_block] = index;
        blocks_.pop_back();
        timestamps_.pop_back();
        block_to_index_.erase(it);
        if (name_ == "shared") {
            // 全局驱逐时，清空所有location信息
            block->location_map.clear();
        } else {
            // 分层驱逐时，仅清除当前tier的location信息
            block->location_map.erase(name_); // 清空block的location信息
        }
    }
}

void RandomLruEvictionPolicy::Clear() {
    // 清空所有blocks的location信息
    for (auto *block : blocks_) {
        if (name_ == "shared") {
            // 全局驱逐时，清空所有location信息
            block->location_map.clear();
        } else {
            // 分层驱逐时，仅清除当前tier的location信息
            block->location_map.erase(name_);
        }
    }
    // 清空所有容器
    blocks_.clear();
    timestamps_.clear();
    block_to_index_.clear();
}

inline size_t RandomLruEvictionPolicy::FastRand(size_t mod) {
    if (mod == 0)
        return 0;
    uint64_t x = xor_state_;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xor_state_ = x;
    return x % mod;
}
} // namespace kv_cache_manager