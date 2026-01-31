#pragma once
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"

namespace kv_cache_manager {

class RandomLruEvictionPolicy : public EvictionPolicy {
private:
    std::string name_;
    std::vector<BlockEntry *> blocks_; // 当前所有块
    std::vector<int64_t> timestamps_;
    std::unordered_map<BlockEntry *, size_t> block_to_index_; // 映射：块 -> index
    uint64_t xor_state_ = 0x12345678ABCDEF01ULL;              // 随机数状态
    RandomLruParams params_;
    size_t sampling_size_;

public:
    explicit RandomLruEvictionPolicy(const std::string &name, const RandomLruParams &params, const int32_t batch_size);
    ~RandomLruEvictionPolicy() override;
    std::string name() const override { return name_; }
    void set_name(const std::string &name) override { name_ = name; }
    void OnBlockWritten(BlockEntry *block) override;

    void OnNodeWritten(std::vector<BlockEntry *> &blocks) override;

    void OnBlockAccessed(BlockEntry *block, int64_t timestamp) override;

    // 驱逐 count 个块（分批，每批按 RandomLRU）
    std::vector<BlockEntry *> EvictBlocks(size_t count) override;
    void Clear() override;
    size_t size() const override { return blocks_.size(); }

private:
    // 随机采样 sample_size 个块
    // 从候选块中选出 k 个 LRU 的块（按 last_access_time 排序）

    std::vector<BlockEntry *> SampleAndPickLru(size_t sample_size, size_t batch);
    // 删除指定 block（O(1）swap-pop）
    void RemoveBlock(BlockEntry *block);

    inline size_t FastRand(size_t mod);
};

} // namespace kv_cache_manager
