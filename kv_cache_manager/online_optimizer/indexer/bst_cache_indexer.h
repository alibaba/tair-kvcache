#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>

#include "kv_cache_manager/online_optimizer/indexer/cache_indexer.h"

namespace kv_cache_manager {

struct TreapNode {
    int64_t key;
    int64_t priority;
    int64_t size;
    TreapNode *left;
    TreapNode *right;

    TreapNode(int64_t k, int64_t p) : key(k), priority(p), size(1), left(nullptr), right(nullptr) {}
};

class AugmentedTreap {
public:
    AugmentedTreap();
    ~AugmentedTreap();

    AugmentedTreap(const AugmentedTreap &) = delete;
    AugmentedTreap &operator=(const AugmentedTreap &) = delete;

    void Insert(int64_t key);
    void Erase(int64_t key);

    int64_t CountGreater(int64_t key) const;
    int64_t Min() const;

    int64_t Size() const;

private:
    static int64_t GetSize(TreapNode *node);
    static void UpdateSize(TreapNode *node);
    static void SplitByKey(TreapNode *node, int64_t key, TreapNode *&left, TreapNode *&right);
    static TreapNode *Merge(TreapNode *left, TreapNode *right);
    static void DestroyTree(TreapNode *node);

    TreapNode *root_;
    std::mt19937_64 rng_;
};

class BSTCacheIndexer : public CacheIndexer {
public:
    explicit BSTCacheIndexer(int64_t max_key_count = 0);

    void Init(const std::vector<double> &capacity_gb,
              int64_t size_full_only,
              int64_t size_full_linear,
              int32_t linear_step) override;

    void ProcessKeys(const std::vector<int64_t> &keys,
                    std::vector<int64_t> &hit_count,
                    int64_t &max_hit_count) override;
    int64_t unique_count() const override;
    int64_t eviction_count() const override { return eviction_count_; }
    int64_t memory_usage_bytes() const override;
    int64_t kv_cache_usage_bytes() const override;

    void PostQueryMaintenance() override;
    bool RemoveKey(int64_t key) override;

    // Exposed for testing: compute stack distance for a single key.
    int64_t ComputeStackDistance(int64_t key);

private:
    void EvictIfExceedsCapacity();
    void DoEvictOne();

    int64_t max_key_count_;
    int64_t avg_bytes_per_block_ = 0;
    std::vector<int64_t> capacity_blocks_;
    AugmentedTreap treap_;
    std::unordered_map<int64_t, int64_t> last_access_;
    std::unordered_map<int64_t, int64_t> reverse_map_;
    int64_t logical_time_ = 0;
    int64_t eviction_count_ = 0;
};

} // namespace kv_cache_manager
