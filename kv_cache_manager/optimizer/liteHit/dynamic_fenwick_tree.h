#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kv_cache_manager {

// Append-only Fenwick (binary indexed) tree that backs LiteHit's reuse-distance
// queries. It is an order-statistics view of the global LRU order: each active
// block marks 1 at its most recent position, and range sums count how many
// distinct newer blocks exist. It is not a cache and not an extra LRU chain.
//
// Growth is amortized: AppendZero adds a logical 0 at the end and seeds the new
// tree node from the current prefix sums, so later Add calls keep every existing
// ancestor consistent.
class DynamicFenwickTree {
public:
    DynamicFenwickTree();

    void AppendZero();
    void Add(std::size_t index, int64_t delta);
    uint64_t PrefixSum(std::size_t index) const;
    std::size_t size() const { return tree_.size() - 1; }
    void Clear();
    uint64_t memory_usage_bytes() const;

private:
    std::vector<int64_t> tree_;
};

} // namespace kv_cache_manager
