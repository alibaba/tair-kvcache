#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kv_cache_manager {

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
