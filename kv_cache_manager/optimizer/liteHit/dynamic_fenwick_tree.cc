#include "kv_cache_manager/optimizer/liteHit/dynamic_fenwick_tree.h"

#include <limits>

namespace kv_cache_manager {

namespace {

uint64_t SaturatingMultiply(uint64_t lhs, uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs * rhs;
}

} // namespace

DynamicFenwickTree::DynamicFenwickTree() : tree_(1, 0) {}

void DynamicFenwickTree::AppendZero() {
    const std::size_t new_index = tree_.size();
    const std::size_t low_bit = new_index & (~new_index + 1);
    const uint64_t range_value = PrefixSum(new_index - 1) - PrefixSum(new_index - low_bit);
    tree_.push_back(static_cast<int64_t>(range_value));
}

void DynamicFenwickTree::Add(std::size_t index, int64_t delta) {
    while (index > 0 && index < tree_.size()) {
        tree_[index] += delta;
        index += index & (~index + 1);
    }
}

uint64_t DynamicFenwickTree::PrefixSum(std::size_t index) const {
    if (index >= tree_.size()) {
        index = tree_.size() - 1;
    }
    int64_t result = 0;
    while (index > 0) {
        result += tree_[index];
        index -= index & (~index + 1);
    }
    return static_cast<uint64_t>(result);
}

void DynamicFenwickTree::Clear() {
    std::vector<int64_t> empty_tree(1, 0);
    tree_.swap(empty_tree);
}

uint64_t DynamicFenwickTree::memory_usage_bytes() const {
    return SaturatingMultiply(static_cast<uint64_t>(tree_.capacity()), sizeof(tree_[0]));
}

} // namespace kv_cache_manager
