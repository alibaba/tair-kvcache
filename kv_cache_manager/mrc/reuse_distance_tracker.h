#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace kv_cache_manager {

// Exact LRU stack distance tracker (Olken's algorithm).
// Maintains a Fenwick tree (binary indexed tree) over logical access
// positions plus a key -> last position map. Each Access() returns the
// number of distinct keys touched since the key's previous access in
// O(log n); an LRU cache of capacity C blocks hits iff distance < C.
// Position space is compacted in place once exhausted, so memory stays
// proportional to the live key count regardless of stream length.
class ReuseDistanceTracker {
public:
    explicit ReuseDistanceTracker(int64_t initial_capacity = 1024);

    ReuseDistanceTracker(const ReuseDistanceTracker &) = delete;
    ReuseDistanceTracker &operator=(const ReuseDistanceTracker &) = delete;

    // Record one access. Returns the LRU stack distance (distinct keys
    // accessed since this key's last access), or -1 for a first-time access.
    int64_t Access(int64_t key);

    // Remove a key from the stack (used by the sampler when tightening its
    // threshold). Returns true if the key was tracked.
    bool Erase(int64_t key);

    int64_t size() const { return static_cast<int64_t>(last_pos_.size()); }
    int64_t memory_usage_bytes() const;

private:
    void FenwickAdd(int64_t pos, int64_t delta);
    int64_t FenwickPrefixSum(int64_t pos) const;
    // Reassign live keys to positions 1..n, growing the tree when more than
    // half of the position space is still live.
    void Compact();

    int64_t capacity_ = 0;                        // usable position slots
    int64_t next_pos_ = 1;                        // next free position (1-based)
    std::vector<int64_t> tree_;                   // Fenwick tree, size capacity_ + 1
    std::unordered_map<int64_t, int64_t> last_pos_; // key -> position
};

} // namespace kv_cache_manager
