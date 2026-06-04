#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace kv_cache_manager {

class HintSuppressor {
public:
    // 返回微秒时间戳。nullptr ⇒ TimestampUtil::GetCurrentTimeUs。
    using ClockFn = std::function<int64_t()>;

    explicit HintSuppressor(size_t capacity = 100000, ClockFn clock = nullptr);

    // window_ms == 0 ⇒ 关闭抑制（始终放行）。target_node 空字符串 ⇒ 视为无效，
    // 返回 true 但不记录条目。
    bool TryEmit(int64_t block_key, const std::string &target_node, uint32_t window_ms);

    size_t Size() const;

private:
    using Key = std::pair<int64_t, std::string>;
    struct KeyHash {
        size_t operator()(const Key &k) const noexcept {
            return std::hash<int64_t>{}(k.first) ^ (std::hash<std::string>{}(k.second) * 0x9e3779b97f4a7c15ULL);
        }
    };
    struct Entry {
        int64_t last_emit_us = 0;
        std::list<Key>::iterator lru_it;
    };

    void EvictIfFullLocked();
    int64_t Now() const;

    mutable std::mutex mu_;
    size_t capacity_;
    ClockFn clock_;
    std::unordered_map<Key, Entry, KeyHash> table_;
    std::list<Key> lru_; // MRU at front, LRU at back
};

} // namespace kv_cache_manager
