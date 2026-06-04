#include "kv_cache_manager/affinity/hint_suppressor.h"

#include "kv_cache_manager/common/timestamp_util.h"

namespace kv_cache_manager {

HintSuppressor::HintSuppressor(size_t capacity, ClockFn clock)
    : capacity_(capacity == 0 ? 1 : capacity), clock_(std::move(clock)) {}

int64_t HintSuppressor::Now() const { return clock_ ? clock_() : TimestampUtil::GetCurrentTimeUs(); }

void HintSuppressor::EvictIfFullLocked() {
    while (table_.size() > capacity_) {
        const Key &victim = lru_.back();
        table_.erase(victim);
        lru_.pop_back();
    }
}

bool HintSuppressor::TryEmit(int64_t block_key, const std::string &target_node, uint32_t window_ms) {
    if (target_node.empty()) {
        return true;
    }
    Key k{block_key, target_node};
    const int64_t now = Now();
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(k);
    if (it != table_.end()) {
        if (window_ms > 0) {
            const int64_t window_us = static_cast<int64_t>(window_ms) * 1000;
            if (now - it->second.last_emit_us < window_us) {
                return false;
            }
        }
        it->second.last_emit_us = now;
        lru_.erase(it->second.lru_it);
        lru_.push_front(it->first);
        it->second.lru_it = lru_.begin();
        return true;
    }
    lru_.push_front(k);
    Entry e;
    e.last_emit_us = now;
    e.lru_it = lru_.begin();
    table_.emplace(std::move(k), std::move(e));
    EvictIfFullLocked();
    return true;
}

size_t HintSuppressor::Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return table_.size();
}

} // namespace kv_cache_manager
