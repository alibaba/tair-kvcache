#include "kv_cache_manager/affinity/frequency_sketch.h"

#include <limits>
#include <chrono>

namespace kv_cache_manager {

int64_t FrequencySketch::Now() const {
    return clock_ ? clock_() : std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint32_t FrequencySketch::DecayedCount(const Entry &entry, int64_t now, uint32_t half_life_ms) {
    if (entry.half_life_ms != half_life_ms) return 0; // Config changed: rebuild evidence.
    if (half_life_ms == 0 || now <= entry.epoch_ms) return entry.count;
    const auto periods = (now - entry.epoch_ms) / half_life_ms;
    return periods >= 32 ? 0 : entry.count >> periods;
}

void FrequencySketch::TouchLocked(const Key &k) {
    auto it = table_.find(k);
    if (it == table_.end()) {
        return;
    }
    lru_.erase(it->second.lru_it);
    lru_.push_front(k);
    it->second.lru_it = lru_.begin();
}

void FrequencySketch::EvictIfFullLocked() {
    while (table_.size() > capacity_) {
        const Key &victim = lru_.back();
        table_.erase(victim);
        lru_.pop_back();
    }
}

void FrequencySketch::Observe(const std::string &caller_node_id, int64_t block_key, const std::string &instance_id, uint32_t half_life_ms) {
    if (caller_node_id.empty()) {
        return; // 空 caller 不参与 F3
    }
    Key k{instance_id, caller_node_id, block_key};
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(k);
    if (it == table_.end()) {
        lru_.push_front(k);
        Entry e;
        e.count = 1;
        e.epoch_ms = Now();
        e.half_life_ms = half_life_ms;
        e.lru_it = lru_.begin();
        table_.emplace(std::move(k), std::move(e));
        EvictIfFullLocked();
    } else {
        const auto now = Now();
        auto &entry = it->second;
        entry.count = DecayedCount(entry, now, half_life_ms);
        if (entry.half_life_ms != half_life_ms || half_life_ms == 0) entry.epoch_ms = now;
        else if (now > entry.epoch_ms) entry.epoch_ms += ((now - entry.epoch_ms) / half_life_ms) * half_life_ms;
        entry.half_life_ms = half_life_ms;
        if (it->second.count < std::numeric_limits<uint32_t>::max()) {
            ++it->second.count;
        }
        // 移到 MRU 位置
        lru_.erase(it->second.lru_it);
        lru_.push_front(it->first);
        it->second.lru_it = lru_.begin();
    }
}

uint32_t FrequencySketch::RemoteCount(const std::string &caller_node_id, int64_t block_key,
                                      const std::string &instance_id, uint32_t half_life_ms) const {
    if (caller_node_id.empty()) {
        return 0;
    }
    Key k{instance_id, caller_node_id, block_key};
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(k);
    return it == table_.end() ? 0 : DecayedCount(it->second, Now(), half_life_ms);
}

void FrequencySketch::Reset(const std::string &caller_node_id, int64_t block_key, const std::string &instance_id) {
    Key k{instance_id, caller_node_id, block_key};
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(k);
    if (it == table_.end()) {
        return;
    }
    lru_.erase(it->second.lru_it);
    table_.erase(it);
}

size_t FrequencySketch::Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return table_.size();
}

} // namespace kv_cache_manager
