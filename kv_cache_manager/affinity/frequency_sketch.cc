#include "kv_cache_manager/affinity/frequency_sketch.h"

namespace kv_cache_manager {

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

void FrequencySketch::Observe(const std::string &caller_node_ip, int64_t block_key) {
    if (caller_node_ip.empty()) {
        return; // 空 caller 不参与 F3
    }
    Key k{caller_node_ip, block_key};
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(k);
    if (it == table_.end()) {
        lru_.push_front(k);
        Entry e;
        e.count = 1;
        e.lru_it = lru_.begin();
        table_.emplace(std::move(k), std::move(e));
        EvictIfFullLocked();
    } else {
        ++it->second.count;
        // 移到 MRU 位置
        lru_.erase(it->second.lru_it);
        lru_.push_front(it->first);
        it->second.lru_it = lru_.begin();
    }
}

uint32_t FrequencySketch::RemoteCount(const std::string &caller_node_ip, int64_t block_key) const {
    if (caller_node_ip.empty()) {
        return 0;
    }
    Key k{caller_node_ip, block_key};
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(k);
    return it == table_.end() ? 0 : it->second.count;
}

void FrequencySketch::Reset(const std::string &caller_node_ip, int64_t block_key) {
    Key k{caller_node_ip, block_key};
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
