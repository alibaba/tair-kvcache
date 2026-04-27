#include "kv_cache_manager/optimizer/eviction_policy/ttl.h"

namespace kv_cache_manager {

TtlEvictionPolicy::TtlEvictionPolicy(const std::string &name, bool fallback_on_pressure)
    : name_(name), fallback_on_pressure_(fallback_on_pressure) {}

TtlEvictionPolicy::~TtlEvictionPolicy() {
    list_.clear();
    node_map_.clear();
}

void TtlEvictionPolicy::OnBlockWritten(BlockEntry *block) {
    if (!block) {
        return;
    }
    auto *node = new ListNode();
    node->payload_ = block;
    list_.push_front(node);
    node_map_[block] = node;
    if (block->last_access_time > last_known_timestamp_) {
        last_known_timestamp_ = block->last_access_time;
    }
    block->ttl_anchor_time = block->last_access_time;
    PushExpireEvent(block);
}

void TtlEvictionPolicy::OnNodeWritten(std::vector<BlockEntry *> &blocks) {
    for (auto *block : blocks) {
        OnBlockWritten(block);
    }
}

void TtlEvictionPolicy::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    OnBlockAccessedWithOptions(block, timestamp, true);
}

void TtlEvictionPolicy::OnBlockAccessedWithOptions(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read) {
    auto it = node_map_.find(block);
    if (it == node_map_.end()) {
        return;
    }
    block->last_access_time = timestamp;
    block->access_count += 1;
    if (timestamp > last_known_timestamp_) {
        last_known_timestamp_ = timestamp;
    }
    list_.move_to_front(it->second);
    if (!refresh_ttl_on_read) {
        return;
    }
    block->ttl_anchor_time = timestamp;
    PushExpireEvent(block);
}

void TtlEvictionPolicy::AdvanceClock(int64_t timestamp) {
    if (timestamp > last_known_timestamp_) {
        last_known_timestamp_ = timestamp;
    }
}

void TtlEvictionPolicy::PushExpireEvent(BlockEntry *block) {
    if (!block || block->ttl_us <= 0) {
        return;
    }
    auto &version = expire_event_version_[block];
    ++version;
    expire_min_heap_.push(ExpireEvent{block->ttl_anchor_time + block->ttl_us, version, block});
    MaybeCompactExpireHeap();
}

bool TtlEvictionPolicy::TryPopOneExpired(BlockEntry *&expired_block) {
    expired_block = nullptr;
    while (!expire_min_heap_.empty()) {
        auto event = expire_min_heap_.top();
        if (event.expire_ts >= last_known_timestamp_) {
            return false;
        }
        expire_min_heap_.pop();

        auto node_it = node_map_.find(event.block);
        if (node_it == node_map_.end()) {
            continue;
        }
        auto version_it = expire_event_version_.find(event.block);
        if (version_it == expire_event_version_.end() || version_it->second != event.version) {
            continue;
        }
        // 防 ABA：即使指针和版本碰巧复用，也必须满足“当前时间已过期”才允许淘汰。
        if (!event.block->IsExpired(last_known_timestamp_)) {
            continue;
        }
        expired_block = event.block;
        return true;
    }
    return false;
}

std::vector<BlockEntry *> TtlEvictionPolicy::HarvestExpiredBlocks() {
    std::vector<BlockEntry *> evicted;
    BlockEntry *expired_block = nullptr;
    while (TryPopOneExpired(expired_block)) {
        auto node_it = node_map_.find(expired_block);
        if (node_it == node_map_.end()) {
            continue;
        }
        auto *node = node_it->second;
        EvictOne(expired_block);
        evicted.push_back(expired_block);
        node_map_.erase(node_it);
        expire_event_version_.erase(expired_block);
        list_.unlink(node);
        delete node;
    }
    return evicted;
}

void TtlEvictionPolicy::MaybeCompactExpireHeap() {
    if (node_map_.empty()) {
        return;
    }
    if (expire_min_heap_.size() > node_map_.size() * 4) {
        RebuildExpireHeap();
    }
}

void TtlEvictionPolicy::RebuildExpireHeap() {
    decltype(expire_min_heap_) new_heap;
    for (const auto &[block, node] : node_map_) {
        (void)node;
        if (!block || block->ttl_us <= 0) {
            continue;
        }
        auto version_it = expire_event_version_.find(block);
        if (version_it == expire_event_version_.end()) {
            continue;
        }
        new_heap.push(ExpireEvent{block->ttl_anchor_time + block->ttl_us, version_it->second, block});
    }
    expire_min_heap_.swap(new_heap);
}

// ============================================================
//  两阶段驱逐：
//  Phase 1 — 清走所有 TTL 过期 block（无视 count）
//  Phase 2 — fallback_on_pressure 开启且不够 count 时，
//            从链表尾部按 last_access_time 最旧优先补足
// ============================================================
std::vector<BlockEntry *> TtlEvictionPolicy::EvictBlocks(size_t count) {
    auto evicted = HarvestExpiredBlocks();
    if (node_map_.empty()) {
        return evicted;
    }

    // ---- Phase 2: LRU 兜底，从尾部取最旧的补足 ----
    if (fallback_on_pressure_ && evicted.size() < count) {
        size_t deficit = count - evicted.size();
        for (size_t i = 0; i < deficit; ++i) {
            auto *tail = static_cast<ListNode *>(list_.getTail());
            if (!tail || !tail->payload_) {
                break;
            }
            EvictOne(tail->payload_);
            evicted.push_back(tail->payload_);
            expire_event_version_.erase(tail->payload_);
            node_map_.erase(tail->payload_);
            list_.unlink(tail);
            delete tail;
        }
    }

    return evicted;
}

std::vector<BlockEntry *> TtlEvictionPolicy::EvictExpired() { return HarvestExpiredBlocks(); }

void TtlEvictionPolicy::EvictOne(BlockEntry *block) {
    if (name_ == "shared") {
        block->location_map.clear();
    } else {
        block->location_map.erase(name_);
    }
}

void TtlEvictionPolicy::Clear() {
    for (auto &[block, node] : node_map_) {
        if (name_ == "shared") {
            block->location_map.clear();
        } else {
            block->location_map.erase(name_);
        }
    }
    list_.clear();
    node_map_.clear();
    expire_event_version_.clear();
    decltype(expire_min_heap_) empty_heap;
    expire_min_heap_.swap(empty_heap);
}

} // namespace kv_cache_manager
