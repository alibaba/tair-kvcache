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
}

void TtlEvictionPolicy::OnNodeWritten(std::vector<BlockEntry *> &blocks) {
    for (auto *block : blocks) {
        OnBlockWritten(block);
    }
}

void TtlEvictionPolicy::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    auto it = node_map_.find(block);
    if (it == node_map_.end()) {
        return;
    }
    block->last_access_time = timestamp;
    block->access_count += 1;
    last_known_timestamp_ = timestamp;
    list_.move_to_front(it->second);
}

// ============================================================
//  两阶段驱逐：
//  Phase 1 — 清走所有 TTL 过期 block（无视 count）
//  Phase 2 — fallback_on_pressure 开启且不够 count 时，
//            从链表尾部按 last_access_time 最旧优先补足
// ============================================================
std::vector<BlockEntry *> TtlEvictionPolicy::EvictBlocks(size_t count) {
    std::vector<BlockEntry *> evicted;
    if (node_map_.empty()) {
        return evicted;
    }

    // ---- Phase 1: 收割所有过期 block ----
    std::vector<ListNode *> expired_nodes;
    size_t remaining = list_.size();
    auto *cursor = static_cast<ListNode *>(list_.getTail());
    while (cursor && remaining > 0) {
        auto *prev = static_cast<ListNode *>(cursor->prev);
        remaining--;
        if (cursor->payload_ && cursor->payload_->IsExpired(last_known_timestamp_)) {
            expired_nodes.push_back(cursor);
        }
        cursor = prev;
    }

    for (auto *node : expired_nodes) {
        EvictOne(node->payload_);
        evicted.push_back(node->payload_);
        node_map_.erase(node->payload_);
        list_.unlink(node);
        delete node;
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
            node_map_.erase(tail->payload_);
            list_.unlink(tail);
            delete tail;
        }
    }

    return evicted;
}

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
}

} // namespace kv_cache_manager
