#include "kv_cache_manager/optimizer/eviction_policy/lru.h"

namespace kv_cache_manager {
LruEvictionPolicy::LruEvictionPolicy(const std::string &name, const LruParams &params) : name_(name), params_(params) {}
LruEvictionPolicy::~LruEvictionPolicy() { node_map_.clear(); }
void LruEvictionPolicy::OnBlockWritten(BlockEntry *block) {
    auto *node = new LRUListNode();
    node->payload_ = block;
    lru_list_.push_front(node);
    node_map_[block] = node;
}

void LruEvictionPolicy::OnNodeWritten(std::vector<BlockEntry *> &blocks) {
    for (auto *block : blocks) {
        OnBlockWritten(block);
    }
}
void LruEvictionPolicy::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    auto it = node_map_.find(block);
    if (it != node_map_.end()) {
        LRUListNode *node = it->second;
        block->last_access_time = timestamp;
        block->access_count += 1;
        lru_list_.move_to_front(node);
    }
}

std::vector<BlockEntry *> LruEvictionPolicy::EvictBlocks(size_t count) {
    std::vector<BlockEntry *> evicted_blocks;
    for (size_t i = 0; i < count; ++i) {
        if (lru_list_.empty()) {
            break;
        }
        LinkedListNode *tail_node = lru_list_.getTail();
        LRUListNode *lru_node = static_cast<LRUListNode *>(tail_node);
        evicted_blocks.push_back(lru_node->payload_);
        node_map_.erase(lru_node->payload_);
        BlockEntry *block = lru_node->payload_;
        if (name_ == "shared") {
            // 全局驱逐时，清空所有location信息
            block->location_map.clear();
        } else {
            // 分层驱逐时，仅清除当前tier的location信息
            block->location_map.erase(name_); // 驱逐时清除该tier的location信息
        }
        lru_list_.remove(tail_node);
    }
    return evicted_blocks;
}

void LruEvictionPolicy::Clear() {
    // 清空所有blocks的location信息
    for (auto &[block, node] : node_map_) {
        if (name_ == "shared") {
            // 全局驱逐时，清空所有location信息
            block->location_map.clear();
        } else {
            // 分层驱逐时，仅清除当前tier的location信息
            block->location_map.erase(name_);
        }
    }
    // 清空LRU链表和映射
    lru_list_.clear();
    node_map_.clear();
}

} // namespace kv_cache_manager