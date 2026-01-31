#include "kv_cache_manager/optimizer/eviction_policy/leaf_aware_lru.h"

#include <algorithm>
#include <set>
#include <vector>
namespace kv_cache_manager {
LeafAwareLruEvictionPolicy::LeafAwareLruEvictionPolicy(const std::string &name, const LruParams &params)
    : name_(name), params_(params) {}

LeafAwareLruEvictionPolicy::~LeafAwareLruEvictionPolicy() {
    for (auto &pair : node_map_) {
        if (leaf_blocks_.find(pair.first) == leaf_blocks_.end()) {
            // 不在链表中的节点需要手动释放
            delete pair.second;
        }
    }
    node_map_.clear();
}

void LeafAwareLruEvictionPolicy::OnBlockWritten(BlockEntry *block) {
    LeafLRUListNode *node = new LeafLRUListNode();
    node->payload_ = block;
    node_map_[block] = node;

    // LeafAwareLRU 核心：只有叶子节点的 block 才加入驱逐链表
    // 非叶子节点（公共前缀）的 block 暂不参与驱逐，等子节点全部驱逐后再激活
    if (block->owner_node != nullptr && block->owner_node->isLeaf()) {
        leaf_lru_list_.push_front(node);
        leaf_blocks_.insert(block);
    }
}

void LeafAwareLruEvictionPolicy::OnNodeWritten(std::vector<BlockEntry *> &blocks) {
    for (auto *block : blocks) {
        OnBlockWritten(block);
    }
}

void LeafAwareLruEvictionPolicy::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    auto node_it = node_map_.find(block);
    if (node_it != node_map_.end()) {
        // 更新所有 block 的访问时间（包括非叶子节点的 block）
        block->last_access_time = timestamp;
        block->access_count++;
        // 只有在驱逐链表中的 block 才需要移动到头部
        auto it = leaf_blocks_.find(block);
        if (it != leaf_blocks_.end()) {
            leaf_lru_list_.move_to_front(node_it->second);
        }
    }
}

void LeafAwareLruEvictionPolicy::insert_sorted_by_priority(LeafLRUListNode *node) {
    if (leaf_lru_list_.empty()) {
        leaf_lru_list_.push_back(node);
        return;
    }

    int64_t node_time = node->payload_->last_access_time;

    // 获取头尾节点的 last_access_time
    auto *head_node = static_cast<LeafLRUListNode *>(leaf_lru_list_.getHead());
    auto *tail_node = static_cast<LeafLRUListNode *>(leaf_lru_list_.getTail());
    int64_t head_time = head_node->payload_->last_access_time;
    int64_t tail_time = tail_node->payload_->last_access_time;

    // 自适应选择遍历方向：离哪端近就从哪端开始
    if (node_time >= head_time) {
        // 比头部还热门，从头部开始遍历
        leaf_lru_list_.insert_sorted(node, LeafLRUListNode::compare);
    } else if (node_time <= tail_time) {
        // 比尾部还冷门，从尾部开始遍历
        leaf_lru_list_.insert_sorted_reverse(node, LeafLRUListNode::compare);
    } else {
        // 在中间位置，比较距离决定方向
        if ((head_time - node_time) < (node_time - tail_time)) {
            leaf_lru_list_.insert_sorted(node, LeafLRUListNode::compare);
        } else {
            leaf_lru_list_.insert_sorted_reverse(node, LeafLRUListNode::compare);
        }
    }
}

std::vector<BlockEntry *> LeafAwareLruEvictionPolicy::EvictBlocks(size_t count) {
    std::vector<BlockEntry *> evicted_blocks;
    // 记录已处理的父节点，避免重复调用 UpdateNodeState
    std::set<RadixTreeNode *> processed_parents;

    while (evicted_blocks.size() < count) {
        if (leaf_lru_list_.empty()) {
            break;
        }

        LinkedListNode *tail_node = leaf_lru_list_.getTail();
        LeafLRUListNode *lru_node = static_cast<LeafLRUListNode *>(tail_node);
        BlockEntry *block = lru_node->payload_;
        auto *owner_node = block->owner_node;

        // 检查 block 所属节点是否仍是叶子（可能因为新插入变成了非叶子）
        // TODO: 可优化为在前缀树 InsertNode/SplitNode 时即时处理，避免每次驱逐都检查
        if (owner_node != nullptr && !owner_node->isDataLeaf()) {
            leaf_lru_list_.unlink(tail_node);
            leaf_blocks_.erase(block);
            continue;
        }

        // 驱逐 block
        evicted_blocks.push_back(block);

        if (name_ == "shared") {
            block->location_map.clear();
        } else {
            block->location_map.erase(name_);
        }

        leaf_blocks_.erase(block);
        node_map_.erase(block);
        leaf_lru_list_.remove(tail_node);

        // 检查父节点是否因此变成 data leaf，如果是则激活它的 blocks
        // 使用 processed_parents 去重：只有 UpdateNodeState 返回 true（真正处理了）才标记
        if (owner_node != nullptr && owner_node->parent != nullptr) {
            RadixTreeNode *parent = owner_node->parent;
            if (processed_parents.find(parent) == processed_parents.end()) {
                if (UpdateNodeState(parent)) {
                    processed_parents.insert(parent);
                }
            }
        }
    }

    return evicted_blocks;
}

bool LeafAwareLruEvictionPolicy::UpdateNodeState(RadixTreeNode *node) {
    if (node == nullptr) {
        return false;
    }

    // 检查节点是否已变成 data leaf（所有子节点的 blocks 都已驱逐）
    if (!node->isDataLeaf()) {
        return false;
    }

    // 节点已变成 data leaf，将其 blocks 按 last_access_time 插入驱逐链表
    // 使用反向遍历插入：父节点的 blocks 通常较老，插入位置靠近尾部，从尾部遍历更快
    bool inserted_any = false;
    for (const auto &block : node->blocks) {
        BlockEntry *block_ptr = block.get();
        auto it = node_map_.find(block_ptr);
        if (it != node_map_.end() && leaf_blocks_.find(block_ptr) == leaf_blocks_.end()) {
            insert_sorted_by_priority(it->second);
            leaf_blocks_.insert(block_ptr);
            inserted_any = true;
        }
    }

    return inserted_any;
}

void LeafAwareLruEvictionPolicy::Clear() {
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
    leaf_lru_list_.clear();
    node_map_.clear();
    leaf_blocks_.clear();
}

} // namespace kv_cache_manager