#include "kv_cache_manager/optimizer/index/radix_tree_index.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_set>

#include "kv_cache_manager/manager/cache_location.h"
namespace kv_cache_manager {
RadixTreeIndex::RadixTreeIndex(const std::string &instance_id, const std::shared_ptr<EvictionPolicy> &eviction_policy) {
    root_ = std::make_unique<RadixTreeNode>();

    eviction_policy_ = eviction_policy;
    instance_id_ = instance_id; // 设置实例ID
}
// TODO 后续改为 记录需要更新信息的node和blockentry，然后统一用一个接口更新
// 这样可以做到反向更新lru链表，避免同一时间戳下先驱逐前缀
std::vector<int64_t> RadixTreeIndex::InsertOnly(const std::vector<int64_t> &block_keys, const int64_t timestamp) {
    if (block_keys.empty()) {
        return block_keys;
    }
    std::vector<int64_t> insert_block_keys = InsertNode(root_.get(), block_keys, timestamp);
    // CleanEmptyLeafNodes(root_.get());
    return insert_block_keys;
}
// 先返回键值，后续看需要location或是针对block的access信息的需求之后再返回BlockEntry指针
// 目前还没有热数据fetch的功能
std::vector<int64_t>
RadixTreeIndex::InsertNode(RadixTreeNode *node, const std::vector<int64_t> &block_keys, const int64_t timestamp) {
    if (block_keys.empty()) {
        return block_keys;
    }
    // 如果当前节点是叶子节点，直接在该节点末尾添加blocks，减少节点生成；
    if (node->isLeaf() && node->parent != nullptr) {
        WriteToTier(node, block_keys, timestamp, nullptr);
        return block_keys;
    }
    // 如果不是叶子节点，继续向下查找合适的子节点插入blocks
    // 情况1:找不到对应子节点，创建新节点插入blocks
    int64_t current_key = block_keys.front();
    auto child_it = node->children.find(current_key);
    if (child_it == node->children.end()) {
        auto new_node = std::make_unique<RadixTreeNode>();
        new_node->parent = node;
        WriteToTier(new_node.get(), block_keys, timestamp, nullptr);
        node->children[current_key] = std::move(new_node);
        return block_keys;
    } else {
        // 情况2:找到对应子节点，继续匹配插入
        RadixTreeNode *child = child_it->second.get();
        std::vector<int64_t> insert_keys;
        std::unordered_map<int64_t, BlockEntry *> evicted_blocks;
        size_t match_len = 0;
        // 找到最长匹配前缀
        while (match_len < child->blocks.size() && match_len < block_keys.size() &&
               child->blocks[match_len]->key == block_keys[match_len]) {
            // 处理已存在但被完全驱逐的blocks
            if (IsBlockEvict(child->blocks[match_len].get())) {
                insert_keys.push_back(block_keys[match_len]);
                evicted_blocks[block_keys[match_len]] = child->blocks[match_len].get();
            }
            match_len++;
        }
        // 处理被驱逐的blocks，block存在但location为空，需要重新写入location
        if (!evicted_blocks.empty()) {
            WriteToTier(child, insert_keys, timestamp, AppendEvictBlocks(std::move(evicted_blocks)));
        }
        if (match_len == child->blocks.size()) {
            // 当前子节点完全匹配，继续向下匹配插入
            auto remain_results =
                InsertNode(child, std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end()), timestamp);
            insert_keys.insert(insert_keys.end(), remain_results.begin(), remain_results.end());
            return insert_keys;
        } else if (match_len == block_keys.size()) {
            // 需要写入的block_keys完全匹配到子节点的部分，直接返回
            return insert_keys;
        } else {
            // 部分匹配，进行节点拆分
            SplitNode(
                child, match_len, std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end()), timestamp);
            auto remain_results = std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end());
            insert_keys.insert(insert_keys.end(), remain_results.begin(), remain_results.end());
            return insert_keys;
        }
    }
}

void RadixTreeIndex::SplitNode(RadixTreeNode *existing_node,
                               size_t split_pos,
                               const std::vector<int64_t> &right_keys,
                               int64_t timestamp) {
    if (split_pos == 0)
        return; // No split needed

    RadixTreeNode *original_parent = existing_node->parent;
    if (!original_parent) {
        return; // Root node, cannot split
    }
    int64_t edge_key = existing_node->blocks.front()->key;
    auto existing_uptr = std::move(original_parent->children[edge_key]);

    auto middle_node = std::make_unique<RadixTreeNode>();
    RadixTreeNode *middle_ptr = middle_node.get();
    middle_ptr->blocks.clear();
    for (size_t i = 0; i < split_pos; ++i) {
        middle_ptr->blocks.push_back(std::move(existing_uptr->blocks[i]));
        if (middle_ptr->blocks.back()) {
            middle_ptr->blocks.back()->owner_node = middle_ptr;
        }
    }
    middle_ptr->parent = original_parent;
    middle_ptr->stat = existing_uptr->stat;

    existing_uptr->blocks.erase(existing_uptr->blocks.begin(), existing_uptr->blocks.begin() + split_pos);
    existing_uptr->parent = middle_ptr;
    middle_ptr->children[existing_uptr->blocks.front()->key] = std::move(existing_uptr);

    if (!right_keys.empty()) {
        auto new_leaf = std::make_unique<RadixTreeNode>();
        new_leaf->parent = middle_ptr;
        WriteToTier(new_leaf.get(), right_keys, timestamp, nullptr);
        middle_ptr->children[right_keys.front()] = std::move(new_leaf);
    }

    original_parent->children[edge_key] = std::move(middle_node);
}
// 同样，这里的PrefixQuery只返回命中key，后续看需求再返回BlockEntry指针等信息
void RadixTreeIndex::PrefixQuery(const std::vector<int64_t> &block_keys,
                                 const BlockMask &block_mask,
                                 const int64_t timestamp,
                                 std::vector<std::vector<int64_t>> &external_hits,
                                 std::vector<std::vector<int64_t>> &internal_hits) {
    if (block_keys.empty()) {
        return;
    }
    std::unordered_set<int64_t> query_keys;
    std::unordered_set<int64_t> mask_keys;
    for (size_t idx = 0; idx < block_keys.size(); idx++) {
        if (IsIndexInMaskRange(block_mask, idx)) {
            mask_keys.insert(block_keys[idx]);
        } else {
            query_keys.insert(block_keys[idx]);
        }
    }

    RadixTreeNode *current_node = root_.get();
    size_t key_idx = 0;

    while (key_idx < block_keys.size()) {
        int64_t current_key = block_keys[key_idx];
        auto child_it = current_node->children.find(current_key);
        if (child_it == current_node->children.end()) {
            break;
        }
        RadixTreeNode *child = child_it->second.get();
        size_t match_len = 0;
        std::vector<int64_t> temp_hits;
        std::vector<int64_t> temp_internal_hits;
        while (match_len < child->blocks.size() && (key_idx + match_len) < block_keys.size() &&
               child->blocks[match_len]->key == block_keys[key_idx + match_len]) {
            if (IsBlockEvict(child->blocks[match_len].get())) {
                break;
            }
            BlockEntry *blk = child->blocks[match_len].get();
            if (query_keys.count(block_keys[key_idx + match_len])) {
                temp_hits.push_back(block_keys[key_idx + match_len]);
                // 访问block，更新存在的tier的访问信息
                OnBlockAccessed(blk, timestamp);
            } else if (mask_keys.count(block_keys[key_idx + match_len])) {
                temp_internal_hits.push_back(block_keys[key_idx + match_len]);
                OnBlockAccessed(blk, timestamp);
            }
            match_len++;
        }
        if (!temp_hits.empty()) {
            external_hits.emplace_back(std::move(temp_hits));
            child->stat.last_access_time = timestamp;
            child->stat.access_count += 1;
        }
        if (!temp_internal_hits.empty()) {
            internal_hits.emplace_back(std::move(temp_internal_hits));
            // 为了保证高频前缀命中，hbm命中也更新节点访问信息
            // 可能会对前缀树可视化产生影响
            child->stat.last_access_time = timestamp;
            child->stat.access_count += 1;
        }
        if (match_len < child->blocks.size()) {
            break;
        } else if ((key_idx + match_len) == block_keys.size()) {
            break;
        }
        current_node = child;
        key_idx += match_len;
    }
}

std::vector<int64_t> RadixTreeIndex::InsertWithQuery(const std::vector<int64_t> &block_keys,
                                                     const int64_t timestamp,
                                                     std::vector<std::vector<int64_t>> &hits) {
    if (block_keys.empty()) {
        return block_keys;
    }
    std::vector<int64_t> insert_block_keys = InsertQuery(root_.get(), block_keys, timestamp, true, hits);
    // CleanEmptyLeafNodes(root_.get());
    return insert_block_keys;
}
// 该接口服务于直接trace分析，在这种场景下无法得知prefill的执行时间，相当于读写同时进行
// 只是为了提高性能，避免先调用PrefixQuery再调用Insert两次遍历前缀树
// 因此合并了两者的逻辑，在查询的同时插入缺失的blocks
// 该接口服务于只包含用户请求的trace，这种请求无法获取decode的内容，并且不区分query或insert
std::vector<int64_t> RadixTreeIndex::InsertQuery(RadixTreeNode *node,
                                                 const std::vector<int64_t> &block_keys,
                                                 const int64_t timestamp,
                                                 bool is_prefix_hit,
                                                 std::vector<std::vector<int64_t>> &hits) {
    if (block_keys.empty()) {
        return block_keys;
    }
    // 如果当前节点是叶子节点，直接在该节点添加blocks；
    // 如果不是叶子节点，继续向下查找合适的子节点插入
    if (node->isLeaf() && node->parent != nullptr) {
        WriteToTier(node, block_keys, timestamp, nullptr);
        return block_keys;
    }
    int64_t current_key = block_keys.front();
    auto child_it = node->children.find(current_key);
    if (child_it == node->children.end()) {
        auto new_node = std::make_unique<RadixTreeNode>();
        new_node->parent = node;
        WriteToTier(new_node.get(), block_keys, timestamp, nullptr);
        node->children[current_key] = std::move(new_node);
        return block_keys;
    } else {
        RadixTreeNode *child = child_it->second.get();
        std::vector<int64_t> insert_keys;
        std::vector<int64_t> temp_hits;
        std::unordered_map<int64_t, BlockEntry *> evicted_blocks;
        size_t match_len = 0;
        while (match_len < child->blocks.size() && match_len < block_keys.size() &&
               child->blocks[match_len]->key == block_keys[match_len]) {
            BlockEntry *blk = child->blocks[match_len].get();
            if (IsBlockEvict(child->blocks[match_len].get())) {
                insert_keys.push_back(block_keys[match_len]);
                evicted_blocks[block_keys[match_len]] = blk;
                is_prefix_hit = false;
            } else if (is_prefix_hit) {
                temp_hits.push_back(block_keys[match_len]);
                OnBlockAccessed(blk, timestamp);
            }
            match_len++;
        }
        if (!evicted_blocks.empty()) {
            WriteToTier(child, insert_keys, timestamp, AppendEvictBlocks(std::move(evicted_blocks)));
        }
        if (!temp_hits.empty()) {
            hits.emplace_back(std::move(temp_hits));
            child->stat.access_count += 1;
            child->stat.last_access_time = timestamp;
        }
        if (match_len == child->blocks.size()) {
            // 当前子节点完全匹配，继续向下匹配插入
            auto remain_results = InsertQuery(child,
                                              std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end()),
                                              timestamp,
                                              is_prefix_hit,
                                              hits);
            insert_keys.insert(insert_keys.end(), remain_results.begin(), remain_results.end());
            return insert_keys;
        } else if (match_len == block_keys.size()) {
            // 需要写入的block_keys完全匹配到子节点的部分，直接返回
            return insert_keys;
        } else {
            // 部分匹配，进行节点拆分
            SplitNode(
                child, match_len, std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end()), timestamp);
            auto remain_results = std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end());
            insert_keys.insert(insert_keys.end(), remain_results.begin(), remain_results.end());
            return insert_keys;
        }
    }
}
void RadixTreeIndex::CleanEmptyBlocks(const std::vector<BlockEntry *> &blocks) {
    std::unordered_set<RadixTreeNode *> nodes_to_check;

    // 步骤 1：收集需要检查的节点
    for (auto *block : blocks) {
        if (block->location_map.empty()) {
            block->ResetAccess();
            auto owner_node = block->owner_node;
            if (owner_node && owner_node->parent) {
                nodes_to_check.insert(owner_node);
            }
        }
    }

    // 步骤 2：多轮删除，直到没有节点被删除
    bool deleted_any = true;
    while (deleted_any) {
        deleted_any = false;
        for (auto it = nodes_to_check.begin(); it != nodes_to_check.end();) {
            auto *node = *it;

            // 检查节点的所有 blocks 是否都被驱逐
            bool all_empty = true;
            for (const auto &block : node->blocks) {
                if (!IsBlockEvict(block.get())) {
                    all_empty = false;
                    break;
                }
            }

            // 检查节点是否是叶子节点（或所有子节点都已被删除）
            bool is_deletable = node->isLeaf(); // 真正的叶子节点

            if (all_empty && !node->blocks.empty() && node->parent && is_deletable) {
                node->parent->children.erase(node->blocks.front()->key);
                it = nodes_to_check.erase(it);
                deleted_any = true;
            } else {
                ++it;
            }
        }
    }
}
std::vector<BlockEntry *>
RadixTreeIndex::AppendNewBlocks(RadixTreeNode *node, const std::vector<int64_t> &block_keys, const int64_t timestamp) {
    auto tier_name = eviction_policy_->name();
    std::vector<BlockEntry *> inserted_blocks;
    inserted_blocks.reserve(block_keys.size());
    for (size_t i = 0; i < block_keys.size(); ++i) {
        auto entry = std::make_unique<BlockEntry>();
        entry->key = block_keys[i];
        entry->writing_time = timestamp;
        entry->last_access_time = timestamp;
        entry->owner_node = node;
        BlockEntry *entry_ptr = entry.get();
        AppendBlockLocation(entry_ptr, tier_name, timestamp);
        node->blocks.emplace_back(std::move(entry));
        inserted_blocks.push_back(entry_ptr);
    }
    return inserted_blocks;
}

void AppendBlockLocation(BlockEntry *block, const std::string &unique_name, int64_t timestamp) {
    if (unique_name == "shared") {
        // 全局驱逐策略，不区分tier，使用统一的location记录
        block->location_map[unique_name] = TierStat();
    } else {
        // 分层驱逐策略，记录具体tier的location信息
        block->location_map[unique_name] = TierStat{0, timestamp, timestamp};
    }
}

RadixTreeIndex::WriteModify RadixTreeIndex::AppendEvictBlocks(std::unordered_map<int64_t, BlockEntry *> blocks_map) {
    return [this, blocks_map = std::move(blocks_map)](const std::vector<int64_t> &block_keys, int64_t timestamp) {
        std::vector<BlockEntry *> revived_blocks;
        revived_blocks.reserve(block_keys.size());

        auto tier_name = eviction_policy_->name();

        for (int64_t key : block_keys) {
            auto it = blocks_map.find(key);
            if (it != blocks_map.end()) {
                BlockEntry *block = it->second;
                block->writing_time = timestamp;
                block->last_access_time = timestamp;
                AppendBlockLocation(block, tier_name, timestamp);
                revived_blocks.push_back(block);
            }
        }
        return revived_blocks;
    };
}

void RadixTreeIndex::WriteToTier(RadixTreeNode *node,
                                 const std::vector<int64_t> &block_keys,
                                 const int64_t timestamp,
                                 RadixTreeIndex::WriteModify cb) {
    // TODO 选择具体tier进行写入
    std::vector<BlockEntry *> inserted_blocks;
    if (!cb) {
        // 节点直接添加新blocks
        inserted_blocks = AppendNewBlocks(node, block_keys, timestamp);
    } else {
        // 节点填充空block 的 location
        inserted_blocks = cb(block_keys, timestamp);
    }
    node->stat.last_access_time = timestamp;
    eviction_policy_->OnNodeWritten(inserted_blocks);
}

void RadixTreeIndex::OnBlockAccessed(BlockEntry *block, int64_t timestamp) {
    // 全局驱逐
    if (eviction_policy_->name() == "shared") {
        eviction_policy_->OnBlockAccessed(block, timestamp);
    } else {
        // 分层驱逐，但目前只创建了第一层的驱逐策略，只对第一层进行驱逐,其他层写进去先不管
        // TODO 后续依照kvcm分层逻辑来实现
        eviction_policy_->OnBlockAccessed(block, timestamp);
        for (auto &location_pair : block->location_map) {
            // 现在不论是不是在第一层命中，都只更新第一层tier的访问信息
            std::string tier_name = eviction_policy_->name();
            if (location_pair.first == tier_name) {
                location_pair.second.access_count += 1;
                location_pair.second.last_access_time = timestamp;
            }
        }
    }
}
// 判断block是否被驱逐，所有location都为空则认为被驱逐
bool RadixTreeIndex::IsBlockEvict(BlockEntry *block) const { return block->location_map.empty(); }

// 导出前缀树用于可视化
RadixTreeIndex::RadixTreeExport RadixTreeIndex::ExportForVisualization() const {
    RadixTreeExport export_data;
    export_data.instance_id = instance_id_;

    if (!root_) {
        return export_data;
    }

    // 使用 BFS 遍历树结构
    std::queue<RadixTreeNode *> node_queue;
    std::unordered_map<RadixTreeNode *, std::string> node_id_map;

    // 生成节点 ID 的辅助函数
    auto generate_node_id = [&node_id_map](RadixTreeNode *node, const std::string &prefix = "") -> std::string {
        std::ostringstream oss;
        oss << prefix << "_" << reinterpret_cast<uintptr_t>(node);
        std::string node_id = oss.str();
        node_id_map[node] = node_id;
        return node_id;
    };

    // 判断 block 是否被缓存
    auto is_block_cached = [](const BlockEntry *block) -> bool {
        return block != nullptr && !block->location_map.empty();
    };

    // 统计变量
    size_t total_nodes = 0;
    size_t total_blocks_count = 0;
    size_t total_cached_blocks_count = 0;

    // 处理根节点
    std::string root_id = generate_node_id(root_.get(), "root");

    RadixTreeExportNode root_node;
    root_node.node_id = root_id;
    root_node.parent_id = "";
    root_node.access_count = 0;
    root_node.last_access_time = 0;
    root_node.total_blocks = std::vector<int64_t>();
    root_node.is_leaf = false;
    root_node.cached_blocks = std::vector<int64_t>();

    export_data.nodes.push_back(root_node);
    total_nodes++;

    // 将根节点的子节点加入队列，并生成它们的 node_id
    for (const auto &child_pair : root_->children) {
        RadixTreeNode *child = child_pair.second.get();
        generate_node_id(child, "node"); // 为子节点生成 ID
        node_queue.push(child);
    }

    // BFS 遍历
    while (!node_queue.empty()) {
        RadixTreeNode *current = node_queue.front();
        node_queue.pop();

        if (!node_id_map.count(current)) {
            // 如果没有 ID，生成一个
            std::string current_id = generate_node_id(current, "node");
        }

        std::string current_id = node_id_map[current];
        std::string parent_id = "";

        // 找到父节点 ID
        if (current->parent && node_id_map.count(current->parent)) {
            parent_id = node_id_map[current->parent];
        } else if (current->parent && !node_id_map.count(current->parent)) {
            // 如果父节点没有 ID，生成一个
            parent_id = generate_node_id(current->parent, "node");
        }

        // 创建导出节点
        RadixTreeExportNode export_node;
        export_node.node_id = current_id;
        export_node.parent_id = parent_id;
        export_node.access_count = current->stat.access_count;
        export_node.last_access_time = current->stat.last_access_time;

        export_node.is_leaf = current->isLeaf();

        // 收集 block 序列
        size_t node_cached_count = 0;
        for (const auto &block : current->blocks) {
            if (!block) {
                // 跳过空指针，避免未定义行为
                std::cerr << "Warning: Found null block pointer in node " << current_id << std::endl;
                continue;
            }
            if (is_block_cached(block.get())) {
                export_node.cached_blocks.push_back(block->key);
                node_cached_count++;
            }
            export_node.total_blocks.push_back(block->key);
        }

        // 更新统计
        total_nodes++;
        total_blocks_count += export_node.total_blocks.size();
        total_cached_blocks_count += export_node.cached_blocks.size();

        // 添加到导出数据
        export_data.nodes.push_back(export_node);

        // 如果有父节点，添加边
        if (!parent_id.empty()) {
            export_data.edges.emplace_back(parent_id, current_id);
        }

        // 将子节点加入队列
        for (const auto &child_pair : current->children) {
            RadixTreeNode *child = child_pair.second.get();
            if (!node_id_map.count(child)) {
                generate_node_id(child, "node");
            }
            node_queue.push(child);
        }
    }

    // 输出统计信息
    std::cout << "=== RadixTree Export Statistics ===" << std::endl;
    std::cout << "Instance ID: " << instance_id_ << std::endl;
    std::cout << "Total Nodes: " << total_nodes << std::endl;
    std::cout << "Total Blocks: " << total_blocks_count << std::endl;
    std::cout << "Total Cached Blocks: " << total_cached_blocks_count << std::endl;
    if (total_blocks_count > 0) {
        std::cout << "Cache Ratio: " << (100.0 * total_cached_blocks_count / total_blocks_count) << "%" << std::endl;
    }
    std::cout << "===================================" << std::endl;

    return export_data;
}

void RadixTreeIndex::Clear() {
    // 清空驱逐策略
    if (eviction_policy_) {
        eviction_policy_->Clear();
    }

    // 重新创建根节点，清空整个树
    root_ = std::make_unique<RadixTreeNode>();
}

} // namespace kv_cache_manager