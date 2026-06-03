#include "kv_cache_manager/optimizer/index/radix_tree_index.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_set>

#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/optimizer/analysis/stats_collector.h"

namespace kv_cache_manager {
namespace {
bool AppendBlockToSingleTier(BlockEntry *block, const std::shared_ptr<EvictionPolicy> &tier_policy, int64_t timestamp) {
    if (!block || !tier_policy) {
        return false;
    }
    const std::string &tier_name = tier_policy->name();
    if (block->location_map.find(tier_name) != block->location_map.end()) {
        return false;
    }
    AppendBlockLocation(block, tier_name, timestamp);
    tier_policy->OnBlockWritten(block);
    return true;
}
} // namespace

// 新构造函数 (多 tier)
RadixTreeIndex::RadixTreeIndex(const std::string &instance_id,
                               std::vector<std::shared_ptr<EvictionPolicy>> tier_policies,
                               TierWriteMode write_mode,
                               int64_t default_ttl_ns,
                               size_t selective_write_threshold,
                               bool tier_access_propagation_enabled,
                               std::vector<TierFlowStrategy> tier_flow_strategies) {
    root_ = std::make_unique<RadixTreeNode>();
    tier_policies_ = std::move(tier_policies);
    for (auto &p : tier_policies_) {
        tier_names_.push_back(p->name());
    }
    InitTierFlowStrategies(
        write_mode, selective_write_threshold, tier_access_propagation_enabled, std::move(tier_flow_strategies));
    instance_id_ = instance_id;
    default_ttl_ns_ = default_ttl_ns;
}

// 兼容构造函数 (单 policy)
RadixTreeIndex::RadixTreeIndex(const std::string &instance_id,
                               const std::shared_ptr<EvictionPolicy> &eviction_policy,
                               int64_t default_ttl_ns) {
    root_ = std::make_unique<RadixTreeNode>();
    tier_policies_.push_back(eviction_policy);
    tier_names_.push_back(eviction_policy->name());
    write_tier_count_ = 1;
    instance_id_ = instance_id;
    default_ttl_ns_ = default_ttl_ns;
}

void RadixTreeIndex::InitTierFlowStrategies(TierWriteMode write_mode,
                                            size_t selective_write_threshold,
                                            bool tier_access_propagation_enabled,
                                            std::vector<TierFlowStrategy> tier_flow_strategies) {
    const size_t edge_count = tier_policies_.size() > 0 ? tier_policies_.size() - 1 : 0;
    if (tier_flow_strategies.size() == edge_count) {
        tier_flow_strategies_ = std::move(tier_flow_strategies);
    } else {
        TierFlowStrategy default_strategy;
        default_strategy.write_mode = write_mode;
        default_strategy.access_propagation_enabled = tier_access_propagation_enabled;
        default_strategy.promote_enabled = false;
        default_strategy.selective_write_threshold = selective_write_threshold;
        tier_flow_strategies_.assign(edge_count, default_strategy);
    }

    write_tier_count_ = CountInitialWriteTiers(tier_policies_.size(), tier_flow_strategies_);
    enable_promote_ = false;
    for (const auto &strategy : tier_flow_strategies_) {
        if (strategy.promote_enabled) {
            enable_promote_ = true;
            break;
        }
    }
}

// TODO 后续改为 记录需要更新信息的node和blockentry，然后统一用一个接口更新
// 这样可以做到反向更新lru链表，避免同一时间戳下先驱逐前缀
RadixTreeIndex::InsertResult
RadixTreeIndex::InsertOnly(const std::vector<int64_t> &block_keys, int64_t timestamp, int64_t ttl_ns) {
    if (block_keys.empty()) {
        return {block_keys};
    }
    // 0 = 使用 default_ttl_ns_, -1 = 禁用(永不过期), >0 = 自定义
    int64_t resolved_ttl = (ttl_ns > 0) ? ttl_ns : (ttl_ns == 0) ? default_ttl_ns_ : 0;
    return InsertNode(root_.get(), block_keys, timestamp, resolved_ttl);
}
// 先返回键值，后续看需要location或是针对block的access信息的需求之后再返回BlockEntry指针
// 目前还没有热数据fetch的功能
RadixTreeIndex::InsertResult RadixTreeIndex::InsertNode(RadixTreeNode *node,
                                                        const std::vector<int64_t> &block_keys,
                                                        int64_t timestamp,
                                                        int64_t ttl_ns) {
    if (block_keys.empty()) {
        return {block_keys};
    }
    // 叶子追加 = 严格前缀包含（树结构保证 B 走完了 A 的全部 blocks）
    if (node->isLeaf() && node->parent != nullptr) {
        WriteToTier(node, block_keys, timestamp, ttl_ns, nullptr);
        return {block_keys};
    }
    int64_t current_key = block_keys.front();
    auto child_it = node->children.find(current_key);
    if (child_it == node->children.end()) {
        auto new_node = std::make_unique<RadixTreeNode>();
        new_node->parent = node;
        WriteToTier(new_node.get(), block_keys, timestamp, ttl_ns, nullptr);
        node->children[current_key] = std::move(new_node);
        return InsertResult{block_keys};
    } else {
        // 情况2:找到对应子节点，继续匹配插入
        RadixTreeNode *child = child_it->second.get();
        std::vector<int64_t> insert_keys;
        std::unordered_map<int64_t, BlockEntry *> evicted_blocks;
        size_t match_len = 0;
        // 找到最长匹配前缀
        while (match_len < child->blocks.size() && match_len < block_keys.size() &&
               child->blocks[match_len]->key == block_keys[match_len]) {
            if (IsBlockEvict(child->blocks[match_len].get(), timestamp)) {
                insert_keys.push_back(block_keys[match_len]);
                evicted_blocks[block_keys[match_len]] = child->blocks[match_len].get();
            }
            match_len++;
        }
        // 处理被驱逐的blocks，block存在但location为空，需要重新写入location
        if (!evicted_blocks.empty()) {
            WriteToTier(child, insert_keys, timestamp, ttl_ns, AppendEvictBlocks(std::move(evicted_blocks), ttl_ns));
        }
        if (match_len == child->blocks.size()) {
            // 完全匹配，递归向下
            auto remain_result = InsertNode(
                child, std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end()), timestamp, ttl_ns);
            insert_keys.insert(
                insert_keys.end(), remain_result.inserted_keys.begin(), remain_result.inserted_keys.end());
            return InsertResult{insert_keys};
        } else if (match_len == block_keys.size()) {
            // keys 完全匹配到子节点的部分前缀
            return InsertResult{insert_keys};
        } else {
            // 部分匹配 → SplitNode
            SplitNode(child,
                      match_len,
                      std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end()),
                      timestamp,
                      ttl_ns);
            auto remain_results = std::vector<int64_t>(block_keys.begin() + match_len, block_keys.end());
            insert_keys.insert(insert_keys.end(), remain_results.begin(), remain_results.end());
            return InsertResult{insert_keys};
        }
    }
}

void RadixTreeIndex::SplitNode(RadixTreeNode *existing_node,
                               size_t split_pos,
                               const std::vector<int64_t> &right_keys,
                               int64_t timestamp,
                               int64_t ttl_ns) {
    if (split_pos == 0)
        return;

    RadixTreeNode *original_parent = existing_node->parent;
    if (!original_parent) {
        return;
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
        WriteToTier(new_leaf.get(), right_keys, timestamp, ttl_ns, nullptr);
        middle_ptr->children[right_keys.front()] = std::move(new_leaf);
    }

    original_parent->children[edge_key] = std::move(middle_node);
}
// 同样，这里的PrefixQuery只返回命中key，后续看需求再返回BlockEntry指针等信息
bool RadixTreeIndex::PrefixQuery(const std::vector<int64_t> &block_keys,
                                 const BlockMask &block_mask,
                                 const int64_t timestamp,
                                 QueryHit *query_hit,
                                 bool refresh_ttl_on_read) {
    bool read_triggered_tier_write = false;
    if (block_keys.empty()) {
        return false;
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
        bool has_remote_hit = false;
        bool has_local_hit = false;
        while (match_len < child->blocks.size() && (key_idx + match_len) < block_keys.size() &&
               child->blocks[match_len]->key == block_keys[key_idx + match_len]) {
            if (IsBlockEvict(child->blocks[match_len].get(), timestamp)) {
                break;
            }
            BlockEntry *blk = child->blocks[match_len].get();
            const bool is_local = IsIndexInMaskRange(block_mask, key_idx + match_len);
            if (is_local) {
                has_local_hit = true;
                RecordTieredHit(blk, false, query_hit);
            } else {
                has_remote_hit = true;
                RecordTieredHit(blk, true, query_hit);
            }
            read_triggered_tier_write =
                OnBlockAccessed(blk, timestamp, refresh_ttl_on_read) || read_triggered_tier_write;
            if (enable_promote_) {
                read_triggered_tier_write = PromoteToHigherTiers(blk, timestamp) || read_triggered_tier_write;
            }
            match_len++;
        }
        if (has_remote_hit || has_local_hit) {
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
    return read_triggered_tier_write;
}

void RadixTreeIndex::CleanEmptyBlocks(const std::vector<BlockEntry *> &blocks,
                                      int64_t eviction_timestamp,
                                      bool use_logical_expire_time) {
    std::unordered_set<RadixTreeNode *> nodes_to_check;

    // 步骤 1：在删除前记录驱逐信息，收集需要检查的节点
    for (auto *block : blocks) {
        if (block->location_map.empty()) {
            int64_t effective_eviction_timestamp = eviction_timestamp;
            if (use_logical_expire_time && block->ttl_ns > 0 && block->ttl_anchor_time >= 0) {
                // TTL 过期清理使用逻辑过期时刻（ttl_anchor + ttl），与 IsExpired 保持同一判据。
                // 不能用 last_access_time：当 ttl_refresh_on_read=false（固定窗口）时，
                // last_access_time 会晚于 ttl_anchor_time，导致 death_time_us 被高估。
                effective_eviction_timestamp = block->ttl_anchor_time + block->ttl_ns;
                if (effective_eviction_timestamp > eviction_timestamp) {
                    effective_eviction_timestamp = eviction_timestamp;
                }
            }
            // 使用真实/逻辑驱逐时间戳记录事件
            if (stats_collector_) {
                stats_collector_->OnBlockEviction(instance_id_, block, effective_eviction_timestamp);
            }

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
                if (!block->location_map.empty()) {
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

std::vector<BlockEntry *> RadixTreeIndex::AppendNewBlocks(RadixTreeNode *node,
                                                          const std::vector<int64_t> &block_keys,
                                                          int64_t timestamp,
                                                          int64_t ttl_ns) {
    std::vector<BlockEntry *> inserted_blocks;
    inserted_blocks.reserve(block_keys.size());
    for (size_t i = 0; i < block_keys.size(); ++i) {
        auto entry = std::make_unique<BlockEntry>();
        entry->key = block_keys[i];
        entry->writing_time = timestamp;
        entry->last_access_time = timestamp;
        entry->ttl_ns = ttl_ns;
        entry->owner_node = node;
        BlockEntry *entry_ptr = entry.get();
        AppendInitialBlockLocations(entry_ptr, timestamp);
        node->blocks.emplace_back(std::move(entry));
        inserted_blocks.push_back(entry_ptr);

        if (stats_collector_) {
            stats_collector_->OnBlockBirth(instance_id_, entry_ptr, timestamp);
        }
    }
    return inserted_blocks;
}

void RadixTreeIndex::AppendInitialBlockLocations(BlockEntry *block, int64_t timestamp) const {
    if (!block) {
        return;
    }
    for (size_t tier_idx = 0; tier_idx < write_tier_count_; ++tier_idx) {
        AppendBlockLocation(block, tier_names_[tier_idx], timestamp);
    }
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

size_t CountInitialWriteTiers(size_t tier_count, const std::vector<TierFlowStrategy> &tier_flow_strategies) {
    if (tier_count == 0) {
        return 0;
    }
    size_t write_tier_count = 1;
    while (write_tier_count < tier_count && write_tier_count - 1 < tier_flow_strategies.size() &&
           tier_flow_strategies[write_tier_count - 1].write_mode == TierWriteMode::WRITE_THROUGH) {
        ++write_tier_count;
    }
    return write_tier_count;
}

bool AppendBlockToTierChain(BlockEntry *block,
                            size_t start_tier_idx,
                            const std::vector<std::shared_ptr<EvictionPolicy>> &tier_policies,
                            const std::vector<TierFlowStrategy> &tier_flow_strategies,
                            int64_t timestamp) {
    if (!block || start_tier_idx >= tier_policies.size()) {
        return false;
    }

    bool wrote_tier = false;
    size_t current_tier_idx = start_tier_idx;
    while (current_tier_idx < tier_policies.size()) {
        wrote_tier = AppendBlockToSingleTier(block, tier_policies[current_tier_idx], timestamp) || wrote_tier;
        if (current_tier_idx >= tier_flow_strategies.size() ||
            tier_flow_strategies[current_tier_idx].write_mode != TierWriteMode::WRITE_THROUGH) {
            break;
        }
        ++current_tier_idx;
    }
    return wrote_tier;
}

RadixTreeIndex::WriteModify RadixTreeIndex::AppendEvictBlocks(std::unordered_map<int64_t, BlockEntry *> blocks_map,
                                                              int64_t ttl_ns) {
    return
        [this, blocks_map = std::move(blocks_map), ttl_ns](const std::vector<int64_t> &block_keys, int64_t timestamp) {
            std::vector<BlockEntry *> revived_blocks;
            revived_blocks.reserve(block_keys.size());

            for (int64_t key : block_keys) {
                auto it = blocks_map.find(key);
                if (it != blocks_map.end()) {
                    BlockEntry *block = it->second;
                    block->writing_time = timestamp;
                    block->last_access_time = timestamp;
                    block->ttl_ns = ttl_ns;
                    AppendInitialBlockLocations(block, timestamp);
                    revived_blocks.push_back(block);

                    if (stats_collector_) {
                        stats_collector_->OnBlockBirth(instance_id_, block, timestamp);
                    }
                }
            }
            return revived_blocks;
        };
}

void RadixTreeIndex::WriteToTier(RadixTreeNode *node,
                                 const std::vector<int64_t> &block_keys,
                                 int64_t timestamp,
                                 int64_t ttl_ns,
                                 RadixTreeIndex::WriteModify cb) {
    std::vector<BlockEntry *> inserted_blocks;
    if (!cb) {
        // 节点直接添加新blocks
        inserted_blocks = AppendNewBlocks(node, block_keys, timestamp, ttl_ns);
    } else {
        // 节点填充空block 的 location
        inserted_blocks = cb(block_keys, timestamp);
    }
    node->stat.last_access_time = timestamp;
    // 根据写入模式注册到对应 tier 的驱逐队列：WRITE_THROUGH=所有层，级联/选择性写入=仅 tier 0
    for (size_t t = 0; t < write_tier_count_; ++t) {
        tier_policies_[t]->OnNodeWritten(inserted_blocks);
    }
}

bool RadixTreeIndex::OnBlockAccessed(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read) {
    // block 级别的统计只更新一次，避免多 tier 场景下重复递增
    block->access_count += 1;
    block->last_access_time = timestamp;

    // 遍历所有 tier：按相邻 edge 的 access_propagation_enabled 决定访问是否继续向下层传播。
    // access_count 仅对"首个命中层"+1（分层读优先读快层，tier 索引最小的持有层视为命中层）
    bool first_hit = true;
    bool propagate_access = true;
    size_t hit_tier_idx = tier_policies_.size();
    size_t hit_tier_access_count = 0;
    for (size_t i = 0; i < tier_policies_.size(); ++i) {
        if (!first_hit && i > 0 && i - 1 < tier_flow_strategies_.size() &&
            !tier_flow_strategies_[i - 1].access_propagation_enabled) {
            propagate_access = false;
        }
        const auto &tier_name = tier_names_[i];
        auto loc_it = block->location_map.find(tier_name);
        if (loc_it != block->location_map.end()) {
            if (first_hit) {
                tier_policies_[i]->OnBlockAccessedWithOptions(block, timestamp, refresh_ttl_on_read);
                loc_it->second.last_access_time = timestamp;
                loc_it->second.access_count += 1;
                hit_tier_idx = i;
                hit_tier_access_count = loc_it->second.access_count;
                first_hit = false;
            } else if (propagate_access) {
                tier_policies_[i]->OnBlockAccessedWithOptions(block, timestamp, refresh_ttl_on_read);
                loc_it->second.last_access_time = timestamp;
            }
        }
    }
    if (hit_tier_idx < tier_flow_strategies_.size() &&
        tier_flow_strategies_[hit_tier_idx].write_mode == TierWriteMode::WRITE_THROUGH_SELECTIVE &&
        hit_tier_access_count >= tier_flow_strategies_[hit_tier_idx].selective_write_threshold) {
        return SelectiveWriteToNextTier(block, hit_tier_idx, timestamp);
    }
    return false;
}

void RadixTreeIndex::RecordTieredHit(BlockEntry *block, bool is_remote, QueryHit *query_hit) const {
    if (!query_hit) {
        return;
    }
    if (is_remote) {
        query_hit->remote_hit_block_num++;
    } else {
        query_hit->local_hit_block_num++;
    }
    // 按 priority 从高到低检查，命中最高优先级的那一层
    if (query_hit->per_tier_hit_block_num.size() < tier_names_.size()) {
        query_hit->per_tier_hit_block_num.resize(tier_names_.size(), 0);
    }
    for (size_t i = 0; i < tier_names_.size(); ++i) {
        if (block->location_map.count(tier_names_[i])) {
            query_hit->per_tier_hit_block_num[i]++;
            break;
        }
    }
}

bool RadixTreeIndex::PromoteToHigherTiers(BlockEntry *block, int64_t timestamp) {
    if (!block || tier_policies_.empty()) {
        return false;
    }
    size_t current_highest_tier = tier_policies_.size();
    for (size_t i = 0; i < tier_policies_.size(); ++i) {
        if (block->location_map.count(tier_names_[i])) {
            current_highest_tier = i;
            break;
        }
    }
    if (current_highest_tier == 0 || current_highest_tier == tier_policies_.size()) {
        return false;
    }
    bool wrote_tier = false;
    for (size_t i = current_highest_tier; i > 0; --i) {
        const size_t higher_tier_idx = i - 1;
        if (higher_tier_idx >= tier_flow_strategies_.size() ||
            !tier_flow_strategies_[higher_tier_idx].promote_enabled) {
            break;
        }
        if (!block->location_map.count(tier_names_[higher_tier_idx])) {
            wrote_tier = AppendBlockToSingleTier(block, tier_policies_[higher_tier_idx], timestamp) || wrote_tier;
        }
    }
    return wrote_tier;
}

bool RadixTreeIndex::SelectiveWriteToNextTier(BlockEntry *block, size_t hit_tier_idx, int64_t timestamp) {
    if (!block || hit_tier_idx >= tier_flow_strategies_.size() ||
        tier_flow_strategies_[hit_tier_idx].write_mode != TierWriteMode::WRITE_THROUGH_SELECTIVE) {
        return false;
    }
    const size_t next_tier_idx = hit_tier_idx + 1;
    if (next_tier_idx >= tier_policies_.size()) {
        return false;
    }
    return AppendBlockToTierChain(block, next_tier_idx, tier_policies_, tier_flow_strategies_, timestamp);
}

// block 逻辑上空了：location 全空（被驱逐）
// V1 语义：
// 1) POLICY_TTL：过期块在读写前通过 EvictExpiredBeforeAccess 物理清理
// 2) 非 TTL 策略：不启用 TTL（既不做前置物理清理，也不做逻辑过期判定）
bool RadixTreeIndex::IsBlockEvict(BlockEntry *block, int64_t timestamp) const {
    (void)timestamp;
    return block->location_map.empty();
}

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
            generate_node_id(current, "node");
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
        for (const auto &block : current->blocks) {
            if (!block) {
                // 跳过空指针，避免未定义行为
                std::cerr << "Warning: Found null block pointer in node " << current_id << std::endl;
                continue;
            }
            if (is_block_cached(block.get())) {
                export_node.cached_blocks.push_back(block->key);
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
    // 清空所有 tier 的驱逐策略
    for (auto &policy : tier_policies_) {
        if (policy) {
            policy->Clear();
        }
    }

    // 重新创建根节点，清空整个树
    root_ = std::make_unique<RadixTreeNode>();
}

} // namespace kv_cache_manager
