#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kv_cache_manager {
enum class EvictionPolicyType {
    POLICY_UNSPECIFIED = 0,
    POLICY_LRU = 1,
    POLICY_RANDOM_LRU = 2,
    POLICY_LEAF_AWARE_LRU = 3,
    POLICY_TTL = 4,
};
enum class EvictionMode {
    EVICTION_MODE_UNSPECIFIED = 0,
    EVICTION_MODE_GROUP_ROUGH = 1,
    EVICTION_MODE_INSTANCE_ROUGH = 2,
    EVICTION_MODE_INSTANCE_PRECISE = 3
};

// Tier 写入模式：仅在 hierarchical_eviction_enabled=true 时生效。
// 只控制 block 在多 tier 间的写入与驱逐流动方式；读访问是否刷新下层由
// tier_strategy.access_propagation_enabled 单独控制。
enum class TierWriteMode {
    WRITE_THROUGH = 0,           // 默认：写入时落所有 tier，各层独立驱逐
    CASCADING = 1,               // 只写 tier 0，tier_i 驱逐出的 block 级联降级到 tier_{i+1}
    WRITE_THROUGH_SELECTIVE = 2, // 初始只写 tier 0，命中热度达到阈值后复制到下一层
};
struct TierFlowStrategy {
    TierWriteMode write_mode = TierWriteMode::WRITE_THROUGH;
    bool access_propagation_enabled = true;
    bool promote_enabled = false;
    size_t selective_write_threshold = 2;
};
struct TierStat {
    size_t access_count = 0;
    int64_t last_access_time = -1;
    int64_t writing_time = -1;
};
using LocationStatMap = std::unordered_map<std::string, TierStat>;

// 前置声明
struct RadixTreeNode;

struct BlockEntry {
    int64_t key;
    LocationStatMap location_map; // key对应的块所在的层级位置以及对应的访问信息
    int64_t writing_time = -1;
    int64_t last_access_time = -1;
    // TTL 续命锚点（与访问统计时间 last_access_time 解耦）。
    // 不变式：仅在 ttl_ns > 0 时有意义；IsExpired 已守卫 ttl_ns <= 0，anchor = -1 不会误判。
    int64_t ttl_anchor_time = -1;
    size_t access_count = 0;
    int64_t ttl_ns = 0;                  // TTL 纳秒，0 = 永不过期
    RadixTreeNode *owner_node = nullptr; // 所属节点指针

    void ResetAccess() {
        access_count = 0;
        last_access_time = -1;
        ttl_anchor_time = -1;
        writing_time = -1;
        ttl_ns = 0;
    }

    bool IsExpired(int64_t current_timestamp) const {
        return ttl_ns > 0 && current_timestamp > ttl_anchor_time + ttl_ns;
    }
};

struct NodeStat {
    size_t access_count = 0;
    int64_t last_access_time = 0;
};

struct RadixTreeNode {
    std::vector<std::unique_ptr<BlockEntry>> blocks; // 连续块的段
    NodeStat stat;
    RadixTreeNode *parent = nullptr;
    std::unordered_map<int64_t, std::unique_ptr<RadixTreeNode>> children;
    bool isLeaf() const { // 辅助判断是否无子节点
        return children.empty();
    }
    bool isDataLeaf() const { // 辅助判断是否无子节点且有数据块
        for (const auto &child_pair : children) {
            for (const auto &block : child_pair.second->blocks) {
                if (!block->location_map.empty()) {
                    return false;
                }
            }
        }
        return true;
    }
};

struct QueryHit {
    size_t local_hit_block_num = 0;
    size_t remote_hit_block_num = 0;
    std::vector<size_t> per_tier_hit_block_num; // indexed by tier priority order
};

EvictionPolicyType ToEvictionPolicyType(const std::string &str);
std::string ToString(const EvictionPolicyType &type);

TierWriteMode ToTierWriteMode(const std::string &str);
bool IsValidTierWriteMode(const std::string &str);
std::string ToString(const TierWriteMode &mode);
} // namespace kv_cache_manager
