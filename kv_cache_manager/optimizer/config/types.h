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
};
enum class TraceType {
    TRACE_UNSPECIFIED = 0,
    TRACE_PUBLISHER_LOG = 1,
    TRACE_QWEN_BAILIAN = 2,
    TRACE_OPTIMIZER_SCHEMA = 3
};

enum class EvictionMode {
    EVICTION_MODE_UNSPECIFIED = 0,
    EVICTION_MODE_GROUP_ROUGH = 1,
    EVICTION_MODE_INSTANCE_ROUGH = 2,
    EVICTION_MODE_INSTANCE_PRECISE = 3
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
    LocationStatMap location_map;   // key对应的块所在的层级位置以及对应的访问信息
    std::vector<int64_t> token_ids; // 可选
    int64_t writing_time = -1;
    int64_t last_access_time = -1;
    size_t access_count = 0;
    RadixTreeNode *owner_node = nullptr; // 所属节点指针

    void ResetAccess() {
        access_count = 0;
        last_access_time = -1;
        writing_time = -1;
    }
};

struct NodeStat {
    size_t access_count = 0;
    int64_t last_access_time = 0;
    int64_t ttl = 250000; // 默认TTL为250000微秒，即250毫秒
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

EvictionPolicyType ToEvictionPolicyType(const std::string &str);
TraceType ToTraceType(const std::string &str);
std::string ToString(const EvictionPolicyType &type);
std::string ToString(const TraceType &type);
} // namespace kv_cache_manager