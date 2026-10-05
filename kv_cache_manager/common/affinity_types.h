#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kv_cache_manager {

struct ReplicaLimits {
    uint32_t max_replicas_per_key = 0; // 0 = unlimited; counts all location reservations
    uint64_t max_instance_bytes = 0;  // 0 = unlimited; includes original copies and WRITING
    uint32_t min_retained_replicas = 1; // Per spec, for node-pressure eviction only
};


// caller 自报节点 (node_id + supernode_id), client / server 共享。
struct CallerNode {
    std::string node_id;
    std::string supernode_id;
};

struct ReplicationSourceSpec {
    std::string spec_name;
    std::string uri;
};

// 复制提示: server affinity 算法产出, 经 proto 透传给 client SDK。
struct ReplicationHint {
    int64_t block_key{0};
    std::string source_uri;
    std::string target_node_id;
    // Complete sources by spec name. source_uri is retained for single-spec SDKs.
    std::vector<ReplicationSourceSpec> source_specs;
};

} // namespace kv_cache_manager
