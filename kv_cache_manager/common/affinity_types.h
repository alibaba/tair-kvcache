#pragma once

#include <cstdint>
#include <string>

namespace kv_cache_manager {

// caller 自报节点 (node_id + supernode_id), client / server 共享。
struct CallerNode {
    std::string node_id;
    std::string supernode_id;
};

// 复制提示: server affinity 算法产出, 经 proto 透传给 client SDK。
struct ReplicationHint {
    int64_t block_key{0};
    std::string source_uri;
    std::string target_node_id;
};

} // namespace kv_cache_manager
