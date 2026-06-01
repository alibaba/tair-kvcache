#pragma once

// affinity v1 §15.1: F3 频率反馈机制层。
//
// Per-(caller_node_ip, block_key) LRU counter，统计 caller 最近对 key 的
// 远端命中次数。每次 GetCacheLocation 处理完一个 key：
//   - 如果 winner 没有 caller 本地的 spec  ⇒ counter += 1
//   - 如果有                              ⇒ 不变（本地已命中）
// PreferLocalStrategy.ShouldEmitReplicationHint 读这个 counter 与
// Params.replication_hot_threshold 比较，决定是否发 hint。
//
// 容量上限：~100 万条 (caller, key) 对，超出按 LRU 淘汰（§15.1 决策）。
// sketch 不持久化，重启后从 0 累积，warm 期分钟级（§18.5 已知行为）。
// 线程安全：所有公开方法持内部 mutex，写少读少负载下争用可接受。

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace kv_cache_manager {

class FrequencySketch {
public:
    // 默认容量 100 万条；上限以 LRU 淘汰。
    explicit FrequencySketch(size_t capacity = 1000000) : capacity_(capacity) {}

    // 记录一次远端命中：counter += 1；若不存在则 init 为 1。
    void Observe(const std::string &caller_node_ip, int64_t block_key);

    // 查询当前 counter 值，不存在返回 0。
    uint32_t RemoteCount(const std::string &caller_node_ip, int64_t block_key) const;

    // 显式重置某 entry（例如 hint 已发出 + 进入 dedup 窗口）。
    void Reset(const std::string &caller_node_ip, int64_t block_key);

    // 测试 / 调试用
    size_t Size() const;

private:
    using Key = std::pair<std::string, int64_t>;
    struct KeyHash {
        size_t operator()(const Key &k) const noexcept {
            // 组合 string hash 和 int64 hash
            return std::hash<std::string>{}(k.first) ^ (std::hash<int64_t>{}(k.second) * 0x9e3779b97f4a7c15ULL);
        }
    };
    struct Entry {
        uint32_t count = 0;
        // 指向 lru_ 中的位置，方便 O(1) 更新
        typename std::list<Key>::iterator lru_it;
    };

    void TouchLocked(const Key &k);
    void EvictIfFullLocked();

    mutable std::mutex mu_;
    size_t capacity_;
    std::unordered_map<Key, Entry, KeyHash> table_;
    std::list<Key> lru_; // 最近访问在 front，淘汰从 back
};

} // namespace kv_cache_manager
