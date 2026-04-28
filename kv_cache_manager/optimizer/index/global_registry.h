#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kv_cache_manager {

// 远程查询返回的注册信息
struct RemoteKeyInfo {
    int64_t register_time_us;       // key 首次注册时间（用于时间差分析）
    std::string source_instance_id; // 当前实际持有该 key 的远程 instance
};

// 重复 block 快照记录
struct DuplicateSnapshot {
    int64_t timestamp_us;
    size_t total_unique_keys;        // 全局不同 block_key 总数
    size_t total_block_copies;       // 所有 block_key 的持有者数之和
    size_t duplicate_block_copies;   // 重复份数 = total_block_copies - total_unique_keys
};

// ============================================================================
// 全局注册表 — 模拟线上 Redis 跨 Instance KVCache 注册
//
// 每个 block_key 维护一个持有该 block 的 instance_id 集合。
// 写入完成后 Register，逐出前 Deregister，查询时用 HasKey 判断远程是否可读。
// ============================================================================
class GlobalRegistry {
public:
    GlobalRegistry() = default;
    ~GlobalRegistry() = default;

    // 写入完成后注册：将 instance_id 加入每个 key 的持有者集合
    void Register(const std::string &instance_id, const std::vector<int64_t> &block_keys, int64_t timestamp_us) {
        auto &keys_set = instance_keys_[instance_id];
        for (int64_t key : block_keys) {
            auto [it, inserted] = registry_[key].insert(instance_id);
            if (inserted) {
                total_block_copies_++;
            }
            keys_set.insert(key);
            // 记录首次注册信息
            if (key_registration_.find(key) == key_registration_.end()) {
                key_registration_[key] = {timestamp_us, instance_id};
            }
        }
        RecordDuplicateSnapshot(timestamp_us);
    }

    // 逐出前反注册：从每个 key 的持有者中移除 instance_id，集合空则删 key
    void Deregister(const std::string &instance_id, const std::vector<int64_t> &block_keys, int64_t timestamp_us) {
        auto inst_it = instance_keys_.find(instance_id);
        for (int64_t key : block_keys) {
            auto it = registry_.find(key);
            if (it != registry_.end()) {
                size_t erased = it->second.erase(instance_id);
                if (erased > 0) {
                    total_block_copies_--;
                }
                if (it->second.empty()) {
                    registry_.erase(it);
                    key_registration_.erase(key);
                }
            }
            if (inst_it != instance_keys_.end()) {
                inst_it->second.erase(key);
            }
        }
        // 如果该 instance 已无任何 key，清理反向索引条目
        if (inst_it != instance_keys_.end() && inst_it->second.empty()) {
            instance_keys_.erase(inst_it);
        }
        RecordDuplicateSnapshot(timestamp_us);
    }

    // 远程查询：key 是否被 exclude_instance_id 以外的 instance 持有
    bool HasKey(int64_t block_key, const std::string &exclude_instance_id) const {
        auto it = registry_.find(block_key);
        if (it == registry_.end()) {
            return false;
        }
        const auto &holders = it->second;
        if (holders.empty()) {
            return false;
        }
        // 检查是否存在至少一个非 exclude 的持有者
        if (holders.size() > 1) {
            return true;
        }
        return holders.find(exclude_instance_id) == holders.end();
    }

    // 远程查询（含注册信息）：命中时返回首次注册时间和实际提供数据的远程 instance
    std::optional<RemoteKeyInfo> QueryRemoteKey(int64_t block_key, const std::string &exclude_instance_id) const {
        auto it = registry_.find(block_key);
        if (it == registry_.end() || it->second.empty()) {
            return std::nullopt;
        }
        // 找一个排除自身后的当前持有者
        const auto &holders = it->second;
        std::string actual_source;
        for (const auto &h : holders) {
            if (h != exclude_instance_id) {
                actual_source = h;
                break;
            }
        }
        if (actual_source.empty()) {
            return std::nullopt; // 只有自身持有
        }
        int64_t reg_time = 0;
        auto reg_it = key_registration_.find(block_key);
        if (reg_it != key_registration_.end()) {
            reg_time = reg_it->second.register_time_us;
        }
        return RemoteKeyInfo{reg_time, actual_source};
    }

    // 清除某 instance 的全部注册（用于 ClearCache）
    void DeregisterInstance(const std::string &instance_id) {
        auto inst_it = instance_keys_.find(instance_id);
        if (inst_it == instance_keys_.end()) {
            return;
        }
        for (int64_t key : inst_it->second) {
            auto reg_it = registry_.find(key);
            if (reg_it != registry_.end()) {
                size_t erased = reg_it->second.erase(instance_id);
                if (erased > 0) {
                    total_block_copies_--;
                }
                if (reg_it->second.empty()) {
                    registry_.erase(reg_it);
                    key_registration_.erase(key);
                }
            }
        }
        instance_keys_.erase(inst_it);
    }

    // 全量清空
    void Clear() {
        registry_.clear();
        instance_keys_.clear();
        key_registration_.clear();
        total_block_copies_ = 0;
        duplicate_snapshots_.clear();
    }

    // 注册表中的 key 总数
    size_t Size() const { return registry_.size(); }

    // 导出重复 block 统计 CSV 到指定目录
    void ExportDuplicateStats(const std::string &output_dir);

private:
    void RecordDuplicateSnapshot(int64_t timestamp_us) {
        size_t unique_keys = registry_.size();
        size_t dup = total_block_copies_ > unique_keys ? total_block_copies_ - unique_keys : 0;
        duplicate_snapshots_.push_back({timestamp_us, unique_keys, total_block_copies_, dup});
    }

    // 正向索引：block_key -> 持有该 key 的 instance_id 集合
    std::unordered_map<int64_t, std::unordered_set<std::string>> registry_;
    // 反向索引：instance_id -> 该 instance 注册的 block_key 集合
    std::unordered_map<std::string, std::unordered_set<int64_t>> instance_keys_;
    // 注册信息：block_key -> 首次注册时间和来源 instance
    std::unordered_map<int64_t, RemoteKeyInfo> key_registration_;
    // 重复统计：增量维护的 block 副本总数
    size_t total_block_copies_ = 0;
    // 重复统计：每次 Register/Deregister 后的快照序列
    std::vector<DuplicateSnapshot> duplicate_snapshots_;
};

} // namespace kv_cache_manager
