#pragma once
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/config/types.h"
namespace kv_cache_manager {

class EvictionPolicy {
public:
    explicit EvictionPolicy(const std::string &name) : name_(name) {}
    virtual ~EvictionPolicy() = default;

    virtual size_t size() const = 0;

    virtual void OnBlockWritten(BlockEntry *block) = 0;
    virtual void OnNodeWritten(std::vector<BlockEntry *> &blocks) = 0;
    virtual void OnBlockAccessed(BlockEntry *block, int64_t timestamp) = 0;
    virtual std::vector<BlockEntry *> EvictBlocks(size_t num_blocks) = 0;

    virtual void Clear() = 0;

    const std::string &name() const { return name_; }
    void set_name(const std::string &name) { name_ = name; }

protected:
    // 统一的 location 清理：shared 模式清空全部，分层模式仅移除当前 tier
    void ClearBlockLocation(BlockEntry *block) const {
        if (name_ == "shared") {
            block->location_map.clear();
        } else {
            block->location_map.erase(name_);
        }
    }

    // 读取 block 在“本策略所属 tier”上的最近访问时间
    // 驱逐排序 / 尾部比较均基于此，实现各层 LRU 独立（不被跨层 block 级统计污染）
    // 若 block 未注册到本层（不应发生，调用方已保证）返回 INT64_MAX
    int64_t GetTierAccessTime(const BlockEntry *block) const {
        if (block == nullptr) {
            return INT64_MAX;
        }
        auto it = block->location_map.find(name_);
        return (it != block->location_map.end()) ? it->second.last_access_time : INT64_MAX;
    }

private:
    std::string name_;
};
} // namespace kv_cache_manager