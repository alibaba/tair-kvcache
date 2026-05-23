#pragma once
#include <climits>
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
    virtual void OnBlockAccessedWithOptions(BlockEntry *block, int64_t timestamp, bool refresh_ttl_on_read) {
        (void)refresh_ttl_on_read;
        OnBlockAccessed(block, timestamp);
    }
    virtual std::vector<BlockEntry *> EvictBlocks(size_t num_blocks) = 0;
    virtual std::vector<BlockEntry *> EvictExpired() { return {}; }
    virtual void Clear() = 0;
    virtual bool NeedCapacityEviction() const { return true; }
    virtual void AdvanceClock(int64_t timestamp) { (void)timestamp; }

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

    // shared 模式本身就是全局池，使用 block 级访问/写入时间排序。
    // 分层模式使用 tier 级时间，实现各层 LRU 独立；若 block 未注册到本层
    // （不应发生，调用方已保证）返回 INT64_MAX。
    int64_t GetTierAccessTime(const BlockEntry *block) const {
        if (block == nullptr) {
            return INT64_MAX;
        }
        auto it = block->location_map.find(name_);
        if (it == block->location_map.end()) {
            return INT64_MAX;
        }
        if (it->second.last_access_time >= 0) {
            return it->second.last_access_time;
        }
        if (it->second.writing_time >= 0) {
            return it->second.writing_time;
        }
        if (name_ == "shared") {
            if (block->last_access_time >= 0) {
                return block->last_access_time;
            }
            if (block->writing_time >= 0) {
                return block->writing_time;
            }
        }
        return INT64_MAX;
    }

private:
    virtual void OnBlockAccessed(BlockEntry *block, int64_t timestamp) = 0;
    std::string name_;
};
} // namespace kv_cache_manager
