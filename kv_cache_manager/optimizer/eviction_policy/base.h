#pragma once
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/config/types.h"
namespace kv_cache_manager {
// TODO 支持更多驱逐策略

class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;

    virtual size_t size() const = 0;

    virtual void OnBlockWritten(BlockEntry *block) = 0;
    virtual void OnNodeWritten(std::vector<BlockEntry *> &blocks) = 0;
    virtual void OnBlockAccessed(BlockEntry *block, int64_t timestamp) = 0;
    virtual std::vector<BlockEntry *> EvictBlocks(size_t num_blocks) = 0;

    // 清空所有缓存的blocks
    virtual void Clear() = 0;

    // TTL 策略 fallback=false 时无需容量驱逐，返回 false
    virtual bool NeedCapacityEviction() const { return true; }
    // 用于区分“TTL 由物理清理保障”与“TTL 仅逻辑判定”两类路径
    virtual bool IsTtlPolicy() const { return false; }
    // 推进策略内部时钟（默认 no-op）
    virtual void AdvanceClock(int64_t timestamp) {}

    virtual std::string name() const = 0;
    virtual void set_name(const std::string &name) = 0;
};
} // namespace kv_cache_manager