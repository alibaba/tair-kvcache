#pragma once
#include <string>

#include "kv_cache_manager/optimizer/analysis/stats_record.h"

namespace kv_cache_manager {

// ============================================================================
// 前置声明
// ============================================================================
struct BlockEntry;
class OptimizerConfig;

// ============================================================================
// 统计子Tracker统一接口
//
// 所有事件方法提供空默认实现，子类只需 override 关心的事件。
// ============================================================================
class StatsTracker {
public:
    virtual ~StatsTracker() = default;

    // ---- 请求级事件 ----
    virtual void OnReadComplete(const std::string &instance_id, const ReadRecord &record) {}
    virtual void OnWriteComplete(const std::string &instance_id, const WriteRecord &record) {}

    // ---- Block级事件 ----
    virtual void OnBlockBirth(const std::string &instance_id, BlockEntry *block, int64_t timestamp) {}
    virtual void OnBlockEviction(const std::string &instance_id, BlockEntry *block, int64_t timestamp) {}

    // ---- 生命周期管理 ----
    virtual void Finalize(const std::string &instance_id, int64_t final_timestamp) {}
    virtual void Export(const std::string &instance_id, const OptimizerConfig &config) {}
    virtual void Reset(const std::string &instance_id) {}

    const std::string &name() const { return name_; }

protected:
    explicit StatsTracker(std::string name) : name_(std::move(name)) {}

private:
    std::string name_;
};

} // namespace kv_cache_manager
