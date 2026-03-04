#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/analysis/stats_tracker.h"

namespace kv_cache_manager {

// ============================================================================
// 统计采集调度中心
//
// 唯一的统计入口：业务层只与 StatsCollector 交互，
// 内部将事件分发给所有已注册的子 Tracker。
// 新增统计维度只需: 1) 实现 StatsTracker 子类  2) RegisterTracker
// ============================================================================
class StatsCollector {
public:
    StatsCollector() = default;
    ~StatsCollector() = default;

    // ---- 注册子Tracker ----
    void RegisterTracker(std::unique_ptr<StatsTracker> tracker) {
        KVCM_LOG_INFO("Registered stats tracker: %s", tracker->name().c_str());
        trackers_.push_back(std::move(tracker));
    }

    template <typename T, typename... Args>
    T *EmplaceTracker(Args &&...args) {
        auto tracker = std::make_unique<T>(std::forward<Args>(args)...);
        T *ptr = tracker.get();
        RegisterTracker(std::move(tracker));
        return ptr;
    }

    // ---- 事件分发 ----
    void OnReadComplete(const std::string &instance_id, const ReadRecord &record) {
        for (auto &t : trackers_) {
            t->OnReadComplete(instance_id, record);
        }
    }

    void OnWriteComplete(const std::string &instance_id, const WriteRecord &record) {
        for (auto &t : trackers_) {
            t->OnWriteComplete(instance_id, record);
        }
    }

    void OnBlockBirth(const std::string &instance_id, BlockEntry *block, int64_t timestamp) {
        for (auto &t : trackers_) {
            t->OnBlockBirth(instance_id, block, timestamp);
        }
    }

    void OnBlockEviction(const std::string &instance_id, BlockEntry *block, int64_t timestamp) {
        for (auto &t : trackers_) {
            t->OnBlockEviction(instance_id, block, timestamp);
        }
    }

    // ---- 生命周期管理 ----
    void FinalizeAll(const std::string &instance_id, int64_t final_timestamp) {
        for (auto &t : trackers_) {
            t->Finalize(instance_id, final_timestamp);
        }
    }

    void ExportAll(const std::string &instance_id, const OptimizerConfig &config) {
        for (auto &t : trackers_) {
            t->Export(instance_id, config);
        }
    }

    void ResetAll(const std::string &instance_id) {
        for (auto &t : trackers_) {
            t->Reset(instance_id);
        }
    }

    // ---- 查询已注册的子Tracker ----
    template <typename T>
    T *GetTracker() const {
        for (auto &t : trackers_) {
            T *ptr = dynamic_cast<T *>(t.get());
            if (ptr) {
                return ptr;
            }
        }
        return nullptr;
    }

    size_t TrackerCount() const { return trackers_.size(); }

    // ---- per-instance 时间戳管理 ----
    void UpdateTimestamp(const std::string &instance_id, int64_t timestamp) {
        auto &ts = last_trace_timestamp_[instance_id];
        ts = std::max(ts, timestamp);
    }

    int64_t GetLastTimestamp(const std::string &instance_id) const {
        auto it = last_trace_timestamp_.find(instance_id);
        return it != last_trace_timestamp_.end() ? it->second : 0;
    }

private:
    std::vector<std::unique_ptr<StatsTracker>> trackers_;
    std::unordered_map<std::string, int64_t> last_trace_timestamp_;
};

} // namespace kv_cache_manager
