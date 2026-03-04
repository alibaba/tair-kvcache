#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/analysis/stats_tracker.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"

namespace kv_cache_manager {

// ============================================================================
// 命中率追踪器
//
// 收集每次读写请求的统计数据，计算并导出命中率时序 CSV。
// ============================================================================
class HitRateTracker : public StatsTracker {
public:
    HitRateTracker();

    void OnReadComplete(const std::string &instance_id, const ReadRecord &record) override;
    void OnWriteComplete(const std::string &instance_id, const WriteRecord &record) override;

    void Export(const std::string &instance_id, const OptimizerConfig &config) override;
    void Reset(const std::string &instance_id) override;

    // 供 OptimizerManager::WriteCache/GetCacheLocation 查询最近一次记录
    const ReadRecord *LastReadRecord(const std::string &instance_id) const;
    const WriteRecord *LastWriteRecord(const std::string &instance_id) const;

private:
    struct InstanceData {
        std::vector<ReadRecord> read_records;
        std::vector<WriteRecord> write_records;
    };

    void ExportHitRates(const std::string &instance_id, const InstanceData &data, const OptimizerConfig &config);

    std::unordered_map<std::string, InstanceData> instance_data_;
};

} // namespace kv_cache_manager
