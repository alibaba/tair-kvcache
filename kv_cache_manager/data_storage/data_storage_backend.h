#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/common_define.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class DataStorageBackend {
public:
    DataStorageBackend() = delete;
    explicit DataStorageBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : metrics_registry_(std::move(metrics_registry)) {}

    virtual ~DataStorageBackend() = default;
    virtual DataStorageType GetType() = 0;
    virtual bool Available() = 0;
    virtual double GetStorageUsageRatio(const std::string &trace_id) const = 0;
    inline bool IsOpen() const { return is_open_.load(std::memory_order_relaxed); }
    inline void SetOpen(bool open) { is_open_.store(open, std::memory_order_relaxed); }
    inline void SetAvailable(bool available) { is_available_.store(available, std::memory_order_relaxed); }
    std::shared_ptr<DataStorageMetricsCollector> GetMetricsCollector() { return metrics_collector_; }
    virtual const StorageConfig &GetStorageConfig() { return config_; }

public:
    virtual ErrorCode Open(const StorageConfig &config, const std::string &trace_id) {
        config_ = config;
        metrics_collector_ = std::make_shared<DataStorageMetricsCollector>(
            metrics_registry_,
            MetricsTags{{"type", ToString(config.type())}, {"unique_name", config.global_unique_name()}});
        if (!metrics_collector_->Init()) {
            metrics_collector_ = nullptr;
        }
        return DoOpen(config, trace_id);
    }
    virtual ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) = 0;
    virtual ErrorCode Close() = 0;
    virtual std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                                     size_t size_per_key,
                                                                     const std::string &trace_id,
                                                                     std::function<void()> cb) = 0;
    virtual std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &trace_id, std::function<void()> cb) = 0;
    virtual std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) = 0;
    virtual std::vector<bool> MightExist(const std::vector<DataStorageUri> &storage_uris) {
        // a low-latency version of Exist()
        // implementation is required to return ASAP;
        // or it should rather return false-positive result, e.g.,
        // all true if low-latency can not be guaranteed
        return std::vector<bool>(storage_uris.size(), true);
    }
    virtual std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) = 0;
    virtual std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) = 0;

    // 跨存储后端复制：把 src_uris[i] 的数据复制到 dst_uris[i]（多层存储迁移 Copy 路径用）。
    // 可选能力，默认不支持（返回逐项 EC_UNIMPLEMENTED）；支持复制的后端（如 TairMempool）按需 override。
    // 同步语义：返回时复制已完成（内部可异步提交+轮询）。
    // 前置：src_uris.size() == dst_uris.size()。
    // 后置：返回 vector.size() 必须等于 src_uris.size()，逐项对应每个 URI 的复制结果。
    //        调用方依赖此长度等式判断完整性；短返回会被视为整体失败（F-27）。
    virtual std::vector<ErrorCode> Copy(const std::vector<DataStorageUri> &src_uris,
                                        const std::vector<DataStorageUri> &dst_uris,
                                        const std::string &trace_id) {
        return std::vector<ErrorCode>(src_uris.size(), EC_UNIMPLEMENTED);
    }

protected:
    inline bool IsAvailable() const { return is_available_.load(std::memory_order_relaxed); }

protected:
    StorageConfig config_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<DataStorageMetricsCollector> metrics_collector_;

private:
    std::atomic_bool is_open_ = false;
    std::atomic_bool is_available_ = false;
};

} // namespace kv_cache_manager
