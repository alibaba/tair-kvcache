#pragma once

#include <atomic>
#include <cstdint>
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
    virtual void SetAvailable(bool available) { is_available_.store(available, std::memory_order_release); }
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
    //        调用方依赖此长度等式判断完整性；短返回会被视为整体失败。
    virtual std::vector<ErrorCode> Copy(const std::vector<DataStorageUri> &src_uris,
                                        const std::vector<DataStorageUri> &dst_uris,
                                        const std::string &trace_id) {
        return std::vector<ErrorCode>(src_uris.size(), EC_UNIMPLEMENTED);
    }

protected:
    inline bool IsAvailable() const { return is_available_.load(std::memory_order_acquire); }

protected:
    StorageConfig config_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<DataStorageMetricsCollector> metrics_collector_;

private:
    std::atomic_bool is_open_ = false;
    std::atomic_bool is_available_ = false;
};

// Side interface for KVMeta object-lifecycle semantics. Keeping it separate
// preserves DataStorageBackend's ABI and the fixed-block KV-cache vtable.
// KVMeta uses singleton allocations, but a backend is not required to expose a
// generation-aware delete API. Backends that cannot safely replay an
// ambiguous delete must say so explicitly; KVMeta will make at most one
// attempt and prefer a possible physical orphan over deleting a reused object.
class KvMetaDataStorageBackendExtension {
public:
    virtual ~KvMetaDataStorageBackendExtension() = default;

    // Performs one synchronous KVMeta delete attempt. EC_OK means the backend
    // reported successful completion of that attempt. It does not, by itself,
    // make an ambiguous transport failure safe to retry.
    virtual std::vector<ErrorCode> DeleteForKvMeta(const std::vector<DataStorageUri> &storage_uris,
                                                   const std::string &trace_id,
                                                   std::function<void()> cb) = 0;

    // True only when replaying the same URI after an unknown outcome cannot
    // delete a successor allocation. Non-retry-safe backends (for example a
    // reusable legacy GA address without a generation token) are attempted at
    // most once, including across leader recovery.
    virtual bool IsKvMetaDeleteRetrySafe() const noexcept = 0;

    // A transport with already-submitted I/O after a failed write returns a
    // positive quarantine. Synchronous exact-object backends return zero.
    virtual std::int64_t GetFailedWriteCleanupGraceSeconds() const noexcept = 0;

    // Backends whose validation policy differs from their fixed-block Create
    // path opt in here. The implementation may still use the same existing
    // storage API; this hook keeps variable-size checks off the main path.
    // A failed Create that may nevertheless have allocated an object MUST
    // return EC_OUTCOME_UNKNOWN with an invalid URI. EC_NOSPC, EC_NOENT,
    // EC_BADARGS, EC_OUT_OF_LIMIT, EC_UNIMPLEMENTED, EC_CONFIG_ERROR, and
    // EC_CORRUPTION are reserved here for failures known to occur before an
    // allocation is created. This distinction lets a transient local
    // dispatch rejection remain retryable without hiding possible orphans.
    virtual bool HasDedicatedKvMetaCreate() const noexcept { return false; }
    virtual std::vector<std::pair<ErrorCode, DataStorageUri>> CreateForKvMeta(const std::vector<std::string> &keys,
                                                                              std::size_t size_per_key,
                                                                              const std::string &trace_id,
                                                                              std::function<void()> cb) {
        (void)size_per_key;
        (void)trace_id;
        (void)cb;
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size(), {EC_UNIMPLEMENTED, DataStorageUri{}});
    }
};

} // namespace kv_cache_manager
