#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/common_define.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

enum class AsyncCopyOutcome : int32_t {
    kSuccess = 0,
    kFailed = 1,
    kCancelled = 2,
    kUnknown = 3,
};

// `error` is diagnostic only. A successful Copy requires terminal + safe.
// Failure/unknown cleanup uses ordinary Delete; an async-capable backend must
// reject deletion while Copy still owns the allocation, without reusing it.
// A timeout never proves that the remote operation has drained.
struct AsyncCopyItemResult {
    AsyncCopyOutcome outcome = AsyncCopyOutcome::kUnknown;
    ErrorCode error = EC_UNKNOWN;
    bool terminal = false;
    bool safe_to_reuse_dst = false;
    std::string backend_task_id;
    std::string detail;
};

struct AsyncCopyBatchResult {
    ErrorCode status = EC_UNKNOWN;
    std::vector<AsyncCopyItemResult> items;
    std::string detail;

    bool AllSucceeded() const {
        return !items.empty() && std::all_of(items.begin(), items.end(), [](const auto &item) {
            return item.terminal && item.safe_to_reuse_dst && item.outcome == AsyncCopyOutcome::kSuccess;
        });
    }
    bool AllTerminalAndSafe() const {
        return !items.empty() && std::all_of(items.begin(), items.end(), [](const auto &item) {
            return item.terminal && item.safe_to_reuse_dst;
        });
    }
};

struct AsyncCopyOptions {
    static constexpr int64_t kMinHttpTimeoutWindowsPerDeadline = 16;

    int64_t operation_deadline_ms = 10 * 60 * 1000;
    int64_t initial_poll_interval_ms = 20;
    int64_t max_poll_interval_ms = 1000;
    // Per-request control-plane budgets.  They are deliberately independent
    // from the end-to-end operation deadline: an HTTP timeout never proves
    // that a remote Copy task is terminal or that its destination is safe.
    int64_t connect_timeout_ms = 1000;
    int64_t submit_timeout_ms = 3000;
    int64_t query_timeout_ms = 3000;
};

using AsyncCopyCompletion = std::function<void(AsyncCopyBatchResult)>;

struct AsyncCopySubmitResult {
    ErrorCode status = EC_UNIMPLEMENTED;
    // Acknowledges local coordinator handoff, not remote Copy completion.
    bool accepted = false;
    // Kept for backends that cannot make local handoff atomic.  Native PACE
    // asynchronous copy always returns false here because no POST happens
    // before the handoff result is returned.
    bool acceptance_unknown = false;
    std::string operation_id;
    std::string detail;
};

class DataStorageBackend {
public:
    DataStorageBackend() = delete;
    explicit DataStorageBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : metrics_registry_(std::move(metrics_registry)) {}

    virtual ~DataStorageBackend() = default;
    virtual DataStorageType GetType() = 0;
    // Whether GC may omit Delete for URIs rejected by Exist/MightExist.
    // Default to deletion: a missing URI does not always imply reclaimed data.
    // This policy does not affect metadata-only cleanup.
    virtual bool ShouldSkipConfirmedMissingBackendDelete() const { return false; }
    virtual bool Available() = 0;
    virtual double GetStorageUsageRatio(const std::string &trace_id) const = 0;
    inline bool IsOpen() const { return is_open_.load(std::memory_order_relaxed); }
    inline void SetOpen(bool open) { is_open_.store(open, std::memory_order_relaxed); }
    virtual void SetAvailable(bool available) { is_available_.store(available, std::memory_order_release); }

    // Returns true if this backend can accept write operations (Create/Delete).
    // When meta-service is unavailable, TairMempoolBackend returns false to
    // indicate read-only mode, while still allowing read operations (Exist/MightExist).
    // Default implementation returns Available() for backward compatibility.
    virtual bool Writable() { return Available(); }

    // True while the backend still serves reads but can no longer reclaim
    // physical objects because its write path is down (read-only degradation).
    // Existence verdicts are untrustworthy in this state and a delete cannot
    // free space, so callers must defer metadata cleanup instead of dropping
    // metadata and leaking storage. Backends without a read-only degraded mode
    // keep the default false, which preserves their existing cleanup behaviour.
    virtual bool InReadOnlyMode() { return false; }

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

    // Opt-in also requires Copy/Delete allocation safety: Delete must return a
    // retryable failure while a Copy may still access that allocation, retaining
    // the allocation and preventing reuse. This applies after caller timeout,
    // cancellation and manager restart; Copy task persistence is not provided.
    virtual bool SupportsAsyncCopy() const { return false; }

    // Native asynchronous copy.  A successful return means the backend owns
    // the completion callback and will invoke it exactly once.  A rejected
    // request has no remote side effect unless acceptance_unknown is true.
    virtual AsyncCopySubmitResult CopyAsync(const std::vector<DataStorageUri> &src_uris,
                                            const std::vector<DataStorageUri> &dst_uris,
                                            const std::string &operation_id,
                                            const std::string &trace_id,
                                            const AsyncCopyOptions &options,
                                            AsyncCopyCompletion completion) {
        (void)dst_uris;
        (void)operation_id;
        (void)trace_id;
        (void)options;
        (void)completion;
        AsyncCopySubmitResult result;
        result.status = src_uris.empty() ? EC_BADARGS : EC_UNIMPLEMENTED;
        result.detail = "backend does not support asynchronous copy";
        return result;
    }

    // Best-effort request only.  EC_OK means the coordinator accepted the
    // cancellation intent, not that PACE proved the target drained.
    virtual ErrorCode RequestCancelAsyncCopy(const std::string &operation_id) {
        (void)operation_id;
        return EC_UNIMPLEMENTED;
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

} // namespace kv_cache_manager
