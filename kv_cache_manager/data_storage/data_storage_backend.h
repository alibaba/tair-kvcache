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

// `error` is diagnostic only.  Target reuse is authorized exclusively by
// `terminal && safe_to_reuse_dst`; callers must never infer safety from an
// ErrorCode, timeout, task age or HTTP status.
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
    int64_t operation_deadline_ms = 10 * 60 * 1000;
    int64_t initial_poll_interval_ms = 20;
    int64_t max_poll_interval_ms = 1000;
};

using AsyncCopyCompletion = std::function<void(AsyncCopyBatchResult)>;

// Result of the short remote submission phase.  This is deliberately
// separate from AsyncCopySubmitResult: CopyAsync() returns after a local
// coordinator handoff, while this result reports whether PACE accepted the
// POST and supplies the durable task handles needed for leader recovery.
struct AsyncCopyRemoteSubmitResult {
    ErrorCode status = EC_UNKNOWN;
    bool accepted = false;
    // True means the POST may have reached PACE but KVCM did not receive an
    // authoritative task handle set.  The destination must remain fenced.
    bool acceptance_unknown = false;
    std::string operation_id;
    std::vector<std::string> backend_task_ids;
    std::string detail;
};

using AsyncCopyRemoteSubmitCompletion = std::function<void(AsyncCopyRemoteSubmitResult)>;

struct AsyncCopySubmitResult {
    ErrorCode status = EC_UNIMPLEMENTED;
    // This flag acknowledges only the local coordinator handoff.  Remote PACE
    // acceptance is reported later through AsyncCopyRemoteSubmitCompletion.
    bool accepted = false;
    // Kept for backends that cannot make local handoff atomic.  Native PACE
    // asynchronous copy always returns false here because no POST happens
    // before the handoff result is returned.
    bool acceptance_unknown = false;
    std::string operation_id;
    std::vector<std::string> backend_task_ids;
    std::string detail;
};

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

    virtual bool SupportsAsyncCopy() const { return false; }

    // Native asynchronous copy.  A successful return means the backend owns
    // the completion callback and will invoke it exactly once.  A rejected
    // request has no remote side effect unless acceptance_unknown is true.
    virtual AsyncCopySubmitResult CopyAsync(const std::vector<DataStorageUri> &src_uris,
                                            const std::vector<DataStorageUri> &dst_uris,
                                            const std::string &operation_id,
                                            const std::string &trace_id,
                                            const AsyncCopyOptions &options,
                                            AsyncCopyRemoteSubmitCompletion remote_submit_completion,
                                            AsyncCopyCompletion completion) {
        (void)dst_uris;
        (void)operation_id;
        (void)trace_id;
        (void)options;
        (void)remote_submit_completion;
        (void)completion;
        AsyncCopySubmitResult result;
        result.status = src_uris.empty() ? EC_BADARGS : EC_UNIMPLEMENTED;
        result.detail = "backend does not support asynchronous copy";
        return result;
    }

    // Reattach a recovered KVCM operation to backend task handles that were
    // durably recorded before a leader restart.  This must query existing
    // tasks only; it must never issue a second physical Copy.
    virtual AsyncCopySubmitResult ResumeAsyncCopy(const std::vector<std::string> &backend_task_ids,
                                                  size_t expected_items,
                                                  const std::string &operation_id,
                                                  const std::string &trace_id,
                                                  const AsyncCopyOptions &options,
                                                  AsyncCopyCompletion completion) {
        (void)backend_task_ids;
        (void)expected_items;
        (void)trace_id;
        (void)options;
        (void)completion;
        AsyncCopySubmitResult result;
        result.operation_id = operation_id;
        result.status = EC_UNIMPLEMENTED;
        result.detail = "backend does not support asynchronous copy recovery";
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
