#pragma once

#include <algorithm>
#include <atomic>
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

struct DataStorageCreateRequest {
    std::vector<std::string> keys;
    size_t size_per_key;
};

using DataStorageCreateResult = std::vector<std::pair<ErrorCode, DataStorageUri>>;

struct DataStorageBatchCreateResult {
    // One result per requested key, grouped in the original request order.
    std::vector<DataStorageCreateResult> results;
    // Allocations that cannot be associated with a key after a malformed response.
    // These are distinct from successful results above; a nonempty list means
    // the batch failed and the caller must reclaim all successful allocations.
    std::vector<DataStorageUri> cleanup_uris;
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
    // Results match the request/key counts and original order, including failures.
    // If a Create response has an invalid count, fail that request and preserve
    // its successful allocations in cleanup_uris for the caller to reclaim.
    // Backends may override the allocation strategy without exposing it to callers.
    // The default preserves request boundaries and stops after the first failure.
    // cb runs once on the calling thread after the entire batch completes.
    virtual DataStorageBatchCreateResult CreateBatch(const std::vector<DataStorageCreateRequest> &requests,
                                                     const std::string &trace_id,
                                                     std::function<void()> cb) {
        DataStorageBatchCreateResult batch_result;
        batch_result.results.resize(requests.size());
        bool failed = false;
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto &request = requests[i];
            auto &result = batch_result.results[i];
            if (request.keys.empty()) {
                continue;
            }
            if (failed) {
                result.assign(request.keys.size(), {EC_ERROR, DataStorageUri{}});
                continue;
            }
            result = Create(request.keys, request.size_per_key, trace_id, nullptr);
            if (result.size() != request.keys.size()) {
                for (auto &[ec, uri] : result) {
                    if (ec == EC_OK) {
                        batch_result.cleanup_uris.push_back(std::move(uri));
                    }
                }
                result.assign(request.keys.size(), {EC_ERROR, DataStorageUri{}});
                failed = true;
            } else {
                failed =
                    std::any_of(result.begin(), result.end(), [](const auto &item) { return item.first != EC_OK; });
            }
        }
        if (cb) {
            cb();
        }
        return batch_result;
    }
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

} // namespace kv_cache_manager
