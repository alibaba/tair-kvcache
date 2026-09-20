#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/config/migration_strategy.h"
#include "kv_cache_manager/config/quota_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/kv_meta_instance.h"
#include "kv_cache_manager/manager/kv_meta_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

void ReplaceKvMetaObjectNonce(CacheLocation &location, char fill) {
    DataStorageUri uri(location.location_specs().front().uri());
    std::string path = uri.GetPath();
    const auto nonce_begin = path.rfind('/');
    if (nonce_begin == std::string::npos || path.size() - nonce_begin - 1 != 32) {
        throw std::logic_error("test location does not contain a KVMeta nonce");
    }
    path.replace(nonce_begin + 1, 32, std::string(32, fill));
    uri.SetPath(path);
    location.mutable_location_specs().front().set_uri(uri.ToUriString());
}

class OverlongCreateNfsBackend : public NfsBackend {
public:
    explicit OverlongCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        std::vector<std::pair<ErrorCode, DataStorageUri>> result;
        for (const char *path : {"/malformed/first", "/malformed/second"}) {
            DataStorageUri uri;
            uri.SetProtocol("file");
            uri.SetPath(path);
            uri.SetParam("size", std::to_string(size_per_key));
            result.emplace_back(EC_OK, std::move(uri));
        }
        DataStorageUri foreign_uri;
        foreign_uri.SetProtocol("dummy");
        foreign_uri.SetPath("/must-not-delete-through-nfs");
        foreign_uri.SetParam("size", std::to_string(size_per_key));
        result.emplace_back(EC_OK, std::move(foreign_uri));
        if (cb) {
            cb();
        }
        return result;
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        deleted_uris.insert(deleted_uris.end(), storage_uris.begin(), storage_uris.end());
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<DataStorageUri> deleted_uris;
};

class WrongSizeCreateNfsBackend : public NfsBackend {
public:
    explicit WrongSizeCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        auto result = NfsBackend::Create(keys, size_per_key, trace_id, std::move(cb));
        if (result.size() == 1 && result[0].first == EC_OK) {
            result[0].second.SetParam("size", std::to_string(size_per_key + 1));
        }
        return result;
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        deleted_uris.insert(deleted_uris.end(), storage_uris.begin(), storage_uris.end());
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<DataStorageUri> deleted_uris;
};

class ForeignObjectCreateNfsBackend : public NfsBackend {
public:
    explicit ForeignObjectCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        DataStorageUri uri;
        uri.SetProtocol("file");
        // Preserve the exact KVMeta suffix while moving it outside the
        // configured backend root. A suffix-only ownership check would grant
        // this forged response physical Delete authority.
        uri.SetPath("/foreign/root/" + keys.front());
        uri.SetParam("size", std::to_string(size_per_key));
        if (cb) {
            cb();
        }
        return {{EC_OK, std::move(uri)}};
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        delete_calls += storage_uris.size();
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::size_t delete_calls{0};
};

class FailedCreateWithUriNfsBackend : public NfsBackend {
public:
    explicit FailedCreateWithUriNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        auto result = NfsBackend::Create(keys, size_per_key, trace_id, std::move(cb));
        if (result.size() == 1) {
            result.front().first = EC_IO_ERROR;
        }
        return result;
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        delete_calls += storage_uris.size();
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::size_t delete_calls{0};
};

class CapacityRejectingNfsBackend : public NfsBackend {
public:
    enum class Mode { kOnce, kAlways };

    CapacityRejectingNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry, Mode mode)
        : NfsBackend(std::move(metrics_registry)), mode_(mode) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        const std::size_t attempt = create_attempts_.fetch_add(1, std::memory_order_acq_rel);
        if (mode_ == Mode::kAlways || attempt == 0) {
            if (cb) {
                cb();
            }
            return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size(), {EC_NOSPC, DataStorageUri{}});
        }
        return NfsBackend::Create(keys, size_per_key, trace_id, std::move(cb));
    }

    std::size_t CreateAttempts() const { return create_attempts_.load(std::memory_order_acquire); }

private:
    Mode mode_;
    std::atomic<std::size_t> create_attempts_{0};
};

class AliasedOverlongCreateNfsBackend : public NfsBackend {
public:
    explicit AliasedOverlongCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        std::vector<std::pair<ErrorCode, DataStorageUri>> result;
        for (const std::size_t logical_size : {size_per_key, size_per_key + 1}) {
            DataStorageUri uri;
            uri.SetProtocol("file");
            uri.SetPath("/malformed/shared-physical-object");
            uri.SetParam("size", std::to_string(logical_size));
            result.emplace_back(EC_OK, std::move(uri));
        }
        if (cb) {
            cb();
        }
        return result;
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        delete_calls += storage_uris.size();
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::size_t delete_calls{0};
};

class LaterMalformedAliasedCreateNfsBackend : public NfsBackend {
public:
    explicit LaterMalformedAliasedCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        ++create_calls_;
        std::vector<std::pair<ErrorCode, DataStorageUri>> result;
        const auto append = [&](const std::string &path) {
            DataStorageUri uri;
            uri.SetProtocol("file");
            uri.SetPath(path);
            uri.SetParam("size", std::to_string(size_per_key));
            result.emplace_back(EC_OK, std::move(uri));
        };
        if (create_calls_ == 1 && keys.size() == 1) {
            // The first singleton response is independently attributable to
            // this Create call and names the exact generated KVMeta key under
            // the backend's configured root.
            result = NfsBackend::Create(keys, size_per_key, trace_id, nullptr);
            if (result.size() == 1 && result.front().first == EC_OK) {
                first_path_ = result.front().second.GetPath();
            }
        } else if (create_calls_ == 2) {
            // The overlong second response aliases the first call's physical
            // allocation and also contains one genuinely new object.
            append(first_path_);
            append("/malformed/new-extra-allocation");
        }
        if (cb) {
            cb();
        }
        return result;
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        for (const auto &uri : storage_uris) {
            deleted_paths.push_back(uri.GetPath());
        }
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<std::string> deleted_paths;
    const std::string &first_path() const { return first_path_; }

private:
    std::size_t create_calls_{0};
    std::string first_path_;
};

class NonSingletonCreateNfsBackend : public NfsBackend {
public:
    explicit NonSingletonCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        DataStorageUri uri;
        uri.SetProtocol("file");
        uri.SetPath("/malformed/shared-file");
        uri.SetParam("size", std::to_string(size_per_key));
        uri.SetParam("blkid", "1");
        if (cb) {
            cb();
        }
        return {{EC_OK, std::move(uri)}};
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        delete_calls += storage_uris.size();
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::size_t delete_calls{0};
};

class DuplicateSingletonCreateTairMempoolBackend : public DataStorageBackend {
public:
    static constexpr const char *kProviderIncarnation = "01234567-89ab-4def-8abc-0123456789ab";

    explicit DuplicateSingletonCreateTairMempoolBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : DataStorageBackend(std::move(metrics_registry)) {}

    DataStorageType GetType() override { return DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL; }
    bool Available() override { return IsOpen() && IsAvailable(); }
    double GetStorageUsageRatio(const std::string &) const override { return 0.0; }

    ErrorCode DoOpen(const StorageConfig &, const std::string &) override {
        SetOpen(true);
        SetAvailable(true);
        return EC_OK;
    }

    ErrorCode Close() override {
        SetAvailable(false);
        SetOpen(false);
        return EC_OK;
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        DataStorageUri uri;
        uri.SetProtocol(kTairMempoolUriScheme);
        uri.SetPath("/424242");
        if (!keys.empty()) {
            uri.SetParam("allocation_token", keys.front());
        }
        uri.SetParam("provider_incarnation", kProviderIncarnation);
        uri.SetParam("size", std::to_string(size_per_key));
        if (cb) {
            cb();
        }
        return {{EC_OK, std::move(uri)}};
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        delete_calls += storage_uris.size();
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::size_t delete_calls{0};

    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<bool>(storage_uris.size(), true);
    }

    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }
};

class ConfigurableMediaTairMempoolBackend : public DataStorageBackend {
public:
    ConfigurableMediaTairMempoolBackend(std::shared_ptr<MetricsRegistry> metrics_registry,
                                        DataStorageType type,
                                        std::uint16_t returned_media_type)
        : DataStorageBackend(std::move(metrics_registry)), type_(type), returned_media_type_(returned_media_type) {}

    DataStorageType GetType() override { return type_; }
    bool Available() override { return IsOpen() && IsAvailable(); }
    double GetStorageUsageRatio(const std::string &) const override { return 0.0; }

    ErrorCode DoOpen(const StorageConfig &, const std::string &) override {
        SetOpen(true);
        SetAvailable(true);
        return EC_OK;
    }

    ErrorCode Close() override {
        SetAvailable(false);
        SetOpen(false);
        return EC_OK;
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        DataStorageUri uri;
        uri.SetProtocol(kTairMempoolUriScheme);
        uri.SetPath("/424242");
        if (!keys.empty()) {
            uri.SetParam("allocation_token", keys.front());
        }
        uri.SetParam("media_type", std::to_string(returned_media_type_));
        uri.SetParam("provider_incarnation", DuplicateSingletonCreateTairMempoolBackend::kProviderIncarnation);
        uri.SetParam("size", std::to_string(size_per_key));
        if (cb) {
            cb();
        }
        return {{EC_OK, std::move(uri)}};
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        delete_calls += storage_uris.size();
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<bool>(storage_uris.size(), true);
    }

    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::size_t delete_calls{0};

private:
    DataStorageType type_;
    std::uint16_t returned_media_type_;
};

class HookedCreateNfsBackend : public NfsBackend {
public:
    explicit HookedCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    void SetOneShotAfterCreateHook(std::function<void()> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        after_create_hook_ = std::move(hook);
    }

    void FailDeletes(bool fail) { fail_deletes_.store(fail, std::memory_order_release); }

    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override {
        delete_attempts_.fetch_add(1, std::memory_order_acq_rel);
        if (!fail_deletes_.load(std::memory_order_acquire)) {
            return NfsBackend::Delete(storage_uris, trace_id, std::move(cb));
        }
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_IO_ERROR);
    }

    std::size_t DeleteAttempts() const { return delete_attempts_.load(std::memory_order_acquire); }

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        auto result = NfsBackend::Create(keys, size_per_key, trace_id, std::move(cb));
        std::function<void()> hook;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            hook = std::move(after_create_hook_);
        }
        if (hook) {
            hook();
        }
        return result;
    }

private:
    std::mutex mutex_;
    std::function<void()> after_create_hook_;
    std::atomic<bool> fail_deletes_{false};
    std::atomic<std::size_t> delete_attempts_{0};
};

class HookedReusedSingletonTairMempoolBackend : public DuplicateSingletonCreateTairMempoolBackend {
public:
    explicit HookedReusedSingletonTairMempoolBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : DuplicateSingletonCreateTairMempoolBackend(std::move(metrics_registry)) {}

    void SetOneShotAfterCreateHook(std::function<void()> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        after_create_hook_ = std::move(hook);
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        auto result = DuplicateSingletonCreateTairMempoolBackend::Create(keys, size_per_key, trace_id, std::move(cb));
        std::function<void()> hook;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            hook = std::move(after_create_hook_);
        }
        if (hook) {
            hook();
        }
        return result;
    }

private:
    std::mutex mutex_;
    std::function<void()> after_create_hook_;
};

class BlockingDeleteNfsBackend : public NfsBackend {
public:
    explicit BlockingDeleteNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        std::unique_lock<std::mutex> lock(mutex_);
        delete_entered_ = true;
        condition_.notify_all();
        condition_.wait(lock, [&]() { return release_delete_; });
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    bool WaitForDelete(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return delete_entered_; });
    }

    void ReleaseDelete() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_delete_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool delete_entered_{false};
    bool release_delete_{false};
};

class QuarantinedWriteNfsBackend : public NfsBackend, public KvMetaDataStorageBackendExtension {
public:
    explicit QuarantinedWriteNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry,
                                        std::int64_t cleanup_grace_seconds = 1)
        : NfsBackend(std::move(metrics_registry)), cleanup_grace_seconds_(cleanup_grace_seconds) {}

    std::int64_t GetFailedWriteCleanupGraceSeconds() const noexcept override { return cleanup_grace_seconds_; }

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        create_calls_.fetch_add(keys.size(), std::memory_order_relaxed);
        return NfsBackend::Create(keys, size_per_key, trace_id, std::move(cb));
    }

    std::vector<ErrorCode> DeleteAndConfirmAbsent(const std::vector<DataStorageUri> &storage_uris,
                                                  const std::string &trace_id,
                                                  std::function<void()> cb) override {
        return Delete(storage_uris, trace_id, std::move(cb));
    }

    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override {
        delete_calls_.fetch_add(storage_uris.size(), std::memory_order_relaxed);
        return NfsBackend::Delete(storage_uris, trace_id, std::move(cb));
    }

    std::size_t DeleteCalls() const noexcept { return delete_calls_.load(std::memory_order_relaxed); }
    std::size_t CreateCalls() const noexcept { return create_calls_.load(std::memory_order_relaxed); }

private:
    const std::int64_t cleanup_grace_seconds_;
    std::atomic<std::size_t> create_calls_{0};
    std::atomic<std::size_t> delete_calls_{0};
};

class ProvisionalCommitNfsBackend : public NfsBackend, public KvMetaDataStorageBackendExtension {
public:
    explicit ProvisionalCommitNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    bool RequiresKvMetaCreateCommit() const noexcept override { return true; }
    std::int64_t GetKvMetaControlRequestTimeoutSeconds() const noexcept override {
        return control_timeout_seconds_;
    }
    std::int64_t GetFailedWriteCleanupGraceSeconds() const noexcept override { return 0; }

    std::vector<ErrorCode> CommitKvMetaCreate(const std::vector<std::string> &allocation_keys,
                                               const std::string &) override {
        ++commit_calls_;
        commit_batches_.push_back(allocation_keys);
        committed_keys_ = allocation_keys;
        if (commit_delay_.count() != 0) {
            std::this_thread::sleep_for(commit_delay_);
        }
        return std::vector<ErrorCode>(allocation_keys.size(), commit_error_);
    }

    std::vector<ErrorCode> DeleteAndConfirmAbsent(const std::vector<DataStorageUri> &storage_uris,
                                                  const std::string &trace_id,
                                                  std::function<void()> cb) override {
        return Delete(storage_uris, trace_id, std::move(cb));
    }

    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override {
        delete_calls_ += storage_uris.size();
        if (delete_error_ != EC_OK) {
            if (cb) {
                cb();
            }
            return std::vector<ErrorCode>(storage_uris.size(), delete_error_);
        }
        return NfsBackend::Delete(storage_uris, trace_id, std::move(cb));
    }

    ErrorCode commit_error_{EC_OK};
    ErrorCode delete_error_{EC_OK};
    std::size_t commit_calls_{0};
    std::size_t delete_calls_{0};
    std::vector<std::string> committed_keys_;
    std::vector<std::vector<std::string>> commit_batches_;
    std::chrono::milliseconds commit_delay_{0};
    std::int64_t control_timeout_seconds_{1};
};

class BlockingThenFailDeleteNfsBackend : public NfsBackend {
public:
    explicit BlockingThenFailDeleteNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        bool fail = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ++delete_attempts_;
            fail = delete_attempts_ > 1;
            if (!fail) {
                first_delete_entered_ = true;
                condition_.notify_all();
                condition_.wait(lock, [&]() { return release_first_delete_; });
            }
        }
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), fail ? EC_IO_ERROR : EC_OK);
    }

    bool WaitForFirstDelete(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return first_delete_entered_; });
    }

    void ReleaseFirstDelete() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_first_delete_ = true;
        condition_.notify_all();
    }

    std::size_t DeleteAttempts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return delete_attempts_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool first_delete_entered_{false};
    bool release_first_delete_{false};
    std::size_t delete_attempts_{0};
};

class FaultingDeleteNfsBackend : public NfsBackend {
public:
    enum class Mode {
        kSuccess,
        kError,
        kShortResult,
        kStandardException,
        kUnknownException,
    };

    FaultingDeleteNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry, Mode mode)
        : NfsBackend(std::move(metrics_registry)), mode_(mode) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        Mode mode;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++delete_attempts_;
            mode = mode_;
            condition_.notify_all();
        }
        if (cb) {
            cb();
        }
        switch (mode) {
        case Mode::kSuccess:
            return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
        case Mode::kError:
            return std::vector<ErrorCode>(storage_uris.size(), EC_IO_ERROR);
        case Mode::kShortResult:
            return std::vector<ErrorCode>(storage_uris.empty() ? 0 : storage_uris.size() - 1, EC_OK);
        case Mode::kStandardException:
            throw std::runtime_error("injected secret provider detail");
        case Mode::kUnknownException:
            throw 17;
        }
        throw std::runtime_error("unreachable fault mode");
    }

    void SetMode(Mode mode) {
        std::lock_guard<std::mutex> lock(mutex_);
        mode_ = mode;
    }

    bool WaitForDeleteAttempts(std::size_t expected, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return delete_attempts_ >= expected; });
    }

    std::size_t DeleteAttempts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return delete_attempts_;
    }

private:
    Mode mode_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t delete_attempts_{0};
};

class ThrowingSecondCreateNfsBackend : public NfsBackend {
public:
    explicit ThrowingSecondCreateNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        ++create_attempts_;
        if (create_attempts_ == 2) {
            throw std::runtime_error("injected secret create detail");
        }
        return NfsBackend::Create(keys, size_per_key, trace_id, std::move(cb));
    }

    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override {
        delete_items_ += storage_uris.size();
        return NfsBackend::Delete(storage_uris, trace_id, std::move(cb));
    }

    std::size_t CreateAttempts() const { return create_attempts_; }
    std::size_t DeleteItems() const { return delete_items_; }

private:
    std::size_t create_attempts_{0};
    std::size_t delete_items_{0};
};

class FailNextMaintenanceDeleteSyncBackend : public MetaLocalBackend {
public:
    void FailNextMaintenanceDeleteSync() {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_next_delete_sync_ = true;
    }

    std::vector<ErrorCode> DeleteLocationsForMaintenance(RequestContext *request_context,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdsPerKey &location_ids) noexcept override {
        auto result = MetaLocalBackend::DeleteLocationsForMaintenance(request_context, keys, location_ids);
        ArmFailureAfterSuccessfulDelete(result);
        return result;
    }

    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept override {
        auto result = MetaLocalBackend::Delete(request_context, keys);
        ArmFailureAfterSuccessfulDelete(result);
        return result;
    }

    bool Sync(const KeyTypeVec &keys) noexcept override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (fail_next_sync_) {
                fail_next_sync_ = false;
                sync_failed_ = true;
                condition_.notify_all();
                return false;
            }
        }
        return MetaLocalBackend::Sync(keys);
    }

    bool WaitForSyncFailure(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return sync_failed_; });
    }

private:
    void ArmFailureAfterSuccessfulDelete(const std::vector<ErrorCode> &result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fail_next_delete_sync_ &&
            std::any_of(result.begin(), result.end(), [](const ErrorCode ec) { return ec == EC_OK; })) {
            fail_next_delete_sync_ = false;
            fail_next_sync_ = true;
        }
    }
    std::mutex mutex_;
    std::condition_variable condition_;
    bool fail_next_delete_sync_{false};
    bool fail_next_sync_{false};
    bool sync_failed_{false};
};

class FailNextMaintenanceUpsertBackend : public MetaLocalBackend {
public:
    void FailNextUpsert() noexcept { fail_next_upsert_.store(true, std::memory_order_release); }

    std::size_t FailedUpserts() const noexcept { return failed_upserts_.load(std::memory_order_acquire); }

    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override {
        if (fail_next_upsert_.exchange(false, std::memory_order_acq_rel)) {
            failed_upserts_.fetch_add(1, std::memory_order_acq_rel);
            return std::vector<ErrorCode>(keys.size(), EC_IO_ERROR);
        }
        return MetaLocalBackend::Upsert(request_context, keys, locations, properties);
    }

private:
    std::atomic<bool> fail_next_upsert_{false};
    std::atomic<std::size_t> failed_upserts_{0};
};

class OversamplingMetaLocalBackend : public MetaLocalBackend {
public:
    ErrorCode SampleReclaimCandidates(RequestContext *request_context,
                                      int64_t count,
                                      ReclaimCandidateVector &out_candidates,
                                      bool require_read_success) noexcept override {
        const auto ec =
            MetaLocalBackend::SampleReclaimCandidates(request_context, count, out_candidates, require_read_success);
        if (ec == EC_OK && !out_candidates.empty()) {
            out_candidates.push_back(out_candidates.front());
        }
        return ec;
    }
};

class ControlledSyncMetaLocalBackend : public MetaLocalBackend {
public:
    void DelaySyncAfter(std::size_t successful_syncs, std::chrono::milliseconds delay) {
        std::lock_guard<std::mutex> lock(mutex_);
        delay_after_syncs_ = successful_syncs;
        delay_ = delay;
    }

    void BlockSyncAfter(std::size_t successful_syncs) {
        std::lock_guard<std::mutex> lock(mutex_);
        block_after_syncs_ = successful_syncs;
        release_blocked_sync_ = false;
    }

    bool WaitForBlockedSyncs(std::size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return blocked_sync_count_ >= count; });
    }

    void ReleaseBlockedSync() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_blocked_sync_ = true;
        condition_.notify_all();
    }

    void FailSyncAfter(std::size_t successful_syncs, std::size_t failure_count = 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_after_syncs_ = failure_count == 0 ? std::nullopt : std::optional<std::size_t>{successful_syncs};
        remaining_failures_ = failure_count;
        sync_failure_count_ = 0;
    }

    bool Sync(const KeyTypeVec &keys) noexcept override {
        std::chrono::milliseconds delay{0};
        bool fail = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (block_after_syncs_) {
                if (*block_after_syncs_ == 0) {
                    block_after_syncs_.reset();
                    ++blocked_sync_count_;
                    condition_.notify_all();
                    // Bound the test fault itself so a failed assertion can
                    // never strand a production-like worker indefinitely.
                    condition_.wait_for(lock, std::chrono::seconds(5), [&]() { return release_blocked_sync_; });
                } else {
                    --*block_after_syncs_;
                }
            }
            if (delay_after_syncs_) {
                if (*delay_after_syncs_ == 0) {
                    delay = delay_;
                    delay_after_syncs_.reset();
                } else {
                    --*delay_after_syncs_;
                }
            }
            if (fail_after_syncs_) {
                if (*fail_after_syncs_ == 0) {
                    fail = true;
                    ++sync_failure_count_;
                    if (remaining_failures_ <= 1) {
                        remaining_failures_ = 0;
                        fail_after_syncs_.reset();
                    } else {
                        --remaining_failures_;
                    }
                    condition_.notify_all();
                } else {
                    --*fail_after_syncs_;
                }
            }
        }
        if (delay.count() > 0) {
            std::this_thread::sleep_for(delay);
        }
        if (fail) {
            return false;
        }
        return MetaLocalBackend::Sync(keys);
    }

    bool WaitForSyncFailure(std::chrono::milliseconds timeout) { return WaitForSyncFailures(1, timeout); }

    bool WaitForSyncFailures(std::size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&]() { return sync_failure_count_ >= count; });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<std::size_t> delay_after_syncs_;
    std::optional<std::size_t> block_after_syncs_;
    std::optional<std::size_t> fail_after_syncs_;
    std::chrono::milliseconds delay_{0};
    bool release_blocked_sync_{false};
    std::size_t remaining_failures_{0};
    std::size_t sync_failure_count_{0};
    std::size_t blocked_sync_count_{0};
};

class ConflictingTargetedReadMetaLocalBackend : public MetaLocalBackend {
public:
    void ConflictOnTargetedRead(std::size_t one_based_call) noexcept {
        targeted_read_count_.store(0, std::memory_order_release);
        conflict_on_call_.store(one_based_call, std::memory_order_release);
    }

    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept override {
        auto result = MetaLocalBackend::GetLocations(request_context, keys, location_ids, out_locations);
        const std::size_t call = targeted_read_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (call != conflict_on_call_.load(std::memory_order_acquire)) {
            return result;
        }
        conflict_on_call_.store(0, std::memory_order_release);
        if (result.size() != 1 || result.front().size() != 1 || result.front().front() != EC_OK ||
            out_locations.size() != 1 || out_locations.front().size() != 1 || !out_locations.front().front()) {
            return result;
        }
        auto replacement = std::make_shared<CacheLocation>(*out_locations.front().front());
        ReplaceKvMetaObjectNonce(*replacement, 'y');
        out_locations.front().front() = std::move(replacement);
        return result;
    }

private:
    std::atomic<std::size_t> targeted_read_count_{0};
    std::atomic<std::size_t> conflict_on_call_{0};
};

class MalformedMooncakeBackend : public DataStorageBackend {
public:
    explicit MalformedMooncakeBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : DataStorageBackend(std::move(metrics_registry)) {}

    DataStorageType GetType() override { return DataStorageType::DATA_STORAGE_TYPE_MOONCAKE; }
    bool Available() override { return IsOpen() && IsAvailable(); }
    double GetStorageUsageRatio(const std::string &) const override { return 0.0; }

    ErrorCode DoOpen(const StorageConfig &, const std::string &) override {
        SetOpen(true);
        SetAvailable(true);
        return EC_OK;
    }

    ErrorCode Close() override {
        SetAvailable(false);
        SetOpen(false);
        return EC_OK;
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        DataStorageUri uri;
        uri.SetProtocol(ToString(GetType()));
        uri.SetPath("/");
        uri.SetParam("size", std::to_string(size_per_key));
        if (cb) {
            cb();
        }
        return {{EC_OK, std::move(uri)}};
    }

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<bool>(storage_uris.size(), true);
    }

    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }
};

} // namespace

class KvMetaManagerTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        ASSERT_TRUE(registry_manager_->Init());

        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        ASSERT_TRUE(cache_manager_->Init());

        StartupConfigLoader loader;
        ASSERT_TRUE(loader.Init(registry_manager_));
        ASSERT_TRUE(loader.Load(""));

        manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
        ASSERT_TRUE(manager_->Init());
        ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, "default", kInstanceId, "emb-test").first);
    }

    void TearDown() override {
        manager_->Shutdown();
        manager_.reset();
        cache_manager_.reset();
        registry_manager_.reset();
        metrics_registry_.reset();
    }

    void CreateReclaimGroup(const std::string &group_name,
                            const std::string &instance_id,
                            std::int64_t capacity,
                            double threshold,
                            std::int32_t delay_before_delete_ms,
                            std::size_t max_key_count = MetaIndexerConfig::kDefaultMaxKeyCount,
                            std::optional<std::int64_t> storage_type_capacity = std::nullopt,
                            ReclaimPolicy reclaim_policy = ReclaimPolicy::POLICY_LRU,
                            ErrorCode expected_registration_ec = EC_OK,
                            DataStorageType storage_type = DataStorageType::DATA_STORAGE_TYPE_NFS,
                            const std::string &storage_name = "nfs_01",
                            bool configure_storage_type_quota = true) {
        const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
        ASSERT_EQ(EC_OK, group_ec);
        ASSERT_TRUE(default_group);
        ASSERT_TRUE(default_group->cache_config());
        ASSERT_TRUE(default_group->cache_config()->reclaim_strategy());

        auto cache_config = std::make_shared<CacheConfig>();
        ASSERT_TRUE(cache_config->FromJsonString(default_group->cache_config()->ToJsonString()));
        auto reclaim_strategy = std::make_shared<CacheReclaimStrategy>(*cache_config->reclaim_strategy());
        TriggerStrategy trigger = reclaim_strategy->trigger_strategy();
        trigger.set_used_percentage(threshold);
        reclaim_strategy->set_trigger_strategy(trigger);
        reclaim_strategy->set_delay_before_delete_ms(delay_before_delete_ms);
        reclaim_strategy->set_reclaim_policy(reclaim_policy);
        reclaim_strategy->set_storage_unique_name(storage_name);
        cache_config->set_reclaim_strategy(reclaim_strategy);
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>(*cache_config->meta_indexer_config());
        meta_indexer_config->SetMaxKeyCount(max_key_count);
        if (max_key_count != MetaIndexerConfig::kDefaultMaxKeyCount) {
            std::size_t mutex_shard_num = 1;
            while (mutex_shard_num <= max_key_count / 2) {
                mutex_shard_num *= 2;
            }
            meta_indexer_config->SetMutexShardNum(mutex_shard_num);
        }
        cache_config->set_meta_indexer_config(meta_indexer_config);

        InstanceGroup object_group(*default_group);
        object_group.set_name(group_name);
        object_group.set_storage_candidates({storage_name});
        object_group.set_global_quota_group_name(group_name + "-quota");
        object_group.set_version(1);
        object_group.set_cache_config(cache_config);
        std::vector<QuotaConfig> type_quotas;
        if (configure_storage_type_quota) {
            type_quotas.emplace_back(storage_type_capacity.value_or(capacity), storage_type);
        }
        object_group.set_quota(InstanceGroupQuota(capacity, type_quotas));
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));
        ASSERT_EQ(expected_registration_ec,
                  manager_->RegisterInstance(&request_context_, group_name, instance_id, "reclaim-test").first);
    }

    FailNextMaintenanceDeleteSyncBackend *InstallFailingSyncBackend(const std::string &instance_id) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        if (!indexer || !indexer->backend_manager_) {
            return nullptr;
        }
        auto config = std::make_shared<MetaStorageBackendConfig>();
        auto backend = std::make_unique<FailNextMaintenanceDeleteSyncBackend>();
        if (backend->Init(KvMetaManager::InternalInstanceId(instance_id), config) != EC_OK ||
            backend->Open() != EC_OK) {
            return nullptr;
        }
        auto *backend_raw = backend.get();
        if (indexer->backend_manager_->persistent_backend_) {
            indexer->backend_manager_->persistent_backend_->Close();
        }
        indexer->backend_manager_->persistent_backend_ = std::move(backend);
        indexer->backend_manager_->cache_backend_.reset();
        return backend_raw;
    }

    FailNextMaintenanceUpsertBackend *InstallFailingUpsertBackend(const std::string &instance_id) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        if (!indexer || !indexer->backend_manager_) {
            return nullptr;
        }
        auto config = std::make_shared<MetaStorageBackendConfig>();
        auto backend = std::make_unique<FailNextMaintenanceUpsertBackend>();
        if (backend->Init(KvMetaManager::InternalInstanceId(instance_id), config) != EC_OK ||
            backend->Open() != EC_OK) {
            return nullptr;
        }
        auto *backend_raw = backend.get();
        if (indexer->backend_manager_->persistent_backend_) {
            indexer->backend_manager_->persistent_backend_->Close();
        }
        indexer->backend_manager_->persistent_backend_ = std::move(backend);
        indexer->backend_manager_->cache_backend_.reset();
        return backend_raw;
    }

    OversamplingMetaLocalBackend *InstallOversamplingBackend(const std::string &instance_id) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        if (!indexer || !indexer->backend_manager_) {
            return nullptr;
        }
        auto config = std::make_shared<MetaStorageBackendConfig>();
        auto backend = std::make_unique<OversamplingMetaLocalBackend>();
        if (backend->Init(KvMetaManager::InternalInstanceId(instance_id), config) != EC_OK ||
            backend->Open() != EC_OK) {
            return nullptr;
        }
        auto *backend_raw = backend.get();
        if (indexer->backend_manager_->persistent_backend_) {
            indexer->backend_manager_->persistent_backend_->Close();
        }
        indexer->backend_manager_->persistent_backend_ = std::move(backend);
        indexer->backend_manager_->cache_backend_.reset();
        return backend_raw;
    }

    ControlledSyncMetaLocalBackend *InstallControlledSyncBackend(const std::string &instance_id) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        if (!indexer || !indexer->backend_manager_) {
            return nullptr;
        }
        auto config = std::make_shared<MetaStorageBackendConfig>();
        auto backend = std::make_unique<ControlledSyncMetaLocalBackend>();
        if (backend->Init(KvMetaManager::InternalInstanceId(instance_id), config) != EC_OK ||
            backend->Open() != EC_OK) {
            return nullptr;
        }
        auto *backend_raw = backend.get();
        if (indexer->backend_manager_->persistent_backend_) {
            indexer->backend_manager_->persistent_backend_->Close();
        }
        indexer->backend_manager_->persistent_backend_ = std::move(backend);
        indexer->backend_manager_->cache_backend_.reset();
        return backend_raw;
    }

    ConflictingTargetedReadMetaLocalBackend *InstallConflictingTargetedReadBackend(const std::string &instance_id) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        if (!indexer || !indexer->backend_manager_) {
            return nullptr;
        }
        auto config = std::make_shared<MetaStorageBackendConfig>();
        auto backend = std::make_unique<ConflictingTargetedReadMetaLocalBackend>();
        if (backend->Init(KvMetaManager::InternalInstanceId(instance_id), config) != EC_OK ||
            backend->Open() != EC_OK) {
            return nullptr;
        }
        auto *backend_raw = backend.get();
        if (indexer->backend_manager_->persistent_backend_) {
            indexer->backend_manager_->persistent_backend_->Close();
        }
        indexer->backend_manager_->persistent_backend_ = std::move(backend);
        indexer->backend_manager_->cache_backend_.reset();
        return backend_raw;
    }

    void CommitObject(const std::string &instance_id, const std::string &key, std::uint64_t size) {
        auto [start_ec, start] = manager_->StartWrite(&request_context_, instance_id, {key}, {size}, 30);
        ASSERT_EQ(EC_OK, start_ec);
        ASSERT_EQ(1, start.locations.size());
        ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, instance_id, start.write_session_id, {true}));
    }

    void MutateObject(const std::string &instance_id,
                      const std::string &key,
                      const std::function<void(CacheLocation &)> &mutation) {
        auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        ASSERT_TRUE(indexer);
        const auto internal_key = KvMetaManager::InternalKey(key);
        const auto location_id = KvMetaManager::StableLocationId(key);
        auto modifier = [&mutation](const std::vector<ErrorCode> &get_ecs,
                                    const LocationIdVector &,
                                    std::size_t,
                                    CacheLocationVector &locations,
                                    PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || get_ecs[0] != EC_OK || locations.size() != 1 || !locations[0]) {
                return {MA_FAIL, {EC_CORRUPTION}};
            }
            auto replacement = std::make_shared<CacheLocation>(*locations[0]);
            mutation(*replacement);
            locations[0] = std::move(replacement);
            return {MA_OK, {EC_OK}};
        };
        const auto result =
            indexer->ReadModifyWriteTargetLocations(&request_context_, {internal_key}, {{location_id}}, modifier);
        ASSERT_EQ(EC_OK, result.ec);
        ASSERT_EQ(1, result.per_location_error_codes.size());
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), result.per_location_error_codes.front());
    }

    static bool WaitUntil(const std::function<bool()> &predicate, std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return predicate();
    }

    static constexpr const char *kInstanceId = "embedding-instance";
    RequestContext request_context_{"kv_meta_manager_test"};
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<KvMetaManager> manager_;
};

TEST_F(KvMetaManagerTest, DynamicSizesAreIndependentAndInvisibleUntilFinish) {
    const std::vector<std::string> keys{"emb-a", "emb-b"};
    const std::vector<std::uint64_t> sizes{17, 33};
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, keys, sizes, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ((std::vector<bool>{false, false}), start.key_mask);
    ASSERT_EQ(2, start.locations.size());
    ASSERT_FALSE(start.write_session_id.empty());
    EXPECT_EQ(2, start.session_item_count);
    EXPECT_EQ(17, start.locations[0].value_size);
    EXPECT_EQ(33, start.locations[1].value_size);
    ASSERT_EQ(1, start.locations[0].specs.size());
    ASSERT_EQ(1, start.locations[1].specs.size());

    const DataStorageUri first_uri(start.locations[0].specs[0].second);
    const DataStorageUri second_uri(start.locations[1].specs[0].second);
    ASSERT_TRUE(first_uri.Valid());
    ASSERT_TRUE(second_uri.Valid());
    // The default NFS backend is configured to pack up to eight keys. The
    // generic path deliberately uses singleton Create calls, so these values
    // still have different physical deletion boundaries.
    EXPECT_NE(first_uri.GetPath(), second_uri.GetPath());
    std::uint64_t first_size = 0;
    std::uint64_t second_size = 0;
    first_uri.GetParamAs<std::uint64_t>("size", first_size);
    second_uri.GetParamAs<std::uint64_t>("size", second_size);
    EXPECT_EQ(17, first_size);
    EXPECT_EQ(33, second_size);

    auto [before_finish_ec, before_finish] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, before_finish_ec);
    ASSERT_EQ(2, before_finish.size());
    EXPECT_FALSE(before_finish[0].found);
    EXPECT_FALSE(before_finish[1].found);

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    ASSERT_TRUE(values[0].found);
    ASSERT_TRUE(values[1].found);
    EXPECT_EQ(17, values[0].location.value_size);
    EXPECT_EQ(33, values[1].location.value_size);

    // Committed generic objects deliberately remain CLS_NEW. The negative
    // timestamp is private to KVMeta and keeps them out of the existing
    // CLS_SERVING reclaimer/migration path.
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    KeyVector internal_keys;
    LocationIdsPerKey location_ids;
    for (const auto &key : keys) {
        internal_keys.push_back(KvMetaManager::InternalKey(key));
        location_ids.push_back({KvMetaManager::StableLocationId(key)});
    }
    LocationsPerKey exact;
    const auto exact_result = indexer->GetLocations(&request_context_, internal_keys, location_ids, exact);
    ASSERT_EQ(2, exact_result.per_location_error_codes.size());
    ASSERT_EQ(2, exact.size());
    for (std::size_t i = 0; i < exact.size(); ++i) {
        ASSERT_EQ(EC_OK, exact_result.per_location_error_codes[i][0]);
        ASSERT_TRUE(exact[i][0]);
        EXPECT_EQ(CLS_NEW, exact[i][0]->status());
        EXPECT_LT(exact[i][0]->create_time(), 0);
    }

    ASSERT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {keys[0]}));
    auto [after_remove_ec, after_remove] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, after_remove_ec);
    EXPECT_FALSE(after_remove[0].found);
    EXPECT_TRUE(after_remove[1].found);
    EXPECT_EQ(33, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, MalformedCreateResponseNeverDeletesUnattributableUris) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<OverlongCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] = manager_->StartWrite(&request_context_, kInstanceId, {"malformed-create"}, {17}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    // Wrong cardinality invalidates the operation-to-result mapping. Even a
    // well-formed returned URI is not sufficient proof that this Create owns
    // it, so compensation deliberately leaves backend orphans instead of
    // risking deletion of an existing object.
    EXPECT_TRUE(malformed->deleted_uris.empty());

    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"malformed-create"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, WrongSizeCreateResponseDeletesTheExactlyAttributedAllocationOnce) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<WrongSizeCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] = manager_->StartWrite(&request_context_, kInstanceId, {"wrong-create-size"}, {17}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    ASSERT_EQ(1, malformed->deleted_uris.size());
    EXPECT_NE(std::string::npos, malformed->deleted_uris[0].GetPath().find("/kvmeta/"));
    EXPECT_EQ("18", malformed->deleted_uris[0].GetParam("size"));

    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"wrong-create-size"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, ForeignCreateResponseIsNeverUsedAsDeleteAuthority) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<ForeignObjectCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] = manager_->StartWrite(&request_context_, kInstanceId, {"foreign-create"}, {17}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    // Exact response cardinality and the exact KVMeta key suffix are not
    // enough: the returned object is outside this backend's configured root.
    EXPECT_EQ(0, malformed->delete_calls);
    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"foreign-create"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, FailedCreateWithAUriIsAmbiguousAndNeverUsedAsDeleteAuthority) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<FailedCreateWithUriNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] =
        manager_->StartWrite(&request_context_, kInstanceId, {"failed-create-with-uri"}, {17}, 30);

    EXPECT_EQ(EC_OUTCOME_UNKNOWN, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    EXPECT_EQ(0, malformed->delete_calls);
    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"failed-create-with-uri"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);
}

TEST_F(KvMetaManagerTest, AliasedMalformedCreateResponseIsNeverUsedAsDeleteAuthority) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<AliasedOverlongCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] =
        manager_->StartWrite(&request_context_, kInstanceId, {"aliased-malformed-create"}, {17}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    EXPECT_EQ(0, malformed->delete_calls);

    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"aliased-malformed-create"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, LaterMalformedCreateDeletesOnlyTheEarlierProvenAllocation) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<LaterMalformedAliasedCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] = manager_->StartWrite(
        &request_context_, kInstanceId, {"valid-first-create", "malformed-second-create"}, {17, 33}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    EXPECT_FALSE(malformed->first_path().empty());
    EXPECT_EQ(1, std::count(malformed->deleted_paths.begin(), malformed->deleted_paths.end(), malformed->first_path()));
    EXPECT_EQ(0,
              std::count(
                  malformed->deleted_paths.begin(), malformed->deleted_paths.end(), "/malformed/new-extra-allocation"));
    EXPECT_EQ(1, malformed->deleted_paths.size());
}

TEST_F(KvMetaManagerTest, RejectsMooncakeAtRegistrationWithoutAProvableDmaDrain) {
    constexpr const char *kGroup = "malformed-mooncake-group";
    constexpr const char *kInstance = "malformed-mooncake-instance";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto malformed = std::make_shared<MalformedMooncakeBackend>(metrics_registry_);
    const StorageConfig mooncake_config(
        DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, "nfs_01", std::make_shared<MooncakeStorageSpec>());
    ASSERT_EQ(EC_OK, malformed->Open(mooncake_config, request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }
    CreateReclaimGroup(kGroup,
                       kInstance,
                       100,
                       0.8,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR,
                       DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);

    // The backend can still be recognized as an owned-object backend for old
    // metadata recovery, but no allocation may be admitted until its client
    // API exposes a completion/drain primitive.
    EXPECT_TRUE(IsKvMetaObjectStorageType(malformed->GetType()));
    EXPECT_FALSE(SupportsKvMetaCallerOwnedBufferLifetime(malformed->GetType()));
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, manager_->GetInstanceInfo(&request_context_, kInstance).first);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RejectsPaceAllocationFromADifferentConfiguredMediaPoolWithoutDeletingIt) {
    constexpr const char *kGroup = "pace-ssd-media-group";
    constexpr const char *kInstance = "pace-ssd-media-instance";
    constexpr const char *kStorage = "pace_ssd_media";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);

    auto wrong_media = std::make_shared<ConfigurableMediaTairMempoolBackend>(
        metrics_registry_, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, kTairMemPoolMediaTypeDram);
    auto pace_spec = std::make_shared<TairMemPoolStorageSpec>();
    pace_spec->set_media_type(kTairMemPoolMediaTypeSsd);
    const StorageConfig pace_config(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, kStorage, pace_spec);
    ASSERT_EQ(EC_OK, wrong_media->Open(pace_config, request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_[kStorage] = wrong_media;
    }
    CreateReclaimGroup(kGroup,
                       kInstance,
                       100,
                       0.8,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_OK,
                       DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD,
                       kStorage);

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstance, {"wrong-media"}, {17}, 30);
    EXPECT_EQ(EC_CORRUPTION, start_ec);
    EXPECT_TRUE(start.locations.empty());
    EXPECT_TRUE(start.write_session_id.empty());
    EXPECT_EQ(0, wrong_media->delete_calls);
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
}

TEST_F(KvMetaManagerTest, CreateProviderExceptionIsContainedAndReleasesEarlierCandidates) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<ThrowingSecondCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }

    const auto [ec, result] =
        manager_->StartWrite(&request_context_, kInstanceId, {"create-before-throw", "create-throws"}, {11, 13}, 30);

    EXPECT_EQ(EC_OUTCOME_UNKNOWN, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_EQ(2, throwing->CreateAttempts());
    EXPECT_EQ(1, throwing->DeleteItems());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"create-before-throw", "create-throws"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RejectsPackedFileMemberWithoutDeletingItsSharedAllocation) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);

    auto malformed = std::make_shared<NonSingletonCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    const auto [ec, result] = manager_->StartWrite(&request_context_, kInstanceId, {"packed-member"}, {17}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    // A non-zero block belongs to a shared file by definition. The malformed
    // backend result is retained rather than risking deletion of other data.
    EXPECT_EQ(0, malformed->delete_calls);
    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"packed-member"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
}

TEST_F(KvMetaManagerTest, RejectsAStorageBackendThatReusesOneSingletonForTwoKeys) {
    constexpr const char *kGroup = "duplicate-pace-allocation-group";
    constexpr const char *kInstance = "duplicate-pace-allocation-instance";
    constexpr const char *kStorage = "duplicate_pace";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto malformed = std::make_shared<DuplicateSingletonCreateTairMempoolBackend>(metrics_registry_);
    auto pace_spec = std::make_shared<TairMemPoolStorageSpec>();
    const StorageConfig pace_config(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, kStorage, pace_spec);
    ASSERT_EQ(EC_OK, malformed->Open(pace_config, request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_[kStorage] = malformed;
    }
    CreateReclaimGroup(kGroup,
                       kInstance,
                       1024,
                       0.8,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_OK,
                       DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL,
                       kStorage);

    // PACE returns opaque addresses, so unlike file-like backends the URI
    // cannot echo the generated logical key. `size` is not part of its
    // physical Delete identity; the reused address must still be rejected
    // before either metadata record is published.
    const auto [ec, result] =
        manager_->StartWrite(&request_context_, kInstance, {"duplicate-uri-a", "duplicate-uri-b"}, {17, 29}, 30);

    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.key_mask.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    EXPECT_EQ(1, malformed->delete_calls);
    const auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"duplicate-uri-a", "duplicate-uri-b"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
}

TEST_F(KvMetaManagerTest, AtomicFinishFailureRollsBackEveryValue) {
    const std::vector<std::string> keys{"atomic-a", "atomic-b"};
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, keys, {64, 128}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(2, start.locations.size());

    // One failed item aborts the complete generic-object transaction.
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, false}));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, keys);
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);

    // Exact metadata was removed, so the same keys can be admitted again.
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, keys, {7, 9}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ((std::vector<bool>{false, false}), retry.key_mask);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false, false}));
}

TEST_F(KvMetaManagerTest, SessionAbortMetadataFailureFailsKvMetaClosedUntilRecovery) {
    constexpr const char *kKey = "session-abort-metadata-sync-fails";
    auto *meta_backend = InstallFailingSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, meta_backend);

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());
    meta_backend->FailNextMaintenanceDeleteSync();

    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    ASSERT_TRUE(meta_backend->WaitForSyncFailure(std::chrono::seconds(2)));

    // Take() consumed the only session owner before rollback. Keep every new
    // generation out until recovery has reconciled the failed absence barrier.
    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"abort-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {31}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RecoveryRebuildsExactDynamicByteUsage) {
    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {"recover-a", "recover-b"}, {17, 33}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(2, start.locations.size());
    ASSERT_EQ(start.locations[0].type, start.locations[1].type);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    indexer->SetStorageUsageByType(start.locations[0].type, 1);
    ASSERT_EQ(1, indexer->GetStorageUsage());

    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(50, indexer->GetStorageUsage());
    EXPECT_EQ(50, indexer->GetStorageUsageByType(start.locations[0].type));
}

TEST_F(KvMetaManagerTest, TrimUsesBoundedMaintenanceBatches) {
    constexpr std::size_t kObjectCount = 257;
    for (std::size_t begin = 0; begin < kObjectCount; begin += manager_->limits().max_batch_items) {
        const std::size_t end = std::min(kObjectCount, begin + manager_->limits().max_batch_items);
        std::vector<std::string> keys;
        std::vector<std::uint64_t> sizes;
        for (std::size_t i = begin; i < end; ++i) {
            keys.push_back("trim-" + std::to_string(i));
            sizes.push_back(1);
        }
        auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, keys, sizes, 30);
        ASSERT_EQ(EC_OK, start_ec);
        ASSERT_EQ(EC_OK,
                  manager_->FinishWrite(&request_context_,
                                        kInstanceId,
                                        start.write_session_id,
                                        std::vector<bool>(start.locations.size(), true)));
    }

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(kObjectCount, indexer->GetStorageUsage());

    // A demotion cancels an unbounded namespace walk before the server waits
    // for KVMeta RPCs. Cancellation is sticky until the next successful
    // leader recovery explicitly resumes maintenance.
    manager_->CancelMaintenance();
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->TrimAll(&request_context_, kInstanceId, false));
    auto [before_resume_ec, before_resume] = manager_->Get(&request_context_, kInstanceId, {"trim-0", "trim-256"});
    ASSERT_EQ(EC_OK, before_resume_ec);
    ASSERT_EQ(2, before_resume.size());
    EXPECT_TRUE(before_resume[0].found);
    EXPECT_TRUE(before_resume[1].found);

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_EQ(0, indexer->GetStorageUsage());

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"trim-0", "trim-128", "trim-256"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(3, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    EXPECT_FALSE(values[2].found);
}

TEST_F(KvMetaManagerTest, TrimRetainsTombstoneAndRecoveryRetriesFailedPhysicalDelete) {
    constexpr const char *kKey = "trim-delete-fails";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {31}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    // The owner first becomes a durable, read-invisible tombstone. A failed
    // physical delete must retain both that URI ledger and its quota charge.
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_EQ(1, failing->DeleteAttempts());
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(31, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    // The current leader is fail-closed, so normal RPC replay cannot race the
    // recovery owner. Leader recovery is explicitly allowed to retry the same
    // immutable allocation generation.
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_EQ(1, failing->DeleteAttempts());
    failing->SetMode(FaultingDeleteNfsBackend::Mode::kSuccess);
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(2, failing->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    EXPECT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, TrimCancellationAfterACommittedBatchIsOutcomeUnknown) {
    constexpr std::size_t kObjectCount = 257;
    for (std::size_t begin = 0; begin < kObjectCount; begin += manager_->limits().max_batch_items) {
        const std::size_t end = std::min(kObjectCount, begin + manager_->limits().max_batch_items);
        std::vector<std::string> keys;
        std::vector<std::uint64_t> sizes;
        for (std::size_t i = begin; i < end; ++i) {
            keys.push_back("trim-cancel-" + std::to_string(i));
            sizes.push_back(1);
        }
        auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, keys, sizes, 30);
        ASSERT_EQ(EC_OK, start_ec);
        ASSERT_EQ(EC_OK,
                  manager_->FinishWrite(&request_context_,
                                        kInstanceId,
                                        start.write_session_id,
                                        std::vector<bool>(start.locations.size(), true)));
    }

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto trim = std::async(std::launch::async, [&]() {
        RequestContext context("trim-partial-cancellation");
        return manager_->TrimAll(&context, kInstanceId, false);
    });
    const bool delete_entered = blocking->WaitForDelete(std::chrono::seconds(2));
    if (!delete_entered) {
        blocking->ReleaseDelete();
    }
    ASSERT_TRUE(delete_entered);
    // DeleteItems removes and Syncs the first 256 metadata records before it
    // enters physical Delete. Demotion at this point must not masquerade as a
    // safe no-op cancellation for the whole 257-object request.
    manager_->CancelMaintenance();
    blocking->ReleaseDelete();
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, trim.get());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(1, indexer->GetStorageUsage());

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_EQ(0, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, TrimMetadataPersistenceFailureFailsKvMetaClosedUntilRecovery) {
    constexpr const char *kKey = "trim-metadata-sync-fails";
    auto *meta_backend = InstallFailingSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, meta_backend);
    CommitObject(kInstanceId, kKey, 31);
    meta_backend->FailNextMaintenanceDeleteSync();

    EXPECT_EQ(EC_OUTCOME_UNKNOWN, manager_->TrimAll(&request_context_, kInstanceId, false));
    ASSERT_TRUE(meta_backend->WaitForSyncFailure(std::chrono::seconds(2)));

    // The per-instance Trim marker is scoped to the call. If the failed
    // metadata barrier did not also close global KVMeta admission, destroying
    // that marker at return would reopen an ABA window immediately.
    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"trim-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    EXPECT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));

    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {37}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, TrimDeletesOnlyTheCapturedGenerationBeforeRejectingAReplacementOwner) {
    constexpr const char *kKey = "trim-replacement-owner";
    auto *meta_backend = InstallConflictingTargetedReadBackend(kInstanceId);
    ASSERT_NE(nullptr, meta_backend);
    CommitObject(kInstanceId, kKey, 31);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = tracking;
    }

    // The captured URI contains an immutable allocation generation. Even if
    // an unsupported actor changes metadata later, deleting that old physical
    // identity cannot target the replacement; the metadata anomaly still
    // closes admission.
    meta_backend->ConflictOnTargetedRead(1);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, manager_->TrimAll(&request_context_, kInstanceId, false));
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    EXPECT_EQ(1, tracking->DeleteAttempts());

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(31, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ExistingKeysAreMaskedAndInflightKeysAreRetryable) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {21}, 30);
    ASSERT_EQ(EC_OK, first_ec);
    ASSERT_EQ((std::vector<bool>{false}), first.key_mask);

    auto [wrong_active_ec, wrong_active] =
        manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {999}, 30);
    // The active reservation is retryable regardless of its provisional
    // size. Only a committed object can make a size mismatch permanent.
    EXPECT_EQ(EC_EXIST, wrong_active_ec);
    EXPECT_TRUE(wrong_active.key_mask.empty());
    EXPECT_TRUE(wrong_active.locations.empty());

    auto [second_ec, second] = manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {21}, 30);
    EXPECT_EQ(EC_EXIST, second_ec);
    EXPECT_TRUE(second.key_mask.empty());
    EXPECT_TRUE(second.locations.empty());
    EXPECT_TRUE(second.write_session_id.empty());

    auto [mixed_ec, mixed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"same-key", "must-not-allocate"}, {21, 7}, 30);
    EXPECT_EQ(EC_EXIST, mixed_ec);
    EXPECT_TRUE(mixed.key_mask.empty());
    EXPECT_TRUE(mixed.locations.empty());
    EXPECT_TRUE(mixed.write_session_id.empty());
    auto [mixed_get_ec, mixed_values] =
        manager_->Get(&request_context_, kInstanceId, {"same-key", "must-not-allocate"});
    ASSERT_EQ(EC_OK, mixed_get_ec);
    ASSERT_EQ(2, mixed_values.size());
    EXPECT_FALSE(mixed_values[0].found);
    EXPECT_FALSE(mixed_values[1].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(21, indexer->GetStorageUsage());

    // A malformed finish request must not consume the valid session.
    EXPECT_EQ(EC_BADARGS, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {}));
    EXPECT_EQ(EC_MISMATCH, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {true, true}));
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {true}));

    auto [wrong_committed_ec, wrong_committed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {42}, 30);
    EXPECT_EQ(EC_MISMATCH, wrong_committed_ec);
    EXPECT_TRUE(wrong_committed.key_mask.empty());
    EXPECT_TRUE(wrong_committed.locations.empty());

    auto [third_ec, third] = manager_->StartWrite(&request_context_, kInstanceId, {"same-key"}, {21}, 30);
    ASSERT_EQ(EC_OK, third_ec);
    EXPECT_EQ((std::vector<bool>{true}), third.key_mask);
    EXPECT_TRUE(third.locations.empty());
}

TEST_F(KvMetaManagerTest, DifferentSizeCanBeWrittenAfterTheActiveWinnerAborts) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"retry-size"}, {21}, 30);
    ASSERT_EQ(EC_OK, first_ec);

    auto [active_ec, active] = manager_->StartWrite(&request_context_, kInstanceId, {"retry-size"}, {42}, 30);
    EXPECT_EQ(EC_EXIST, active_ec);
    EXPECT_TRUE(active.locations.empty());

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {false}));
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"retry-size"}, {42}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(1, retry.locations.size());
    EXPECT_EQ(42, retry.locations.front().value_size);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, DifferentSizeConcurrentActiveWinnerRemainsRetryableAfterConditionalInsertRace) {
    constexpr const char *kKey = "different-size-conditional-insert-race";
    auto concurrent_manager = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
    ASSERT_TRUE(concurrent_manager->Init());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto hooked = std::make_shared<HookedCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, hooked->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = hooked;
    }

    std::optional<std::pair<ErrorCode, KvMetaManager::StartWriteResult>> winner;
    hooked->SetOneShotAfterCreateHook([&]() {
        RequestContext winner_context("different-size-race-winner");
        winner = concurrent_manager->StartWrite(&winner_context, kInstanceId, {kKey}, {42}, 30);
    });

    // The first manager has already observed a miss and allocated 21 bytes
    // when the independent manager publishes a 42-byte active reservation.
    // The conditional insert loser must report the transient active state,
    // not a permanent mismatch based on the winner's provisional size.
    auto [loser_ec, loser] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {21}, 30);
    ASSERT_TRUE(winner.has_value());
    ASSERT_EQ(EC_OK, winner->first);
    ASSERT_FALSE(winner->second.write_session_id.empty());
    EXPECT_EQ(EC_EXIST, loser_ec);
    EXPECT_TRUE(loser.locations.empty());
    EXPECT_TRUE(loser.write_session_id.empty());

    EXPECT_EQ(
        EC_OK,
        concurrent_manager->FinishWrite(&request_context_, kInstanceId, winner->second.write_session_id, {false}));
    concurrent_manager->Shutdown();
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ConditionalInsertRollbackViewFailureFailsKvMetaClosedUntilRecovery) {
    constexpr const char *kKey = "conditional-insert-view-fails";
    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    auto concurrent_manager = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
    ASSERT_TRUE(concurrent_manager->Init());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto hooked = std::make_shared<HookedCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, hooked->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = hooked;
    }

    std::optional<std::pair<ErrorCode, KvMetaManager::StartWriteResult>> winner;
    hooked->SetOneShotAfterCreateHook([&]() {
        RequestContext winner_context("conditional-insert-view-winner");
        winner = concurrent_manager->StartWrite(&winner_context, kInstanceId, {kKey}, {42}, 30);
        // The winner's reservation is durable. Fail the loser's barrier before
        // it classifies its allocations against the conditional-insert result.
        controlled_sync->FailSyncAfter(0);
    });

    auto [loser_ec, loser] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {21}, 30);
    ASSERT_TRUE(winner.has_value());
    ASSERT_EQ(EC_OK, winner->first);
    ASSERT_FALSE(winner->second.write_session_id.empty());
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, loser_ec);
    EXPECT_TRUE(loser.locations.empty());
    EXPECT_TRUE(loser.write_session_id.empty());
    EXPECT_TRUE(controlled_sync->WaitForSyncFailure(std::chrono::seconds(2)));

    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"insert-view-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    // The independently published owner is still finalizable while this
    // manager is fail-closed.
    EXPECT_EQ(
        EC_OK,
        concurrent_manager->FinishWrite(&request_context_, kInstanceId, winner->second.write_session_id, {false}));
    concurrent_manager->Shutdown();
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {23}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ConditionalInsertRollbackNeverDeletesAnUnexpectedSameAllocationOwner) {
    constexpr const char *kGroup = "same-pace-allocation-race-group";
    constexpr const char *kInstance = "same-pace-allocation-race-instance";
    constexpr const char *kStorage = "same_allocation_pace";
    constexpr const char *kKey = "conditional-insert-same-allocation-owner";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto reused = std::make_shared<HookedReusedSingletonTairMempoolBackend>(metrics_registry_);
    auto pace_spec = std::make_shared<TairMemPoolStorageSpec>();
    const StorageConfig pace_config(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, kStorage, pace_spec);
    ASSERT_EQ(EC_OK, reused->Open(pace_config, request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_[kStorage] = reused;
    }
    CreateReclaimGroup(kGroup,
                       kInstance,
                       1024,
                       0.8,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_OK,
                       DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL,
                       kStorage);
    auto concurrent_manager = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
    ASSERT_TRUE(concurrent_manager->Init());

    std::optional<std::pair<ErrorCode, KvMetaManager::StartWriteResult>> winner;
    reused->SetOneShotAfterCreateHook([&]() {
        RequestContext winner_context("conditional-insert-same-allocation-winner");
        winner = concurrent_manager->StartWrite(&winner_context, kInstance, {kKey}, {42}, 29);
    });

    // The injected backend violates singleton ownership across two Create
    // calls. It also varies the non-identity size query parameter, proving
    // alias detection follows PACE's opaque delete address, not the complete
    // metadata URI. The loser must retain the winner and close
    // admission; physical identity is not authority to CAS-delete metadata.
    auto [loser_ec, loser] = manager_->StartWrite(&request_context_, kInstance, {kKey}, {41}, 30);
    ASSERT_TRUE(winner.has_value());
    ASSERT_EQ(EC_OK, winner->first);
    ASSERT_FALSE(winner->second.write_session_id.empty());
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, loser_ec);
    EXPECT_TRUE(loser.locations.empty());
    EXPECT_TRUE(loser.write_session_id.empty());
    EXPECT_EQ(0, reused->delete_calls);

    // If loser rollback had erased the winner, its exact abort would fail.
    EXPECT_EQ(EC_OK,
              concurrent_manager->FinishWrite(&request_context_, kInstance, winner->second.write_session_id, {false}));
    EXPECT_EQ(1, reused->delete_calls);
    concurrent_manager->Shutdown();

    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstance, {"same-allocation-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstance, {kKey}, {43}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ConditionalInsertRollbackFailsClosedOnAnUnexpectedReplacementOwner) {
    constexpr const char *kRaceKey = "conditional-insert-replacement-race";
    constexpr const char *kReplacedKey = "conditional-insert-replaced-owner";
    constexpr std::uint64_t kReplacedSize = 31;
    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    auto concurrent_manager = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
    ASSERT_TRUE(concurrent_manager->Init());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto hooked = std::make_shared<HookedCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, hooked->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = hooked;
    }

    std::optional<std::pair<ErrorCode, KvMetaManager::StartWriteResult>> winner;
    hooked->SetOneShotAfterCreateHook([&]() {
        RequestContext winner_context("conditional-insert-replacement-winner");
        winner = concurrent_manager->StartWrite(&winner_context, kInstanceId, {kRaceKey}, {41}, 30);
    });
    // The winner consumes the first barrier. Block the loser's rollback
    // barrier after its own second key has been conditionally inserted.
    controlled_sync->BlockSyncAfter(1);
    auto loser = std::async(std::launch::async, [&]() {
        RequestContext loser_context("conditional-insert-replacement-loser");
        return manager_->StartWrite(&loser_context, kInstanceId, {kRaceKey, kReplacedKey}, {21, kReplacedSize}, 30);
    });
    ASSERT_TRUE(controlled_sync->WaitForBlockedSyncs(1, std::chrono::seconds(2)));
    ASSERT_TRUE(winner.has_value());
    ASSERT_EQ(EC_OK, winner->first);

    // Simulate an independent manager replacing metadata for the allocation
    // that the loser proved it had inserted. The replacement uses a distinct
    // physical path, so loser cleanup may release only its old allocation; it
    // must preserve this owner and fail closed for counter/ownership recovery.
    MutateObject(kInstanceId, kReplacedKey, [](CacheLocation &location) {
        location.set_create_time(-123);
        ReplaceKvMetaObjectNonce(location, 'x');
    });
    controlled_sync->ReleaseBlockedSync();

    const auto [loser_ec, loser_result] = loser.get();
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, loser_ec);
    EXPECT_TRUE(loser_result.locations.empty());
    EXPECT_TRUE(loser_result.write_session_id.empty());

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kReplacedKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
    EXPECT_EQ(kReplacedSize, values.front().location.value_size);

    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"replacement-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    EXPECT_EQ(
        EC_OK,
        concurrent_manager->FinishWrite(&request_context_, kInstanceId, winner->second.write_session_id, {false}));
    concurrent_manager->Shutdown();
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {kReplacedKey}));
}

TEST_F(KvMetaManagerTest, ConditionalInsertRollbackFailsClosedOnAnUnexpectedMissingOwner) {
    constexpr const char *kRaceKey = "conditional-insert-missing-race";
    constexpr const char *kMissingKey = "conditional-insert-missing-owner";
    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    auto concurrent_manager = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
    ASSERT_TRUE(concurrent_manager->Init());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto hooked = std::make_shared<HookedCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, hooked->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = hooked;
    }

    std::optional<std::pair<ErrorCode, KvMetaManager::StartWriteResult>> winner;
    hooked->SetOneShotAfterCreateHook([&]() {
        RequestContext winner_context("conditional-insert-missing-winner");
        winner = concurrent_manager->StartWrite(&winner_context, kInstanceId, {kRaceKey}, {41}, 30);
    });
    controlled_sync->BlockSyncAfter(1);
    auto loser = std::async(std::launch::async, [&]() {
        RequestContext loser_context("conditional-insert-missing-loser");
        return manager_->StartWrite(&loser_context, kInstanceId, {kRaceKey, kMissingKey}, {21, 31}, 30);
    });
    ASSERT_TRUE(controlled_sync->WaitForBlockedSyncs(1, std::chrono::seconds(2)));
    ASSERT_TRUE(winner.has_value());
    ASSERT_EQ(EC_OK, winner->first);

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto missing_internal_key = KvMetaManager::InternalKey(kMissingKey);
    const auto delete_result = indexer->Delete(&request_context_, {missing_internal_key});
    ASSERT_EQ(EC_OK, delete_result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_result.error_codes);
    ASSERT_TRUE(indexer->Sync({missing_internal_key}));
    controlled_sync->ReleaseBlockedSync();

    const auto [loser_ec, loser_result] = loser.get();
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, loser_ec);
    EXPECT_TRUE(loser_result.locations.empty());
    EXPECT_TRUE(loser_result.write_session_id.empty());
    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"missing-owner-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    EXPECT_EQ(
        EC_OK,
        concurrent_manager->FinishWrite(&request_context_, kInstanceId, winner->second.write_session_id, {false}));
    concurrent_manager->Shutdown();
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kMissingKey}, {37}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ConditionalInsertLoserNeverDeletesACommittedSameAllocationWinner) {
    constexpr const char *kGroup = "committed-pace-allocation-race-group";
    constexpr const char *kInstance = "committed-pace-allocation-race-instance";
    constexpr const char *kStorage = "committed_same_allocation_pace";
    constexpr const char *kKey = "conditional-insert-committed-same-allocation";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto reused = std::make_shared<HookedReusedSingletonTairMempoolBackend>(metrics_registry_);
    auto pace_spec = std::make_shared<TairMemPoolStorageSpec>();
    const StorageConfig pace_config(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, kStorage, pace_spec);
    ASSERT_EQ(EC_OK, reused->Open(pace_config, request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_[kStorage] = reused;
    }
    CreateReclaimGroup(kGroup,
                       kInstance,
                       1024,
                       0.8,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_OK,
                       DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL,
                       kStorage);
    auto concurrent_manager = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_);
    ASSERT_TRUE(concurrent_manager->Init());

    std::optional<std::pair<ErrorCode, KvMetaManager::StartWriteResult>> winner;
    ErrorCode winner_finish_ec = EC_ERROR;
    reused->SetOneShotAfterCreateHook([&]() {
        RequestContext winner_context("conditional-insert-committed-same-allocation-winner");
        winner = concurrent_manager->StartWrite(&winner_context, kInstance, {kKey}, {41}, 29);
        if (winner->first == EC_OK && !winner->second.write_session_id.empty()) {
            winner_finish_ec =
                concurrent_manager->FinishWrite(&winner_context, kInstance, winner->second.write_session_id, {true});
        }
    });

    // A normal same-size committed race is a hit, but this backend returned
    // the winner's exact URI for the loser's independent Create. Cleaning the
    // apparent loser would therefore delete the committed object's bytes.
    auto [loser_ec, loser] = manager_->StartWrite(&request_context_, kInstance, {kKey}, {41}, 30);
    ASSERT_TRUE(winner.has_value());
    ASSERT_EQ(EC_OK, winner->first);
    ASSERT_EQ(EC_OK, winner_finish_ec);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, loser_ec);
    EXPECT_TRUE(loser.locations.empty());
    EXPECT_TRUE(loser.write_session_id.empty());
    EXPECT_EQ(0, reused->delete_calls);

    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
    EXPECT_EQ(41, values.front().location.value_size);

    concurrent_manager->Shutdown();
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstance, {kKey}));
}

TEST_F(KvMetaManagerTest, ReservationRollbackFailureFailsKvMetaClosedUntilRecovery) {
    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    // The first failure makes the newly inserted reservation durability
    // barrier ambiguous. The second failure prevents the compensating metadata
    // delete from proving absence, and no write session exists yet to own it.
    controlled_sync->FailSyncAfter(0, 2);

    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {"reservation-rollback-unknown"}, {17}, 30);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, start_ec);
    EXPECT_TRUE(start.write_session_id.empty());
    EXPECT_TRUE(start.locations.empty());
    EXPECT_TRUE(controlled_sync->WaitForSyncFailures(2, std::chrono::seconds(2)));

    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"must-wait-for-recovery"}, {19}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"after-recovery"}, {23}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReservationPhysicalCleanupFailureFailsKvMetaClosedUntilRecovery) {
    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    controlled_sync->FailSyncAfter(0);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    // The reservation barrier fails, then compensation persists an immediate
    // tombstone. A failed physical release must retain that charged cleanup
    // ledger and close admission instead of leaking one object per retry.
    auto [failed_ec, failed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"rollback-physical-orphan"}, {17}, 30);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, failed_ec);
    EXPECT_TRUE(failed.locations.empty());
    EXPECT_EQ(1, failing->DeleteAttempts());

    auto [closed_ec, closed] = manager_->StartWrite(&request_context_, kInstanceId, {"admission-must-close"}, {19}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"after-recovery"}, {19}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {true}));
}

TEST_F(KvMetaManagerTest, TrimRejectsActiveSessionsWithoutDeletingCommittedValues) {
    auto [committed_ec, committed] = manager_->StartWrite(&request_context_, kInstanceId, {"trim-committed"}, {13}, 30);
    ASSERT_EQ(EC_OK, committed_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, committed.write_session_id, {true}));

    auto [active_ec, active] = manager_->StartWrite(&request_context_, kInstanceId, {"trim-active"}, {17}, 30);
    ASSERT_EQ(EC_OK, active_ec);
    ASSERT_FALSE(active.write_session_id.empty());

    EXPECT_EQ(EC_EXIST, manager_->TrimAll(&request_context_, kInstanceId, false));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"trim-committed", "trim-active"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_FALSE(values[1].found);

    // Rejection does not consume the writer's session. Once the caller ends
    // it, the same explicit Trim can safely remove the whole namespace.
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, active.write_session_id, {false}));
    ASSERT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
    auto [after_ec, after] = manager_->Get(&request_context_, kInstanceId, {"trim-committed"});
    ASSERT_EQ(EC_OK, after_ec);
    ASSERT_EQ(1, after.size());
    EXPECT_FALSE(after[0].found);
}

TEST_F(KvMetaManagerTest, TrimWaitBarrierIncludesSessionFinalization) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"finalizing-trim"}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto finish = std::async(std::launch::async, [&]() {
        RequestContext context("finish-while-trim");
        return manager_->FinishWrite(&context, kInstanceId, start.write_session_id, {false});
    });
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    auto trim = std::async(std::launch::async, [&]() {
        RequestContext context("trim-while-finalizing");
        return manager_->TrimAll(&context, kInstanceId, false);
    });
    const auto trim_status = trim.wait_for(std::chrono::seconds(1));
    blocking->ReleaseDelete();

    EXPECT_EQ(std::future_status::ready, trim_status);
    EXPECT_EQ(EC_EXIST, trim.get());
    EXPECT_EQ(EC_OK, finish.get());
    EXPECT_EQ(EC_OK, manager_->TrimAll(&request_context_, kInstanceId, false));
}

TEST_F(KvMetaManagerTest, TrimFenceDoesNotHoldTheGroupShardAcrossStorageIo) {
    constexpr const char *kPeerInstance = "trim-peer-instance";
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, "default", kPeerInstance, "peer").first);
    CommitObject(kInstanceId, "trim-slow-delete", 17);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto trim = std::async(std::launch::async, [&]() {
        RequestContext context("slow-trim");
        return manager_->TrimAll(&context, kInstanceId, false);
    });
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    auto same_instance_write = std::async(std::launch::async, [&]() {
        RequestContext context("write-during-same-instance-trim");
        return manager_->StartWrite(&context, kInstanceId, {"must-be-fenced"}, {19}, 30).first;
    });
    auto duplicate_trim = std::async(std::launch::async, [&]() {
        RequestContext context("duplicate-trim");
        return manager_->TrimAll(&context, kInstanceId, false);
    });
    auto same_instance_remove = std::async(std::launch::async, [&]() {
        RequestContext context("remove-during-same-instance-trim");
        return manager_->Remove(&context, kInstanceId, {"must-be-fenced"});
    });
    auto peer_write = std::async(std::launch::async, [&]() {
        RequestContext context("peer-write-during-trim");
        return manager_->StartWrite(&context, kPeerInstance, {"peer-value"}, {23}, 30);
    });

    const auto same_status = same_instance_write.wait_for(std::chrono::seconds(1));
    const auto duplicate_status = duplicate_trim.wait_for(std::chrono::seconds(1));
    const auto remove_status = same_instance_remove.wait_for(std::chrono::seconds(1));
    const auto peer_status = peer_write.wait_for(std::chrono::seconds(1));
    blocking->ReleaseDelete();

    ASSERT_EQ(std::future_status::ready, same_status);
    EXPECT_EQ(EC_EXIST, same_instance_write.get());
    ASSERT_EQ(std::future_status::ready, duplicate_status);
    EXPECT_EQ(EC_EXIST, duplicate_trim.get());
    ASSERT_EQ(std::future_status::ready, remove_status);
    EXPECT_EQ(EC_EXIST, same_instance_remove.get());
    ASSERT_EQ(std::future_status::ready, peer_status);
    auto [peer_ec, peer_start] = peer_write.get();
    ASSERT_EQ(EC_OK, peer_ec);
    ASSERT_FALSE(peer_start.write_session_id.empty());

    EXPECT_EQ(EC_OK, trim.get());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kPeerInstance, peer_start.write_session_id, {false}));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RemoveDoesNotInvalidateAnActiveWriteSession) {
    auto [committed_start_ec, committed_start] =
        manager_->StartWrite(&request_context_, kInstanceId, {"committed-remove-guard"}, {13}, 30);
    ASSERT_EQ(EC_OK, committed_start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, committed_start.write_session_id, {true}));

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"active-remove"}, {21}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    EXPECT_EQ(EC_EXIST, manager_->Remove(&request_context_, kInstanceId, {"committed-remove-guard", "active-remove"}));

    // Batch validation finishes before DeleteItems, so the committed key is
    // preserved when a later key in the same request is still active.
    auto [before_finish_ec, before_finish] =
        manager_->Get(&request_context_, kInstanceId, {"committed-remove-guard", "active-remove"});
    ASSERT_EQ(EC_OK, before_finish_ec);
    ASSERT_EQ(2, before_finish.size());
    EXPECT_TRUE(before_finish[0].found);
    EXPECT_FALSE(before_finish[1].found);

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"active-remove"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {"committed-remove-guard", "active-remove"}));
}

TEST_F(KvMetaManagerTest, RemoveFinishesPhysicalDeleteBeforeReadmittingTheKey) {
    constexpr const char *key = "remove-recreate";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {key}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto remove = std::async(std::launch::async, [&]() {
        RequestContext context("remove-before-recreate");
        return manager_->Remove(&context, kInstanceId, {key});
    });
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    auto recreate = std::async(std::launch::async, [&]() {
        RequestContext context("recreate-after-remove");
        return manager_->StartWrite(&context, kInstanceId, {key}, {19}, 30);
    });
    // Remove still owns the KVMeta group shard while the old allocation is
    // being deleted, so a new generation cannot reach backend allocation yet.
    EXPECT_EQ(std::future_status::timeout, recreate.wait_for(std::chrono::milliseconds(100)));

    blocking->ReleaseDelete();
    EXPECT_EQ(EC_OK, remove.get());
    auto [recreate_ec, recreated] = recreate.get();
    ASSERT_EQ(EC_OK, recreate_ec);
    ASSERT_EQ((std::vector<bool>{false}), recreated.key_mask);
    ASSERT_EQ(1, recreated.locations.size());
    ASSERT_FALSE(recreated.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, recreated.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RemovePhysicalDeleteExceptionRetainsARecoveryLedger) {
    constexpr const char *kKey = "remove-delete-throws";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_,
                                                               FaultingDeleteNfsBackend::Mode::kStandardException);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }

    // The provider exception is contained, but the durable tombstone and its
    // quota charge remain until recovery confirms physical absence.
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, manager_->Remove(&request_context_, kInstanceId, {kKey}));
    EXPECT_EQ(1, throwing->DeleteAttempts());
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(23, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    // Ordinary client replay is fenced. Recovery alone retries the immutable
    // physical identity and erases the tombstone after success.
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->Remove(&request_context_, kInstanceId, {kKey}));
    EXPECT_EQ(1, throwing->DeleteAttempts());
    throwing->SetMode(FaultingDeleteNfsBackend::Mode::kSuccess);
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(2, throwing->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [recreate_ec, recreated] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 30);
    ASSERT_EQ(EC_OK, recreate_ec);
    ASSERT_FALSE(recreated.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, recreated.write_session_id, {false}));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RemoveNeverDeletesPhysicalAllocationWhenTombstoneUpsertFails) {
    constexpr const char *kKey = "remove-tombstone-upsert-fails";
    auto *meta_backend = InstallFailingUpsertBackend(kInstanceId);
    ASSERT_NE(nullptr, meta_backend);
    CommitObject(kInstanceId, kKey, 29);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kSuccess);
    ASSERT_EQ(EC_OK, tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = tracking;
    }

    meta_backend->FailNextUpsert();
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, manager_->Remove(&request_context_, kInstanceId, {kKey}));
    EXPECT_EQ(1, meta_backend->FailedUpserts());
    EXPECT_EQ(0, tracking->DeleteAttempts());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));

    // Observing the expected owner inside the modifier is insufficient: only
    // an EC_OK tombstone write authorizes physical deletion. The failed write
    // leaves both the readable owner and its quota charge intact.
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values[0].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(29, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RemoveReturnsUnknownWhenMetadataPersistenceIsAmbiguous) {
    constexpr const char *kKey = "remove-metadata-sync-fails";
    auto *meta_backend = InstallFailingSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, meta_backend);
    CommitObject(kInstanceId, kKey, 31);
    meta_backend->FailNextMaintenanceDeleteSync();

    EXPECT_EQ(EC_OUTCOME_UNKNOWN, manager_->Remove(&request_context_, kInstanceId, {kKey}));
    ASSERT_TRUE(meta_backend->WaitForSyncFailure(std::chrono::seconds(2)));

    // The in-memory view has changed, but the failed barrier means the caller
    // cannot infer the durable state and must not blindly replay the mutation.
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);

    // Outcome-unknown is not sufficient by itself: another client must not be
    // allowed to allocate a successor while durable metadata may still own the
    // old reusable-address URI.
    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"remove-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->Remove(&request_context_, kInstanceId, {kKey}));

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {37}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RemoveDeletesOnlyTheCapturedGenerationBeforeRejectingAReplacementOwner) {
    constexpr const char *kKey = "remove-replacement-owner";
    auto *meta_backend = InstallConflictingTargetedReadBackend(kInstanceId);
    ASSERT_NE(nullptr, meta_backend);
    CommitObject(kInstanceId, kKey, 31);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = tracking;
    }

    // The first targeted read captures the immutable physical generation. A
    // later unsupported metadata replacement cannot make exact deletion target
    // the successor, but it must still fail the metadata transaction closed.
    meta_backend->ConflictOnTargetedRead(2);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, manager_->Remove(&request_context_, kInstanceId, {kKey}));
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    EXPECT_EQ(1, tracking->DeleteAttempts());

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(31, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RollbackFinishesPhysicalDeleteBeforeReadmittingTheKey) {
    constexpr const char *key = "rollback-recreate";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {key}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    auto rollback = std::async(std::launch::async, [&]() {
        RequestContext context("rollback-before-recreate");
        return manager_->FinishWrite(&context, kInstanceId, start.write_session_id, {false});
    });
    // A read-invisible tombstone is already durable when physical delete
    // blocks. The group shard also prevents a new generation from being
    // allocated until this generation is confirmed absent and finalized.
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    auto recreate = std::async(std::launch::async, [&]() {
        RequestContext context("recreate-after-rollback");
        return manager_->StartWrite(&context, kInstanceId, {key}, {19}, 30);
    });
    EXPECT_EQ(std::future_status::timeout, recreate.wait_for(std::chrono::milliseconds(100)));

    blocking->ReleaseDelete();
    EXPECT_EQ(EC_OK, rollback.get());
    auto [recreate_ec, recreated] = recreate.get();
    ASSERT_EQ(EC_OK, recreate_ec);
    ASSERT_EQ((std::vector<bool>{false}), recreated.key_mask);
    ASSERT_EQ(1, recreated.locations.size());
    ASSERT_FALSE(recreated.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, recreated.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RollbackContainsPhysicalDeleteExceptionAndRecoveryRetriesIt) {
    constexpr const char *kKey = "rollback-delete-throws";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_,
                                                               FaultingDeleteNfsBackend::Mode::kStandardException);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }

    // The tombstone-first rollback contains the provider exception and trips
    // the circuit breaker without losing its restart-safe URI ledger.
    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    EXPECT_EQ(1, throwing->DeleteAttempts());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(23, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 30).first);
    throwing->SetMode(FaultingDeleteNfsBackend::Mode::kSuccess);
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(2, throwing->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [recreate_ec, recreated] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 30);
    ASSERT_EQ(EC_OK, recreate_ec);
    ASSERT_EQ((std::vector<bool>{false}), recreated.key_mask);
    ASSERT_FALSE(recreated.write_session_id.empty());
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, recreated.write_session_id, {false}));
    EXPECT_EQ(0, indexer->GetStorageUsage());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RollbackRetainsLedgerAfterMalformedPhysicalDeleteResult) {
    constexpr const char *kKey = "rollback-short-delete-result";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto malformed =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kShortResult);
    ASSERT_EQ(EC_OK, malformed->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = malformed;
    }

    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    EXPECT_EQ(1, malformed->DeleteAttempts());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(17, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    malformed->SetMode(FaultingDeleteNfsBackend::Mode::kSuccess);
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(2, malformed->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RollbackMissingOwnerFailsClosedAndRebuildsUsage) {
    constexpr const char *kKey = "rollback-missing-owner";
    constexpr std::uint64_t kSize = 31;
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {kSize}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(kSize, indexer->GetStorageUsage());
    const auto internal_key = KvMetaManager::InternalKey(kKey);
    const auto delete_result = indexer->Delete(&request_context_, {internal_key});
    ASSERT_EQ(EC_OK, delete_result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_result.error_codes);
    ASSERT_TRUE(indexer->Sync({internal_key}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = tracking;
    }

    // A raw/split-brain metadata deletion bypasses KVMeta's byte-accounting
    // transition. Absence is not physical-delete authority: another actor may
    // already have freed and reused the address. Abort must preserve the URI
    // as an orphan and close admission until recovery rebuilds accounting.
    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    EXPECT_EQ(0, tracking->DeleteAttempts());
    EXPECT_EQ(kSize, indexer->GetStorageUsage());
    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"rollback-counter-fence"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {37}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, PartialCommitWithIncompleteRollbackReturnsUnknownOutcome) {
    constexpr const char *kFirstKey = "partial-commit-first";
    constexpr const char *kSecondKey = "partial-commit-second";
    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {kFirstKey, kSecondKey}, {17, 23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto second_key = KvMetaManager::InternalKey(kSecondKey);
    const auto delete_result = indexer->Delete(&request_context_, {second_key});
    ASSERT_EQ(EC_OK, delete_result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_result.error_codes);
    ASSERT_TRUE(indexer->Sync({second_key}));

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));
    EXPECT_GE(failing->DeleteAttempts(), 1);

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kFirstKey, kSecondKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, PartialCommitMissingOwnerFailsClosedAndRebuildsUsage) {
    constexpr const char *kFirstKey = "partial-missing-owner-first";
    constexpr const char *kSecondKey = "partial-missing-owner-second";
    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {kFirstKey, kSecondKey}, {17, 23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto second_key = KvMetaManager::InternalKey(kSecondKey);
    const auto delete_result = indexer->Delete(&request_context_, {second_key});
    ASSERT_EQ(EC_OK, delete_result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_result.error_codes);
    ASSERT_TRUE(indexer->Sync({second_key}));

    // The session charged both allocations before publication. If one owner
    // disappears outside the supported state machine, rollback can release
    // its physical allocation but cannot know whether another actor already
    // adjusted the byte counter. Recovery must rebuild the authoritative sum.
    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));
    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"partial-owner-counter-fence"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] =
        manager_->StartWrite(&request_context_, kInstanceId, {kFirstKey, kSecondKey}, {19, 29}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false, false}));
}

TEST_F(KvMetaManagerTest, PartialCommitViewFailureFailsKvMetaClosedUntilRecovery) {
    constexpr const char *kFirstKey = "partial-view-first";
    constexpr const char *kSecondKey = "partial-view-second";
    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);

    auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {kFirstKey, kSecondKey}, {17, 23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto second_key = KvMetaManager::InternalKey(kSecondKey);
    const auto delete_result = indexer->Delete(&request_context_, {second_key});
    ASSERT_EQ(EC_OK, delete_result.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_result.error_codes);
    ASSERT_TRUE(indexer->Sync({second_key}));

    // The first item commits in memory and the missing second item fails the
    // batch CAS. If the barrier used to classify rollback ownership then
    // fails, no physical allocation can safely be declared unreferenced.
    controlled_sync->FailSyncAfter(0);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true, true}));
    ASSERT_TRUE(controlled_sync->WaitForSyncFailure(std::chrono::seconds(2)));

    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"commit-view-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());

    // Recovery may preserve the committed prefix or retire an active item;
    // either way it must make subsequent exact-key operations safe.
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {kFirstKey, kSecondKey}));
    auto [retry_ec, retry] =
        manager_->StartWrite(&request_context_, kInstanceId, {kFirstKey, kSecondKey}, {19, 29}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false, false}));
}

TEST_F(KvMetaManagerTest, CommitRollbackNeverDeletesAnUnexpectedOwnerOfTheSameAllocation) {
    constexpr const char *kKey = "commit-unexpected-same-allocation-owner";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {41}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    // Simulate an independent actor replacing the exact metadata while still
    // referring to the same physical URI. It is neither the active value nor
    // the committed value this session is authorized to delete.
    MutateObject(kInstanceId, kKey, [](CacheLocation &location) { location.set_create_time(-123); });

    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    // The unexpected owner must survive rollback. Treating physical identity
    // as metadata ownership would erase it and silently corrupt the cache.
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
    EXPECT_EQ(41, values.front().location.value_size);

    auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"same-allocation-must-wait-for-recovery"}, {7}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {kKey}));
}

TEST_F(KvMetaManagerTest, ExpiredFinishReturnsUnknownAndFailsClosedWhenCleanupFails) {
    auto [blocker_ec, blocker] =
        manager_->StartWrite(&request_context_, kInstanceId, {"expired-cleanup-blocker"}, {11}, 1);
    ASSERT_EQ(EC_OK, blocker_ec);
    auto [target_ec, target] =
        manager_->StartWrite(&request_context_, kInstanceId, {"expired-cleanup-failure"}, {19}, 2);
    ASSERT_EQ(EC_OK, target_ec);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingThenFailDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }
    ASSERT_TRUE(blocking->WaitForFirstDelete(std::chrono::seconds(3)));

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    auto finish = std::async(std::launch::async, [&]() {
        RequestContext context("expired-finish-with-cleanup-failure");
        return manager_->FinishWrite(&context, kInstanceId, target.write_session_id, {true});
    });
    EXPECT_EQ(std::future_status::timeout, finish.wait_for(std::chrono::milliseconds(100)));
    blocking->ReleaseFirstDelete();
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, finish.get());
    EXPECT_EQ(2, blocking->DeleteAttempts());
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&request_context_, kInstanceId, blocker.write_session_id, {true}));
    EXPECT_EQ(EC_SERVICE_NOT_LEADER,
              manager_->StartWrite(&request_context_, kInstanceId, {"blocked-after-expiry-cleanup"}, {7}, 30).first);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ExpiredSessionIsCleanedBeforeTheKeyCanBeWrittenAgain) {
    constexpr const char *kKey = "expires-and-retries";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    ErrorCode retry_ec = EC_UNKNOWN;
    KvMetaManager::StartWriteResult retry;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        auto attempt = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 30);
        retry_ec = attempt.first;
        retry = std::move(attempt.second);
        if (retry_ec == EC_OK && !retry.write_session_id.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while (std::chrono::steady_clock::now() < deadline);

    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, FailedWriteOnQuarantinedBackendKeepsItsAddressUntilCleanupDeadline) {
    constexpr const char *kKey = "quarantined-failed-write";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto quarantined = std::make_shared<QuarantinedWriteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, quarantined->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = quarantined;
    }

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {37}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(37, indexer->GetStorageUsage());

    // A reported transfer failure aborts the session but must not release a
    // reusable address while the backend's late-I/O quarantine is active.
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    EXPECT_EQ(0, quarantined->DeleteCalls());
    EXPECT_EQ(EC_EXIST, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
    EXPECT_EQ(EC_EXIST, manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {37}, 30).first);
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);
    EXPECT_EQ(37, indexer->GetStorageUsage());

    ASSERT_TRUE(WaitUntil([&]() { return quarantined->DeleteCalls() == 1 && indexer->GetStorageUsage() == 0; },
                          std::chrono::seconds(4)));
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {37}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {true}));
    ASSERT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {kKey}));

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ProvisionalBackendCommitsOnlyAfterDurableMetadataReservation) {
    constexpr const char *kKey = "provisional-commit-success";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto provisional = std::make_shared<ProvisionalCommitNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, provisional->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = provisional;
    }

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {41}, 30);
    ASSERT_EQ(start_ec, EC_OK);
    ASSERT_FALSE(start.write_session_id.empty());
    ASSERT_EQ(provisional->commit_calls_, 1u);
    ASSERT_EQ(provisional->committed_keys_.size(), 1u);
    EXPECT_TRUE(HasCanonicalKvMetaObjectKey(provisional->committed_keys_.front()));
    EXPECT_EQ(provisional->delete_calls_, 0u);
    EXPECT_EQ(manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}), EC_OK);
    EXPECT_EQ(manager_->Remove(&request_context_, kInstanceId, {kKey}), EC_OK);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ProvisionalBackendCommitsBatchAsDeadlineBoundedSingletons) {
    constexpr const char *kFirstKey = "provisional-batch-commit-first";
    constexpr const char *kSecondKey = "provisional-batch-commit-second";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto provisional = std::make_shared<ProvisionalCommitNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, provisional->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = provisional;
    }

    auto [start_ec, start] = manager_->StartWrite(
        &request_context_, kInstanceId, {kFirstKey, kSecondKey}, {31, 37}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());
    ASSERT_EQ(2u, provisional->commit_calls_);
    ASSERT_EQ(2u, provisional->commit_batches_.size());
    EXPECT_EQ(1u, provisional->commit_batches_[0].size());
    EXPECT_EQ(1u, provisional->commit_batches_[1].size());
    EXPECT_NE(provisional->commit_batches_[0][0], provisional->commit_batches_[1][0]);
    EXPECT_EQ(EC_OK,
              manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false, false}));

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ProvisionalBatchStopsCommittingWhenTheWriteDeadlineExpires) {
    constexpr const char *kFirstKey = "provisional-expired-commit-first";
    constexpr const char *kSecondKey = "provisional-expired-commit-second";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto provisional = std::make_shared<ProvisionalCommitNfsBackend>(metrics_registry_);
    provisional->commit_delay_ = std::chrono::milliseconds(1100);
    ASSERT_EQ(EC_OK, provisional->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = provisional;
    }

    const auto [start_ec, start] = manager_->StartWrite(
        &request_context_, kInstanceId, {kFirstKey, kSecondKey}, {31, 37}, 1);
    EXPECT_EQ(EC_TIMEOUT, start_ec);
    EXPECT_TRUE(start.write_session_id.empty());
    EXPECT_EQ(1u, provisional->commit_calls_);
    EXPECT_EQ(2u, provisional->delete_calls_);

    auto [get_ec, values] = manager_->Get(
        &request_context_, kInstanceId, {kFirstKey, kSecondKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2u, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ProvisionalBackendNeverCommitsBeforeMetadataPersistenceBarrier) {
    constexpr const char *kKey = "provisional-commit-after-sync";
    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    controlled_sync->BlockSyncAfter(0);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto provisional = std::make_shared<ProvisionalCommitNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, provisional->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = provisional;
    }

    auto start_future = std::async(std::launch::async, [&]() {
        RequestContext context("provisional-commit-ordering");
        return manager_->StartWrite(&context, kInstanceId, {kKey}, {47}, 30);
    });
    ASSERT_TRUE(controlled_sync->WaitForBlockedSyncs(1, std::chrono::seconds(3)));
    // The reservation exists in the in-memory index at this point, but the
    // persistence barrier has not completed. A commit here would convert a
    // crash into an unreachable, non-leased physical allocation.
    EXPECT_EQ(0u, provisional->commit_calls_);
    controlled_sync->ReleaseBlockedSync();

    ASSERT_EQ(std::future_status::ready, start_future.wait_for(std::chrono::seconds(3)));
    auto [start_ec, start] = start_future.get();
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());
    EXPECT_EQ(1u, provisional->commit_calls_);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, FailedProvisionalCommitRollsBackMetadataAndPhysicalAllocation) {
    constexpr const char *kKey = "provisional-commit-failure";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto provisional = std::make_shared<ProvisionalCommitNfsBackend>(metrics_registry_);
    provisional->commit_error_ = EC_IO_ERROR;
    ASSERT_EQ(EC_OK, provisional->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = provisional;
    }

    const auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {43}, 30);
    EXPECT_EQ(start_ec, EC_IO_ERROR);
    EXPECT_TRUE(start.write_session_id.empty());
    EXPECT_EQ(provisional->commit_calls_, 1u);
    EXPECT_EQ(provisional->delete_calls_, 1u);
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(get_ec, EC_OK);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_FALSE(values.front().found);
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(
        KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(indexer->GetStorageUsage(), 0u);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, FailedProvisionalCommitRetainsTombstoneWhenPhysicalRollbackFails) {
    constexpr const char *kKey = "provisional-commit-and-delete-failure";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto provisional = std::make_shared<ProvisionalCommitNfsBackend>(metrics_registry_);
    provisional->commit_error_ = EC_IO_ERROR;
    provisional->delete_error_ = EC_IO_ERROR;
    ASSERT_EQ(EC_OK, provisional->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = provisional;
    }

    const auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {53}, 30);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, start_ec);
    EXPECT_TRUE(start.write_session_id.empty());
    EXPECT_EQ(1u, provisional->commit_calls_);
    EXPECT_EQ(1u, provisional->delete_calls_);
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(
        KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    CacheLocationMapVector maps;
    const auto metadata = indexer->GetLocationMapsForMaintenance(
        &request_context_, {KvMetaManager::InternalKey(kKey)}, maps);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), metadata.error_codes);
    ASSERT_EQ(1u, maps.size());
    const auto location = maps.front().find(KvMetaManager::StableLocationId(kKey));
    ASSERT_NE(maps.front().end(), location);
    ASSERT_TRUE(location->second);
    EXPECT_EQ(CLS_DELETING, location->second->status());

    provisional->delete_error_ = EC_OK;
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(2u, provisional->delete_calls_);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1u, values.size());
    EXPECT_FALSE(values.front().found);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, LateSuccessfulFinishOnQuarantinedBackendCannotPublishOrReleaseEarly) {
    constexpr const char *kKey = "quarantined-late-success";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto quarantined = std::make_shared<QuarantinedWriteNfsBackend>(metrics_registry_, 2);
    ASSERT_EQ(EC_OK, quarantined->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = quarantined;
    }

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {43}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);

    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    // The data plane may report success after the client-visible commit lease,
    // but publishing it would violate timeout/recovery ordering. The reusable
    // allocation remains hidden and charged until the backend grace expires.
    EXPECT_EQ(EC_TIMEOUT, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
    EXPECT_EQ(EC_EXIST, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
    EXPECT_EQ(0, quarantined->DeleteCalls());
    EXPECT_EQ(43, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);

    ASSERT_TRUE(WaitUntil([&]() { return quarantined->DeleteCalls() == 1 && indexer->GetStorageUsage() == 0; },
                          std::chrono::seconds(5)));
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {43}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {true}));
    ASSERT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {kKey}));

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ExpiryRetainsTombstoneForRecoveryAfterFailedPhysicalDelete) {
    constexpr const char *kKey = "expiry-delete-fails";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {29}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    ASSERT_TRUE(failing->WaitForDeleteAttempts(1, std::chrono::seconds(3)));
    // The active leader does not spin on the failed delete; it closes EMB and
    // leaves the durable URI tombstone to the recovery owner.
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    EXPECT_EQ(1, failing->DeleteAttempts());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(29, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {31}, 30).first);
    failing->SetMode(FaultingDeleteNfsBackend::Mode::kSuccess);
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(2, failing->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {31}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_FALSE(retry.write_session_id.empty());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ExpiryWorkerContainsUnknownDeleteExceptionAndRecoveryRetriesIt) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"expiry-throws-a"}, {37}, 1);
    ASSERT_EQ(EC_OK, first_ec);
    ASSERT_FALSE(first.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_,
                                                               FaultingDeleteNfsBackend::Mode::kUnknownException);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }

    ASSERT_TRUE(throwing->WaitForDeleteAttempts(1, std::chrono::seconds(3)));
    ASSERT_TRUE(WaitUntil([&]() { return manager_->maintenance_cancelled_.load(std::memory_order_acquire); },
                          std::chrono::seconds(2)));
    auto [second_ec, second] = manager_->StartWrite(&request_context_, kInstanceId, {"expiry-throws-b"}, {41}, 1);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, second_ec);
    EXPECT_TRUE(second.write_session_id.empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    EXPECT_EQ(1, throwing->DeleteAttempts());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(37, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"expiry-throws-a", "expiry-throws-b"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(2, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_FALSE(values[1].found);
    throwing->SetMode(FaultingDeleteNfsBackend::Mode::kSuccess);
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(2, throwing->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, FinishCannotCommitAfterItsLeaseDeadlineWhileExpiryIsBusy) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"expiry-blocker"}, {11}, 1);
    ASSERT_EQ(EC_OK, first_ec);
    auto [late_ec, late] = manager_->StartWrite(&request_context_, kInstanceId, {"late-finish"}, {13}, 1);
    ASSERT_EQ(EC_OK, late_ec);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    // The expiry worker is now blocked cleaning the first session, so the
    // second session remains addressable even after its own deadline. A late
    // Finish must take it as expired and clean it, never commit it.
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(3)));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    auto finish = std::async(std::launch::async, [&]() {
        RequestContext context("late-finish-after-deadline");
        return manager_->FinishWrite(&context, kInstanceId, late.write_session_id, {true});
    });
    EXPECT_EQ(std::future_status::timeout, finish.wait_for(std::chrono::milliseconds(100)));
    blocking->ReleaseDelete();
    EXPECT_EQ(EC_TIMEOUT, finish.get());

    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {true}));
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto usage_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (indexer->GetStorageUsage() != 0 && std::chrono::steady_clock::now() < usage_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, OversizedSessionIdIsRejectedWithoutConsumingTheSession) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"bounded-session-id"}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    EXPECT_EQ(EC_BADARGS,
              manager_->FinishWrite(&request_context_,
                                    kInstanceId,
                                    std::string(manager_->limits().max_write_session_id_bytes + 1, 'x'),
                                    {true}));
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
}

TEST_F(KvMetaManagerTest, FinishRechecksLeaseAfterWaitingForTheGroupShard) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"finish-lock-wait"}, {17}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto [instance_ec, instance_info] = manager_->GetValidatedInstanceInfo(&request_context_, kInstanceId);
    ASSERT_EQ(EC_OK, instance_ec);
    ASSERT_TRUE(instance_info);
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % manager_->quota_admission_mutexes_.size();

    std::promise<void> shard_locked;
    auto release_shard = shard_locked.get_future();
    std::thread blocker([&]() {
        std::unique_lock<std::mutex> lock(manager_->quota_admission_mutexes_[quota_shard]);
        shard_locked.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    });
    release_shard.wait();

    RequestContext finish_context("finish-after-group-lock-wait");
    EXPECT_EQ(EC_TIMEOUT, manager_->FinishWrite(&finish_context, kInstanceId, start.write_session_id, {true}));
    blocker.join();

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"finish-lock-wait"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, ExactIdentityAndStorageSchemeAreValidated) {
    const std::string key = "owned-key";
    const auto internal_key = KvMetaManager::InternalKey(key);
    const std::string location_id = KvMetaManager::StableLocationId(key);
    EXPECT_TRUE(manager_->IsOwnedLocation(internal_key, location_id));
    EXPECT_FALSE(manager_->IsOwnedLocation(KvMetaManager::InternalKey("another-key"), location_id));
    EXPECT_FALSE(manager_->IsOwnedLocation(internal_key, "kvmeta:v1:6F"));
    EXPECT_FALSE(manager_->IsOwnedLocation(internal_key, "kvmeta:v1:0"));

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {key}, {31}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    auto corrupt_scheme = [](const std::vector<ErrorCode> &get_ecs,
                             const LocationIdVector &,
                             std::size_t,
                             CacheLocationVector &locations,
                             PropertyMap &) -> LocationModifierResult {
        if (get_ecs.size() != 1 || get_ecs[0] != EC_OK || locations.size() != 1 || !locations[0]) {
            return {MA_FAIL, {EC_CORRUPTION}};
        }
        auto replacement = std::make_shared<CacheLocation>(*locations[0]);
        DataStorageUri uri(replacement->location_specs().front().uri());
        uri.SetProtocol("dummy");
        replacement->mutable_location_specs().front().set_uri(uri.ToUriString());
        locations[0] = std::move(replacement);
        return {MA_OK, {EC_OK}};
    };
    const auto rmw =
        indexer->ReadModifyWriteTargetLocations(&request_context_, {internal_key}, {{location_id}}, corrupt_scheme);
    ASSERT_EQ(EC_OK, rmw.ec);
    ASSERT_EQ(1, rmw.per_location_error_codes.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), rmw.per_location_error_codes[0]);

    EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstanceId, {key}).first);
    EXPECT_EQ(EC_CORRUPTION, manager_->Remove(&request_context_, kInstanceId, {key}));
}

TEST_F(KvMetaManagerTest, PersistedObjectUriAndAccountingMustRemainUnambiguous) {
    const std::vector<std::string> keys{"duplicate-size",
                                        "fragment",
                                        "accounting-mismatch",
                                        "event-report",
                                        "oversized-uri",
                                        "missing-file-path",
                                        "root-file-path",
                                        "dot-file-path",
                                        "parent-file-path",
                                        "empty-segment-file-path",
                                        "trailing-slash-file-path",
                                        "foreign-namespace-path",
                                        "userinfo-authority",
                                        "port-authority",
                                        "implicit-query-value"};
    for (const auto &key : keys) {
        CommitObject(kInstanceId, key, 31);
    }

    MutateObject(kInstanceId, "duplicate-size", [](CacheLocation &location) {
        auto &spec = location.mutable_location_specs().front();
        spec.set_uri(spec.uri() + "&size=31");
    });
    MutateObject(kInstanceId, "fragment", [](CacheLocation &location) {
        auto &spec = location.mutable_location_specs().front();
        spec.set_uri(spec.uri() + "#ignored-by-some-parsers");
    });
    MutateObject(
        kInstanceId, "accounting-mismatch", [](CacheLocation &location) { location.set_validated_total_size(30); });
    MutateObject(kInstanceId, "event-report", [](CacheLocation &location) {
        location.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    });
    MutateObject(kInstanceId, "oversized-uri", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs/" + std::string(kMaxKvMetaLocationUriBytes, 'x') +
                                                          "?size=31");
    });
    MutateObject(kInstanceId, "missing-file-path", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs_01?size=31");
    });
    MutateObject(kInstanceId, "root-file-path", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs_01/?size=31");
    });
    MutateObject(kInstanceId, "dot-file-path", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs_01/dir/./object?size=31");
    });
    MutateObject(kInstanceId, "parent-file-path", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs_01/dir/../object?size=31");
    });
    MutateObject(kInstanceId, "empty-segment-file-path", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs_01/dir//object?size=31");
    });
    MutateObject(kInstanceId, "trailing-slash-file-path", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs_01/dir/object/?size=31");
    });
    MutateObject(kInstanceId, "foreign-namespace-path", [](CacheLocation &location) {
        auto &spec = location.mutable_location_specs().front();
        DataStorageUri uri(spec.uri());
        const auto namespace_position = uri.GetPath().find("/kvmeta/");
        ASSERT_NE(std::string::npos, namespace_position);
        uri.SetPath("/foreign/root" + uri.GetPath().substr(namespace_position));
        spec.set_uri(uri.ToUriString());
    });
    MutateObject(kInstanceId, "userinfo-authority", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://owner@nfs_01/object?size=31");
    });
    MutateObject(kInstanceId, "port-authority", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs_01:7/object?size=31");
    });
    MutateObject(kInstanceId, "implicit-query-value", [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri(location.location_specs().front().uri() + "&ownership-flag");
    });

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto delete_tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, delete_tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = delete_tracking;
    }

    for (const auto &key : keys) {
        SCOPED_TRACE(key);
        EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstanceId, {key}).first);
        EXPECT_EQ(EC_CORRUPTION, manager_->Remove(&request_context_, kInstanceId, {key}));
    }
    EXPECT_EQ(0, delete_tracking->DeleteAttempts());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, MooncakeOwnedUriRequiresANonEmptyPhysicalObjectKey) {
    const std::vector<std::string> keys{"missing-mooncake-key", "empty-mooncake-key", "valid-mooncake-key"};
    const std::vector<std::uint64_t> sizes{11, 13, 17};
    for (std::size_t i = 0; i < keys.size(); ++i) {
        CommitObject(kInstanceId, keys[i], sizes[i]);
    }
    const auto [source_ec, source_values] = manager_->Get(&request_context_, kInstanceId, {keys[2]});
    ASSERT_EQ(EC_OK, source_ec);
    ASSERT_EQ(1, source_values.size());
    ASSERT_TRUE(source_values[0].found);
    ASSERT_EQ(1, source_values[0].location.specs.size());
    const DataStorageUri source_uri(source_values[0].location.specs[0].second);
    const auto namespace_position = source_uri.GetPath().find("/kvmeta/");
    ASSERT_NE(std::string::npos, namespace_position);
    const std::string valid_physical_key = source_uri.GetPath().substr(namespace_position + 1);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto mooncake = std::make_shared<MalformedMooncakeBackend>(metrics_registry_);
    const StorageConfig mooncake_config(
        DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, "nfs_01", std::make_shared<MooncakeStorageSpec>());
    ASSERT_EQ(EC_OK, mooncake->Open(mooncake_config, request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = mooncake;
    }

    MutateObject(kInstanceId, keys[0], [&](CacheLocation &location) {
        location.set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
        location.mutable_location_specs().front().set_uri("mooncake://nfs_01/object?size=" + std::to_string(sizes[0]));
    });
    MutateObject(kInstanceId, keys[1], [&](CacheLocation &location) {
        location.set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
        location.mutable_location_specs().front().set_uri("mooncake://nfs_01/object?key=&size=" +
                                                          std::to_string(sizes[1]));
    });
    MutateObject(kInstanceId, keys[2], [&](CacheLocation &location) {
        location.set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
        location.mutable_location_specs().front().set_uri("mooncake://nfs_01/object?key=" + valid_physical_key +
                                                          "&size=" + std::to_string(sizes[2]));
    });

    EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstanceId, {keys[0]}).first);
    EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstanceId, {keys[1]}).first);
    auto [valid_ec, valid] = manager_->Get(&request_context_, kInstanceId, {keys[2]});
    ASSERT_EQ(EC_OK, valid_ec);
    ASSERT_EQ(1, valid.size());
    EXPECT_TRUE(valid.front().found);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RejectsAmbiguousOrUnboundedRequestsBeforeAllocation) {
    auto [duplicate_ec, duplicate] = manager_->StartWrite(&request_context_, kInstanceId, {"dup", "dup"}, {1, 2}, 30);
    EXPECT_EQ(EC_DUPLICATE_ENTITY, duplicate_ec);
    EXPECT_TRUE(duplicate.locations.empty());

    auto [size_count_ec, size_count] = manager_->StartWrite(&request_context_, kInstanceId, {"key"}, {}, 30);
    EXPECT_EQ(EC_BADARGS, size_count_ec);
    EXPECT_TRUE(size_count.locations.empty());

    auto [zero_size_ec, zero_size] = manager_->StartWrite(&request_context_, kInstanceId, {"key"}, {0}, 30);
    EXPECT_EQ(EC_OUT_OF_LIMIT, zero_size_ec);
    EXPECT_TRUE(zero_size.locations.empty());

    auto [timeout_ec, timeout] = manager_->StartWrite(&request_context_, kInstanceId, {"key"}, {1}, 0);
    EXPECT_EQ(EC_BADARGS, timeout_ec);
    EXPECT_TRUE(timeout.locations.empty());

    // Keep the overflow check correct for a custom configuration where one
    // value may fit max_value_bytes but cannot fit max_batch_bytes.
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_value_bytes = 16;
    limits.max_batch_bytes = 8;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());
    auto [batch_limit_ec, batch_limit] =
        manager_->StartWrite(&request_context_, kInstanceId, {"too-large-for-batch"}, {9}, 30);
    EXPECT_EQ(EC_OUT_OF_LIMIT, batch_limit_ec);
    EXPECT_TRUE(batch_limit.locations.empty());
}

TEST_F(KvMetaManagerTest, RejectsWriteTimeoutLimitOutsideTheProtocolRange) {
    KvMetaManager::Limits limits;
    limits.max_write_timeout_seconds = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
    KvMetaManager invalid_manager(cache_manager_, registry_manager_, limits);

    EXPECT_FALSE(invalid_manager.Init());

    limits.max_write_timeout_seconds = std::numeric_limits<std::int32_t>::max();
    limits.max_failed_write_cleanup_grace_seconds = 1;
    KvMetaManager overflowing_lease_manager(cache_manager_, registry_manager_, limits);
    EXPECT_FALSE(overflowing_lease_manager.Init());

    limits.max_write_timeout_seconds = 1;
    limits.max_failed_write_cleanup_grace_seconds = -1;
    KvMetaManager negative_grace_manager(cache_manager_, registry_manager_, limits);
    EXPECT_FALSE(negative_grace_manager.Init());
}

TEST_F(KvMetaManagerTest, RejectsBackendCleanupGraceAboveServerLimitBeforeAllocation) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto unsafe = std::make_shared<QuarantinedWriteNfsBackend>(metrics_registry_, 181);
    ASSERT_EQ(EC_OK, unsafe->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = unsafe;
    }

    auto [ec, result] = manager_->StartWrite(&request_context_, kInstanceId, {"unsafe-grace"}, {31}, 30);
    EXPECT_EQ(EC_CORRUPTION, ec);
    EXPECT_TRUE(result.write_session_id.empty());
    EXPECT_TRUE(result.locations.empty());
    EXPECT_EQ(0, unsafe->CreateCalls());
    EXPECT_EQ(0, unsafe->DeleteCalls());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(0, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RejectsLocationUriLimitOutsideTheClientContract) {
    KvMetaManager::Limits limits;
    limits.max_location_uri_bytes = 0;
    KvMetaManager zero_limit_manager(cache_manager_, registry_manager_, limits);
    EXPECT_FALSE(zero_limit_manager.Init());

    limits.max_location_uri_bytes = kMaxKvMetaLocationUriBytes + 1;
    KvMetaManager oversized_limit_manager(cache_manager_, registry_manager_, limits);
    EXPECT_FALSE(oversized_limit_manager.Init());
}

TEST_F(KvMetaManagerTest, RecoveryCanBeCancelledWithoutTouchingMetadata) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"committed"}, {23}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));

    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->DoRecover([]() { return true; }));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"committed"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values[0].found);
}

TEST_F(KvMetaManagerTest, DemotionDefersUnboundedSessionCleanupToRecovery) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"demoted-active"}, {19}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    LocationsPerKey active_location;
    const auto active_result = indexer->GetLocations(&request_context_,
                                                     {KvMetaManager::InternalKey("demoted-active")},
                                                     {{KvMetaManager::StableLocationId("demoted-active")}},
                                                     active_location);
    ASSERT_EQ(1, active_result.per_location_error_codes.size());
    ASSERT_EQ(1, active_location.size());
    ASSERT_EQ(1, active_location[0].size());
    ASSERT_TRUE(active_location[0][0]);
    // Tagged positive values persist the write-lease deadline. This is well
    // above any ordinary Unix timestamp in microseconds.
    EXPECT_GT(active_location[0][0]->create_time(), std::int64_t{1} << 62);

    manager_->DoCleanup();
    EXPECT_EQ(EC_NOENT, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {true}));
    auto [hidden_ec, hidden] = manager_->Get(&request_context_, kInstanceId, {"demoted-active"});
    ASSERT_EQ(EC_OK, hidden_ec);
    ASSERT_EQ(1, hidden.size());
    EXPECT_FALSE(hidden[0].found);

    const auto recovery_start = std::chrono::steady_clock::now();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    const auto recovery_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - recovery_start);
    // Recovery must not immediately delete a lease that an old leader has
    // already handed to a client. Keep the lower bound loose for slow ASAN
    // hosts while still distinguishing it from the old eager deletion.
    EXPECT_GE(recovery_elapsed.count(), 100);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"demoted-active"}, {19}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ((std::vector<bool>{false}), retry.key_mask);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RecoveryProtectsUntaggedActiveMarkerDuringRollingUpgrade) {
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"legacy-active"}, {19}, 30);
    ASSERT_EQ(EC_OK, start_ec);

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    const auto internal_key = KvMetaManager::InternalKey("legacy-active");
    const auto location_id = KvMetaManager::StableLocationId("legacy-active");
    auto replace_with_legacy_marker = [](const std::vector<ErrorCode> &get_ecs,
                                         const LocationIdVector &,
                                         std::size_t,
                                         CacheLocationVector &locations,
                                         PropertyMap &) -> LocationModifierResult {
        if (get_ecs.size() != 1 || get_ecs[0] != EC_OK || locations.size() != 1 || !locations[0]) {
            return {MA_FAIL, {EC_CORRUPTION}};
        }
        auto legacy = std::make_shared<CacheLocation>(*locations[0]);
        legacy->set_create_time(std::max<std::int64_t>(1, TimestampUtil::GetCurrentTimeUs()));
        locations[0] = std::move(legacy);
        return {MA_OK, {EC_OK}};
    };
    const auto rmw = indexer->ReadModifyWriteTargetLocations(
        &request_context_, {internal_key}, {{location_id}}, replace_with_legacy_marker);
    ASSERT_EQ(EC_OK, rmw.ec);
    ASSERT_TRUE(indexer->Sync({internal_key}));

    manager_->DoCleanup();
    const auto recovery_start = std::chrono::steady_clock::now();
    const auto abort_after_poll = [&]() {
        return std::chrono::steady_clock::now() - recovery_start >= std::chrono::milliseconds(150);
    };
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->DoRecover(abort_after_poll));

    LocationsPerKey still_active;
    const auto active_result = indexer->GetLocations(&request_context_, {internal_key}, {{location_id}}, still_active);
    ASSERT_EQ(1, active_result.per_location_error_codes.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), active_result.per_location_error_codes[0]);
    ASSERT_EQ(1, still_active.size());
    ASSERT_EQ(1, still_active[0].size());
    ASSERT_TRUE(still_active[0][0]);
    EXPECT_GT(still_active[0][0]->create_time(), 0);
    EXPECT_LT(still_active[0][0]->create_time(), std::int64_t{1} << 62);
}

TEST_F(KvMetaManagerTest, RecoveryRetainsItsLedgerAndRetriesAfterPhysicalDeleteException) {
    constexpr const char *kKey = "recovery-delete-throws";
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {41}, 1);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_FALSE(start.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto throwing = std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_,
                                                               FaultingDeleteNfsBackend::Mode::kStandardException);
    ASSERT_EQ(EC_OK, throwing->Open(original->GetStorageConfig(), request_context_.trace_id()));

    manager_->DoCleanup();
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = throwing;
    }
    EXPECT_EQ(EC_IO_ERROR, manager_->DoRecover());
    EXPECT_EQ(1, throwing->DeleteAttempts());

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstanceId));
    ASSERT_TRUE(indexer);
    LocationsPerKey active_location;
    const auto active_result = indexer->GetLocations(&request_context_,
                                                     {KvMetaManager::InternalKey(kKey)},
                                                     {{KvMetaManager::StableLocationId(kKey)}},
                                                     active_location);
    ASSERT_EQ(1, active_result.per_location_error_codes.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), active_result.per_location_error_codes[0]);
    ASSERT_EQ(1, active_location.size());
    ASSERT_EQ(1, active_location[0].size());
    ASSERT_TRUE(active_location[0][0]);
    // The stale, read-invisible owner remains the durable URI ledger while
    // physical cleanup is unresolved. A later recovery safely retries it.
    EXPECT_EQ(41, indexer->GetStorageUsage());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
    // After backend recovery, a fresh pass replays the immutable allocation
    // identity, confirms absence, and only then removes metadata.
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(1, throwing->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {kKey}, {43}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RecoveryRetainsEveryCleanupLedgerAcrossPhysicalFailures) {
    constexpr const char *kPeerInstance = "recovery-physical-failure-peer";
    constexpr const char *kFirstKey = "recovery-physical-failure-first";
    constexpr const char *kPeerKey = "recovery-physical-failure-second";
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, "default", kPeerInstance, "").first);

    const auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {kFirstKey}, {41}, 1);
    ASSERT_EQ(EC_OK, first_ec);
    ASSERT_FALSE(first.write_session_id.empty());
    const auto [peer_ec, peer] = manager_->StartWrite(&request_context_, kPeerInstance, {kPeerKey}, {43}, 1);
    ASSERT_EQ(EC_OK, peer_ec);
    ASSERT_FALSE(peer.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));

    manager_->DoCleanup();
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }
    EXPECT_EQ(EC_IO_ERROR, manager_->DoRecover());
    EXPECT_EQ(2, failing->DeleteAttempts());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));

    const auto metadata_present = [&](const std::string &instance_id, const std::string &key) {
        const auto indexer =
            cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(instance_id));
        if (!indexer) {
            ADD_FAILURE() << "missing test indexer";
            return false;
        }
        LocationsPerKey locations;
        const auto result = indexer->GetLocations(
            &request_context_, {KvMetaManager::InternalKey(key)}, {{KvMetaManager::StableLocationId(key)}}, locations);
        if (result.per_location_error_codes.size() != 1 || result.per_location_error_codes.front().size() != 1 ||
            locations.size() != 1 || locations.front().size() != 1) {
            ADD_FAILURE() << "malformed test metadata result";
            return false;
        }
        const ErrorCode location_ec = result.per_location_error_codes.front().front();
        if (location_ec == EC_OK) {
            EXPECT_TRUE(locations.front().front());
            return locations.front().front() != nullptr;
        }
        EXPECT_EQ(EC_NOENT, location_ec);
        EXPECT_FALSE(locations.front().front());
        return false;
    };

    // Each failed physical cleanup retains its own durable URI ledger; no
    // instance becomes an unreachable orphan even though recovery scans the
    // complete namespace.
    const bool first_present = metadata_present(kInstanceId, kFirstKey);
    const bool peer_present = metadata_present(kPeerInstance, kPeerKey);
    EXPECT_TRUE(first_present);
    EXPECT_TRUE(peer_present);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(2, failing->DeleteAttempts());
    EXPECT_FALSE(metadata_present(kInstanceId, kFirstKey));
    EXPECT_FALSE(metadata_present(kPeerInstance, kPeerKey));
    ASSERT_TRUE(manager_->ResumeMaintenance());
}

TEST_F(KvMetaManagerTest, CancellationClosesSessionAdmissionBeforeWorkerJoin) {
    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"before-cancel"}, {19}, 30);
    ASSERT_EQ(EC_OK, first_ec);

    manager_->CancelMaintenance();
    auto [cancelled_ec, cancelled] = manager_->StartWrite(&request_context_, kInstanceId, {"after-cancel"}, {23}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, cancelled_ec);
    EXPECT_TRUE(cancelled.locations.empty());
    EXPECT_FALSE(manager_->ResumeMaintenance());

    // An already admitted Finish may still drain cleanly. No new session can
    // be published after cancellation, and DoCleanup performs the final join.
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {false}));
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"after-cancel"}, {23}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, CancellationAfterAllocationWithFailedCleanupIsOutcomeUnknownAndFailsClosed) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto hooked = std::make_shared<HookedCreateNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, hooked->Open(original->GetStorageConfig(), request_context_.trace_id()));
    hooked->FailDeletes(true);
    hooked->SetOneShotAfterCreateHook([this]() { manager_->CancelMaintenance(); });
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = hooked;
    }

    const auto [start_ec, start] =
        manager_->StartWrite(&request_context_, kInstanceId, {"demoted-after-allocation"}, {17}, 30);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, start_ec);
    EXPECT_TRUE(start.key_mask.empty());
    EXPECT_TRUE(start.locations.empty());
    EXPECT_TRUE(start.write_session_id.empty());
    EXPECT_EQ(1, hooked->DeleteAttempts());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));

    // SERVER_NOT_LEADER would permit endpoint failover and another physical
    // allocation.  Once cleanup is uncertain, this manager instead remains
    // closed until recovery and exposes no retryable result for the first call.
    const auto [closed_ec, closed] =
        manager_->StartWrite(&request_context_, kInstanceId, {"must-not-fail-over"}, {19}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    const auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"demoted-after-allocation"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ActiveSessionCountIsBoundedBeforeAllocation) {
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_active_write_sessions = 1;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());

    auto [first_ec, first] = manager_->StartWrite(&request_context_, kInstanceId, {"session-a"}, {7}, 30);
    ASSERT_EQ(EC_OK, first_ec);
    auto [second_ec, second] = manager_->StartWrite(&request_context_, kInstanceId, {"session-b"}, {9}, 30);
    EXPECT_EQ(EC_NOSPC, second_ec);
    EXPECT_TRUE(second.key_mask.empty());
    EXPECT_TRUE(second.locations.empty());

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, first.write_session_id, {false}));
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"session-b"}, {9}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, SessionPublicationRaceDurablyRollsBackBeforeReturningNoSpace) {
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_active_write_sessions = 1;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());

    const auto shard_count = manager_->quota_admission_mutexes_.size();
    const auto default_shard = std::hash<std::string>{}("default") % shard_count;
    std::string peer_group;
    for (std::size_t i = 0; i < shard_count * 2 && peer_group.empty(); ++i) {
        const std::string candidate = "session-publish-peer-" + std::to_string(i);
        if (std::hash<std::string>{}(candidate) % shard_count != default_shard) {
            peer_group = candidate;
        }
    }
    ASSERT_FALSE(peer_group.empty());
    constexpr const char *kPeerInstance = "session-publish-peer-instance";
    CreateReclaimGroup(peer_group, kPeerInstance, 1024 * 1024, 0.8, 0);

    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    controlled_sync->BlockSyncAfter(0);

    auto raced_start = std::async(std::launch::async, [&]() {
        RequestContext context("session-publication-race-cleanup");
        return manager_->StartWrite(&context, kInstanceId, {"raced-session-cleanup"}, {17}, 30);
    });
    const bool sync_blocked = controlled_sync->WaitForBlockedSyncs(1, std::chrono::seconds(2));
    if (!sync_blocked) {
        controlled_sync->ReleaseBlockedSync();
    }
    ASSERT_TRUE(sync_blocked);

    // The first request passed Availability() and persisted its reservation,
    // but has not published a session. A different quota shard can fill the
    // one-entry global session table in that interval.
    auto [peer_ec, peer] = manager_->StartWrite(&request_context_, kPeerInstance, {"session-table-owner"}, {13}, 30);
    controlled_sync->ReleaseBlockedSync();
    const auto [raced_ec, raced] = raced_start.get();

    ASSERT_EQ(EC_OK, peer_ec);
    ASSERT_FALSE(peer.write_session_id.empty());
    EXPECT_EQ(EC_NOSPC, raced_ec);
    EXPECT_TRUE(raced.locations.empty());
    EXPECT_TRUE(raced.write_session_id.empty());

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"raced-session-cleanup"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kPeerInstance, peer.write_session_id, {false}));
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"raced-session-cleanup"}, {17}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, SessionPublicationRollbackFailureFailsKvMetaClosed) {
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_active_write_sessions = 1;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());

    const auto shard_count = manager_->quota_admission_mutexes_.size();
    const auto default_shard = std::hash<std::string>{}("default") % shard_count;
    std::string peer_group;
    for (std::size_t i = 0; i < shard_count * 2 && peer_group.empty(); ++i) {
        const std::string candidate = "session-publish-failure-peer-" + std::to_string(i);
        if (std::hash<std::string>{}(candidate) % shard_count != default_shard) {
            peer_group = candidate;
        }
    }
    ASSERT_FALSE(peer_group.empty());
    constexpr const char *kPeerInstance = "session-publish-failure-peer-instance";
    CreateReclaimGroup(peer_group, kPeerInstance, 1024 * 1024, 0.8, 0);

    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    controlled_sync->BlockSyncAfter(0);
    // The blocked call is the successful reservation barrier. Once Put()
    // loses the session-table race, fail the compensating absence barrier.
    controlled_sync->FailSyncAfter(1);

    auto raced_start = std::async(std::launch::async, [&]() {
        RequestContext context("session-publication-race-unknown");
        return manager_->StartWrite(&context, kInstanceId, {"raced-session-unknown"}, {17}, 30);
    });
    const bool sync_blocked = controlled_sync->WaitForBlockedSyncs(1, std::chrono::seconds(2));
    if (!sync_blocked) {
        controlled_sync->ReleaseBlockedSync();
    }
    ASSERT_TRUE(sync_blocked);

    auto [peer_ec, peer] = manager_->StartWrite(&request_context_, kPeerInstance, {"session-table-owner"}, {13}, 30);
    controlled_sync->ReleaseBlockedSync();
    const auto [raced_ec, raced] = raced_start.get();

    ASSERT_EQ(EC_OK, peer_ec);
    ASSERT_FALSE(peer.write_session_id.empty());
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, raced_ec);
    EXPECT_TRUE(raced.locations.empty());
    EXPECT_TRUE(raced.write_session_id.empty());
    EXPECT_TRUE(controlled_sync->WaitForSyncFailure(std::chrono::seconds(2)));

    auto [closed_ec, closed] = manager_->StartWrite(&request_context_, kInstanceId, {"closed-after-race"}, {1}, 30);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, closed_ec);
    EXPECT_TRUE(closed.locations.empty());

    // Already-published sessions remain finalizable after admission closes.
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kPeerInstance, peer.write_session_id, {false}));
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstanceId, {"raced-session-unknown"}, {17}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, SessionPublicationPhysicalCleanupFailureFailsKvMetaClosed) {
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_active_write_sessions = 1;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());

    const auto shard_count = manager_->quota_admission_mutexes_.size();
    const auto default_shard = std::hash<std::string>{}("default") % shard_count;
    std::string peer_group;
    for (std::size_t i = 0; i < shard_count * 2 && peer_group.empty(); ++i) {
        const std::string candidate = "session-publish-physical-peer-" + std::to_string(i);
        if (std::hash<std::string>{}(candidate) % shard_count != default_shard) {
            peer_group = candidate;
        }
    }
    ASSERT_FALSE(peer_group.empty());
    constexpr const char *kPeerInstance = "session-publish-physical-peer-instance";
    CreateReclaimGroup(peer_group, kPeerInstance, 1024 * 1024, 0.8, 0);

    auto *controlled_sync = InstallControlledSyncBackend(kInstanceId);
    ASSERT_NE(nullptr, controlled_sync);
    controlled_sync->BlockSyncAfter(0);

    auto raced_start = std::async(std::launch::async, [&]() {
        RequestContext context("session-publication-physical-cleanup-unknown");
        return manager_->StartWrite(&context, kInstanceId, {"raced-session-physical-unknown"}, {17}, 30);
    });
    const bool sync_blocked = controlled_sync->WaitForBlockedSyncs(1, std::chrono::seconds(2));
    if (!sync_blocked) {
        controlled_sync->ReleaseBlockedSync();
    }
    ASSERT_TRUE(sync_blocked);

    // Fill the one-entry session table from another quota shard after the
    // raced request has allocated and persisted its reservation. Replace only
    // the physical backend before releasing that barrier, so the compensating
    // metadata delete succeeds but its one-shot allocation release does not.
    auto [peer_ec, peer] = manager_->StartWrite(&request_context_, kPeerInstance, {"session-table-owner"}, {13}, 30);
    ASSERT_EQ(EC_OK, peer_ec);
    ASSERT_FALSE(peer.write_session_id.empty());

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    controlled_sync->ReleaseBlockedSync();
    const auto [raced_ec, raced] = raced_start.get();
    EXPECT_EQ(EC_OUTCOME_UNKNOWN, raced_ec);
    EXPECT_TRUE(raced.locations.empty());
    EXPECT_TRUE(raced.write_session_id.empty());
    EXPECT_EQ(1, failing->DeleteAttempts());
    EXPECT_TRUE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstanceId, {"raced-session-physical-unknown"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);
    EXPECT_EQ(EC_SERVICE_NOT_LEADER,
              manager_->StartWrite(&request_context_, kInstanceId, {"closed-after-physical-race"}, {1}, 30).first);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
    // A session published before the circuit breaker tripped remains
    // explicitly finalizable; no new session may be admitted meanwhile.
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kPeerInstance, peer.write_session_id, {false}));
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [retry_ec, retry] =
        manager_->StartWrite(&request_context_, kInstanceId, {"raced-session-physical-unknown"}, {17}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RejectsOrdinaryRegistrationIntoAKvMetaGroup) {
    ModelDeployment deployment;
    deployment.set_model_name("ordinary-kv-cache");
    deployment.set_dtype("fp16");
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);
    EXPECT_EQ(EC_BADARGS,
              cache_manager_
                  ->RegisterInstance(&request_context_,
                                     "default",
                                     "ordinary-instance",
                                     1,
                                     {LocationSpecInfo("value", 1)},
                                     deployment,
                                     {},
                                     CacheManager::QueryType::QT_BATCH_GET)
                  .first);
    EXPECT_EQ(nullptr, registry_manager_->GetInstanceInfo(&request_context_, "ordinary-instance"));

    // A rejected legacy registration cannot poison admission or reclamation
    // for the already-valid KVMeta group.
    EXPECT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, "default", "another-object-instance", "").first);
    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstanceId, {"must-not-share-quota"}, {1}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstanceId, start.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, RejectsKvMetaRegistrationIntoAnOrdinaryGroup) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup ordinary_group(*default_group);
    ordinary_group.set_name("ordinary-only-group");
    ordinary_group.set_global_quota_group_name("ordinary-only-quota");
    ordinary_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, ordinary_group));

    ModelDeployment deployment;
    deployment.set_model_name("ordinary-kv-cache");
    deployment.set_dtype("fp16");
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);
    ASSERT_EQ(EC_OK,
              cache_manager_
                  ->RegisterInstance(&request_context_,
                                     ordinary_group.name(),
                                     "ordinary-first",
                                     1,
                                     {LocationSpecInfo("value", 1)},
                                     deployment,
                                     {},
                                     CacheManager::QueryType::QT_BATCH_GET)
                  .first);
    EXPECT_EQ(EC_BADARGS,
              manager_->RegisterInstance(&request_context_, ordinary_group.name(), "kvmeta-must-not-mix", "").first);
}

TEST_F(KvMetaManagerTest, ExistingOrdinaryInstanceInLegacyMixedGroupCanStillRecover) {
    ModelDeployment deployment;
    deployment.set_model_name("legacy-ordinary-kv-cache");
    deployment.set_dtype("fp16");
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);
    const std::vector<LocationSpecInfo> specs{LocationSpecInfo("value", 1)};

    // Simulate persisted state produced before bidirectional group reservation
    // existed. Recovery must recreate the ordinary indexer without performing
    // another registry mutation or making the main KV-cache path unavailable.
    ASSERT_EQ(EC_OK,
              registry_manager_->RegisterInstance(&request_context_,
                                                  "default",
                                                  "legacy-ordinary",
                                                  1,
                                                  specs,
                                                  deployment,
                                                  {},
                                                  static_cast<std::int32_t>(CacheManager::QueryType::QT_BATCH_GET)));
    EXPECT_EQ(EC_OK,
              cache_manager_
                  ->RegisterInstance(&request_context_,
                                     "default",
                                     "legacy-ordinary",
                                     1,
                                     specs,
                                     deployment,
                                     {},
                                     CacheManager::QueryType::QT_BATCH_GET)
                  .first);
    EXPECT_NE(nullptr, cache_manager_->meta_indexer_manager()->GetMetaIndexer("legacy-ordinary"));

    // Only the optional KVMeta side fails closed until operators repair the
    // historical group; a new member of either type is still rejected.
    EXPECT_EQ(EC_BADARGS, manager_->StartWrite(&request_context_, kInstanceId, {"mixed-group"}, {1}, 30).first);
    EXPECT_EQ(EC_BADARGS,
              cache_manager_
                  ->RegisterInstance(&request_context_,
                                     "default",
                                     "new-ordinary-must-not-extend-mixed-group",
                                     1,
                                     specs,
                                     deployment,
                                     {},
                                     CacheManager::QueryType::QT_BATCH_GET)
                  .first);
    EXPECT_EQ(
        EC_BADARGS,
        manager_->RegisterInstance(&request_context_, "default", "new-kvmeta-must-not-extend-mixed-group", "").first);
}

TEST_F(KvMetaManagerTest, ConcurrentMixedRegistrationCannotCreateAMixedGroup) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup empty_group(*default_group);
    empty_group.set_name("concurrent-group-kind-reservation");
    empty_group.set_global_quota_group_name("concurrent-group-kind-quota");
    empty_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, empty_group));

    ModelDeployment deployment;
    deployment.set_model_name("ordinary-kv-cache");
    deployment.set_dtype("fp16");
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);

    std::promise<void> start_signal;
    const auto start_gate = start_signal.get_future().share();
    auto ordinary = std::async(std::launch::async, [&, start_gate]() {
        start_gate.wait();
        RequestContext context("concurrent-ordinary-registration");
        return cache_manager_
            ->RegisterInstance(&context,
                               empty_group.name(),
                               "concurrent-ordinary-instance",
                               1,
                               {LocationSpecInfo("value", 1)},
                               deployment,
                               {},
                               CacheManager::QueryType::QT_BATCH_GET)
            .first;
    });
    auto kv_meta = std::async(std::launch::async, [&, start_gate]() {
        start_gate.wait();
        RequestContext context("concurrent-kvmeta-registration");
        return manager_->RegisterInstance(&context, empty_group.name(), "concurrent-kvmeta-instance", "").first;
    });
    start_signal.set_value();

    const ErrorCode ordinary_ec = ordinary.get();
    const ErrorCode kv_meta_ec = kv_meta.get();
    EXPECT_EQ(1, static_cast<int>(ordinary_ec == EC_OK) + static_cast<int>(kv_meta_ec == EC_OK));
    EXPECT_TRUE((ordinary_ec == EC_OK && kv_meta_ec == EC_BADARGS) ||
                (ordinary_ec == EC_BADARGS && kv_meta_ec == EC_OK));

    const auto [instances_ec, instances] = registry_manager_->ListInstanceInfo(&request_context_, empty_group.name());
    ASSERT_EQ(EC_OK, instances_ec);
    ASSERT_EQ(1, instances.size());
    ASSERT_TRUE(instances.front());
    EXPECT_EQ(kv_meta_ec == EC_OK, HasKvMetaReservedInstancePrefix(instances.front()->instance_id()));
}

TEST_F(KvMetaManagerTest, ExactValueSizesAreIncludedInByteAdmission) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup object_group(*default_group);
    object_group.set_name("small-object-group");
    object_group.set_global_quota_group_name("small-object-quota");
    object_group.set_version(1);
    object_group.set_quota(InstanceGroupQuota(20, {QuotaConfig(20, DataStorageType::DATA_STORAGE_TYPE_NFS)}));
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));
    ASSERT_EQ(EC_OK,
              manager_->RegisterInstance(&request_context_, "small-object-group", "small-object-instance", "").first);

    auto [oversized_ec, oversized] =
        manager_->StartWrite(&request_context_, "small-object-instance", {"a", "b"}, {17, 4}, 30);
    EXPECT_EQ(EC_NOSPC, oversized_ec);
    EXPECT_TRUE(oversized.locations.empty());

    auto [start_ec, start] = manager_->StartWrite(&request_context_, "small-object-instance", {"a"}, {17}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, "small-object-instance", start.write_session_id, {true}));
    auto [remaining_ec, remaining] = manager_->StartWrite(&request_context_, "small-object-instance", {"b"}, {4}, 30);
    EXPECT_EQ(EC_NOSPC, remaining_ec);
    EXPECT_TRUE(remaining.locations.empty());

    auto [fill_ec, fill] = manager_->StartWrite(&request_context_, "small-object-instance", {"b"}, {3}, 30);
    ASSERT_EQ(EC_OK, fill_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, "small-object-instance", fill.write_session_id, {true}));
    auto [full_ec, full] = manager_->StartWrite(&request_context_, "small-object-instance", {"c"}, {1}, 30);
    EXPECT_EQ(EC_NOSPC, full_ec);
    EXPECT_TRUE(full.locations.empty());
}

TEST_F(KvMetaManagerTest, ConcurrentStartsCannotOvershootExactByteQuota) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup object_group(*default_group);
    object_group.set_name("concurrent-object-group");
    object_group.set_global_quota_group_name("concurrent-object-quota");
    object_group.set_version(1);
    object_group.set_quota(InstanceGroupQuota(20, {QuotaConfig(20, DataStorageType::DATA_STORAGE_TYPE_NFS)}));
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));
    ASSERT_EQ(EC_OK,
              manager_->RegisterInstance(&request_context_, "concurrent-object-group", "concurrent-object-instance", "")
                  .first);

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::array<ErrorCode, 2> errors{EC_UNKNOWN, EC_UNKNOWN};
    std::array<KvMetaManager::StartWriteResult, 2> results;
    std::array<std::thread, 2> workers;
    for (std::size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i]() {
            RequestContext context("kv_meta_concurrent_admission_" + std::to_string(i));
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto [ec, result] =
                manager_->StartWrite(&context, "concurrent-object-instance", {"key-" + std::to_string(i)}, {15}, 30);
            errors[i] = ec;
            results[i] = std::move(result);
        });
    }
    while (ready.load(std::memory_order_acquire) != static_cast<int>(workers.size())) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto &worker : workers) {
        worker.join();
    }

    const std::size_t success_count = static_cast<std::size_t>(std::count(errors.begin(), errors.end(), EC_OK));
    const std::size_t quota_failure_count =
        static_cast<std::size_t>(std::count(errors.begin(), errors.end(), EC_NOSPC));
    EXPECT_EQ(1, success_count);
    EXPECT_EQ(1, quota_failure_count);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(
        KvMetaManager::InternalInstanceId("concurrent-object-instance"));
    ASSERT_TRUE(indexer);
    EXPECT_EQ(15, indexer->GetStorageUsage());
    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (errors[i] == EC_OK) {
            ASSERT_EQ((std::vector<bool>{false}), results[i].key_mask);
            ASSERT_EQ(1, results[i].locations.size());
            EXPECT_EQ(15, results[i].locations.front().value_size);
            ASSERT_EQ(EC_OK,
                      manager_->FinishWrite(
                          &request_context_, "concurrent-object-instance", results[i].write_session_id, {false}));
        } else {
            EXPECT_TRUE(results[i].key_mask.empty());
            EXPECT_TRUE(results[i].locations.empty());
        }
    }
    EXPECT_EQ(0, indexer->GetStorageUsage());
}

TEST_F(KvMetaManagerTest, ReclaimerEvictsTheLeastRecentlyUsedCommittedObjectAtTheByteWatermark) {
    constexpr const char *kGroup = "reclaim-lru-group";
    constexpr const char *kInstance = "reclaim-lru-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);

    CommitObject(kInstance, "touched-old", 30);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "oldest-untouched", 30);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "newest", 30);
    // Touch the object that was oldest before this read. This must move it
    // behind both untouched objects in the eviction order; touching the
    // newest object would not prove that Get refreshes LRU heat.
    auto [touch_ec, touch] = manager_->Get(&request_context_, kInstance, {"touched-old"});
    ASSERT_EQ(EC_OK, touch_ec);
    ASSERT_EQ(1, touch.size());
    ASSERT_TRUE(touch[0].found);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(90, indexer->GetStorageUsage());
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 60; }, std::chrono::seconds(2)));
    ASSERT_TRUE(
        WaitUntil([&]() { return metrics_registry_->GetGauge("kv_meta_reclaimer.pending_object_count").Get() == 0; },
                  std::chrono::seconds(2)));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"touched-old", "oldest-untouched", "newest"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(3, values.size());
    EXPECT_TRUE(values[0].found);
    EXPECT_FALSE(values[1].found);
    EXPECT_TRUE(values[2].found);
    EXPECT_GE(metrics_registry_->GetCounter("kv_meta_reclaimer.round_count").Get(), 1);
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.retired_object_count").Get());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.reclaimed_object_count").Get());
    EXPECT_EQ(30, metrics_registry_->GetCounter("kv_meta_reclaimer.reclaimed_bytes").Get());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_attempted_object_count").Get());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_object_count").Get());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_bytes").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.pending_bytes").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.blocked_group_count").Get());
}

TEST_F(KvMetaManagerTest, ReclaimerNeverDeletesACorruptRootPathOwnershipRecord) {
    constexpr const char *kGroup = "reclaim-corrupt-root-group";
    constexpr const char *kInstance = "reclaim-corrupt-root-instance";
    constexpr const char *kKey = "corrupt-root-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, kKey, 90);
    MutateObject(kInstance, kKey, [](CacheLocation &location) {
        location.mutable_location_specs().front().set_uri("file://nfs_01/?size=90");
    });

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = tracking;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const double errors_before = metrics_registry_->GetCounter("kv_meta_reclaimer.error_count").Get();
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(
        [&]() { return metrics_registry_->GetCounter("kv_meta_reclaimer.error_count").Get() > errors_before; },
        std::chrono::seconds(2)));
    manager_->CancelMaintenance();

    EXPECT_EQ(0, tracking->DeleteAttempts());
    EXPECT_EQ(90, indexer->GetStorageUsage());
    EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstance, {kKey}).first);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ReclaimerDetectsAnUnexpectedMissingOwnerAfterDeletingTheCapturedGeneration) {
    constexpr const char *kGroup = "reclaim-missing-owner-group";
    constexpr const char *kInstance = "reclaim-missing-owner-instance";
    constexpr const char *kKey = "reclaim-missing-owner-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 500);
    CommitObject(kInstance, kKey, 90);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kSuccess);
    ASSERT_EQ(EC_OK, tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = tracking;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto internal_key = KvMetaManager::InternalKey(kKey);
    const auto location_id = KvMetaManager::StableLocationId(kKey);
    const auto has_finite_retirement_fence = [&]() {
        LocationsPerKey locations;
        const auto result = indexer->GetLocations(&request_context_, {internal_key}, {{location_id}}, locations);
        return result.ec == EC_OK && locations.size() == 1 && locations.front().size() == 1 &&
               locations.front().front() && locations.front().front()->status() == CLS_DELETING &&
               locations.front().front()->create_time() != std::numeric_limits<std::int64_t>::max();
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(has_finite_retirement_fence, std::chrono::seconds(2)));

    // Model an unsupported actor removing the retired owner before this
    // worker's exact metadata delete. The captured URI identifies an immutable
    // allocation generation, so physical cleanup remains safe; the unexpected
    // metadata transition must still fail closed instead of changing usage.
    const auto raw_delete = indexer->Delete(&request_context_, {internal_key});
    ASSERT_EQ(EC_OK, raw_delete.ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), raw_delete.error_codes);
    ASSERT_TRUE(indexer->Sync({internal_key}));
    ASSERT_TRUE(WaitUntil([&]() { return manager_->maintenance_cancelled_.load(std::memory_order_acquire); },
                          std::chrono::seconds(2)));
    EXPECT_EQ(1, tracking->DeleteAttempts());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
    manager_->DoCleanup();
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    ASSERT_TRUE(manager_->ResumeMaintenance());
}

TEST_F(KvMetaManagerTest, ReclaimerDeletesOnlyTheCapturedGenerationBeforeRejectingAReplacementOwner) {
    constexpr const char *kGroup = "reclaim-replaced-owner-group";
    constexpr const char *kInstance = "reclaim-replaced-owner-instance";
    constexpr const char *kKey = "reclaim-replaced-owner-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 500);
    CommitObject(kInstance, kKey, 90);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kSuccess);
    ASSERT_EQ(EC_OK, tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = tracking;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto internal_key = KvMetaManager::InternalKey(kKey);
    const auto location_id = KvMetaManager::StableLocationId(kKey);
    const auto has_finite_retirement_fence = [&]() {
        LocationsPerKey locations;
        const auto result = indexer->GetLocations(&request_context_, {internal_key}, {{location_id}}, locations);
        return result.ec == EC_OK && locations.size() == 1 && locations.front().size() == 1 &&
               locations.front().front() && locations.front().front()->status() == CLS_DELETING &&
               locations.front().front()->create_time() != std::numeric_limits<std::int64_t>::max();
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(has_finite_retirement_fence, std::chrono::seconds(2)));

    // Model a split-brain/out-of-protocol writer replacing the exact retired
    // value under the stable location id. A compare mismatch is definitive
    // ownership conflict, not a transient condition to retry indefinitely.
    MutateObject(kInstance, kKey, [](CacheLocation &location) { ReplaceKvMetaObjectNonce(location, 'w'); });
    ASSERT_TRUE(indexer->Sync({internal_key}));
    ASSERT_TRUE(WaitUntil([&]() { return manager_->maintenance_cancelled_.load(std::memory_order_acquire); },
                          std::chrono::seconds(2)));

    EXPECT_EQ(1, tracking->DeleteAttempts());
    EXPECT_EQ(90, indexer->GetStorageUsage());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ReclaimerRetiresMetadataBeforeWaitingForTheReadGracePeriod) {
    constexpr const char *kGroup = "reclaim-grace-group";
    constexpr const char *kInstance = "reclaim-grace-instance";
    constexpr const char *kKey = "grace-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 250);
    CommitObject(kInstance, kKey, 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto internal_key = KvMetaManager::InternalKey(kKey);
    const auto location_id = KvMetaManager::StableLocationId(kKey);
    const auto read_status = [&]() {
        CacheLocationMapVector maps;
        const auto result = indexer->GetLocationMapsForMaintenance(&request_context_, {internal_key}, maps);
        if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
            return CLS_NOT_FOUND;
        }
        const auto it = maps[0].find(location_id);
        return it == maps[0].end() || !it->second ? CLS_NOT_FOUND : it->second->status();
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return read_status() == CLS_DELETING; }, std::chrono::seconds(2)));

    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_EQ(90, indexer->GetStorageUsage());
    EXPECT_EQ(EC_EXIST, manager_->TrimAll(&request_context_, kInstance, false));

    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    EXPECT_EQ(CLS_NOT_FOUND, read_status());
}

TEST_F(KvMetaManagerTest, ReclaimerAnchorsPersistedGraceAfterEveryReaderFence) {
    constexpr const char *kGroup = "reclaim-persisted-grace-group";
    constexpr const char *kFirstInstance = "a-reclaim-persisted-grace-instance";
    constexpr const char *kLastInstance = "z-reclaim-persisted-grace-instance";
    constexpr const char *kLastKey = "last-reader-fence";
    constexpr std::int64_t kDeadlineTag = std::int64_t{1} << 62;
    constexpr auto kGrace = std::chrono::milliseconds(500);
    CreateReclaimGroup(kGroup, kFirstInstance, 100, 0.0, static_cast<std::int32_t>(kGrace.count()));
    auto *slow_sync = InstallControlledSyncBackend(kFirstInstance);
    ASSERT_NE(nullptr, slow_sync);
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, kGroup, kLastInstance, "reclaim-test").first);
    CommitObject(kFirstInstance, "first-reader-fence", 10);
    CommitObject(kLastInstance, kLastKey, 10);

    // The old implementation chose one persisted deadline before retiring the
    // first instance. Delay its post-fence durability barrier (the first Sync
    // is the maintenance pre-read barrier): the finite deadline must be chosen
    // only after every instance has durably stopped serving its URI.
    slow_sync->DelaySyncAfter(1, std::chrono::milliseconds(250));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 10));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&request_context_, 10));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto last_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kLastInstance));
    ASSERT_TRUE(last_indexer);
    std::int64_t persisted_marker = 0;
    ASSERT_TRUE(WaitUntil(
        [&]() {
            CacheLocationMapVector maps;
            const auto result = last_indexer->GetLocationMapsForMaintenance(
                &request_context_, {KvMetaManager::InternalKey(kLastKey)}, maps);
            if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
                return false;
            }
            const auto it = maps[0].find(KvMetaManager::StableLocationId(kLastKey));
            if (it == maps[0].end() || !it->second || it->second->status() != CLS_DELETING ||
                it->second->create_time() == std::numeric_limits<std::int64_t>::max()) {
                return false;
            }
            persisted_marker = it->second->create_time();
            return persisted_marker > kDeadlineTag;
        },
        std::chrono::seconds(2)));

    const std::int64_t remaining_grace_us = persisted_marker - kDeadlineTag - TimestampUtil::GetCurrentTimeUs();
    EXPECT_GE(remaining_grace_us, std::chrono::duration_cast<std::chrono::microseconds>(kGrace).count() - 100'000);
    auto first_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kFirstInstance));
    ASSERT_TRUE(first_indexer);
    ASSERT_TRUE(
        WaitUntil([&]() { return first_indexer->GetStorageUsage() == 0 && last_indexer->GetStorageUsage() == 0; },
                  std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerFenceSyncFailureFailsClosedAndRecoveryPreservesTheGrace) {
    constexpr const char *kGroup = "reclaim-fence-sync-failure-group";
    constexpr const char *kInstance = "reclaim-fence-sync-failure-instance";
    constexpr const char *kKey = "reader-fence-sync-failure";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    auto *controlled_sync = InstallControlledSyncBackend(kInstance);
    ASSERT_NE(nullptr, controlled_sync);
    CommitObject(kInstance, kKey, 90);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto faulting =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, faulting->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = faulting;
    }

    // The first Sync is the maintenance pre-read barrier. Fail the explicit
    // post-CAS fence barrier and verify that no finite grace or physical
    // deletion can be published from an uncertain metadata outcome.
    controlled_sync->FailSyncAfter(1);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(controlled_sync->WaitForSyncFailure(std::chrono::seconds(2)));
    ASSERT_TRUE(WaitUntil(
        [&]() {
            return manager_->StartWrite(&request_context_, kInstance, {"must-be-rejected"}, {1}, 30).first ==
                   EC_SERVICE_NOT_LEADER;
        },
        std::chrono::seconds(2)));

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    CacheLocationMapVector maps;
    const auto result =
        indexer->GetLocationMapsForMaintenance(&request_context_, {KvMetaManager::InternalKey(kKey)}, maps);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), result.error_codes);
    ASSERT_EQ(1, maps.size());
    const auto it = maps.front().find(KvMetaManager::StableLocationId(kKey));
    ASSERT_NE(maps.front().end(), it);
    ASSERT_TRUE(it->second);
    EXPECT_EQ(CLS_DELETING, it->second->status());
    EXPECT_EQ(std::numeric_limits<std::int64_t>::max(), it->second->create_time());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(0, faulting->DeleteAttempts());

    // Model a promotion before phase 2. Recovery must treat the transitional
    // marker conservatively, remain cancellable, and never delete it eagerly.
    manager_->DoCleanup();
    const auto recovery_start = std::chrono::steady_clock::now();
    const auto abort_after_poll = [&]() {
        return std::chrono::steady_clock::now() - recovery_start >= std::chrono::milliseconds(150);
    };
    EXPECT_EQ(EC_SERVICE_NOT_LEADER, manager_->DoRecover(abort_after_poll));
    EXPECT_EQ(0, faulting->DeleteAttempts());

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ReclaimerNeverEvictsAnActiveWriteAndReclaimsAfterCommit) {
    constexpr const char *kGroup = "reclaim-active-group";
    constexpr const char *kInstance = "reclaim-active-instance";
    constexpr const char *kKey = "active-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstance, {kKey}, {90}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    LocationsPerKey locations;
    auto exact = indexer->GetLocations(
        &request_context_, {KvMetaManager::InternalKey(kKey)}, {{KvMetaManager::StableLocationId(kKey)}}, locations);
    ASSERT_EQ(EC_OK, exact.ec);
    ASSERT_EQ(1, locations.size());
    ASSERT_EQ(1, locations[0].size());
    ASSERT_TRUE(locations[0][0]);
    EXPECT_EQ(CLS_NEW, locations[0][0]->status());
    EXPECT_GT(locations[0][0]->create_time(), 0);
    EXPECT_EQ(90, indexer->GetStorageUsage());

    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, start.write_session_id, {true}));
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerRetainsItsLedgerAndRetriesPhysicalDeleteUntilConfirmedAbsent) {
    constexpr const char *kGroup = "reclaim-delete-failure-group";
    constexpr const char *kInstance = "reclaim-delete-failure-instance";
    constexpr const char *kKey = "delete-failure-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, kKey, 90);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    std::shared_ptr<DataStorageBackend> original;
    auto faulting =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        original = storage_manager->storage_map_.at("nfs_01");
        ASSERT_EQ(EC_OK, faulting->Open(original->GetStorageConfig(), request_context_.trace_id()));
        storage_manager->storage_map_["nfs_01"] = faulting;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(faulting->WaitForDeleteAttempts(1, std::chrono::seconds(2)));
    ASSERT_TRUE(faulting->WaitForDeleteAttempts(2, std::chrono::seconds(2)));

    // No backend failure can release logical quota or erase the only durable
    // URI ledger. The value is already hidden by the tombstone, but all 90
    // bytes remain charged until physical absence is confirmed.
    EXPECT_EQ(90, indexer->GetStorageUsage());
    EXPECT_FALSE(manager_->maintenance_cancelled_.load(std::memory_order_acquire));
    CacheLocationMapVector maps;
    const auto metadata = indexer->GetLocationMapsForMaintenance(
        &request_context_, {KvMetaManager::InternalKey(kKey)}, maps);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), metadata.error_codes);
    ASSERT_EQ(1, maps.size());
    const auto location_it = maps.front().find(KvMetaManager::StableLocationId(kKey));
    ASSERT_NE(maps.front().end(), location_it);
    ASSERT_TRUE(location_it->second);
    EXPECT_EQ(CLS_DELETING, location_it->second->status());

    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);

    faulting->SetMode(FaultingDeleteNfsBackend::Mode::kSuccess);
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(3)));
    EXPECT_GE(metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_attempted_object_count").Get(), 3);
    EXPECT_GE(metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_object_count").Get(), 2);
    EXPECT_GE(metrics_registry_->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_bytes").Get(), 180);
    auto [open_ec, opened] =
        manager_->StartWrite(&request_context_, kInstance, {"capacity-is-reusable"}, {7}, 30);
    ASSERT_EQ(EC_OK, open_ec);
    ASSERT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, opened.write_session_id, {false}));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ReclaimerPhysicalDeleteDoesNotHoldTheEmbGroupAdmissionShard) {
    constexpr const char *kGroup = "reclaim-nonblocking-io-group";
    constexpr const char *kInstance = "reclaim-nonblocking-io-instance";
    constexpr const char *kRetiredKey = "slow-delete-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, kRetiredKey, 90);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto blocking = std::make_shared<BlockingDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, blocking->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = blocking;
    }

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(blocking->WaitForDelete(std::chrono::seconds(2)));

    // The same key is fenced by its durable tombstone, but unrelated capacity
    // in the group remains usable while the backend is slow. This is the
    // isolation boundary that keeps GC I/O off the EMB request critical path.
    auto [same_key_ec, same_key] = manager_->StartWrite(&request_context_, kInstance, {kRetiredKey}, {90}, 30);
    EXPECT_EQ(EC_EXIST, same_key_ec);
    EXPECT_TRUE(same_key.locations.empty());

    auto unrelated = std::async(std::launch::async, [&]() {
        RequestContext context("put-during-reclaim-physical-delete");
        return manager_->StartWrite(&context, kInstance, {"unrelated-small-object"}, {5}, 30);
    });
    ASSERT_EQ(std::future_status::ready, unrelated.wait_for(std::chrono::milliseconds(500)));
    auto [unrelated_ec, unrelated_start] = unrelated.get();
    ASSERT_EQ(EC_OK, unrelated_ec);
    ASSERT_FALSE(unrelated_start.write_session_id.empty());

    blocking->ReleaseDelete();
    EXPECT_EQ(EC_OK,
              manager_->FinishWrite(
                  &request_context_, kInstance, unrelated_start.write_session_id, std::vector<bool>{false}));
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RecoveryResumesAReclaimerPhysicalDeleteFromTheDurableTombstone) {
    constexpr const char *kGroup = "reclaim-restart-retry-group";
    constexpr const char *kInstance = "reclaim-restart-retry-instance";
    constexpr const char *kKey = "reclaim-restart-retry-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, kKey, 90);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto faulting =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, faulting->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = faulting;
    }

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(faulting->WaitForDeleteAttempts(1, std::chrono::seconds(2)));
    EXPECT_EQ(90, indexer->GetStorageUsage());

    // Drop every volatile pending queue entry as a demotion/restart would.
    // Recovery must rediscover the persisted URI and continue the same
    // idempotent physical-first transaction.
    manager_->DoCleanup();
    faulting->SetMode(FaultingDeleteNfsBackend::Mode::kSuccess);
    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_GE(faulting->DeleteAttempts(), 2);
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ReclaimerPendingCreditPreventsOverEvictionDuringTheGracePeriod) {
    constexpr const char *kGroup = "reclaim-credit-group";
    constexpr const char *kInstance = "reclaim-credit-instance";
    const std::vector<std::string> keys{"credit-old", "credit-middle", "credit-new"};
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 1000);
    for (const auto &key : keys) {
        CommitObject(kInstance, key, 30);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto retired_count = [&]() {
        std::size_t count = 0;
        for (const auto &key : keys) {
            CacheLocationMapVector maps;
            const auto result =
                indexer->GetLocationMapsForMaintenance(&request_context_, {KvMetaManager::InternalKey(key)}, maps);
            if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
                continue;
            }
            const auto it = maps[0].find(KvMetaManager::StableLocationId(key));
            if (it != maps[0].end() && it->second && it->second->status() == CLS_DELETING) {
                ++count;
            }
        }
        return count;
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return retired_count() == 1; }, std::chrono::seconds(2)));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(1, retired_count());
    EXPECT_EQ(90, indexer->GetStorageUsage());

    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 60; }, std::chrono::seconds(2)));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, keys);
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(3, values.size());
    EXPECT_EQ(2, std::count_if(values.begin(), values.end(), [](const auto &value) { return value.found; }));
}

TEST_F(KvMetaManagerTest, ReclaimerShrinksALargeBatchToThePendingByteBudget) {
    constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
    constexpr std::uint64_t kTiB = 1024ULL * kGiB;
    constexpr std::size_t kObjectCount = 4097;
    constexpr std::uint64_t kPendingBytes = 4ULL * kTiB;
    constexpr const char *kGroup = "reclaim-pending-byte-budget-group";
    constexpr const char *kInstance = "reclaim-pending-byte-budget-instance";

    // Allow each setup RPC to create 64 maximum-sized values. The production
    // per-object limit remains 1 GiB; only this test's request aggregate limit
    // is widened so the real 4 TiB pending boundary is practical to exercise.
    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_batch_bytes = 64ULL * kGiB;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());
    CreateReclaimGroup(kGroup, kInstance, 8LL * static_cast<std::int64_t>(kTiB), 0.0, 10'000);

    for (std::size_t begin = 0; begin < kObjectCount; begin += limits.max_batch_items) {
        const std::size_t count = std::min(limits.max_batch_items, kObjectCount - begin);
        std::vector<std::string> keys;
        keys.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            keys.push_back("pending-byte-" + std::to_string(begin + i));
        }
        auto [start_ec, start] =
            manager_->StartWrite(&request_context_, kInstance, keys, std::vector<std::uint64_t>(count, kGiB), 30);
        ASSERT_EQ(EC_OK, start_ec);
        ASSERT_EQ(count, start.locations.size());
        ASSERT_EQ(EC_OK,
                  manager_->FinishWrite(
                      &request_context_, kInstance, start.write_session_id, std::vector<bool>(count, true)));
    }

    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, kObjectCount));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&request_context_, kObjectCount));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    // 4097 one-GiB candidates exceed the 4-TiB pending bound by one object.
    // The worker must retire the largest safe prefix instead of rejecting the
    // same oversized vector forever.
    ASSERT_TRUE(WaitUntil(
        [&]() { return metrics_registry_->GetCounter("kv_meta_reclaimer.retired_object_count").Get() == 4096; },
        std::chrono::seconds(10)));
    EXPECT_DOUBLE_EQ(static_cast<double>(kPendingBytes),
                     metrics_registry_->GetGauge("kv_meta_reclaimer.pending_bytes").Get());
    EXPECT_DOUBLE_EQ(4096, metrics_registry_->GetGauge("kv_meta_reclaimer.pending_object_count").Get());
}

TEST_F(KvMetaManagerTest, ReclaimerGracePeriodForOneGroupDoesNotBlockAnotherGroup) {
    constexpr const char *kSlowGroup = "reclaim-a-slow-group";
    constexpr const char *kSlowInstance = "reclaim-a-slow-instance";
    constexpr const char *kFastGroup = "reclaim-z-fast-group";
    constexpr const char *kFastInstance = "reclaim-z-fast-instance";
    constexpr const char *kSlowKey = "slow-object";
    CreateReclaimGroup(kSlowGroup, kSlowInstance, 100, 0.8, 750);
    CreateReclaimGroup(kFastGroup, kFastInstance, 100, 0.8, 0);
    CommitObject(kSlowInstance, kSlowKey, 90);
    CommitObject(kFastInstance, "fast-object", 90);

    auto slow_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kSlowInstance));
    auto fast_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kFastInstance));
    ASSERT_TRUE(slow_indexer);
    ASSERT_TRUE(fast_indexer);
    const auto slow_is_retired = [&]() {
        CacheLocationMapVector maps;
        const auto result = slow_indexer->GetLocationMapsForMaintenance(
            &request_context_, {KvMetaManager::InternalKey(kSlowKey)}, maps);
        if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
            return false;
        }
        const auto it = maps[0].find(KvMetaManager::StableLocationId(kSlowKey));
        return it != maps[0].end() && it->second && it->second->status() == CLS_DELETING;
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(slow_is_retired, std::chrono::seconds(2)));
    ASSERT_TRUE(WaitUntil([&]() { return fast_indexer->GetStorageUsage() == 0; }, std::chrono::milliseconds(250)));
    EXPECT_EQ(90, slow_indexer->GetStorageUsage());
    ASSERT_TRUE(WaitUntil([&]() { return slow_indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerDoesNotRepeatConfirmedPhysicalDeleteWhileRetryingMetadataSync) {
    constexpr const char *kGroup = "reclaim-sync-failure-group";
    constexpr const char *kInstance = "reclaim-sync-failure-instance";
    constexpr const char *kKey = "sync-failure-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    auto *meta_backend = InstallFailingSyncBackend(kInstance);
    ASSERT_NE(nullptr, meta_backend);
    CommitObject(kInstance, kKey, 90);
    meta_backend->FailNextMaintenanceDeleteSync();

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    std::shared_ptr<DataStorageBackend> original;
    auto tracking =
        std::make_shared<FaultingDeleteNfsBackend>(metrics_registry_, FaultingDeleteNfsBackend::Mode::kSuccess);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        original = storage_manager->storage_map_.at("nfs_01");
        ASSERT_EQ(EC_OK, tracking->Open(original->GetStorageConfig(), request_context_.trace_id()));
        storage_manager->storage_map_["nfs_01"] = tracking;
    }
    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 200);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(meta_backend->WaitForSyncFailure(std::chrono::seconds(2)));
    ASSERT_TRUE(
        WaitUntil([&]() { return metrics_registry_->GetGauge("kv_meta_reclaimer.blocked_group_count").Get() == 1; },
                  std::chrono::seconds(2)));

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"new-object"}, {10}, 30);
    EXPECT_EQ(EC_EXIST, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    EXPECT_EQ(1, tracking->DeleteAttempts());

    ASSERT_TRUE(WaitUntil(
        [&]() { return metrics_registry_->GetGauge("kv_meta_reclaimer.pending_object_count").Get() == 0; },
        std::chrono::seconds(3)));
    EXPECT_EQ(1, tracking->DeleteAttempts());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }

    auto [start_ec, start] = manager_->StartWrite(&request_context_, kInstance, {"new-object"}, {10}, 30);
    ASSERT_EQ(EC_OK, start_ec);
    ASSERT_EQ(1, start.locations.size());
    EXPECT_GE(metrics_registry_->GetCounter("kv_meta_reclaimer.retry_count").Get(), 1);
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.blocked_group_count").Get());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, start.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerHonorsTheIndependentKeyCountWatermark) {
    constexpr const char *kGroup = "reclaim-key-count-group";
    constexpr const char *kInstance = "reclaim-key-count-instance";
    CreateReclaimGroup(kGroup, kInstance, 10'000, 0.8, 0, 4);
    CommitObject(kInstance, "key-old", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "key-middle", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "key-new", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, "key-newest", 10);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(4, indexer->GetKeyCount());
    ASSERT_EQ(40, indexer->GetStorageUsage());
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetKeyCount() == 3; }, std::chrono::seconds(2)));
    EXPECT_EQ(30, indexer->GetStorageUsage());

    auto [get_ec, values] =
        manager_->Get(&request_context_, kInstance, {"key-old", "key-middle", "key-new", "key-newest"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(4, values.size());
    EXPECT_FALSE(values[0].found);
    EXPECT_TRUE(values[1].found);
    EXPECT_TRUE(values[2].found);
    EXPECT_TRUE(values[3].found);
}

TEST_F(KvMetaManagerTest, ReclaimerHonorsTheStorageTypeByteWatermark) {
    constexpr const char *kGroup = "reclaim-storage-type-group";
    constexpr const char *kInstance = "reclaim-storage-type-instance";
    // Group usage is only 9%, but NFS usage is 90% of its independent quota.
    CreateReclaimGroup(kGroup, kInstance, 1000, 0.8, 0, MetaIndexerConfig::kDefaultMaxKeyCount, 100);
    CommitObject(kInstance, "type-pressure", 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(90, indexer->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerCreatesHeadroomForARejectedRequestBelowTheWatermark) {
    constexpr const char *kGroup = "reclaim-admission-demand-group";
    constexpr const char *kInstance = "reclaim-admission-demand-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 1.0, 0);
    CommitObject(kInstance, "old-object", 70);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_EQ(70, indexer->GetStorageUsage());

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"large-new-object"}, {40}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());

    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get());
    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstance, {"large-new-object"}, {40}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(1, retry.locations.size());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerCreatesStorageTypeHeadroomBelowTheGroupWatermark) {
    constexpr const char *kGroup = "reclaim-type-admission-demand-group";
    constexpr const char *kInstance = "reclaim-type-admission-demand-instance";
    CreateReclaimGroup(kGroup, kInstance, 1000, 1.0, 0, MetaIndexerConfig::kDefaultMaxKeyCount, 100);
    CommitObject(kInstance, "old-type-object", 70);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_EQ(70, indexer->GetStorageUsage());

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"new-type-object"}, {40}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));

    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstance, {"new-type-object"}, {40}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerCreatesPhysicalHeadroomAfterAuthoritativeBackendNoSpace) {
    constexpr const char *kGroup = "reclaim-backend-capacity-group";
    constexpr const char *kInstance = "reclaim-backend-capacity-instance";
    CreateReclaimGroup(kGroup, kInstance, 1000, 1.0, 0);
    CommitObject(kInstance, "old-backend-object", 70);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto capacity = std::make_shared<CapacityRejectingNfsBackend>(
        metrics_registry_, CapacityRejectingNfsBackend::Mode::kOnce);
    ASSERT_EQ(EC_OK, capacity->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = capacity;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    // Both logical quotas have ample headroom. Only the backend's explicit
    // EC_NOSPC should trigger this type-targeted eviction.
    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"new-backend-object"}, {40}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    EXPECT_EQ(1, capacity->CreateAttempts());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.backend_capacity_demand_count").Get());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    ASSERT_TRUE(WaitUntil(
        [&]() { return metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get() == 0; },
        std::chrono::seconds(2)));

    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kInstance, {"new-backend-object"}, {40}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(1, retry.locations.size());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, retry.write_session_id, {false}));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, BackendCapacityReclaimDoesNotRequireAStorageTypeQuota) {
    constexpr const char *kGroup = "reclaim-backend-capacity-without-type-quota-group";
    constexpr const char *kInstance = "reclaim-backend-capacity-without-type-quota-instance";
    CreateReclaimGroup(kGroup,
                       kInstance,
                       1000,
                       1.0,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_OK,
                       DataStorageType::DATA_STORAGE_TYPE_NFS,
                       "nfs_01",
                       false);
    CommitObject(kInstance, "old-backend-object", 70);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto capacity = std::make_shared<CapacityRejectingNfsBackend>(
        metrics_registry_, CapacityRejectingNfsBackend::Mode::kOnce);
    ASSERT_EQ(EC_OK, capacity->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = capacity;
    }

    auto indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [blocked_ec, blocked] =
        manager_->StartWrite(&request_context_, kInstance, {"new-backend-object"}, {40}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    EXPECT_EQ(1, capacity->CreateAttempts());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    ASSERT_TRUE(WaitUntil(
        [&]() { return metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get() == 0; },
        std::chrono::seconds(2)));

    auto [retry_ec, retry] =
        manager_->StartWrite(&request_context_, kInstance, {"new-backend-object"}, {40}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    ASSERT_EQ(1, retry.locations.size());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kInstance, retry.write_session_id, {false}));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, BackendCapacityRetriesDoNotDuplicateInFlightEviction) {
    constexpr const char *kGroup = "reclaim-backend-retry-group";
    constexpr const char *kInstance = "reclaim-backend-retry-instance";
    constexpr const char *kOldA = "backend-retry-old-a";
    constexpr const char *kOldB = "backend-retry-old-b";
    CreateReclaimGroup(kGroup, kInstance, 1000, 1.0, 500);
    CommitObject(kInstance, kOldA, 30);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kInstance, kOldB, 30);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto capacity = std::make_shared<CapacityRejectingNfsBackend>(
        metrics_registry_, CapacityRejectingNfsBackend::Mode::kAlways);
    ASSERT_EQ(EC_OK, capacity->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = capacity;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto retired_count = [&]() {
        std::size_t count = 0;
        for (const char *key : {kOldA, kOldB}) {
            CacheLocationMapVector maps;
            const auto result = indexer->GetLocationMapsForMaintenance(
                &request_context_, {KvMetaManager::InternalKey(key)}, maps);
            if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
                continue;
            }
            const auto it = maps.front().find(KvMetaManager::StableLocationId(key));
            if (it != maps.front().end() && it->second && it->second->status() == CLS_DELETING) {
                ++count;
            }
        }
        return count;
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    EXPECT_EQ(EC_NOSPC,
              manager_->StartWrite(&request_context_, kInstance, {"backend-retry-new"}, {20}, 30).first);
    ASSERT_TRUE(WaitUntil([&]() { return retired_count() == 1; }, std::chrono::seconds(2)));
    EXPECT_DOUBLE_EQ(20, metrics_registry_->GetGauge("kv_meta_reclaimer.backend_capacity_demand_bytes").Get());

    // Re-publishing the same failure while the first object is inside reader
    // grace must be covered by its pending bytes, not retire the second key.
    EXPECT_EQ(EC_NOSPC,
              manager_->StartWrite(&request_context_, kInstance, {"backend-retry-new"}, {20}, 30).first);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_EQ(1, retired_count());
    EXPECT_EQ(60, indexer->GetStorageUsage());
    EXPECT_DOUBLE_EQ(20, metrics_registry_->GetGauge("kv_meta_reclaimer.backend_capacity_demand_bytes").Get());

    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 30; }, std::chrono::seconds(2)));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(30, indexer->GetStorageUsage());
    EXPECT_EQ(0, retired_count());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.backend_capacity_demand_bytes").Get());
    ASSERT_TRUE(WaitUntil(
        [&]() { return metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get() == 0; },
        std::chrono::seconds(2)));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, ReclaimerTargetsKeyAdmissionPressureToTheFullInstance) {
    constexpr const char *kGroup = "reclaim-key-admission-demand-group";
    constexpr const char *kFullInstance = "z-reclaim-key-admission-full-instance";
    constexpr const char *kPeerInstance = "a-reclaim-key-admission-peer-instance";
    CreateReclaimGroup(kGroup, kFullInstance, 10'000, 1.0, 0, 2);
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, kGroup, kPeerInstance, "reclaim-test").first);

    // Make the peer object globally oldest. Targeted pressure must still free
    // a primary metadata key from the full instance, not evict the peer.
    CommitObject(kPeerInstance, "peer-oldest", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kFullInstance, "full-old", 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kFullInstance, "full-new", 10);

    auto full_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kFullInstance));
    auto peer_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kPeerInstance));
    ASSERT_TRUE(full_indexer);
    ASSERT_TRUE(peer_indexer);
    ASSERT_EQ(2, full_indexer->GetKeyCount());
    ASSERT_EQ(1, peer_indexer->GetKeyCount());
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 1));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&request_context_, 1));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 1000);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kFullInstance, {"third-key"}, {10}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    // The admission wake must sample the targeted (lexically last) instance
    // immediately rather than waiting one full idle interval for rotation.
    ASSERT_TRUE(WaitUntil([&]() { return full_indexer->GetKeyCount() == 1; }, std::chrono::milliseconds(500)));
    EXPECT_EQ(1, peer_indexer->GetKeyCount());
    auto [peer_get_ec, peer_values] = manager_->Get(&request_context_, kPeerInstance, {"peer-oldest"});
    ASSERT_EQ(EC_OK, peer_get_ec);
    ASSERT_EQ(1, peer_values.size());
    EXPECT_TRUE(peer_values.front().found);

    auto [retry_ec, retry] = manager_->StartWrite(&request_context_, kFullInstance, {"third-key"}, {10}, 30);
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kFullInstance, retry.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerUsesTargetedKeyEvictionToAlsoSatisfyGroupAndTypePressure) {
    constexpr const char *kGroup = "reclaim-overlapping-pressure-group";
    constexpr const char *kFullInstance = "reclaim-overlapping-pressure-full-instance";
    constexpr const char *kPeerInstance = "reclaim-overlapping-pressure-peer-instance";
    CreateReclaimGroup(kGroup, kFullInstance, 100, 0.9, 0, 2);
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, kGroup, kPeerInstance, "reclaim-test").first);

    // The peer is globally oldest, but retiring it would satisfy only byte
    // pressure. Retiring one object from the full instance satisfies the
    // targeted key demand and the group/type byte pressure at the same time.
    CommitObject(kPeerInstance, "peer-oldest", 35);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kFullInstance, "full-old", 30);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CommitObject(kFullInstance, "full-new", 30);

    auto full_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kFullInstance));
    auto peer_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kPeerInstance));
    ASSERT_TRUE(full_indexer);
    ASSERT_TRUE(peer_indexer);
    cache_manager_->PauseReclaimer();
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kFullInstance, {"third-key"}, {1}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    EXPECT_EQ(1, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());

    cache_manager_->ResumeReclaimer();
    ASSERT_TRUE(WaitUntil([&]() { return full_indexer->GetKeyCount() == 1; }, std::chrono::seconds(2)));
    EXPECT_EQ(1, peer_indexer->GetKeyCount());
    EXPECT_EQ(35, peer_indexer->GetStorageUsage());
    auto [peer_get_ec, peer_values] = manager_->Get(&request_context_, kPeerInstance, {"peer-oldest"});
    ASSERT_EQ(EC_OK, peer_get_ec);
    ASSERT_EQ(1, peer_values.size());
    EXPECT_TRUE(peer_values.front().found);
}

TEST_F(KvMetaManagerTest, ImpossibleAdmissionDemandDoesNotEvictUsefulCacheEntries) {
    constexpr const char *kGroup = "reclaim-impossible-admission-group";
    constexpr const char *kInstance = "reclaim-impossible-admission-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 1.0, 0);
    CommitObject(kInstance, "must-stay", 70);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"cannot-fit"}, {101}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(70, indexer->GetStorageUsage());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"must-stay"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, ImpossibleStorageTypeAdmissionDoesNotEvictUsefulCacheEntries) {
    constexpr const char *kGroup = "reclaim-impossible-type-admission-group";
    constexpr const char *kInstance = "reclaim-impossible-type-admission-instance";
    CreateReclaimGroup(kGroup, kInstance, 1000, 1.0, 0, MetaIndexerConfig::kDefaultMaxKeyCount, 100);
    CommitObject(kInstance, "must-stay", 70);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    auto [blocked_ec, blocked] = manager_->StartWrite(&request_context_, kInstance, {"cannot-fit-type"}, {101}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(70, indexer->GetStorageUsage());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"must-stay"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, ImpossibleKeyAdmissionDemandDoesNotEvictUsefulCacheEntries) {
    constexpr const char *kGroup = "reclaim-impossible-key-admission-group";
    constexpr const char *kInstance = "reclaim-impossible-key-admission-instance";
    CreateReclaimGroup(kGroup, kInstance, 1000, 1.0, 0, 2);
    CommitObject(kInstance, "must-stay", 10);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    auto [blocked_ec, blocked] =
        manager_->StartWrite(&request_context_, kInstance, {"new-a", "new-b", "new-c"}, {1, 1, 1}, 30);
    EXPECT_EQ(EC_NOSPC, blocked_ec);
    EXPECT_TRUE(blocked.locations.empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(1, indexer->GetKeyCount());
    EXPECT_EQ(10, indexer->GetStorageUsage());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.admission_demand_count").Get());
    EXPECT_DOUBLE_EQ(0, metrics_registry_->GetGauge("kv_meta_reclaimer.admission_demand_group_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"must-stay"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, ReclaimerSupportsAZeroWatermark) {
    constexpr const char *kGroup = "reclaim-zero-watermark-group";
    constexpr const char *kInstance = "reclaim-zero-watermark-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.0, 0);
    CommitObject(kInstance, "remove-at-zero", 10);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerSharesTheOperationalPauseSwitch) {
    constexpr const char *kGroup = "reclaim-pause-group";
    constexpr const char *kInstance = "reclaim-pause-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, "paused-object", 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    cache_manager_->PauseReclaimer();
    ASSERT_TRUE(manager_->ResumeMaintenance());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(90, indexer->GetStorageUsage());

    cache_manager_->ResumeReclaimer();
    ASSERT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST_F(KvMetaManagerTest, ReclaimerRotatesAOneKeySampleBudgetAcrossInstances) {
    constexpr const char *kGroup = "reclaim-sample-rotation-group";
    constexpr const char *kActiveInstance = "a-reclaim-active-instance";
    constexpr const char *kCommittedInstance = "z-reclaim-committed-instance";
    CreateReclaimGroup(kGroup, kActiveInstance, 100, 0.8, 0);
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(&request_context_, kGroup, kCommittedInstance, "reclaim-test").first);

    auto [active_ec, active] = manager_->StartWrite(&request_context_, kActiveInstance, {"active"}, {45}, 30);
    ASSERT_EQ(EC_OK, active_ec);
    CommitObject(kCommittedInstance, "committed", 45);

    auto active_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kActiveInstance));
    auto committed_indexer =
        cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kCommittedInstance));
    ASSERT_TRUE(active_indexer);
    ASSERT_TRUE(committed_indexer);
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 1));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&request_context_, 1));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());

    ASSERT_TRUE(WaitUntil([&]() { return committed_indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
    EXPECT_EQ(45, active_indexer->GetStorageUsage());
    EXPECT_EQ(EC_OK, manager_->FinishWrite(&request_context_, kActiveInstance, active.write_session_id, {false}));
}

TEST_F(KvMetaManagerTest, ReclaimerRejectsAnOverlongBackendSampleWithoutDeleting) {
    constexpr const char *kGroup = "reclaim-overlong-sample-group";
    constexpr const char *kInstance = "reclaim-overlong-sample-instance";
    constexpr const char *kKey = "must-not-be-deleted";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    ASSERT_NE(nullptr, InstallOversamplingBackend(kInstance));
    CommitObject(kInstance, kKey, 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 1));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil([&]() { return metrics_registry_->GetCounter("kv_meta_reclaimer.error_count").Get() != 0; },
                          std::chrono::seconds(2)));
    cache_manager_->PauseReclaimer();

    EXPECT_EQ(90, indexer->GetStorageUsage());
    EXPECT_EQ(0, metrics_registry_->GetCounter("kv_meta_reclaimer.retired_object_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, ReclaimerIsolatesACorruptCandidateAndStillMakesProgress) {
    constexpr const char *kGroup = "reclaim-corrupt-candidate-group";
    constexpr const char *kInstance = "reclaim-corrupt-candidate-instance";
    constexpr const char *kCorruptKey = "corrupt-object";
    constexpr const char *kValidKey = "valid-object";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.5, 0);
    CommitObject(kInstance, kCorruptKey, 30);
    CommitObject(kInstance, kValidKey, 30);
    MutateObject(kInstance, kCorruptKey, [](CacheLocation &location) {
        auto &spec = location.mutable_location_specs().front();
        spec.set_uri(spec.uri() + "&size=30");
    });

    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetSamplingSize(&request_context_, 10));
    ASSERT_EQ(EC_OK, cache_manager_->cache_reclaimer()->SetBatchingSize(&request_context_, 10));
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(
        [&]() {
            auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kValidKey});
            return get_ec == EC_OK && values.size() == 1 && !values.front().found;
        },
        std::chrono::seconds(2)));
    cache_manager_->PauseReclaimer();

    EXPECT_NE(0, metrics_registry_->GetCounter("kv_meta_reclaimer.error_count").Get());
    EXPECT_NE(0, metrics_registry_->GetCounter("kv_meta_reclaimer.reclaimed_object_count").Get());
    EXPECT_EQ(EC_CORRUPTION, manager_->Get(&request_context_, kInstance, {kCorruptKey}).first);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsAnUnsupportedReclaimPolicy) {
    constexpr const char *kGroup = "reclaim-unsupported-policy-group";
    constexpr const char *kInstance = "reclaim-unsupported-policy-instance";
    CreateReclaimGroup(kGroup,
                       kInstance,
                       100,
                       0.8,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_TTL,
                       EC_CONFIG_ERROR);
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, manager_->GetInstanceInfo(&request_context_, kInstance).first);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsInvalidReclaimBounds) {
    CreateReclaimGroup("reclaim-watermark-negative-group",
                       "reclaim-watermark-negative-instance",
                       100,
                       -0.01,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-watermark-high-group",
                       "reclaim-watermark-high-instance",
                       100,
                       1.01,
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-watermark-infinity-group",
                       "reclaim-watermark-infinity-instance",
                       100,
                       std::numeric_limits<double>::infinity(),
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-watermark-nan-group",
                       "reclaim-watermark-nan-instance",
                       100,
                       std::numeric_limits<double>::quiet_NaN(),
                       0,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-negative-delay-group",
                       "reclaim-negative-delay-instance",
                       100,
                       0.8,
                       -1,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
    CreateReclaimGroup("reclaim-excessive-delay-group",
                       "reclaim-excessive-delay-instance",
                       100,
                       0.8,
                       1'800'001,
                       MetaIndexerConfig::kDefaultMaxKeyCount,
                       std::nullopt,
                       ReclaimPolicy::POLICY_LRU,
                       EC_CONFIG_ERROR);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsStorageWithoutExactObjectOwnership) {
    constexpr const char *kStorage = "external-event-report";
    auto event_spec = std::make_shared<EventReportStorageSpec>();
    StorageConfig event_config(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, kStorage, event_spec);
    ASSERT_EQ(EC_OK,
              registry_manager_->data_storage_manager()->RegisterStorage(&request_context_, kStorage, event_config));

    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup object_group(*default_group);
    object_group.set_name("event-report-object-group");
    object_group.set_global_quota_group_name("event-report-object-quota");
    object_group.set_storage_candidates({kStorage});
    object_group.set_quota(
        InstanceGroupQuota(100, {QuotaConfig(100, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5)}));
    object_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, object_group));

    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_
            ->RegisterInstance(&request_context_, object_group.name(), "event-report-object-instance", "reclaim-test")
            .first);

    constexpr const char *kHotGroup = "hot-storage-object-group";
    constexpr const char *kHotInstance = "hot-storage-object-instance";
    CreateReclaimGroup(kHotGroup, kHotInstance, 100, 0.8, 0);
    CommitObject(kHotInstance, "existing", 17);
    const auto [hot_group_ec, hot_group] = registry_manager_->GetInstanceGroup(&request_context_, kHotGroup);
    ASSERT_EQ(EC_OK, hot_group_ec);
    ASSERT_TRUE(hot_group);
    InstanceGroup updated_hot_group(*hot_group);
    updated_hot_group.set_storage_candidates({kStorage});
    updated_hot_group.set_version(hot_group->version() + 1);
    ASSERT_EQ(EC_OK,
              registry_manager_->UpdateInstanceGroup(&request_context_, updated_hot_group, hot_group->version()));

    auto [hit_ec, hit] = manager_->StartWrite(&request_context_, kHotInstance, {"existing"}, {17}, 30);
    ASSERT_EQ(EC_OK, hit_ec);
    EXPECT_EQ((std::vector<bool>{true}), hit.key_mask);
    EXPECT_EQ(EC_CONFIG_ERROR, manager_->StartWrite(&request_context_, kHotInstance, {"new-object"}, {17}, 30).first);
    EXPECT_EQ(EC_OK, registry_manager_->data_storage_manager()->UnRegisterStorage(kStorage));
}

TEST_F(KvMetaManagerTest, RegistrationReturnsOnlyExactObjectStorageCandidates) {
    constexpr const char *kMigrationOnlyStorage = "external-event-report-migration";
    auto event_spec = std::make_shared<EventReportStorageSpec>();
    StorageConfig event_config(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, kMigrationOnlyStorage, event_spec);
    ASSERT_EQ(EC_OK,
              registry_manager_->data_storage_manager()->RegisterStorage(
                  &request_context_, kMigrationOnlyStorage, event_config));

    const auto [group_ec, current_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(current_group);
    ASSERT_TRUE(current_group->cache_config());
    auto cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(cache_config->FromJsonString(current_group->cache_config()->ToJsonString()));
    auto migration = std::make_shared<MigrationStrategy>();
    migration->set_source_storage_name("nfs_01");
    migration->set_target_storage_name(kMigrationOnlyStorage);
    migration->set_trigger_threshold(0.5);
    MigrationMethods methods;
    methods.mutable_mark().set_enabled(true);
    migration->set_methods(methods);
    migration->set_retention(MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE);
    cache_config->set_migration_strategies({migration});

    InstanceGroup updated_group(*current_group);
    updated_group.set_cache_config(cache_config);
    updated_group.set_version(current_group->version() + 1);
    ASSERT_EQ(EC_OK,
              registry_manager_->UpdateInstanceGroup(&request_context_, updated_group, current_group->version()));

    auto [register_ec, storage_configs] =
        manager_->RegisterInstance(&request_context_, "default", "second-embedding-instance", "emb-test");
    ASSERT_EQ(EC_OK, register_ec);
    std::vector<std::shared_ptr<StorageConfig>> parsed_configs;
    ASSERT_TRUE(Jsonizable::FromJsonString(storage_configs, parsed_configs));
    ASSERT_EQ(1, parsed_configs.size());
    ASSERT_TRUE(parsed_configs.front());
    EXPECT_EQ("nfs_01", parsed_configs.front()->global_unique_name());
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, parsed_configs.front()->type());

    // The idempotent existing-instance path must apply the same filtering;
    // otherwise a client restart could fail even though initial registration
    // succeeded with the exact same group configuration.
    auto [reregister_ec, reregistered_storage_configs] =
        manager_->RegisterInstance(&request_context_, "default", "second-embedding-instance", "emb-test");
    ASSERT_EQ(EC_OK, reregister_ec);
    parsed_configs.clear();
    ASSERT_TRUE(Jsonizable::FromJsonString(reregistered_storage_configs, parsed_configs));
    ASSERT_EQ(1, parsed_configs.size());
    ASSERT_TRUE(parsed_configs.front());
    EXPECT_EQ("nfs_01", parsed_configs.front()->global_unique_name());
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, parsed_configs.front()->type());
}

TEST_F(KvMetaManagerTest, RegistrationRejectsUnboundedProvisionalControlPlaneTimeouts) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto provisional = std::make_shared<ProvisionalCommitNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, provisional->Open(original->GetStorageConfig(), request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = provisional;
    }

    for (const std::int64_t timeout :
         {std::int64_t{0}, kKvMetaMaxExactControlRpcTimeoutSeconds + 1}) {
        provisional->control_timeout_seconds_ = timeout;
        EXPECT_EQ(EC_CONFIG_ERROR,
                  manager_->RegisterInstance(
                      &request_context_, "default", "unsafe-provisional-timeout", "").first);
    }
    provisional->control_timeout_seconds_ = kKvMetaMaxExactControlRpcTimeoutSeconds;
    EXPECT_EQ(EC_OK,
              manager_->RegisterInstance(
                  &request_context_, "default", "bounded-provisional-timeout", "").first);

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaManagerTest, RegistrationRejectsDuplicateStorageCandidates) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    ASSERT_FALSE(default_group->storage_candidates().empty());
    InstanceGroup duplicate_group(*default_group);
    duplicate_group.set_name("duplicate-storage-object-group");
    duplicate_group.set_global_quota_group_name("duplicate-storage-object-quota");
    duplicate_group.set_storage_candidates(
        {default_group->storage_candidates().front(), default_group->storage_candidates().front()});
    duplicate_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, duplicate_group));
    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_->RegisterInstance(&request_context_, duplicate_group.name(), "duplicate-storage-object-instance", "")
            .first);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsUriUnsafeStorageCandidateNames) {
    constexpr const char *kStorage = "unsafe:kvmeta-storage";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    ASSERT_EQ(EC_OK, storage_manager->RegisterStorage(&request_context_, kStorage, original->GetStorageConfig()));

    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup unsafe_group(*default_group);
    unsafe_group.set_name("unsafe-storage-object-group");
    unsafe_group.set_global_quota_group_name("unsafe-storage-object-quota");
    unsafe_group.set_storage_candidates({kStorage});
    unsafe_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, unsafe_group));

    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_->RegisterInstance(&request_context_, unsafe_group.name(), "unsafe-storage-object-instance", "").first);
    EXPECT_EQ(EC_OK, storage_manager->UnRegisterStorage(kStorage));
}

TEST_F(KvMetaManagerTest, RegistrationRejectsStorageWhoseConfiguredIdentityDiffersFromItsRegistryName) {
    constexpr const char *kStorageAlias = "kvmeta-storage-alias";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    ASSERT_NE(kStorageAlias, original->GetStorageConfig().global_unique_name());
    ASSERT_EQ(EC_OK, storage_manager->RegisterStorage(&request_context_, kStorageAlias, original->GetStorageConfig()));

    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup aliased_group(*default_group);
    aliased_group.set_name("aliased-storage-object-group");
    aliased_group.set_global_quota_group_name("aliased-storage-object-quota");
    aliased_group.set_storage_candidates({kStorageAlias});
    aliased_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, aliased_group));

    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->RegisterInstance(&request_context_, aliased_group.name(), "aliased-storage-object-instance", "")
                  .first);
    EXPECT_EQ(EC_OK, storage_manager->UnRegisterStorage(kStorageAlias));
}

TEST_F(KvMetaManagerTest, RegistrationRejectsAnUnsafeExactObjectNamespace) {
    constexpr const char *kStorage = "unsafe-kvmeta-root";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto spec = std::make_shared<NfsStorageSpec>();
    // NfsBackend concatenates root_path and object key. Without the trailing
    // separator this would allocate `/tmp/rootkvmeta/...`, which cannot be
    // proven to belong to the reserved `/kvmeta/` namespace afterward.
    spec->set_root_path("/tmp/unsafe-kvmeta-root");
    spec->set_key_count_per_file(1);
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, kStorage, spec);
    ASSERT_EQ(EC_OK, storage_manager->RegisterStorage(&request_context_, kStorage, config));

    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup unsafe_group(*default_group);
    unsafe_group.set_name("unsafe-root-object-group");
    unsafe_group.set_global_quota_group_name("unsafe-root-object-quota");
    unsafe_group.set_storage_candidates({kStorage});
    unsafe_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, unsafe_group));

    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_->RegisterInstance(&request_context_, unsafe_group.name(), "unsafe-root-object-instance", "").first);
    EXPECT_EQ(EC_OK, storage_manager->UnRegisterStorage(kStorage));
}

TEST_F(KvMetaManagerTest, RegistrationUsesTheConfiguredUriLimitForTheWorstCaseObjectName) {
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto backend = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(backend);
    ASSERT_TRUE(HasSafeConfiguredKvMetaNamespace(backend->GetStorageConfig()));
    ASSERT_FALSE(HasSafeConfiguredKvMetaNamespace(backend->GetStorageConfig(), 64));
    ASSERT_FALSE(HasSafeConfiguredKvMetaNamespace(backend->GetStorageConfig(), kMaxKvMetaLocationUriBytes + 1));

    manager_->Shutdown();
    KvMetaManager::Limits limits;
    limits.max_location_uri_bytes = 64;
    manager_ = std::make_unique<KvMetaManager>(cache_manager_, registry_manager_, limits);
    ASSERT_TRUE(manager_->Init());

    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->RegisterInstance(&request_context_, "default", "uri-limit-object-instance", "").first);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsAStorageConfigWithTheWrongTypedSpec) {
    constexpr const char *kStorage = "wrong-typed-mooncake";
    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);

    auto nfs_spec = std::make_shared<NfsStorageSpec>();
    nfs_spec->set_root_path("/tmp/wrong-typed-mooncake/");
    nfs_spec->set_key_count_per_file(1);
    const StorageConfig malformed_config(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, kStorage, nfs_spec);
    auto malformed_backend = std::make_shared<MalformedMooncakeBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, malformed_backend->Open(malformed_config, request_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_[kStorage] = malformed_backend;
    }

    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    InstanceGroup malformed_group(*default_group);
    malformed_group.set_name("wrong-typed-storage-object-group");
    malformed_group.set_global_quota_group_name("wrong-typed-storage-object-quota");
    malformed_group.set_storage_candidates({kStorage});
    malformed_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, malformed_group));

    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_->RegisterInstance(&request_context_, malformed_group.name(), "wrong-typed-storage-object-instance", "")
            .first);
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_.erase(kStorage);
    }
    EXPECT_EQ(EC_OK, malformed_backend->Close());
}

TEST_F(KvMetaManagerTest, RegistrationRejectsMissingStorageCandidates) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);

    const auto create_and_register = [&](const std::string &group_name,
                                         const std::string &instance_id,
                                         std::vector<std::string> storage_candidates) {
        InstanceGroup group(*default_group);
        group.set_name(group_name);
        group.set_global_quota_group_name(group_name + "-quota");
        group.set_storage_candidates(std::move(storage_candidates));
        group.set_version(1);
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, group));
        EXPECT_EQ(EC_CONFIG_ERROR,
                  manager_->RegisterInstance(&request_context_, group_name, instance_id, "reclaim-test").first);
    };

    create_and_register("empty-storage-object-group", "empty-storage-object-instance", {});
    create_and_register(
        "missing-storage-object-group", "missing-storage-object-instance", {"unregistered-kvmeta-storage"});
}

TEST_F(KvMetaManagerTest, RegistrationRejectsMetadataBackendWithoutReadHeatTracking) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    ASSERT_TRUE(default_group->cache_config());
    ASSERT_TRUE(default_group->cache_config()->meta_indexer_config());

    auto cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(cache_config->FromJsonString(default_group->cache_config()->ToJsonString()));
    auto indexer_config = std::make_shared<MetaIndexerConfig>(*cache_config->meta_indexer_config());
    auto redis_config = std::make_shared<MetaStorageBackendConfig>(META_REDIS_BACKEND_TYPE_STR);
    redis_config->SetStorageUri("redis://127.0.0.1:6379");
    indexer_config->SetMetaStorageBackendConfig(redis_config);
    cache_config->set_meta_indexer_config(indexer_config);

    InstanceGroup redis_group(*default_group);
    redis_group.set_name("redis-metadata-object-group");
    redis_group.set_global_quota_group_name("redis-metadata-object-quota");
    redis_group.set_cache_config(cache_config);
    redis_group.set_version(1);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, redis_group));

    // Direct Redis metadata does not update PROPERTY_LRU_TIME on Get. KVMeta
    // must reject it instead of advertising POLICY_LRU while evicting by
    // sampled/tie order. This guard is specific to KVMeta registration.
    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_->RegisterInstance(&request_context_, redis_group.name(), "redis-metadata-object-instance", "").first);
}

TEST_F(KvMetaManagerTest, RegistrationRejectsInvalidCachedMetadataHotLayer) {
    const auto [group_ec, default_group] = registry_manager_->GetInstanceGroup(&request_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(default_group);
    ASSERT_TRUE(default_group->cache_config());
    ASSERT_TRUE(default_group->cache_config()->meta_indexer_config());

    const auto make_group = [&](const std::string &name, const std::string &storage_uri) {
        auto cache_config = std::make_shared<CacheConfig>();
        EXPECT_TRUE(cache_config->FromJsonString(default_group->cache_config()->ToJsonString()));
        auto indexer_config = std::make_shared<MetaIndexerConfig>(*cache_config->meta_indexer_config());
        auto cached_config = std::make_shared<MetaStorageBackendConfig>(META_CACHED_BACKEND_TYPE_STR);
        cached_config->SetStorageUri(storage_uri);
        indexer_config->SetMetaStorageBackendConfig(cached_config);
        cache_config->set_meta_indexer_config(indexer_config);

        InstanceGroup group(*default_group);
        group.set_name(name);
        group.set_global_quota_group_name(name + "-quota");
        group.set_cache_config(cache_config);
        group.set_version(1);
        return group;
    };

    // Redis requires a real endpoint even when cached mode would otherwise
    // default to redis/local. An empty URI must not be certified as durable or
    // as an operational read-heat source.
    auto empty_uri_group = make_group("cached-empty-uri-object-group", "");
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, empty_uri_group));
    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_->RegisterInstance(&request_context_, empty_uri_group.name(), "cached-empty-uri-object-instance", "")
            .first);

    // MetaStorageBackendFactory supports only a local hot layer. Validate the
    // nested mode now so a registry hot update cannot leave KVMeta admitting
    // writes under a configuration the active indexer could not reopen.
    auto invalid_hot_group =
        make_group("cached-invalid-hot-object-group", "redis://127.0.0.1:6379?persistent_type=redis&cache_type=redis");
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context_, invalid_hot_group));
    EXPECT_EQ(
        EC_CONFIG_ERROR,
        manager_
            ->RegisterInstance(&request_context_, invalid_hot_group.name(), "cached-invalid-hot-object-instance", "")
            .first);
}

TEST_F(KvMetaManagerTest, ZeroReclaimerTuningBlocksOnlyNewAllocation) {
    CommitObject(kInstanceId, "existing", 17);
    const auto cache_reclaimer = cache_manager_->cache_reclaimer();
    ASSERT_NE(nullptr, cache_reclaimer);
    const auto original_sampling = cache_reclaimer->GetSamplingSize(&request_context_);
    const auto original_batching = cache_reclaimer->GetBatchingSize(&request_context_);

    ASSERT_EQ(EC_OK, cache_reclaimer->SetSamplingSize(&request_context_, 0));
    auto [hit_ec, hit] = manager_->StartWrite(&request_context_, kInstanceId, {"existing"}, {17}, 30);
    ASSERT_EQ(EC_OK, hit_ec);
    EXPECT_EQ((std::vector<bool>{true}), hit.key_mask);
    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->StartWrite(&request_context_, kInstanceId, {"sampling-disabled"}, {17}, 30).first);
    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->RegisterInstance(&request_context_, "default", "sampling-disabled-instance", "").first);

    ASSERT_EQ(EC_OK, cache_reclaimer->SetSamplingSize(&request_context_, original_sampling));
    ASSERT_EQ(EC_OK, cache_reclaimer->SetBatchingSize(&request_context_, 0));
    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->StartWrite(&request_context_, kInstanceId, {"batching-disabled"}, {17}, 30).first);

    // A broken GC knob closes admission, but never closes the operator's
    // cleanup path for objects already present in the cache.
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstanceId, {"existing"}));
    ASSERT_EQ(EC_OK, cache_reclaimer->SetBatchingSize(&request_context_, original_batching));
}

TEST_F(KvMetaManagerTest, InvalidHotUpdatedReclaimPolicyBlocksOnlyNewAllocation) {
    constexpr const char *kGroup = "reclaim-hot-update-group";
    constexpr const char *kInstance = "reclaim-hot-update-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, "existing", 17);

    const auto [group_ec, current_group] = registry_manager_->GetInstanceGroup(&request_context_, kGroup);
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(current_group);
    ASSERT_TRUE(current_group->cache_config());
    ASSERT_TRUE(current_group->cache_config()->reclaim_strategy());
    auto updated_cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(updated_cache_config->FromJsonString(current_group->cache_config()->ToJsonString()));
    auto unsupported_strategy = std::make_shared<CacheReclaimStrategy>(*updated_cache_config->reclaim_strategy());
    unsupported_strategy->set_reclaim_policy(ReclaimPolicy::POLICY_TTL);
    updated_cache_config->set_reclaim_strategy(unsupported_strategy);
    InstanceGroup updated_group(*current_group);
    updated_group.set_cache_config(updated_cache_config);
    updated_group.set_version(current_group->version() + 1);
    ASSERT_EQ(EC_OK,
              registry_manager_->UpdateInstanceGroup(&request_context_, updated_group, current_group->version()));

    auto [hit_ec, hit] = manager_->StartWrite(&request_context_, kInstance, {"existing"}, {17}, 30);
    ASSERT_EQ(EC_OK, hit_ec);
    EXPECT_EQ((std::vector<bool>{true}), hit.key_mask);
    EXPECT_TRUE(hit.locations.empty());
    EXPECT_TRUE(hit.write_session_id.empty());

    auto [miss_ec, miss] = manager_->StartWrite(&request_context_, kInstance, {"new"}, {17}, 30);
    EXPECT_EQ(EC_CONFIG_ERROR, miss_ec);
    EXPECT_TRUE(miss.locations.empty());
    EXPECT_TRUE(miss.write_session_id.empty());
    EXPECT_EQ(EC_CONFIG_ERROR,
              manager_->RegisterInstance(&request_context_, kGroup, "another-instance", "reclaim-test").first);

    // Configuration failure closes admission, not cleanup: operators must
    // still be able to drain already committed cache objects safely.
    EXPECT_EQ(EC_OK, manager_->Remove(&request_context_, kInstance, {"existing"}));
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"existing"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values.front().found);
}

TEST_F(KvMetaManagerTest, HotUpdatedMetadataWithoutReadHeatStopsAdmissionAndAutomaticReclaim) {
    constexpr const char *kGroup = "reclaim-hot-metadata-group";
    constexpr const char *kInstance = "reclaim-hot-metadata-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, "existing", 90);

    const auto [group_ec, current_group] = registry_manager_->GetInstanceGroup(&request_context_, kGroup);
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(current_group);
    ASSERT_TRUE(current_group->cache_config());
    ASSERT_TRUE(current_group->cache_config()->meta_indexer_config());
    auto updated_cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(updated_cache_config->FromJsonString(current_group->cache_config()->ToJsonString()));
    auto updated_indexer_config = std::make_shared<MetaIndexerConfig>(*updated_cache_config->meta_indexer_config());
    auto redis_config = std::make_shared<MetaStorageBackendConfig>(META_REDIS_BACKEND_TYPE_STR);
    redis_config->SetStorageUri("redis://127.0.0.1:6379");
    updated_indexer_config->SetMetaStorageBackendConfig(redis_config);
    updated_cache_config->set_meta_indexer_config(updated_indexer_config);
    InstanceGroup updated_group(*current_group);
    updated_group.set_cache_config(updated_cache_config);
    updated_group.set_version(current_group->version() + 1);
    ASSERT_EQ(EC_OK,
              registry_manager_->UpdateInstanceGroup(&request_context_, updated_group, current_group->version()));

    auto [hit_ec, hit] = manager_->StartWrite(&request_context_, kInstance, {"existing"}, {90}, 30);
    ASSERT_EQ(EC_OK, hit_ec);
    EXPECT_EQ((std::vector<bool>{true}), hit.key_mask);
    EXPECT_EQ(EC_CONFIG_ERROR, manager_->StartWrite(&request_context_, kInstance, {"new-object"}, {10}, 30).first);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const double rounds_before = metrics_registry_->GetCounter("kv_meta_reclaimer.round_count").Get();
    const double retired_before = metrics_registry_->GetCounter("kv_meta_reclaimer.retired_object_count").Get();
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(
        [&]() { return metrics_registry_->GetCounter("kv_meta_reclaimer.round_count").Get() > rounds_before; },
        std::chrono::seconds(2)));

    // The hot update changed only registry configuration; the already-open
    // local indexer would still be capable of deleting the object. Verify the
    // Reclaimer consults the current group contract and refuses to pretend
    // that direct Redis provides LRU read heat.
    EXPECT_EQ(90, indexer->GetStorageUsage());
    EXPECT_EQ(retired_before, metrics_registry_->GetCounter("kv_meta_reclaimer.retired_object_count").Get());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {"existing"});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_TRUE(values.front().found);
}

TEST_F(KvMetaManagerTest, RecoveryCompletesAReclaimerRetirementLeftByDemotion) {
    constexpr const char *kGroup = "reclaim-recovery-group";
    constexpr const char *kInstance = "reclaim-recovery-instance";
    constexpr const char *kKey = "retired-before-demotion";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 250);
    CommitObject(kInstance, kKey, 90);

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    const auto internal_key = KvMetaManager::InternalKey(kKey);
    const auto location_id = KvMetaManager::StableLocationId(kKey);
    const auto is_retired = [&]() {
        CacheLocationMapVector maps;
        const auto result = indexer->GetLocationMapsForMaintenance(&request_context_, {internal_key}, maps);
        if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || maps.size() != 1) {
            return false;
        }
        const auto it = maps[0].find(location_id);
        return it != maps[0].end() && it->second && it->second->status() == CLS_DELETING;
    };

    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    ASSERT_TRUE(WaitUntil(is_retired, std::chrono::seconds(2)));
    manager_->CancelMaintenance();
    manager_->DoCleanup();

    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ(0, indexer->GetStorageUsage());
    auto [get_ec, values] = manager_->Get(&request_context_, kInstance, {kKey});
    ASSERT_EQ(EC_OK, get_ec);
    ASSERT_EQ(1, values.size());
    EXPECT_FALSE(values[0].found);
    ASSERT_TRUE(manager_->ResumeMaintenance());
}

TEST_F(KvMetaManagerTest, RecoveryRebuildsTheBoundedKvMetaReclaimerGroupSet) {
    constexpr const char *kGroup = "reclaim-recovered-group";
    constexpr const char *kInstance = "reclaim-recovered-instance";
    CreateReclaimGroup(kGroup, kInstance, 100, 0.8, 0);
    CommitObject(kInstance, "recovered-object", 90);

    // Model a fresh process: the runtime-only discovery set is empty, while
    // the reserved KVMeta instance and its committed metadata are durable.
    manager_->ReplaceKvMetaGroups({});
    EXPECT_TRUE(manager_->SnapshotKvMetaGroups().empty());

    ASSERT_EQ(EC_OK, manager_->DoRecover());
    EXPECT_EQ((std::vector<std::string>{"default", kGroup}), manager_->SnapshotKvMetaGroups());

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(KvMetaManager::InternalInstanceId(kInstance));
    ASSERT_TRUE(indexer);
    ASSERT_EQ(90, indexer->GetStorageUsage());
    cache_manager_->cache_reclaimer()->SetSleepIntervalMs(&request_context_, 5);
    ASSERT_TRUE(manager_->ResumeMaintenance());
    EXPECT_TRUE(WaitUntil([&]() { return indexer->GetStorageUsage() == 0; }, std::chrono::seconds(2)));
}

TEST(KvMetaInstanceMarkerTest, RequiresTheCompleteReservedSchema) {
    ModelDeployment deployment;
    deployment.set_model_name(std::string(kKvMetaModelName));
    deployment.set_dtype(std::string(kKvMetaDtype));
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);
    deployment.set_extra(std::string(kKvMetaDeploymentExtra));
    InstanceInfo instance("quota",
                          "objects",
                          std::string(kKvMetaInternalInstancePrefix) + "6964",
                          1,
                          {LocationSpecInfo(std::string(kKvMetaValueSpecName), 1)},
                          deployment,
                          {},
                          1);
    EXPECT_TRUE(IsKvMetaInstance(instance));

    instance.set_block_size(2);
    EXPECT_FALSE(IsKvMetaInstance(instance));
    instance.set_block_size(1);
    instance.set_instance_id(std::string(kKvMetaInternalInstancePrefix) + "not-hex");
    EXPECT_FALSE(IsKvMetaInstance(instance));
}

} // namespace kv_cache_manager
