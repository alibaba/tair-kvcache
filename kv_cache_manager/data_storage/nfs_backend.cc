#include "nfs_backend.h"

#include <cerrno>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

void PruneEmptyKvMetaObjectParents(const std::filesystem::path &object_path) noexcept {
    // A canonical KVMeta key is kvmeta/<instance-hash>/<key-hash>/<nonce>.
    // Reclaim the two per-instance/per-key directories after the leaf is
    // absent, but deliberately retain the reserved kvmeta namespace root.
    // Concurrent generations make remove report directory_not_empty; that is
    // an expected stop condition and the last generation will retry pruning.
    try {
        std::filesystem::path candidate = object_path.parent_path();
        for (int level = 0; level < 2; ++level) {
            std::error_code ec;
            const bool removed = std::filesystem::remove(candidate, ec);
            if (!removed) {
                if (ec && ec != std::errc::directory_not_empty && ec != std::errc::no_such_file_or_directory) {
                    KVCM_INTERVAL_LOG_WARN(10,
                                           "KVMeta NFS could not prune empty object directory, path: [%s], msg: [%s]",
                                           candidate.string().c_str(),
                                           ec.message().c_str());
                }
                break;
            }
            candidate = candidate.parent_path();
        }
    } catch (const std::exception &e) {
        // The exact object is already confirmed absent. Parent pruning is a
        // best-effort inode optimization and must never turn that terminal
        // state into an uncertain delete/retry loop.
        KVCM_INTERVAL_LOG_WARN(10, "KVMeta NFS parent pruning caught exception: [%s]", e.what());
    } catch (...) { KVCM_INTERVAL_LOG_WARN(10, "KVMeta NFS parent pruning caught unknown exception"); }
}

} // namespace

NfsBackend::NfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

DataStorageType NfsBackend::GetType() { return DataStorageType::DATA_STORAGE_TYPE_NFS; }

bool NfsBackend::Available() { return IsOpen() && IsAvailable(); }

double NfsBackend::GetStorageUsageRatio(const std::string &trace_id) const { return 0.0; }

ErrorCode NfsBackend::DoOpen(const StorageConfig &storage_config, const std::string &trace_id) {
    if (auto cfg = std::dynamic_pointer_cast<NfsStorageSpec>(storage_config.storage_spec())) {
        spec_ = *cfg;
    } else {
        KVCM_LOG_WARN("unexpected config type, storage config: [%s]", storage_config.ToString().c_str());
        return EC_ERROR;
    }
    KVCM_LOG_INFO("open nfs backend success, config: [%s]", spec_.ToString().c_str());
    SetOpen(true);
    SetAvailable(true);
    return EC_OK;
};

ErrorCode NfsBackend::Close() {
    KVCM_LOG_INFO("close nfs backend");
    SetOpen(false);
    SetAvailable(false);
    return EC_OK;
};

std::vector<std::pair<ErrorCode, DataStorageUri>> NfsBackend::Create(const std::vector<std::string> &keys,
                                                                     size_t size_per_key,
                                                                     const std::string &trace_id,
                                                                     std::function<void()> cb) {
    std::vector<std::pair<ErrorCode, DataStorageUri>> result;
    std::vector<std::vector<std::string>> batches;
    int32_t batch_size = spec_.key_count_per_file();
    batch_size = batch_size <= 0 ? 1 : batch_size;
    size_t total_key_count = keys.size();
    for (size_t start = 0; start < total_key_count; start += batch_size) {
        size_t end = std::min(start + batch_size, total_key_count);
        batches.emplace_back(keys.begin() + start, keys.begin() + end);
    }
    for (auto &batch : batches) {
        DataStorageUri storage_uri;
        storage_uri.SetProtocol(ToString(GetType()));
        if (batch.size() > 1) {
            std::string combine_key = StringUtil::Join(batch, "|");
            std::string hash_str = StringUtil::Uint64ToHex(Hash64(combine_key.c_str(), combine_key.size(), 42));
            storage_uri.SetPath(spec_.root_path() + batch[0] + "_" + hash_str);
        } else {
            storage_uri.SetPath(spec_.root_path() + batch[0]);
        }
        storage_uri.SetParam("size", std::to_string(size_per_key));
        for (size_t j = 0; j < batch.size(); ++j) {
            if (batch_size > 1) {
                storage_uri.SetParam("blkid", std::to_string(j));
            }
            result.push_back({EC_OK, storage_uri});
        }
    }
    if (cb) {
        cb();
    }
    return result;
}

std::vector<ErrorCode> NfsBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                          const std::string &trace_id,
                                          std::function<void()> cb) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}

std::vector<ErrorCode> NfsBackend::DeleteAndConfirmAbsent(const std::vector<DataStorageUri> &storage_uris,
                                                          const std::string &trace_id,
                                                          std::function<void()> cb) {
    std::vector<ErrorCode> results(storage_uris.size(), EC_CORRUPTION);
    std::vector<bool> valid(storage_uris.size(), false);
    const StorageConfig &config = GetStorageConfig();
    for (std::size_t i = 0; i < storage_uris.size(); ++i) {
        const auto &uri = storage_uris[i];
        valid[i] = HasCanonicalKvMetaAuthority(uri) &&
                   ToBaseType(ToDataStorageType(uri.GetProtocol())) == DataStorageType::DATA_STORAGE_TYPE_NFS &&
                   HasOwnedKvMetaAllocationShape(uri, DataStorageType::DATA_STORAGE_TYPE_NFS) &&
                   UriMatchesConfiguredKvMetaNamespace(uri, DataStorageType::DATA_STORAGE_TYPE_NFS, config);
    }

    // Preserve fault injection and any future legacy NFS bookkeeping by
    // dispatching through the virtual method. A successful legacy response is
    // only an attempt acknowledgement; exact KVMeta deletion still performs
    // and verifies the filesystem operation below.
    std::vector<DataStorageUri> valid_uris;
    std::vector<std::size_t> valid_indices;
    valid_uris.reserve(storage_uris.size());
    valid_indices.reserve(storage_uris.size());
    for (std::size_t i = 0; i < storage_uris.size(); ++i) {
        if (valid[i]) {
            valid_uris.push_back(storage_uris[i]);
            valid_indices.push_back(i);
        }
    }
    const auto attempted = valid_uris.empty() ? std::vector<ErrorCode>{} : Delete(valid_uris, trace_id, nullptr);
    if (attempted.size() == valid_uris.size()) {
        for (std::size_t i = 0; i < valid_uris.size(); ++i) {
            const std::size_t result_index = valid_indices[i];
            if (attempted[i] != EC_OK && attempted[i] != EC_NOENT) {
                results[result_index] = attempted[i];
                continue;
            }
            const std::string &object_path = valid_uris[i].GetPath();
            struct stat object_stat;
            if (::lstat(object_path.c_str(), &object_stat) == 0) {
                // An exact NFS allocation is a singleton file. Never let
                // generic filesystem removal reinterpret a corrupt empty
                // directory as a valid generation and recursively erode the
                // namespace shape.
                if (S_ISDIR(object_stat.st_mode)) {
                    KVCM_LOG_ERROR("KVMeta NFS exact delete rejected a directory, path: [%s]", object_path.c_str());
                    results[result_index] = EC_IO_ERROR;
                    continue;
                }
                if (::unlink(object_path.c_str()) != 0 && errno != ENOENT) {
                    KVCM_LOG_ERROR("KVMeta NFS exact unlink failed, path: [%s], msg: [%s]",
                                   object_path.c_str(),
                                   std::strerror(errno));
                    results[result_index] = EC_IO_ERROR;
                    continue;
                }
            } else if (errno != ENOENT) {
                KVCM_LOG_ERROR(
                    "KVMeta NFS exact lstat failed, path: [%s], msg: [%s]", object_path.c_str(), std::strerror(errno));
                results[result_index] = EC_IO_ERROR;
                continue;
            }
            // lstat(ENOENT) proves only current namespace visibility. Persist
            // the unlink (or an already-absent retry) before allowing KVCM to
            // erase the tombstone; otherwise a host/server crash could leave
            // an untracked orphan after metadata finalization.
            if (!SyncKvMetaDeleteDirectory(object_path)) {
                results[result_index] = EC_IO_ERROR;
                continue;
            }
            errno = 0;
            const int confirm_result = ::lstat(object_path.c_str(), &object_stat);
            const int confirm_errno = errno;
            if (confirm_result == 0 || confirm_errno != ENOENT) {
                KVCM_LOG_ERROR("KVMeta NFS exact delete could not confirm absence, path: [%s], msg: [%s]",
                               object_path.c_str(),
                               confirm_result == 0 ? "path still exists" : std::strerror(confirm_errno));
                results[result_index] = EC_IO_ERROR;
                continue;
            }
            results[result_index] = EC_OK;
            PruneEmptyKvMetaObjectParents(object_path);
        }
    } else {
        for (const std::size_t index : valid_indices) {
            results[index] = EC_MISMATCH;
        }
    }
    if (cb) {
        cb();
    }
    return results;
}

bool NfsBackend::SyncKvMetaDeleteDirectory(const std::string &object_path) const noexcept {
    try {
        // Walk key-hash, instance-hash, kvmeta, then the configured root. The
        // root fallback is required when a write session obtained its URI but
        // failed before the client ever materialized kvmeta/ or the object.
        std::filesystem::path directory = std::filesystem::path(object_path).parent_path();
        for (int level = 0; level < 4; ++level) {
            const std::string directory_string = directory.string();
            errno = 0;
            const int fd = ::open(directory_string.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
            if (fd >= 0) {
                errno = 0;
                const int sync_result = ::fsync(fd);
                const int sync_errno = errno;
                errno = 0;
                const int close_result = ::close(fd);
                const int close_errno = errno;
                if (sync_result == 0 && close_result == 0) {
                    return true;
                }
                KVCM_LOG_ERROR("KVMeta NFS could not persist exact-delete directory, path: [%s], msg: [%s]",
                               directory_string.c_str(),
                               std::strerror(sync_result != 0 ? sync_errno : close_errno));
                return false;
            }
            const int open_errno = errno;
            if (open_errno != ENOENT) {
                KVCM_LOG_ERROR("KVMeta NFS could not open exact-delete directory, path: [%s], msg: [%s]",
                               directory_string.c_str(),
                               std::strerror(open_errno));
                return false;
            }
            directory = directory.parent_path();
        }
        KVCM_LOG_ERROR("KVMeta NFS exact-delete namespace disappeared before its durability barrier, object: [%s]",
                       object_path.c_str());
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("KVMeta NFS exact-delete directory sync caught exception: [%s]", e.what());
    } catch (...) { KVCM_LOG_ERROR("KVMeta NFS exact-delete directory sync caught unknown exception"); }
    return false;
}

std::vector<bool> NfsBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<bool> result(storage_uris.size(), true);
    // not supported yet
    return result;
}
std::vector<ErrorCode> NfsBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}
std::vector<ErrorCode> NfsBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}

} // namespace kv_cache_manager
