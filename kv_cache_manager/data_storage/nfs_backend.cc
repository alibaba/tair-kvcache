#include "nfs_backend.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <utility>

#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

class ScopedFdArray {
public:
    ~ScopedFdArray() {
        for (int &fd : fds_) {
            Close(fd);
        }
    }

    int &operator[](std::size_t index) noexcept { return fds_[index]; }
    int operator[](std::size_t index) const noexcept { return fds_[index]; }

    static void Close(int &fd) noexcept {
        if (fd >= 0) {
            (void)::close(fd);
            fd = -1;
        }
    }

private:
    std::array<int, 4> fds_{{-1, -1, -1, -1}};
};

} // namespace

NfsBackend::NfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

NfsBackend::~NfsBackend() {
    std::lock_guard<std::mutex> lock(kv_meta_admission_root_mutex_);
    if (kv_meta_admission_root_fd_ >= 0) {
        (void)::close(kv_meta_admission_root_fd_);
        kv_meta_admission_root_fd_ = -1;
    }
}

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
    {
        std::lock_guard<std::mutex> lock(kv_meta_admission_root_mutex_);
        kv_meta_admission_root_synced_.store(false, std::memory_order_release);
        kv_meta_admission_root_device_.store(0, std::memory_order_relaxed);
        kv_meta_admission_root_inode_.store(0, std::memory_order_relaxed);
        if (kv_meta_admission_root_fd_ >= 0) {
            (void)::close(kv_meta_admission_root_fd_);
            kv_meta_admission_root_fd_ = -1;
        }
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
    {
        std::lock_guard<std::mutex> lock(kv_meta_admission_root_mutex_);
        kv_meta_admission_root_synced_.store(false, std::memory_order_release);
        kv_meta_admission_root_device_.store(0, std::memory_order_relaxed);
        kv_meta_admission_root_inode_.store(0, std::memory_order_relaxed);
        if (kv_meta_admission_root_fd_ >= 0) {
            (void)::close(kv_meta_admission_root_fd_);
            kv_meta_admission_root_fd_ = -1;
        }
    }
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

std::vector<std::pair<ErrorCode, DataStorageUri>> NfsBackend::CreateForKvMeta(const std::vector<std::string> &keys,
                                                                              std::size_t size_per_key,
                                                                              const std::string &trace_id,
                                                                              std::function<void()> cb) {
    if (keys.empty()) {
        return {};
    }
    if (keys.size() != 1 || size_per_key == 0 || !HasCanonicalKvMetaNfsRootPath(spec_.root_path()) ||
        std::any_of(keys.begin(), keys.end(), [](const auto &key) { return !HasCanonicalKvMetaObjectKey(key); })) {
        KVCM_LOG_ERROR("KVMeta NFS exact create requires one canonical object, non-zero size, and a canonical "
                       "absolute configured root");
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size(), {EC_BADARGS, DataStorageUri{}});
    }
    const CreatePreflightResult capacity =
        PreflightKvMetaCreate({CreatePreflightItem{keys.front(), static_cast<std::uint64_t>(size_per_key)}});
    if (capacity.ec != EC_OK) {
        // No object or namespace child has been created yet. EC_NOSPC is
        // therefore authoritative backend-pressure evidence and can safely
        // trigger exact-backend cache reclamation; probe failures remain
        // control-plane I/O errors and must not evict useful entries.
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size(), {capacity.ec, DataStorageUri{}});
    }
    return Create(keys, size_per_key, trace_id, std::move(cb));
}

KvMetaDataStorageBackendExtension::CreatePreflightResult
NfsBackend::PreflightKvMetaCreate(const std::vector<CreatePreflightItem> &items) {
    if (items.empty() || std::any_of(items.begin(), items.end(), [](const CreatePreflightItem &item) {
            return item.value_size == 0 || !HasCanonicalKvMetaObjectKey(item.allocation_key);
        })) {
        return {EC_BADARGS, 0, 0};
    }
    if (!SyncKvMetaAdmissionRoot()) {
        // Returning no URI proves that this failed admission did not allocate
        // a generation. The manager can reject it without creating a cleanup
        // tombstone or charging quota for an object that never became usable.
        return {EC_IO_ERROR, 0, 0};
    }
    return CheckKvMetaAdmissionCapacity(items);
}

KvMetaDataStorageBackendExtension::CreatePreflightResult
NfsBackend::CheckKvMetaAdmissionCapacity(const std::vector<CreatePreflightItem> &items) const noexcept {
    if (items.empty() || std::any_of(items.begin(), items.end(), [](const CreatePreflightItem &item) {
            return item.value_size == 0 || !HasCanonicalKvMetaObjectKey(item.allocation_key);
        })) {
        return {EC_BADARGS, 0, 0};
    }
    try {
        std::lock_guard<std::mutex> lock(kv_meta_admission_root_mutex_);
        if (kv_meta_admission_root_fd_ < 0 || !kv_meta_admission_root_synced_.load(std::memory_order_acquire)) {
            return {EC_IO_ERROR, 0, 0};
        }
        struct statvfs capacity{};
        errno = 0;
        if (::fstatvfs(kv_meta_admission_root_fd_, &capacity) != 0) {
            KVCM_LOG_ERROR("KVMeta NFS capacity probe failed, root: [%s], msg: [%s]",
                           spec_.root_path().c_str(),
                           std::strerror(errno));
            return {EC_IO_ERROR, 0, 0};
        }
        std::uint64_t missing_namespace_directories = 0;
        if (!CountMissingKvMetaNamespaceDirectories(kv_meta_admission_root_fd_, items, missing_namespace_directories)) {
            return {EC_IO_ERROR, 0, 0};
        }
        std::vector<std::uint64_t> value_sizes;
        value_sizes.reserve(items.size());
        std::uint64_t requested_bytes = 0;
        for (const CreatePreflightItem &item : items) {
            const std::uint64_t value_size = item.value_size;
            value_sizes.push_back(value_size);
            requested_bytes = value_size > std::numeric_limits<std::uint64_t>::max() - requested_bytes
                                  ? std::numeric_limits<std::uint64_t>::max()
                                  : requested_bytes + value_size;
        }
        const std::uint64_t block_size = capacity.f_frsize != 0 ? capacity.f_frsize : capacity.f_bsize;
        const std::uint64_t available_blocks = capacity.f_bavail;
        const std::uint64_t available_bytes =
            block_size != 0 && available_blocks > std::numeric_limits<std::uint64_t>::max() / block_size
                ? std::numeric_limits<std::uint64_t>::max()
                : available_blocks * block_size;
        const auto result = ComputeKvMetaCapacityPressure(value_sizes,
                                                          missing_namespace_directories,
                                                          block_size,
                                                          available_blocks,
                                                          capacity.f_files != 0,
                                                          capacity.f_favail);
        if (result.ec == EC_NOSPC) {
            KVCM_INTERVAL_LOG_WARN(10,
                                   "KVMeta NFS admission rejected by physical capacity, root: [%s], "
                                   "requested_bytes: [%lu], available_bytes: [%lu], reclaim_bytes: [%lu], "
                                   "requested_objects: [%zu], missing_namespace_directories: [%lu], "
                                   "available_inodes: [%lu], reclaim_objects: [%lu]",
                                   spec_.root_path().c_str(),
                                   requested_bytes,
                                   available_bytes,
                                   result.reclaim_bytes,
                                   value_sizes.size(),
                                   missing_namespace_directories,
                                   capacity.f_files == 0 ? std::numeric_limits<std::uint64_t>::max()
                                                         : static_cast<std::uint64_t>(capacity.f_favail),
                                   result.reclaim_objects);
        }
        return result;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("KVMeta NFS capacity probe caught exception: [%s]", e.what());
    } catch (...) {
        KVCM_LOG_ERROR("KVMeta NFS capacity probe caught unknown exception");
    }
    return {EC_IO_ERROR, 0, 0};
}

KvMetaDataStorageBackendExtension::CreatePreflightResult
NfsBackend::ComputeKvMetaCapacityPressure(const std::vector<std::uint64_t> &value_sizes,
                                          std::uint64_t missing_namespace_directories,
                                          std::uint64_t block_size,
                                          std::uint64_t available_blocks,
                                          bool inode_capacity_known,
                                          std::uint64_t available_inodes) noexcept {
    if (value_sizes.empty() ||
        std::any_of(value_sizes.begin(), value_sizes.end(), [](std::uint64_t size) { return size == 0; })) {
        return {EC_BADARGS, 0, 0};
    }
    if (block_size == 0) {
        // A filesystem that reports neither f_frsize nor f_bsize did not give
        // authoritative capacity evidence. Treat it as a probe failure, never
        // as pressure that is allowed to evict healthy cache entries.
        return {EC_IO_ERROR, 0, 0};
    }
    const std::uint64_t available_bytes =
        block_size != 0 && available_blocks > std::numeric_limits<std::uint64_t>::max() / block_size
            ? std::numeric_limits<std::uint64_t>::max()
            : available_blocks * block_size;

    // fallocate consumes filesystem allocation units, not just logical value
    // bytes. Preserve the complete variable-size distribution and sum
    // ceil(size / block_size) * block_size with checked arithmetic; total
    // bytes plus object count cannot distinguish one nearly-full block from
    // several tiny independently rounded files.
    std::uint64_t requested_bytes = 0;
    std::uint64_t required_physical_bytes = 0;
    for (const std::uint64_t value_size : value_sizes) {
        if (value_size > std::numeric_limits<std::uint64_t>::max() - requested_bytes) {
            return {EC_OUT_OF_LIMIT, 0, 0};
        }
        requested_bytes += value_size;
        std::uint64_t physical_size = value_size;
        const std::uint64_t remainder = value_size % block_size;
        const std::uint64_t padding = remainder == 0 ? 0 : block_size - remainder;
        if (padding > std::numeric_limits<std::uint64_t>::max() - value_size) {
            return {EC_OUT_OF_LIMIT, 0, 0};
        }
        physical_size += padding;
        if (physical_size > std::numeric_limits<std::uint64_t>::max() - required_physical_bytes) {
            return {EC_OUT_OF_LIMIT, 0, 0};
        }
        required_physical_bytes += physical_size;
    }
    if (missing_namespace_directories > std::numeric_limits<std::uint64_t>::max() / block_size) {
        return {EC_OUT_OF_LIMIT, 0, 0};
    }
    const std::uint64_t namespace_physical_bytes = missing_namespace_directories * block_size;
    if (namespace_physical_bytes > std::numeric_limits<std::uint64_t>::max() - required_physical_bytes) {
        return {EC_OUT_OF_LIMIT, 0, 0};
    }
    required_physical_bytes += namespace_physical_bytes;
    const std::uint64_t physical_byte_shortage =
        required_physical_bytes > available_bytes ? required_physical_bytes - available_bytes : 0;
    // One bounded retry target is expressed in logical cache bytes. The next
    // authoritative preflight observes real free blocks again, while a tiny
    // value cannot turn one allocation-unit deficit into arbitrary eviction.
    const std::uint64_t reclaim_bytes = std::min(requested_bytes, physical_byte_shortage);

    // Every generation needs one file inode. Add only the namespace
    // directories that the anchored, descriptor-relative probe proved absent;
    // shared existing directories must not turn a large batch into a false
    // inode-pressure signal.
    const std::uint64_t requested_objects = value_sizes.size();
    if (missing_namespace_directories > std::numeric_limits<std::uint64_t>::max() - requested_objects) {
        return {EC_OUT_OF_LIMIT, 0, 0};
    }
    const std::uint64_t required_inodes = requested_objects + missing_namespace_directories;
    const std::uint64_t effective_available_inodes =
        inode_capacity_known ? available_inodes : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t inode_shortage =
        required_inodes > effective_available_inodes ? required_inodes - effective_available_inodes : 0;
    const std::uint64_t reclaim_objects = std::min(requested_objects, inode_shortage);
    return reclaim_bytes != 0 || reclaim_objects != 0 ? CreatePreflightResult{EC_NOSPC, reclaim_bytes, reclaim_objects}
                                                      : CreatePreflightResult{};
}

bool NfsBackend::CountMissingKvMetaNamespaceDirectories(int anchored_root_fd,
                                                        const std::vector<CreatePreflightItem> &items,
                                                        std::uint64_t &missing_directories) noexcept {
    missing_directories = 0;
    try {
        if (anchored_root_fd < 0) {
            return false;
        }
        std::set<std::string> missing_paths;
        for (const CreatePreflightItem &item : items) {
            if (!HasCanonicalKvMetaObjectKey(item.allocation_key)) {
                return false;
            }
            std::array<std::string_view, 4> components;
            std::size_t component_begin = 0;
            for (std::size_t i = 0; i < components.size(); ++i) {
                const std::size_t component_end = item.allocation_key.find('/', component_begin);
                if ((i + 1 < components.size() && component_end == std::string::npos) ||
                    (i + 1 == components.size() && component_end != std::string::npos)) {
                    return false;
                }
                const std::size_t end = component_end == std::string::npos ? item.allocation_key.size() : component_end;
                components[i] = std::string_view(item.allocation_key).substr(component_begin, end - component_begin);
                component_begin = end + 1;
            }

            ScopedFdArray descriptors;
            errno = 0;
            descriptors[0] = ::fcntl(anchored_root_fd, F_DUPFD_CLOEXEC, 0);
            if (descriptors[0] < 0) {
                KVCM_LOG_ERROR("KVMeta NFS capacity probe could not duplicate its anchored root, msg: [%s]",
                               std::strerror(errno));
                return false;
            }
            std::string relative_path;
            for (std::size_t i = 0; i < 3; ++i) {
                if (!relative_path.empty()) {
                    relative_path.push_back('/');
                }
                relative_path.append(components[i]);
                const std::string component(components[i]);
                errno = 0;
                descriptors[i + 1] =
                    ::openat(descriptors[i], component.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
                if (descriptors[i + 1] >= 0) {
                    continue;
                }
                const int open_errno = errno;
                if (open_errno != ENOENT) {
                    KVCM_LOG_ERROR("KVMeta NFS capacity probe rejected a non-directory or symlinked namespace "
                                   "component, key: [%s], component: [%s], msg: [%s]",
                                   item.allocation_key.c_str(),
                                   component.c_str(),
                                   std::strerror(open_errno));
                    return false;
                }
                missing_paths.insert(relative_path);
                for (std::size_t missing_index = i + 1; missing_index < 3; ++missing_index) {
                    relative_path.push_back('/');
                    relative_path.append(components[missing_index]);
                    missing_paths.insert(relative_path);
                }
                break;
            }
        }
        missing_directories = static_cast<std::uint64_t>(missing_paths.size());
        return true;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("KVMeta NFS namespace capacity probe caught exception: [%s]", e.what());
    } catch (...) {
        KVCM_LOG_ERROR("KVMeta NFS namespace capacity probe caught unknown exception");
    }
    return false;
}

bool NfsBackend::SyncKvMetaAdmissionRoot() const noexcept {
    try {
        // Exact-object admission is isolated from the historical fixed-block
        // Create path. Serialize the small control-plane probe so Close/Open
        // cannot invalidate the durable directory anchor while it is checked.
        std::lock_guard<std::mutex> lock(kv_meta_admission_root_mutex_);
        std::uint64_t observed_device = 0;
        std::uint64_t observed_inode = 0;
        if (!ProbeKvMetaAdmissionRoot(observed_device, observed_inode)) {
            return false;
        }
        if (kv_meta_admission_root_synced_.load(std::memory_order_acquire)) {
            struct stat anchored_stat{};
            const bool anchor_valid = kv_meta_admission_root_fd_ >= 0 &&
                                      ::fstat(kv_meta_admission_root_fd_, &anchored_stat) == 0 &&
                                      S_ISDIR(anchored_stat.st_mode);
            const bool matches = anchor_valid && observed_device == static_cast<std::uint64_t>(anchored_stat.st_dev) &&
                                 observed_inode == static_cast<std::uint64_t>(anchored_stat.st_ino) &&
                                 observed_device == kv_meta_admission_root_device_.load(std::memory_order_relaxed) &&
                                 observed_inode == kv_meta_admission_root_inode_.load(std::memory_order_relaxed);
            if (!matches) {
                KVCM_LOG_ERROR("KVMeta NFS admission root identity changed; reopen the backend after restoring the "
                               "configured mount");
            }
            return matches;
        }
        if (!PersistKvMetaAdmissionRoot()) {
            return false;
        }
        int durable_fd = -1;
        std::uint64_t durable_device = 0;
        std::uint64_t durable_inode = 0;
        if (!OpenKvMetaAdmissionRoot(durable_fd, durable_device, durable_inode) || durable_device != observed_device ||
            durable_inode != observed_inode) {
            if (durable_fd >= 0) {
                (void)::close(durable_fd);
            }
            KVCM_LOG_ERROR("KVMeta NFS admission root identity changed while establishing durability");
            return false;
        }
        // Close the remaining rename/unmount race between opening the anchor
        // and publishing it. An open anchor keeps its inode alive, so a path
        // replacement cannot masquerade as the same generation afterwards.
        std::uint64_t confirmed_device = 0;
        std::uint64_t confirmed_inode = 0;
        if (!ProbeKvMetaAdmissionRoot(confirmed_device, confirmed_inode) || confirmed_device != durable_device ||
            confirmed_inode != durable_inode) {
            (void)::close(durable_fd);
            KVCM_LOG_ERROR("KVMeta NFS admission root changed before its durability anchor was published");
            return false;
        }
        kv_meta_admission_root_fd_ = durable_fd;
        kv_meta_admission_root_device_.store(durable_device, std::memory_order_relaxed);
        kv_meta_admission_root_inode_.store(durable_inode, std::memory_order_relaxed);
        kv_meta_admission_root_synced_.store(true, std::memory_order_release);
        return true;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("KVMeta NFS admission root state caught exception: [%s]", e.what());
    } catch (...) {
        KVCM_LOG_ERROR("KVMeta NFS admission root state caught unknown exception");
    }
    return false;
}

bool NfsBackend::ProbeKvMetaAdmissionRoot(std::uint64_t &device, std::uint64_t &inode) const noexcept {
    int fd = -1;
    if (!OpenKvMetaAdmissionRoot(fd, device, inode)) {
        return false;
    }
    errno = 0;
    const int close_result = ::close(fd);
    if (close_result != 0) {
        KVCM_LOG_ERROR("KVMeta NFS admission root probe close failed, msg: [%s]", std::strerror(errno));
        device = 0;
        inode = 0;
        return false;
    }
    return true;
}

bool NfsBackend::OpenKvMetaAdmissionRoot(int &fd, std::uint64_t &device, std::uint64_t &inode) const noexcept {
    fd = -1;
    device = 0;
    inode = 0;
    try {
        std::string root_path = spec_.root_path();
        while (root_path.size() > 1 && root_path.back() == '/') {
            root_path.pop_back();
        }
        errno = 0;
        fd = ::open(root_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) {
            KVCM_LOG_ERROR("KVMeta NFS admission root is unavailable, path: [%s], msg: [%s]",
                           root_path.c_str(),
                           std::strerror(errno));
            return false;
        }
        struct stat root_stat{};
        errno = 0;
        const int stat_result = ::fstat(fd, &root_stat);
        const int stat_errno = errno;
        errno = 0;
        if (stat_result != 0 || !S_ISDIR(root_stat.st_mode)) {
            KVCM_LOG_ERROR("KVMeta NFS admission root probe failed, path: [%s], msg: [%s]",
                           root_path.c_str(),
                           stat_result != 0 ? std::strerror(stat_errno) : "path is not a directory");
            (void)::close(fd);
            fd = -1;
            return false;
        }
        device = static_cast<std::uint64_t>(root_stat.st_dev);
        inode = static_cast<std::uint64_t>(root_stat.st_ino);
        return true;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("KVMeta NFS admission root probe caught exception: [%s]", e.what());
    } catch (...) {
        KVCM_LOG_ERROR("KVMeta NFS admission root probe caught unknown exception");
    }
    if (fd >= 0) {
        (void)::close(fd);
        fd = -1;
    }
    return false;
}

bool NfsBackend::PersistKvMetaAdmissionRoot() const noexcept {
    try {
        std::string root_path = spec_.root_path();
        // Remove trailing separators so O_NOFOLLOW applies to the configured
        // root itself rather than following a final symlink through `/`.
        while (root_path.size() > 1 && root_path.back() == '/') {
            root_path.pop_back();
        }
        errno = 0;
        const int fd = ::open(root_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) {
            KVCM_LOG_ERROR("KVMeta NFS admission root is unavailable, path: [%s], msg: [%s]",
                           root_path.c_str(),
                           std::strerror(errno));
            return false;
        }
        errno = 0;
        const int sync_result = ::fsync(fd);
        const int sync_errno = errno;
        errno = 0;
        const int close_result = ::close(fd);
        const int close_errno = errno;
        if (sync_result != 0 || close_result != 0) {
            KVCM_LOG_ERROR("KVMeta NFS admission root is not durable, path: [%s], msg: [%s]",
                           root_path.c_str(),
                           std::strerror(sync_result != 0 ? sync_errno : close_errno));
            return false;
        }
        return true;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("KVMeta NFS admission root sync caught exception: [%s]", e.what());
    } catch (...) {
        KVCM_LOG_ERROR("KVMeta NFS admission root sync caught unknown exception");
    }
    return false;
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
            if (!DeleteExactKvMetaObject(object_path)) {
                results[result_index] = EC_IO_ERROR;
                continue;
            }
            // Keep the virtual barrier as a deterministic fault-injection
            // seam. DeleteExactKvMetaObject has already persisted the exact
            // openat/unlinkat parent descriptor; this second barrier preserves
            // existing backend hooks without weakening path containment.
            if (!SyncKvMetaDeleteDirectory(object_path)) {
                results[result_index] = EC_IO_ERROR;
                continue;
            }
            results[result_index] = EC_OK;
            PruneExactKvMetaObjectParents(object_path);
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

bool NfsBackend::DeleteExactKvMetaObject(const std::string &object_path) const noexcept {
    try {
        std::string_view object_key;
        if (!TryGetCanonicalKvMetaObjectKeyFromPath(object_path, object_key) || spec_.root_path().empty() ||
            spec_.root_path().back() != '/' || object_path != spec_.root_path() + std::string(object_key)) {
            KVCM_LOG_ERROR("KVMeta NFS exact delete received a path outside its canonical namespace: [%s]",
                           object_path.c_str());
            return false;
        }
        // Recovery can delete before this process has admitted a new object,
        // so establish (or verify) the same durable root anchor lazily here.
        if (!SyncKvMetaAdmissionRoot()) {
            return false;
        }

        ScopedFdArray descriptors;
        {
            std::lock_guard<std::mutex> lock(kv_meta_admission_root_mutex_);
            if (kv_meta_admission_root_fd_ < 0 ||
                (descriptors[0] = ::fcntl(kv_meta_admission_root_fd_, F_DUPFD_CLOEXEC, 0)) < 0) {
                KVCM_LOG_ERROR("KVMeta NFS could not duplicate its anchored root, path: [%s], msg: [%s]",
                               spec_.root_path().c_str(),
                               std::strerror(errno));
                return false;
            }
        }

        std::array<std::string_view, 4> components;
        std::size_t component_begin = 0;
        for (std::size_t i = 0; i < components.size(); ++i) {
            const std::size_t component_end = object_key.find('/', component_begin);
            if ((i + 1 < components.size() && component_end == std::string_view::npos) ||
                (i + 1 == components.size() && component_end != std::string_view::npos)) {
                return false;
            }
            const std::size_t end = component_end == std::string_view::npos ? object_key.size() : component_end;
            components[i] = object_key.substr(component_begin, end - component_begin);
            component_begin = end + 1;
        }

        // Hold every ancestor descriptor. Besides preventing symlink
        // traversal, this gives pruning exact parent/name pairs after unlink.
        for (std::size_t i = 0; i < 3; ++i) {
            const std::string component(components[i]);
            errno = 0;
            descriptors[i + 1] =
                ::openat(descriptors[i], component.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
            if (descriptors[i + 1] >= 0) {
                continue;
            }
            const int open_errno = errno;
            if (open_errno == ENOENT) {
                // Missing ancestor proves the leaf absent in this anchored
                // namespace. Persist that namespace observation before the
                // cleanup ledger is removed.
                if (::fsync(descriptors[i]) == 0) {
                    return true;
                }
                KVCM_LOG_ERROR("KVMeta NFS could not persist an absent exact-object ancestor, path: [%s], msg: [%s]",
                               object_path.c_str(),
                               std::strerror(errno));
                return false;
            }
            KVCM_LOG_ERROR("KVMeta NFS exact delete rejected a missing, non-directory, or symlinked ancestor, "
                           "path: [%s], component: [%s], msg: [%s]",
                           object_path.c_str(),
                           component.c_str(),
                           std::strerror(open_errno));
            return false;
        }

        const std::string leaf(components[3]);
        struct stat object_stat{};
        errno = 0;
        const int stat_result = ::fstatat(descriptors[3], leaf.c_str(), &object_stat, AT_SYMLINK_NOFOLLOW);
        if (stat_result == 0) {
            // Only a regular file can be a materialized NFS EMB generation.
            // Reject directories and symlinks instead of converting namespace
            // corruption into a successful cache eviction.
            if (!S_ISREG(object_stat.st_mode)) {
                KVCM_LOG_ERROR("KVMeta NFS exact delete rejected a non-regular generation, path: [%s]",
                               object_path.c_str());
                return false;
            }
            errno = 0;
            if (::unlinkat(descriptors[3], leaf.c_str(), 0) != 0 && errno != ENOENT) {
                KVCM_LOG_ERROR("KVMeta NFS exact unlinkat failed, path: [%s], msg: [%s]",
                               object_path.c_str(),
                               std::strerror(errno));
                return false;
            }
        } else if (errno != ENOENT) {
            KVCM_LOG_ERROR(
                "KVMeta NFS exact fstatat failed, path: [%s], msg: [%s]", object_path.c_str(), std::strerror(errno));
            return false;
        }

        if (::fsync(descriptors[3]) != 0) {
            KVCM_LOG_ERROR("KVMeta NFS could not persist exact unlink, path: [%s], msg: [%s]",
                           object_path.c_str(),
                           std::strerror(errno));
            return false;
        }
        errno = 0;
        if (::fstatat(descriptors[3], leaf.c_str(), &object_stat, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT) {
            KVCM_LOG_ERROR("KVMeta NFS exact delete could not confirm anchored absence, path: [%s], msg: [%s]",
                           object_path.c_str(),
                           errno == 0 ? "path still exists" : std::strerror(errno));
            return false;
        }

        return true;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("KVMeta NFS anchored exact delete caught exception: [%s]", e.what());
    } catch (...) {
        KVCM_LOG_ERROR("KVMeta NFS anchored exact delete caught unknown exception");
    }
    return false;
}

void NfsBackend::PruneExactKvMetaObjectParents(const std::string &object_path) const noexcept {
    try {
        std::string_view object_key;
        if (!TryGetCanonicalKvMetaObjectKeyFromPath(object_path, object_key) || spec_.root_path().empty() ||
            object_path != spec_.root_path() + std::string(object_key) || !SyncKvMetaAdmissionRoot()) {
            return;
        }
        std::array<std::string_view, 4> components;
        std::size_t component_begin = 0;
        for (std::size_t i = 0; i < components.size(); ++i) {
            const std::size_t component_end = object_key.find('/', component_begin);
            const std::size_t end = component_end == std::string_view::npos ? object_key.size() : component_end;
            components[i] = object_key.substr(component_begin, end - component_begin);
            component_begin = end + 1;
        }

        ScopedFdArray descriptors;
        {
            std::lock_guard<std::mutex> lock(kv_meta_admission_root_mutex_);
            if (kv_meta_admission_root_fd_ < 0 ||
                (descriptors[0] = ::fcntl(kv_meta_admission_root_fd_, F_DUPFD_CLOEXEC, 0)) < 0) {
                return;
            }
        }
        for (std::size_t i = 0; i < 3; ++i) {
            const std::string component(components[i]);
            descriptors[i + 1] =
                ::openat(descriptors[i], component.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
            if (descriptors[i + 1] < 0) {
                return;
            }
        }

        // Best-effort, descriptor-relative pruning avoids an inode leak for
        // one-directory-per-key while never following a replacement path.
        // Concurrent generations legitimately leave directories non-empty.
        ScopedFdArray::Close(descriptors[3]);
        errno = 0;
        if (::unlinkat(descriptors[2], std::string(components[2]).c_str(), AT_REMOVEDIR) == 0) {
            (void)::fsync(descriptors[2]);
            ScopedFdArray::Close(descriptors[2]);
            errno = 0;
            if (::unlinkat(descriptors[1], std::string(components[1]).c_str(), AT_REMOVEDIR) == 0) {
                (void)::fsync(descriptors[1]);
            } else if (errno != ENOTEMPTY && errno != EEXIST && errno != ENOENT) {
                KVCM_INTERVAL_LOG_WARN(10,
                                       "KVMeta NFS could not prune empty instance directory, path: [%s], msg: [%s]",
                                       object_path.c_str(),
                                       std::strerror(errno));
            }
        } else if (errno != ENOTEMPTY && errno != EEXIST && errno != ENOENT) {
            KVCM_INTERVAL_LOG_WARN(10,
                                   "KVMeta NFS could not prune empty key directory, path: [%s], msg: [%s]",
                                   object_path.c_str(),
                                   std::strerror(errno));
        }
    } catch (const std::exception &e) {
        KVCM_INTERVAL_LOG_WARN(10, "KVMeta NFS anchored parent pruning caught exception: [%s]", e.what());
    } catch (...) {
        KVCM_INTERVAL_LOG_WARN(10, "KVMeta NFS anchored parent pruning caught unknown exception");
    }
}

bool NfsBackend::SyncKvMetaDeleteDirectory(const std::string &object_path) const noexcept {
    try {
        std::string_view object_key;
        if (!TryGetCanonicalKvMetaObjectKeyFromPath(object_path, object_key) || spec_.root_path().empty() ||
            spec_.root_path().back() != '/' || object_path != spec_.root_path() + std::string(object_key)) {
            return false;
        }
        // DeleteExactKvMetaObject already fsyncs the exact openat parent that
        // owns the leaf. Do not reopen an absolute child path here: O_NOFOLLOW
        // protects only the final component and would reintroduce intermediate
        // symlink traversal after the anchored deletion. Retain this virtual
        // method solely as a deterministic post-delete fault-injection seam;
        // the base implementation revalidates the durable root anchor.
        return SyncKvMetaAdmissionRoot();
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("KVMeta NFS exact-delete directory sync caught exception: [%s]", e.what());
    } catch (...) {
        KVCM_LOG_ERROR("KVMeta NFS exact-delete directory sync caught unknown exception");
    }
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
