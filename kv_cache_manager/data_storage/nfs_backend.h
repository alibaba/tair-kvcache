#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

class MetricsRegistry;

class NfsBackend : public DataStorageBackend, public KvMetaDataStorageBackendExtension {
public:
    NfsBackend() = delete;
    explicit NfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~NfsBackend() override;
    DataStorageType GetType() override;
    bool Available() override;
    double GetStorageUsageRatio(const std::string &trace_id) const override;

public:
    ErrorCode DoOpen(const StorageConfig &storage_config, const std::string &trace_id) override;
    ErrorCode Close() override;

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override;
    // The exact-object side path establishes a durable, pre-mounted namespace
    // anchor before returning any allocation URI. Legacy fixed-block Create
    // deliberately keeps its historical metadata-only behavior.
    bool HasDedicatedKvMetaCreate() const noexcept override { return true; }
    CreatePreflightResult PreflightKvMetaCreate(const std::vector<CreatePreflightItem> &items) override;
    std::vector<std::pair<ErrorCode, DataStorageUri>> CreateForKvMeta(const std::vector<std::string> &keys,
                                                                      std::size_t size_per_key,
                                                                      const std::string &trace_id,
                                                                      std::function<void()> cb) override;
    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override;
    // Keep the legacy Delete behavior unchanged for fixed-block callers. The
    // KVMeta side interface performs and verifies exact singleton-file removal.
    std::vector<ErrorCode> DeleteAndConfirmAbsent(const std::vector<DataStorageUri> &storage_uris,
                                                  const std::string &trace_id,
                                                  std::function<void()> cb) override;
    std::int64_t GetFailedWriteCleanupGraceSeconds() const noexcept override { return 0; }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override;

protected:
    // Require and persist the operator-provisioned namespace root before an
    // exact reservation is admitted. This must never create a missing mount
    // root. The lower-level persistence operation is virtual only for
    // deterministic fault injection.
    bool SyncKvMetaAdmissionRoot() const noexcept;
    bool ProbeKvMetaAdmissionRoot(std::uint64_t &device, std::uint64_t &inode) const noexcept;
    bool OpenKvMetaAdmissionRoot(int &fd, std::uint64_t &device, std::uint64_t &inode) const noexcept;
    virtual bool PersistKvMetaAdmissionRoot() const noexcept;
    // Exact-only physical capacity probe. It runs after the durable root has
    // been anchored and before any URI is returned, so EC_NOSPC proves that
    // the failed request allocated no generation. Virtual only for fault
    // injection; fixed-block NFS Create never enters this path.
    virtual CreatePreflightResult
    CheckKvMetaAdmissionCapacity(const std::vector<CreatePreflightItem> &items) const noexcept;
    // Pure capacity policy shared by the real fstatvfs path and focused unit
    // tests. `inode_capacity_known=false` models filesystems that report
    // f_files=0 and therefore have no meaningful inode budget.
    static CreatePreflightResult ComputeKvMetaCapacityPressure(const std::vector<std::uint64_t> &value_sizes,
                                                               std::uint64_t missing_namespace_directories,
                                                               std::uint64_t block_size,
                                                               std::uint64_t available_blocks,
                                                               bool inode_capacity_known,
                                                               std::uint64_t available_inodes) noexcept;
    // Read-only descriptor-relative namespace accounting. The caller owns a
    // stable directory descriptor for the duration of the call; duplicate
    // shared ancestors across a batch are counted once.
    static bool CountMissingKvMetaNamespaceDirectories(int anchored_root_fd,
                                                       const std::vector<CreatePreflightItem> &items,
                                                       std::uint64_t &missing_directories) noexcept;
    // Persist the namespace mutation before KVMeta is allowed to erase its
    // durable cleanup ledger. Virtual only for deterministic fault injection.
    virtual bool SyncKvMetaDeleteDirectory(const std::string &object_path) const noexcept;

private:
    // Delete one exact KVMeta generation relative to the durable namespace
    // root descriptor. Every child component is opened with O_NOFOLLOW, so a
    // corrupted or replaced intermediate directory can never redirect GC
    // outside the configured cache namespace. The helper also fsyncs the
    // owning directory before it reports authoritative absence.
    bool DeleteExactKvMetaObject(const std::string &object_path) const noexcept;
    void PruneExactKvMetaObjectParents(const std::string &object_path) const noexcept;

    NfsStorageSpec spec_;
    // Root durability is a backend-level deployment invariant, not an
    // object-level mutation. Establish it lazily on the first exact admission
    // so a batch of singleton EMB allocations does not fsync the same NFS
    // directory once per object.
    mutable std::mutex kv_meta_admission_root_mutex_;
    mutable std::atomic<bool> kv_meta_admission_root_synced_{false};
    mutable std::atomic<std::uint64_t> kv_meta_admission_root_device_{0};
    mutable std::atomic<std::uint64_t> kv_meta_admission_root_inode_{0};
    // Keep the accepted directory open for the backend lifetime. Besides
    // providing a stable comparison target, the open reference prevents a
    // removed directory's inode from being recycled and defeating a dev/inode
    // replacement check on the next admission.
    mutable int kv_meta_admission_root_fd_{-1};
};

} // namespace kv_cache_manager
