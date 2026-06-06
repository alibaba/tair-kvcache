#pragma once

#include <libnuraft/cluster_config.hxx>
#include <libnuraft/snapshot.hxx>
#include <libnuraft/state_machine.hxx>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/meta_cache_base_backend.h"

namespace kv_cache_manager {
namespace raft_meta {

// MetaStateMachine routes every replicated log entry (decoded via
// raft_log_codec into a LogOp) to a per-instance MetaCacheBaseBackend. The
// state machine is shared by every Instance hosted in this kvcm process — one
// raft group per process, one in-memory backend per Instance, keys never leak
// across Instances (Instance isolation, see CLAUDE.md).
//
// Backends are created on demand: the first commit() that names a previously
// unseen instance_id calls the supplied factory closure to materialise its
// inner backend. This keeps the state machine decoupled from how Instances
// are configured or registered — the RaftCoordinator owns the closure.
//
// Reads served by MetaRaftBackend bypass raft and hit the same per-instance
// backend directly (linearisable on the leader since all writes flow here).
//
// Snapshots are a full dump of every per-instance backend, serialised into
// one file at <snapshot_dir>/snapshot_<last_log_idx>_<term>. Logical snapshot
// transport uses a single object (id 1 carries the bytes; id 0 is a
// sentinel). Phase 1 metadata fits comfortably in one chunk; chunking is
// future work alongside LMDB-backed log.
class MetaStateMachine : public nuraft::state_machine {
public:
    // Factory closure: maps an instance_id to a fully-initialised backend
    // for that Instance. Called under the state machine's internal lock; must
    // be reentrancy-safe enough that the caller can serialise creation by
    // Instance (the state machine memoises the result).
    using BackendFactory = std::function<std::shared_ptr<MetaCacheBaseBackend>(const std::string &)>;

    MetaStateMachine(BackendFactory factory, std::string snapshot_dir);
    ~MetaStateMachine() override;

    nuraft::ptr<nuraft::buffer> commit(const nuraft::ulong log_idx, nuraft::buffer &data) override;

    // Snapshot interfaces.
    void create_snapshot(nuraft::snapshot &s,
                         nuraft::async_result<bool>::handler_type &when_done) override;
    bool apply_snapshot(nuraft::snapshot &s) override;
    int read_logical_snp_obj(nuraft::snapshot &s,
                             void *&user_snp_ctx,
                             nuraft::ulong obj_id,
                             nuraft::ptr<nuraft::buffer> &data_out,
                             bool &is_last_obj) override;
    void save_logical_snp_obj(nuraft::snapshot &s,
                              nuraft::ulong &obj_id,
                              nuraft::buffer &data,
                              bool is_first_obj,
                              bool is_last_obj) override;
    void free_user_snp_ctx(void *&user_snp_ctx) override;
    nuraft::ptr<nuraft::snapshot> last_snapshot() override;
    nuraft::ulong last_commit_index() override;

    // Read access for one Instance — used by MetaRaftBackend to serve reads
    // without going through raft. Returns nullptr if the Instance has never
    // received a write (i.e. has no commit applied yet).
    std::shared_ptr<MetaCacheBaseBackend> GetBackend(const std::string &instance_id) const;

    // Lookup-or-create: same as GetBackend but materialises via the factory
    // closure on miss. MetaRaftBackend uses this when it must answer reads
    // for Instances that exist on the leader but haven't yet been observed
    // by this state machine (e.g. a follower that just caught up).
    std::shared_ptr<MetaCacheBaseBackend> GetOrCreateBackend(const std::string &instance_id);

    // Registry data access — used by RegistryRaftBackend.
    ErrorCode RegistryLoad(const std::string &key, std::map<std::string, std::string> &out) const;
    void RegistryClear();

    // Callback fired after every registry commit (kRegistrySave/kRegistryDelete)
    // on ALL nodes (leader and followers). Used to keep RegistryManager's
    // in-memory state in sync without requiring DoRecover on re-election.
    // is_save=true: key was saved with the given fields.
    // is_save=false: key was deleted (fields is empty).
    using RegistryCommitCallback =
        std::function<void(bool is_save, const std::string &key, const std::map<std::string, std::string> &fields)>;
    void SetRegistryCommitCallback(RegistryCommitCallback cb);

private:
    // Encode every per-instance backend into one buffer:
    //   u8  version
    //   u32 instance_count
    //   for each instance: str instance_id, then the per-instance dump
    //     (u32 key_count, [i64 key, locations, properties]*).
    // Format diverges from raft_log_codec on purpose so the two can evolve
    // independently.
    nuraft::ptr<nuraft::buffer> SerializeAll() const;
    bool DeserializeAll(nuraft::buffer &buf);

    std::string SnapshotPath(nuraft::ulong last_log_idx, nuraft::ulong term) const;
    static bool EnsureDir(const std::string &dir);

    BackendFactory backend_factory_;
    std::string snapshot_dir_;

    mutable std::shared_mutex backends_mu_;
    std::unordered_map<std::string, std::shared_ptr<MetaCacheBaseBackend>> backends_;

    mutable std::mutex snapshot_mutex_;
    nuraft::ptr<nuraft::snapshot> last_snapshot_;

    // Tracks the highest log index applied via commit(), exposed to raft via
    // last_commit_index() for catch-up bookkeeping.
    std::atomic<nuraft::ulong> last_commit_index_{0};

    // Registry data: replicated KV pairs for InstanceGroup / Instance /
    // Storage / Account configuration. Separate from per-instance meta
    // backends — this is global config, not per-instance cache metadata.
    mutable std::shared_mutex registry_mu_;
    std::unordered_map<std::string, std::map<std::string, std::string>> registry_store_;

    std::mutex registry_cb_mu_;
    RegistryCommitCallback registry_commit_cb_;
};

} // namespace raft_meta
} // namespace kv_cache_manager
