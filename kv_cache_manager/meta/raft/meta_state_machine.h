#pragma once

#include <libnuraft/cluster_config.hxx>
#include <libnuraft/snapshot.hxx>
#include <libnuraft/state_machine.hxx>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/meta_cache_base_backend.h"

namespace kv_cache_manager {
namespace raft_meta {

// MetaStateMachine wraps the in-memory MetaLocalBackend that holds the
// authoritative meta state for the Raft group. Every replicated log entry is
// decoded by raft_log_codec and dispatched to the backend's write API; reads
// served by MetaRaftBackend bypass raft and hit this same backend directly
// (linearisable on the leader by virtue of all writes flowing through here).
//
// Snapshots are full-state dumps of the MetaLocalBackend, serialised to a
// single file at <snapshot_dir>/snapshot_<last_log_idx>_<term>. Logical
// snapshot transport is implemented as a single object: object id 0 carries
// the complete snapshot bytes — simpler than chunking and adequate for the
// metadata sizes Phase 1 targets.
class MetaStateMachine : public nuraft::state_machine {
public:
    // `backend` must be already-initialised and Open()ed; this state machine
    // takes shared ownership and dispatches all writes to it.
    // `snapshot_dir` is the directory where snapshot files live; created if
    // missing.
    MetaStateMachine(std::shared_ptr<MetaCacheBaseBackend> backend, std::string snapshot_dir);
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

    // Direct access to the underlying meta backend — MetaRaftBackend uses it
    // to serve reads without going through raft.
    MetaCacheBaseBackend *backend() const { return backend_.get(); }

private:
    // Encode the entire MetaLocalBackend state to a single byte buffer using
    // the same serialisation primitives as raft_log_codec (length-prefixed
    // entries, JSON for CacheLocation). Separate format from log entries so
    // we can evolve them independently.
    nuraft::ptr<nuraft::buffer> SerializeBackend() const;
    // Reverse of SerializeBackend — clears the backend then reinserts.
    bool DeserializeIntoBackend(nuraft::buffer &buf);

    std::string SnapshotPath(nuraft::ulong last_log_idx, nuraft::ulong term) const;
    static bool EnsureDir(const std::string &dir);

    std::shared_ptr<MetaCacheBaseBackend> backend_;
    std::string snapshot_dir_;

    mutable std::mutex snapshot_mutex_;
    nuraft::ptr<nuraft::snapshot> last_snapshot_;

    // Tracks the highest log index applied via commit(), exposed to raft via
    // last_commit_index() for catch-up bookkeeping.
    std::atomic<nuraft::ulong> last_commit_index_{0};
};

} // namespace raft_meta
} // namespace kv_cache_manager
