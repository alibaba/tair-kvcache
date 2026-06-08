#pragma once

#include <libnuraft/launcher.hxx>
#include <libnuraft/raft_server.hxx>

#include <atomic>
#include <climits>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/raft/lmdb_log_store.h"
#include "kv_cache_manager/meta/raft/meta_state_machine.h"
#include "kv_cache_manager/meta/raft/meta_state_mgr.h"

namespace kv_cache_manager {
namespace raft_meta {

// RaftCoordinator owns the *process-wide* raft cluster: log store, state
// machine, state manager, launcher, and the live raft_server. One kvcm
// process is one raft node — every Instance hosted in this process replicates
// through the same raft group, with Instance isolation enforced by the
// state machine routing each LogOp by op.instance_id to a per-instance
// MetaLocalBackend.
//
// Lifecycle (typical, when storage_backend_type == "raft"):
//   1. Server::Init builds a RaftCoordinator::Config from ServerConfig.raft
//      and calls Start() — this is the *first* moment raft is alive in the
//      process. cb_func (leader change callback) MUST be installed via
//      init_options.raft_callback_ BEFORE the launcher fires; we set it in
//      Start(), so SetLeadershipCallback() must be called before Start().
//   2. Server publishes the coordinator as the process singleton via
//      SetInstance() so MetaRaftBackend instances created later (one per
//      Instance) can call GetInstance() during their Init() and become thin
//      proxies that route writes through this coordinator.
//   3. RaftLeaderElector (PR 5-rev) also pulls the singleton, registers a
//      leadership callback, and exposes IsLeader / LeaderId via the
//      ILeaderElector interface.
//   4. Server::Stop calls Stop(), which shuts the launcher down.
//
// Thread safety: GetOrCreateBackend / AppendAndWait / IsLeader / LeaderId
// are safe to call concurrently. Start / Stop / SetInstance are not — they
// are expected to be called once during process bring-up / tear-down.
class RaftCoordinator {
public:
    struct PeerSpec {
        int32_t server_id = 0;
        std::string endpoint;
        std::string aux;
        bool is_learner = false;
    };

    struct Config {
        // Raft identity.
        int32_t server_id = 0;
        std::string self_endpoint; // host:port the raft RPC layer binds.
        int port = 0;              // Numeric port part of self_endpoint; what asio listens on.
        std::vector<PeerSpec> peers;
        std::string self_aux; // Goes into our srv_config.aux (NodeEndpointInfo JSON).

        // Persistent paths under data_dir: <data_dir>/state, <data_dir>/snapshot.
        std::string data_dir;

        // Raft tuning.
        int snapshot_distance = 100000;
        int election_timeout_lower = 150;
        int election_timeout_upper = 300;
        int heart_beat_interval = 20;

        // Default tunables passed to per-instance MetaLocalBackend created by
        // the state machine factory. Each Instance gets its own backend; the
        // coordinator holds the defaults.
        //
        // The state machine's MetaLocalBackend is the authoritative store (no
        // Redis behind it), so capacity must be large enough that
        // strict_capacity_limit never rejects a committed write. We default
        // to SIZE_MAX / (1024*1024) so the LRU cache is effectively unbounded.
        // TODO: replace MetaLocalBackend with a disk-backed store (e.g. LevelDB)
        //       so state machine memory is bounded by disk rather than RAM.
        size_t local_capacity = SIZE_MAX / (1024ULL * 1024);
        int32_t local_num_shard_bits = 10;
        int32_t local_sample_times = 10;
    };

    using LeadershipCallback = std::function<void(bool is_leader)>;

    RaftCoordinator();
    ~RaftCoordinator();

    RaftCoordinator(const RaftCoordinator &) = delete;
    RaftCoordinator &operator=(const RaftCoordinator &) = delete;

    // Brings the raft node up. Returns EC_OK on success. Fails fast (and
    // returns EC_BADARGS) on malformed config, or EC_ERROR if the launcher
    // could not bind the port / build the server.
    ErrorCode Start(const Config &config);

    // Tears the raft node down. Idempotent.
    void Stop();

    // True after Start() succeeded and before Stop() ran.
    bool IsRunning() const { return running_.load(); }

    // Append one LogOp-encoded buffer and block until commit (or rejection).
    // Returns EC_OK on commit, EC_BADARGS if this node is not the leader,
    // EC_ERROR on raft transport / timeout failures.
    ErrorCode AppendAndWait(nuraft::ptr<nuraft::buffer> data);

    // Append a kNoOp entry and block until it commits. Guarantees that all
    // previously appended log entries have been applied to the state machine.
    // Must be called from the leader after election to drain the log before
    // reading state machine state.
    ErrorCode Barrier();

    // Per-instance backend access. The state machine creates these on demand
    // via the factory closure installed in Start(); GetOrCreateBackend either
    // returns an existing one or asks the state machine to materialise it.
    std::shared_ptr<MetaCacheBaseBackend> GetOrCreateBackend(const std::string &instance_id);

    // Registry data read. Delegates to the state machine's RegistryLoad.
    ErrorCode RegistryLoad(const std::string &key, std::map<std::string, std::string> &out) const;

    // Raft introspection.
    bool IsLeader() const;
    int32_t LeaderId() const;
    int32_t SelfId() const { return config_.server_id; }
    const Config &GetConfig() const { return config_; }
    std::string GetPeerAux(int32_t server_id) const;

    // Leadership change subscriber. The callback fires from the NuRaft
    // raft_callback_ context — keep it short and non-blocking. Must be set
    // BEFORE Start() because NuRaft's init_options.raft_callback_ is captured
    // at launcher init time. Calling this after Start() is a no-op.
    void SetLeadershipCallback(LeadershipCallback cb);

    // Registry commit callback passthrough. Must be called AFTER Start()
    // (state machine exists only after BuildInner). Fires on every registry
    // commit on all nodes (leader and followers).
    void SetRegistryCommitCallback(MetaStateMachine::RegistryCommitCallback cb);

    // Test hook: lets unit tests drive the state machine without bringing the
    // raft launcher up. Returns the same state machine that Start() will
    // hand to the launcher.
    nuraft::ptr<MetaStateMachine> StateMachineForTest() const { return state_machine_; }

    // Process singleton. MetaRaftBackend::Init pulls the coordinator via
    // GetInstance() so it can register itself as a thin proxy without the
    // factory layer needing to know about raft. The ServerConfig wiring sets
    // the instance during Server::Init; tests that don't need a singleton can
    // just construct a coordinator on the stack.
    static void SetInstance(RaftCoordinator *coord);
    static RaftCoordinator *GetInstance();

private:
    // Translate Config into the state-mgr / state-machine inputs and build
    // the inner objects. Called from Start() before the launcher fires.
    ErrorCode BuildInner();

    // Backend factory closure handed to MetaStateMachine. Materialises a
    // MetaLocalBackend per Instance using the coordinator's default tunables.
    std::shared_ptr<MetaCacheBaseBackend> MakeBackend(const std::string &instance_id);

    Config config_;
    std::atomic<bool> running_{false};

    // Owned raft objects.
    nuraft::ptr<LmdbLogStore> log_store_;
    nuraft::ptr<MetaStateMachine> state_machine_;
    nuraft::ptr<MetaStateMgr> state_mgr_;
    std::unique_ptr<nuraft::raft_launcher> launcher_;
    nuraft::ptr<nuraft::raft_server> raft_server_;

    // Leadership callback set before Start.
    std::mutex cb_mu_;
    LeadershipCallback leadership_cb_;
};

} // namespace raft_meta
} // namespace kv_cache_manager
