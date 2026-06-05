#pragma once

#include <libnuraft/cluster_config.hxx>
#include <libnuraft/log_store.hxx>
#include <libnuraft/srv_config.hxx>
#include <libnuraft/srv_state.hxx>
#include <libnuraft/state_mgr.hxx>

#include <mutex>
#include <string>
#include <vector>

namespace kv_cache_manager {
namespace raft_meta {

// Description of one peer learned from KVCM config (id, endpoint, optional
// learner flag, opaque aux). `endpoint` is the host:port the raft RPC layer
// will bind/connect on — distinct from KVCM's gRPC service port. `aux` is
// stored verbatim into srv_config and gets replicated through cluster_config
// like everything else; we use it to carry a JSON-serialised
// NodeEndpointInfo (gRPC port, HTTP port, …) so any peer can answer
// "where's the leader's gRPC?" by reading the leader's srv_config.aux.
struct PeerEntry {
    int32_t server_id = 0;
    std::string endpoint;
    std::string aux;
    bool is_learner = false;
};

// File-backed implementation of nuraft::state_mgr. Persists srv_state and
// cluster_config to a small directory under <data_dir>/raft/state so the
// node can rejoin its term/vote and replicated config after restart.
//
// load_log_store returns the log store the caller wired in via the
// constructor — MetaRaftBackend owns the actual MetaLogStore instance.
class MetaStateMgr : public nuraft::state_mgr {
public:
    MetaStateMgr(int32_t server_id,
                 std::string self_endpoint,
                 std::string self_aux,
                 std::vector<PeerEntry> initial_peers,
                 std::string state_dir,
                 nuraft::ptr<nuraft::log_store> log_store);
    ~MetaStateMgr() override = default;

    nuraft::ptr<nuraft::cluster_config> load_config() override;
    void save_config(const nuraft::cluster_config &config) override;
    void save_state(const nuraft::srv_state &state) override;
    nuraft::ptr<nuraft::srv_state> read_state() override;
    nuraft::ptr<nuraft::log_store> load_log_store() override;
    nuraft::int32 server_id() override;
    void system_exit(const int exit_code) override;

private:
    // Build a fresh cluster_config from the constructor-supplied peer list.
    // Used when no on-disk config has been written yet.
    nuraft::ptr<nuraft::cluster_config> BuildDefaultConfig() const;

    // Atomic file rewrite: write to <path>.tmp then rename. Returns false on
    // any IO failure (logged at error level by the caller's KVCM logger).
    static bool AtomicWriteFile(const std::string &path, const void *data, size_t len);
    static bool ReadFile(const std::string &path, std::string &out);
    static bool EnsureDir(const std::string &dir);

    std::string ConfigPath() const;
    std::string StatePath() const;

    int32_t server_id_;
    std::string self_endpoint_;
    std::string self_aux_;
    std::vector<PeerEntry> initial_peers_;
    std::string state_dir_;
    nuraft::ptr<nuraft::log_store> log_store_;
    std::mutex mutex_;
};

} // namespace raft_meta
} // namespace kv_cache_manager
