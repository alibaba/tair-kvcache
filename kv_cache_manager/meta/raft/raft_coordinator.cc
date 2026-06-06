#include "kv_cache_manager/meta/raft/raft_coordinator.h"

#include <libnuraft/asio_service.hxx>
#include <libnuraft/asio_service_options.hxx>
#include <libnuraft/async.hxx>
#include <libnuraft/callback.hxx>
#include <libnuraft/raft_params.hxx>

#include <atomic>
#include <sstream>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_local_backend.h"

namespace kv_cache_manager {
namespace raft_meta {

using nuraft::asio_service;
using nuraft::buffer;
using nuraft::cb_func;
using nuraft::cmd_result;
using nuraft::cmd_result_code;
using nuraft::cs_new;
using nuraft::ptr;
using nuraft::raft_launcher;
using nuraft::raft_params;
using nuraft::raft_server;

namespace {

// Process-wide singleton pointer. Set/cleared by Server::Init/Stop in the
// raft branch; tests that exercise multiple coordinators in the same process
// (e.g. coordinator basic-lifecycle test) skip the singleton.
std::atomic<RaftCoordinator *> g_instance{nullptr};

ErrorCode CmdCodeToEc(cmd_result_code code) {
    switch (code) {
    case cmd_result_code::OK:
        return EC_OK;
    case cmd_result_code::NOT_LEADER:
        return EC_BADARGS;
    case cmd_result_code::TIMEOUT:
    case cmd_result_code::CANCELLED:
    default:
        return EC_ERROR;
    }
}

} // namespace

RaftCoordinator::RaftCoordinator() = default;

RaftCoordinator::~RaftCoordinator() {
    if (running_.load()) {
        Stop();
    }
}

void RaftCoordinator::SetLeadershipCallback(LeadershipCallback cb) {
    std::lock_guard<std::mutex> g(cb_mu_);
    leadership_cb_ = std::move(cb);
}

std::shared_ptr<MetaCacheBaseBackend> RaftCoordinator::MakeBackend(const std::string &instance_id) {
    auto inner_cfg = std::make_shared<MetaStorageBackendConfig>(META_LOCAL_BACKEND_TYPE_STR);
    {
        std::ostringstream local_uri;
        local_uri << "local://?capacity=" << config_.local_capacity
                  << "&num_shard_bits=" << config_.local_num_shard_bits
                  << "&sample_times=" << config_.local_sample_times;
        inner_cfg->SetStorageUri(local_uri.str());
    }
    auto backend = std::make_shared<MetaLocalBackend>();
    if (backend->Init(instance_id, inner_cfg) != EC_OK) {
        KVCM_LOG_ERROR("RaftCoordinator: inner MetaLocalBackend Init failed for instance[%s]", instance_id.c_str());
        return nullptr;
    }
    if (backend->Open() != EC_OK) {
        KVCM_LOG_ERROR("RaftCoordinator: inner MetaLocalBackend Open failed for instance[%s]", instance_id.c_str());
        return nullptr;
    }
    return backend;
}

ErrorCode RaftCoordinator::BuildInner() {
    if (config_.server_id <= 0 || config_.self_endpoint.empty() || config_.data_dir.empty()) {
        KVCM_LOG_ERROR("RaftCoordinator: invalid config (server_id[%d] endpoint[%s] data_dir[%s])",
                       config_.server_id,
                       config_.self_endpoint.c_str(),
                       config_.data_dir.c_str());
        return EC_BADARGS;
    }

    log_store_ = cs_new<LmdbLogStore>(config_.data_dir + "/raft_log");

    // State machine routes by instance_id; the factory closure pulls
    // tunables from the coordinator config, so per-Instance setup stays
    // consistent across the process.
    state_machine_ = cs_new<MetaStateMachine>(
        [this](const std::string &iid) -> std::shared_ptr<MetaCacheBaseBackend> { return MakeBackend(iid); },
        config_.data_dir + "/snapshot");

    std::vector<PeerEntry> peers;
    peers.reserve(config_.peers.size());
    for (const auto &p : config_.peers) {
        PeerEntry pe;
        pe.server_id = p.server_id;
        pe.endpoint = p.endpoint;
        pe.aux = p.aux;
        pe.is_learner = p.is_learner;
        peers.push_back(std::move(pe));
    }
    state_mgr_ = cs_new<MetaStateMgr>(config_.server_id,
                                      config_.self_endpoint,
                                      config_.self_aux,
                                      std::move(peers),
                                      config_.data_dir + "/state",
                                      log_store_);
    return EC_OK;
}

ErrorCode RaftCoordinator::Start(const Config &config) {
    if (running_.load()) {
        KVCM_LOG_ERROR("RaftCoordinator: already running");
        return EC_ERROR;
    }
    config_ = config;
    if (auto rc = BuildInner(); rc != EC_OK) {
        return rc;
    }

    raft_params params;
    params.with_election_timeout_lower(config_.election_timeout_lower)
        .with_election_timeout_upper(config_.election_timeout_upper)
        .with_hb_interval(config_.heart_beat_interval)
        .with_snapshot_enabled(config_.snapshot_distance);

    asio_service::options asio_opts;
    raft_server::init_options init_opts;
    // Phase 1: all clusters are fresh (no persistent state). Every node
    // must be allowed to campaign so that an initial leader can emerge.
    // skip_initial_election_timeout_ = true would prevent all nodes from
    // starting elections in a fresh cluster (no existing leader to send
    // heartbeats). Leave it false.
    init_opts.skip_initial_election_timeout_ = false;

    // Capture leadership callback at launcher init time. NuRaft has no
    // post-init way to install raft_callback_, so we read leadership_cb_
    // here under cb_mu_ and bake it into a stable lambda. Subsequent
    // SetLeadershipCallback calls become no-ops by design.
    LeadershipCallback cb;
    {
        std::lock_guard<std::mutex> g(cb_mu_);
        cb = leadership_cb_;
    }
    init_opts.raft_callback_ = [cb](cb_func::Type type, cb_func::Param * /*param*/) -> cb_func::ReturnCode {
        if (!cb) {
            return cb_func::ReturnCode::Ok;
        }
        if (type == cb_func::Type::BecomeLeader) {
            cb(true);
        } else if (type == cb_func::Type::BecomeFollower || type == cb_func::Type::RemovedFromCluster) {
            cb(false);
        }
        return cb_func::ReturnCode::Ok;
    };

    launcher_ = std::make_unique<raft_launcher>();
    raft_server_ =
        launcher_->init(state_machine_, state_mgr_, /*logger=*/nullptr, config_.port, asio_opts, params, init_opts);
    if (!raft_server_) {
        KVCM_LOG_ERROR("RaftCoordinator: raft_launcher init failed for server[%d] port[%d]",
                       config_.server_id,
                       config_.port);
        // Drop the half-built objects so a retry can rebuild cleanly.
        launcher_.reset();
        state_mgr_.reset();
        state_machine_.reset();
        log_store_.reset();
        return EC_ERROR;
    }
    running_.store(true);
    KVCM_LOG_INFO("RaftCoordinator: started, server_id[%d] endpoint[%s] peers[%zu] data_dir[%s]",
                  config_.server_id,
                  config_.self_endpoint.c_str(),
                  config_.peers.size(),
                  config_.data_dir.c_str());
    return EC_OK;
}

void RaftCoordinator::Stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (launcher_) {
        launcher_->shutdown();
        launcher_.reset();
    }
    raft_server_.reset();
    // Keep state_machine_ / state_mgr_ / log_store_ around so any in-flight
    // reader still sees consistent state until the coordinator dies; they
    // are dropped in the dtor.
}

ErrorCode RaftCoordinator::AppendAndWait(ptr<buffer> data) {
    if (!running_.load() || !raft_server_) {
        return EC_ERROR;
    }
    if (!raft_server_->is_leader()) {
        return EC_BADARGS;
    }
    std::vector<ptr<buffer>> logs{data};
    auto result = raft_server_->append_entries(logs);
    if (!result) {
        return EC_ERROR;
    }
    if (!result->get_accepted()) {
        return CmdCodeToEc(result->get_result_code());
    }
    result->get();
    return CmdCodeToEc(result->get_result_code());
}

std::shared_ptr<MetaCacheBaseBackend> RaftCoordinator::GetOrCreateBackend(const std::string &instance_id) {
    if (!state_machine_) {
        return nullptr;
    }
    return state_machine_->GetOrCreateBackend(instance_id);
}

ErrorCode RaftCoordinator::RegistryLoad(const std::string &key,
                                        std::map<std::string, std::string> &out) const {
    if (!state_machine_) {
        return EC_ERROR;
    }
    return state_machine_->RegistryLoad(key, out);
}

bool RaftCoordinator::IsLeader() const { return raft_server_ && raft_server_->is_leader(); }

int32_t RaftCoordinator::LeaderId() const { return raft_server_ ? raft_server_->get_leader() : -1; }

std::string RaftCoordinator::GetPeerAux(int32_t server_id) const {
    if (!raft_server_) {
        return {};
    }
    return raft_server_->get_aux(server_id);
}

void RaftCoordinator::SetRegistryCommitCallback(MetaStateMachine::RegistryCommitCallback cb) {
    if (state_machine_) {
        state_machine_->SetRegistryCommitCallback(std::move(cb));
    }
}

void RaftCoordinator::SetInstance(RaftCoordinator *coord) { g_instance.store(coord); }

RaftCoordinator *RaftCoordinator::GetInstance() { return g_instance.load(); }

} // namespace raft_meta
} // namespace kv_cache_manager
