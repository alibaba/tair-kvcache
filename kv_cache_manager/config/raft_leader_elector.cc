#include "kv_cache_manager/config/raft_leader_elector.h"

#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/node_endpoint_info.h"
#include "kv_cache_manager/meta/raft/raft_coordinator.h"

namespace kv_cache_manager {

RaftLeaderElector::RaftLeaderElector(const std::string &node_id) : node_id_(node_id) {}

RaftLeaderElector::~RaftLeaderElector() {
    if (started_.load()) {
        Stop();
    }
}

raft_meta::RaftCoordinator *RaftLeaderElector::Coordinator() const {
    if (coordinator_override_) {
        return coordinator_override_;
    }
    return raft_meta::RaftCoordinator::GetInstance();
}

bool RaftLeaderElector::Start() {
    if (started_.load()) {
        return true;
    }
    if (!become_leader_handler_ || !no_longer_leader_handler_) {
        KVCM_LOG_ERROR("RaftLeaderElector: handlers must be set before Start");
        return false;
    }
    auto *coord = Coordinator();
    if (!coord) {
        KVCM_LOG_ERROR("RaftLeaderElector: no RaftCoordinator available");
        return false;
    }

    coord->SetLeadershipCallback([this](bool is_leader) {
        if (is_leader) {
            KVCM_LOG_INFO("RaftLeaderElector: became leader (node[%s])", node_id_.c_str());
            if (become_leader_handler_) {
                become_leader_handler_();
            }
        } else {
            KVCM_LOG_INFO("RaftLeaderElector: no longer leader (node[%s])", node_id_.c_str());
            if (no_longer_leader_handler_) {
                no_longer_leader_handler_();
            }
        }
    });

    started_.store(true);
    KVCM_LOG_INFO("RaftLeaderElector: started, node_id[%s]", node_id_.c_str());
    return true;
}

void RaftLeaderElector::Stop() {
    if (!started_.exchange(false)) {
        return;
    }
    KVCM_LOG_INFO("RaftLeaderElector: stopped, node_id[%s]", node_id_.c_str());
}

bool RaftLeaderElector::IsLeader() const {
    auto *coord = Coordinator();
    return coord && coord->IsLeader();
}

void RaftLeaderElector::SetBecomeLeaderHandler(const HandlerFuncType &handler) {
    become_leader_handler_ = handler;
}

void RaftLeaderElector::SetNoLongerLeaderHandler(const HandlerFuncType &handler) {
    no_longer_leader_handler_ = handler;
}

std::string RaftLeaderElector::GetLeaderNodeID() const {
    auto *coord = Coordinator();
    if (!coord) {
        return {};
    }
    int32_t leader_id = coord->LeaderId();
    if (leader_id < 0) {
        return {};
    }
    NodeEndpointInfo info;
    if (ParseNodeInfoFromAux(leader_id, info) == EC_OK) {
        return info.node_id();
    }
    return std::to_string(leader_id);
}

std::string RaftLeaderElector::GetSelfNodeID() const { return node_id_; }

ErrorCode RaftLeaderElector::SetSelfNodeInfo(const NodeEndpointInfo &node_info) {
    // In raft mode, NodeEndpointInfo is carried in srv_config.aux and
    // replicated through cluster_config. The aux is baked into the
    // RaftCoordinator::Config.self_aux at startup time, so there is no
    // mutable KV store to write to. We cache the info locally so
    // GetSelfNodeInfo returns it.
    std::lock_guard<std::mutex> g(self_info_mu_);
    self_node_info_ = std::make_unique<NodeEndpointInfo>(node_info);
    return EC_OK;
}

ErrorCode RaftLeaderElector::GetSelfNodeInfo(NodeEndpointInfo &out_node_info) const {
    std::lock_guard<std::mutex> g(self_info_mu_);
    if (!self_node_info_) {
        return EC_NOENT;
    }
    out_node_info = *self_node_info_;
    return EC_OK;
}

ErrorCode RaftLeaderElector::ParseNodeInfoFromAux(int32_t server_id, NodeEndpointInfo &out) const {
    auto *coord = Coordinator();
    if (!coord) {
        return EC_ERROR;
    }
    std::string aux = coord->GetPeerAux(server_id);
    if (aux.empty()) {
        return EC_NOENT;
    }
    if (!out.FromJsonString(aux)) {
        KVCM_LOG_ERROR("RaftLeaderElector: failed to parse NodeEndpointInfo from aux for server[%d]", server_id);
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode RaftLeaderElector::GetNodeInfo(const std::string &node_id, NodeEndpointInfo &out_node_info) {
    if (node_id == node_id_) {
        return GetSelfNodeInfo(out_node_info);
    }
    auto *coord = Coordinator();
    if (!coord) {
        return EC_ERROR;
    }
    // NodeEndpointInfo.node_id is embedded in the aux JSON; we need to
    // scan all known servers. For the common case (looking up the leader),
    // GetLeaderNodeInfo is faster.
    // Phase 1 only has a small fixed cluster, so a linear scan is fine.
    const auto &peers = coord->GetConfig().peers;
    for (const auto &p : peers) {
        NodeEndpointInfo info;
        if (ParseNodeInfoFromAux(p.server_id, info) == EC_OK && info.node_id() == node_id) {
            out_node_info = info;
            return EC_OK;
        }
    }
    return EC_NOENT;
}

ErrorCode RaftLeaderElector::GetLeaderNodeInfo(NodeEndpointInfo &out_node_info) {
    auto *coord = Coordinator();
    if (!coord) {
        return EC_ERROR;
    }
    int32_t leader_id = coord->LeaderId();
    if (leader_id < 0) {
        return EC_NOENT;
    }
    if (leader_id == coord->SelfId()) {
        return GetSelfNodeInfo(out_node_info);
    }
    return ParseNodeInfoFromAux(leader_id, out_node_info);
}

void RaftLeaderElector::Demote() {
    // Raft leadership is managed by the consensus protocol; explicit
    // demotion is not supported in Phase 1. Log and no-op.
    KVCM_LOG_WARN("RaftLeaderElector: Demote() is a no-op in raft mode");
}

int64_t RaftLeaderElector::GetLastLoopTimeUs() const { return -1; }

int64_t RaftLeaderElector::GetLeaseExpirationTime() const { return -1; }

int64_t RaftLeaderElector::GetForbidCampaignLeaderTimeMs() const { return 0; }

void RaftLeaderElector::SetForbidCampaignLeaderTimeMs(int64_t /*forbid_time*/) {}

void RaftLeaderElector::SetCoordinatorForTest(raft_meta::RaftCoordinator *coord) {
    coordinator_override_ = coord;
}

} // namespace kv_cache_manager
