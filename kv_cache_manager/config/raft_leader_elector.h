#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/leader_elector_interface.h"

namespace kv_cache_manager {

namespace raft_meta {
class RaftCoordinator;
} // namespace raft_meta

// ILeaderElector backed by the process-wide RaftCoordinator. Leadership is
// determined by the raft consensus protocol — no external coordination
// backend needed. Node endpoint info is carried in each srv_config.aux
// JSON, replicated through the raft cluster_config.
//
// Usage (mirrors LeaseLockLeaderElector surface):
//   1. Construct with node_id (or derive from RaftCoordinator config).
//   2. SetBecomeLeaderHandler / SetNoLongerLeaderHandler.
//   3. Start() — installs a leadership callback on the coordinator.
//   4. Stop() — unregisters.
//
// Thread safety: same contract as ILeaderElector — SetBecomeLeaderHandler /
// SetNoLongerLeaderHandler / Start / Stop from one thread; IsLeader /
// GetLeaderNodeID / GetNodeInfo from any thread.
class RaftLeaderElector : public ILeaderElector {
public:
    explicit RaftLeaderElector(const std::string &node_id);
    ~RaftLeaderElector() override;

    RaftLeaderElector(const RaftLeaderElector &) = delete;
    RaftLeaderElector &operator=(const RaftLeaderElector &) = delete;

    bool Start() override;
    void Stop() override;

    bool IsLeader() const override;

    void SetBecomeLeaderHandler(const HandlerFuncType &handler) override;
    void SetNoLongerLeaderHandler(const HandlerFuncType &handler) override;

    std::string GetLeaderNodeID() const override;
    std::string GetSelfNodeID() const override;

    ErrorCode SetSelfNodeInfo(const NodeEndpointInfo &node_info) override;
    ErrorCode GetSelfNodeInfo(NodeEndpointInfo &out_node_info) const override;
    ErrorCode GetNodeInfo(const std::string &node_id, NodeEndpointInfo &out_node_info) override;
    ErrorCode GetLeaderNodeInfo(NodeEndpointInfo &out_node_info) override;

    void Demote() override;

    int64_t GetLastLoopTimeUs() const override;
    int64_t GetLeaseExpirationTime() const override;
    int64_t GetForbidCampaignLeaderTimeMs() const override;
    void SetForbidCampaignLeaderTimeMs(int64_t forbid_time) override;

    void SetCoordinatorForTest(raft_meta::RaftCoordinator *coord);

private:
    raft_meta::RaftCoordinator *Coordinator() const;
    ErrorCode ParseNodeInfoFromAux(int32_t server_id, NodeEndpointInfo &out) const;

    std::string node_id_;
    std::atomic<bool> started_{false};
    raft_meta::RaftCoordinator *coordinator_override_{nullptr};

    HandlerFuncType become_leader_handler_;
    HandlerFuncType no_longer_leader_handler_;

    mutable std::mutex self_info_mu_;
    std::unique_ptr<NodeEndpointInfo> self_node_info_;
};

} // namespace kv_cache_manager
