#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {

class NodeEndpointInfo;

// Abstract interface decoupling upper layers (Server / MetaServiceImpl /
// AdminServiceImpl) from any concrete leader election mechanism.
// Implementations: LeaderElector (lease-lock based, current default) and
// future RaftLeaderElector (NuRaft based).
class ILeaderElector {
public:
    using HandlerFuncType = std::function<void()>;

    virtual ~ILeaderElector() = default;

    virtual bool Start() = 0;
    virtual void Stop() = 0;

    virtual bool IsLeader() const = 0;

    // Callbacks must be set before Start().
    virtual void SetBecomeLeaderHandler(const HandlerFuncType &handler) = 0;
    virtual void SetNoLongerLeaderHandler(const HandlerFuncType &handler) = 0;

    virtual std::string GetLeaderNodeID() const = 0;
    virtual std::string GetSelfNodeID() const = 0;

    virtual ErrorCode SetSelfNodeInfo(const NodeEndpointInfo &node_info) = 0;
    virtual ErrorCode GetSelfNodeInfo(NodeEndpointInfo &out_node_info) const = 0;
    virtual ErrorCode GetNodeInfo(const std::string &node_id, NodeEndpointInfo &out_node_info) = 0;
    virtual ErrorCode GetLeaderNodeInfo(NodeEndpointInfo &out_node_info) = 0;

    virtual void Demote() = 0;

    // Diagnostic / control surface exposed via Admin API. Lease-lock specific
    // semantics; non-lease-lock implementations may return defaults / no-op.
    virtual int64_t GetLastLoopTimeUs() const = 0;
    virtual int64_t GetLeaseExpirationTime() const = 0;
    virtual int64_t GetForbidCampaignLeaderTimeMs() const = 0;
    virtual void SetForbidCampaignLeaderTimeMs(int64_t forbid_time) = 0;
};

using ILeaderElectorPtr = std::shared_ptr<ILeaderElector>;

} // namespace kv_cache_manager
