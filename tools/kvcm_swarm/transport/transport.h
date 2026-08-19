// Transport contract shared by the plaintext HTTP and insecure gRPC paths.
//
// The transport owns permit admission, submit-time phase attribution and
// latency measurement, and preserves raw batch responses and error codes.
// It never interprets sessions, selectors, location owners or check semantics,
// and never hides a retry.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"
#include "tools/kvcm_swarm/evidence/histogram.h"
#include "tools/kvcm_swarm/evidence/observation.h"
#include "tools/kvcm_swarm/evidence/sink.h"
#include "tools/kvcm_swarm/protocol/api.h"
#include "tools/kvcm_swarm/runtime/admission.h"
#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/runtime/executor.h"
#include "tools/kvcm_swarm/runtime/stop_token.h"

namespace google {
namespace protobuf {
class Message;
} // namespace protobuf
} // namespace google

namespace kvcm_swarm {

using kv_cache_manager::async_rpc::Api;
using kv_cache_manager::async_rpc::EndpointSet;
using kv_cache_manager::async_rpc::RpcResult;
using kv_cache_manager::async_rpc::TransportError;
using kv_cache_manager::async_rpc::TransportErrorName;
using kv_cache_manager::async_rpc::TransportKind;
using kv_cache_manager::async_rpc::TransportKindName;
using TransportLimits = kv_cache_manager::async_rpc::ClientLimits;

// Tracks the current run phase. RPC observations are attributed at the moment
// the request is submitted; a completion never rewrites the phase.
class PhaseSource {
public:
    Phase Current() const { return phase_.load(std::memory_order_relaxed); }
    void Set(Phase phase) { phase_.store(phase, std::memory_order_relaxed); }

private:
    std::atomic<Phase> phase_{Phase::kValidate};
};

struct ClientIdentity {
    std::string behavior_type;
    std::string behavior_id;
    std::string process_id; // empty for behavior-level contexts
};

struct EndpointStats {
    std::string endpoint;
    std::string role; // "meta" or "admin"
    uint64_t channels = 0;
    uint64_t connections_current = 0;
    uint64_t connections_peak = 0;
    uint64_t connections_created = 0;
    uint64_t connections_reused = 0;
    uint64_t in_flight_current = 0;
    uint64_t in_flight_peak = 0;
    Histogram establish_latency_ms;
};

struct TransportContextStats {
    ClientIdentity identity;
    TransportKind kind = TransportKind::kHttp;
    std::vector<EndpointStats> endpoints;
};

struct CallOptions {
    TrafficLane lane = TrafficLane::kBusiness;
    TimePoint deadline{};
    // Time at which the operation became due; the gap to the actual submit
    // time is reported as queue delay.
    TimePoint planned_at{};
    StopToken stop;
};

class ClientTransportContext {
public:
    virtual ~ClientTransportContext() = default;

    virtual Task<RpcResult> Call(Api api,
                                 const google::protobuf::Message &request,
                                 google::protobuf::Message *response,
                                 CallOptions options) = 0;

    // Leader discovery may move the meta endpoint of this context only.
    virtual void SetMetaEndpoint(const std::string &endpoint) = 0;
    virtual std::string MetaEndpoint() const = 0;

    virtual TransportContextStats Stats() const = 0;
    virtual void Shutdown() = 0;
};

} // namespace kvcm_swarm
