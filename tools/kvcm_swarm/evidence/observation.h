// Evidence value types shared by the runtime, the transport and behaviors.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"
#include "tools/kvcm_swarm/runtime/admission.h"
#include "tools/kvcm_swarm/runtime/clock.h"

namespace kvcm_swarm {

enum class Phase {
    kValidate,
    kPreflight,
    kInitialize,
    kWarmup,
    kSteady,
    kDrain,
    kReport,
};

const char *PhaseName(Phase phase);

using kv_cache_manager::async_rpc::IsUncertain;
using kv_cache_manager::async_rpc::RpcResult;
using kv_cache_manager::async_rpc::TransportError;
using kv_cache_manager::async_rpc::TransportErrorName;

struct RpcObservation {
    std::string behavior_type;
    std::string behavior_id;
    std::string process_id; // empty when not process-scoped
    std::string api;
    Phase phase = Phase::kSteady;
    TrafficLane lane = TrafficLane::kBusiness;
    Duration queue_delay{}; // planned time -> actual submit time
    Duration permit_wait{};
    RpcResult result;
};

enum class CheckStatus {
    kPass,
    kFail,
    kNotRun,
    kInconclusive
};

const char *CheckStatusName(CheckStatus status);

struct InvariantObservation {
    std::string behavior_type;
    std::string check_name;
    CheckStatus status = CheckStatus::kNotRun;
    uint64_t checked = 0;
    uint64_t violations = 0;
    std::string reason;
    std::map<std::string, int64_t> counters;
    std::vector<std::string> detail_preview;
};

} // namespace kvcm_swarm
