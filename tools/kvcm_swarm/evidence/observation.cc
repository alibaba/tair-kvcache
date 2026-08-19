#include "tools/kvcm_swarm/evidence/observation.h"

namespace kvcm_swarm {

const char *PhaseName(Phase phase) {
    switch (phase) {
    case Phase::kValidate:
        return "validate";
    case Phase::kPreflight:
        return "preflight";
    case Phase::kInitialize:
        return "initialize";
    case Phase::kWarmup:
        return "warmup";
    case Phase::kSteady:
        return "steady";
    case Phase::kDrain:
        return "drain";
    case Phase::kReport:
        return "report";
    }
    return "unknown";
}

const char *CheckStatusName(CheckStatus status) {
    switch (status) {
    case CheckStatus::kPass:
        return "PASS";
    case CheckStatus::kFail:
        return "FAIL";
    case CheckStatus::kNotRun:
        return "NOT_RUN";
    case CheckStatus::kInconclusive:
        return "INCONCLUSIVE";
    }
    return "UNKNOWN";
}

} // namespace kvcm_swarm
