// Preflight: verifies the real link with an isolated temporary identity and
// temporary keys before any workload starts.
//
// Preflight never creates or updates storages or instance groups. It is the
// only path allowed to call RemoveCache, and only for the tiny temporary cold
// key it created itself.
#pragma once

#include <string>
#include <vector>

#include "tools/kvcm_swarm/evidence/report.h"
#include "tools/kvcm_swarm/scenario/config.h"
#include "tools/kvcm_swarm/transport/transport_provider.h"

namespace kvcm_swarm {

class PreflightRunner {
public:
    PreflightRunner(const ScenarioConfig &config, TransportProvider &transports)
        : config_(config), transports_(transports) {}

    Task<PreflightReport> Run(TimePoint deadline);

private:
    const ScenarioConfig &config_;
    TransportProvider &transports_;
};

} // namespace kvcm_swarm
