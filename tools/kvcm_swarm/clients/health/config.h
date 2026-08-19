// health_probe configuration. Independent of V6D in every respect.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/client_behavior.h"
#include "tools/kvcm_swarm/runtime/clock.h"

namespace kvcm_swarm {

struct HealthProbeConfig {
    Duration interval = std::chrono::seconds(3);
    // C5 bound: a CheckHealth response must arrive within this deadline.
    Duration probe_deadline = std::chrono::seconds(1);
    uint32_t streams = 1;
};

bool ParseHealthProbeConfig(const BehaviorSpec &spec, HealthProbeConfig *config, std::vector<std::string> *errors);

} // namespace kvcm_swarm
