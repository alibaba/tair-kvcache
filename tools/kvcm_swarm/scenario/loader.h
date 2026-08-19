// Local configuration validation: no transport is created and no RPC is sent.
#pragma once

#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/registry.h"
#include "tools/kvcm_swarm/scenario/config.h"

namespace kvcm_swarm {

struct LoadResult {
    bool ok = false;
    ScenarioConfig config;
    std::vector<std::string> errors;
};

LoadResult LoadScenarioFromJson(const std::string &json, const BehaviorRegistry &registry);
LoadResult LoadScenarioFromFile(const std::string &path, const BehaviorRegistry &registry);

} // namespace kvcm_swarm
