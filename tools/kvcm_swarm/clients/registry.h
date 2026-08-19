// Compile-time behavior registry. No dynamic plugin ABI, no generic client DSL.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/client_behavior.h"

namespace kvcm_swarm {

class BehaviorRegistry {
public:
    void Register(std::string type, std::unique_ptr<BehaviorFactory> factory);
    const BehaviorFactory *Find(const std::string &type) const;
    std::vector<std::string> Types() const;

private:
    std::map<std::string, std::unique_ptr<BehaviorFactory>> factories_;
};

// Registers every behavior shipped today: v6d_deployment and health_probe.
// `event_reporter` will be added here as its own top-level behavior.
BehaviorRegistry MakeDefaultRegistry();

} // namespace kvcm_swarm
