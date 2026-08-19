#include "tools/kvcm_swarm/clients/registry.h"

#include "tools/kvcm_swarm/clients/health/health_probe.h"
#include "tools/kvcm_swarm/clients/v6d/deployment.h"

namespace kvcm_swarm {

void BehaviorRegistry::Register(std::string type, std::unique_ptr<BehaviorFactory> factory) {
    factories_[std::move(type)] = std::move(factory);
}

const BehaviorFactory *BehaviorRegistry::Find(const std::string &type) const {
    const auto it = factories_.find(type);
    return it == factories_.end() ? nullptr : it->second.get();
}

std::vector<std::string> BehaviorRegistry::Types() const {
    std::vector<std::string> types;
    types.reserve(factories_.size());
    for (const auto &entry : factories_) {
        types.push_back(entry.first);
    }
    return types;
}

BehaviorRegistry MakeDefaultRegistry() {
    BehaviorRegistry registry;
    registry.Register("v6d_deployment", MakeV6dDeploymentFactory());
    registry.Register("health_probe", MakeHealthProbeFactory());
    return registry;
}

} // namespace kvcm_swarm
