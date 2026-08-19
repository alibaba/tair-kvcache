// A test-only behavior that proves the common layer is behavior-agnostic.
//
// It links only against the public behavior contract and RuntimeServices: no
// V6D workload, cache, check or expected-location type is reachable from here.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "tools/kvcm_swarm/clients/client_behavior.h"

namespace kvcm_swarm {

// Emits generic timer facts and, optionally, one CheckHealth control RPC.
std::unique_ptr<BehaviorFactory> MakeFakeBehaviorFactory();

} // namespace kvcm_swarm
