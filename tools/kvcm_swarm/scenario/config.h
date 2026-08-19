// Immutable effective run configuration.
//
// There is exactly one configuration entry point: a single JSON document. No
// base + overlay merging. CI scaling happens in the fixture, which writes the
// final values here; the C++ tool never rescales anything.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/client_behavior.h"
#include "tools/kvcm_swarm/runtime/admission.h"
#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/scenario/config_node.h"
#include "tools/kvcm_swarm/transport/transport.h"

namespace kvcm_swarm {

struct RuntimeConfig {
    Duration warmup = std::chrono::seconds(5);
    Duration steady = std::chrono::seconds(30);
    Duration drain_timeout = std::chrono::seconds(20);
    uint32_t workers = 4;
    AdmissionLimits limits;
    // Fixed, small pools of threads that wait for network events.
    uint32_t reactor_threads = 2;
    uint32_t grpc_completion_queues = 2;
    TransportLimits transport;
};

struct InstanceGroupTarget {
    std::string name;
    uint64_t quota_bytes = 0;
};

struct TargetConfig {
    EndpointSet endpoints;
    std::map<std::string, InstanceGroupTarget> instance_groups;
};

struct EvidenceConfig {
    std::string output_json;
    std::string violations_jsonl;
    std::string markdown_summary;
};

struct ScenarioConfig {
    std::string name;
    uint64_t seed = 0;
    RuntimeConfig runtime;
    TargetConfig target;
    std::vector<BehaviorSpec> behaviors;
    EvidenceConfig evidence;
    bool preflight_enabled = true;
    // Raw configuration document, echoed into the report.
    ConfigNode document;
};

} // namespace kvcm_swarm
