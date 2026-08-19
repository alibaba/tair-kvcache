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

#include "kv_cache_manager/common/jsonizable.h"
#include "tools/kvcm_swarm/clients/client_behavior.h"
#include "tools/kvcm_swarm/runtime/admission.h"
#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/transport/transport.h"

namespace kvcm_swarm {

struct RuntimeConfig : public kv_cache_manager::Jsonizable {
    Duration warmup = std::chrono::seconds(5);
    Duration steady = std::chrono::seconds(30);
    Duration drain_timeout = std::chrono::seconds(20);
    uint32_t workers = 4;
    AdmissionLimits limits;
    // Fixed, small pools of threads that wait for network events.
    uint32_t reactor_threads = 2;
    uint32_t grpc_completion_queues = 2;
    TransportLimits transport;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
};

struct InstanceGroupTarget : public kv_cache_manager::Jsonizable {
    std::string name;
    uint64_t quota_bytes = 0;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
};

struct TargetConfig : public kv_cache_manager::Jsonizable {
    EndpointSet endpoints;
    std::map<std::string, InstanceGroupTarget> instance_groups;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
};

struct EvidenceConfig : public kv_cache_manager::Jsonizable {
    std::string output_json;
    std::string violations_jsonl;
    std::string markdown_summary;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
};

struct ScenarioConfig {
    std::string name;
    uint64_t seed = 0;
    RuntimeConfig runtime;
    TargetConfig target;
    std::vector<BehaviorSpec> behaviors;
    EvidenceConfig evidence;
    bool preflight_enabled = true;
};

} // namespace kvcm_swarm
