// health_probe configuration. Independent of V6D in every respect.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/client_behavior.h"
#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/scenario/json_config.h"

namespace kvcm_swarm {

struct HealthProbeConfig : public JsonConfig {
    Duration interval = std::chrono::seconds(3);
    // C5 bound: a CheckHealth response must arrive within this deadline.
    Duration probe_deadline = std::chrono::seconds(1);
    uint32_t streams = 1;

    bool FromRapidValue(const rapidjson::Value &value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool Validate(std::vector<std::string> *errors);

private:
    uint64_t streams_json_ = 1;
};

bool ParseHealthProbeConfig(const BehaviorSpec &spec, HealthProbeConfig *config, std::vector<std::string> *errors);

} // namespace kvcm_swarm
