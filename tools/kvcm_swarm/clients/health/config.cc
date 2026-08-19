#include "tools/kvcm_swarm/clients/health/config.h"

#include "tools/kvcm_swarm/scenario/config_reader.h"

namespace kvcm_swarm {

bool ParseHealthProbeConfig(const BehaviorSpec &spec, HealthProbeConfig *config, std::vector<std::string> *errors) {
    const size_t before = errors->size();
    ConfigReader reader(spec.config, errors);
    if (!spec.config.IsObject()) {
        errors->push_back("behaviors[" + spec.id + "].config: must be an object");
        return false;
    }
    config->interval = reader.RequiredDuration("interval");
    config->probe_deadline = reader.OptionalDuration("probe_deadline", std::chrono::seconds(1));
    config->streams = static_cast<uint32_t>(reader.OptionalUint("streams", 1));

    if (config->interval <= Duration::zero()) {
        reader.ErrorAt("interval", "must be positive");
    }
    if (config->probe_deadline <= Duration::zero()) {
        reader.ErrorAt("probe_deadline", "must be positive");
    }
    if (config->streams == 0) {
        reader.ErrorAt("streams", "must be at least 1");
    }
    std::vector<std::string> unknown;
    spec.config.CollectUnknown(&unknown);
    for (const auto &key : unknown) {
        errors->push_back("unknown configuration field: " + key);
    }
    return errors->size() == before;
}

} // namespace kvcm_swarm
