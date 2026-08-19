#include "tools/kvcm_swarm/clients/health/config.h"

#include <limits>

#include "tools/kvcm_swarm/scenario/duration.h"

namespace kvcm_swarm {
namespace {

bool DecodeDuration(const rapidjson::Value &value, Duration *out, std::string *error) {
    if (!value.IsString()) {
        *error = "must be a duration string such as \"10ms\"";
        return false;
    }
    return ParseDuration(std::string(value.GetString(), value.GetStringLength()), out, error);
}

void WriteDuration(rapidjson::Writer<rapidjson::StringBuffer> &writer, const char *key, Duration value) {
    const std::string text = FormatDuration(value);
    writer.Key(key);
    writer.String(text.data(), static_cast<rapidjson::SizeType>(text.size()), false);
}

} // namespace

bool HealthProbeConfig::FromRapidValue(const rapidjson::Value &value) {
    if (!BeginObject(value, {"interval", "probe_deadline", "streams"})) {
        return false;
    }
    RequiredCustom(value, "interval", interval, DecodeDuration);
    OptionalCustom(value, "probe_deadline", probe_deadline, Duration(std::chrono::seconds(1)), DecodeDuration);
    Optional(value, "streams", streams_json_, uint64_t{1});
    return true;
}

bool HealthProbeConfig::Validate(std::vector<std::string> *errors) {
    const size_t before = errors->size();
    if (streams_json_ > std::numeric_limits<uint32_t>::max()) {
        errors->push_back("streams: must fit in uint32");
    } else {
        streams = static_cast<uint32_t>(streams_json_);
    }
    if (interval <= Duration::zero()) {
        errors->push_back("interval: must be positive");
    }
    if (probe_deadline <= Duration::zero()) {
        errors->push_back("probe_deadline: must be positive");
    }
    if (streams == 0) {
        errors->push_back("streams: must be at least 1");
    }
    return errors->size() == before;
}

void HealthProbeConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    WriteDuration(writer, "interval", interval);
    WriteDuration(writer, "probe_deadline", probe_deadline);
    Put(writer, "streams", streams);
}

bool ParseHealthProbeConfig(const BehaviorSpec &spec, HealthProbeConfig *config, std::vector<std::string> *errors) {
    const size_t before = errors->size();
    *config = HealthProbeConfig();
    std::string parse_error;
    if (!config->FromJsonString(spec.config_json, &parse_error)) {
        if (!parse_error.empty()) {
            errors->push_back(parse_error);
        }
        config->AppendJsonErrors("", errors);
        return false;
    }
    config->AppendJsonErrors("", errors);
    if (errors->size() != before) {
        return false;
    }
    config->Validate(errors);
    return errors->size() == before;
}

} // namespace kvcm_swarm
