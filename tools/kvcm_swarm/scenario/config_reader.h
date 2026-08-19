// Error-accumulating typed reader over ConfigNode.
//
// Every read either succeeds or records a precise, path-qualified error, so
// local validation reports all configuration problems at once and creates no
// transports and sends no RPCs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/runtime/sample_spec.h"
#include "tools/kvcm_swarm/scenario/config_node.h"

namespace kvcm_swarm {

// Parses "10ms", "1.5s", "2m", "500us", "3h". Bare numbers are rejected:
// units must be explicit.
bool ParseDuration(const std::string &text, Duration *out, std::string *error);
std::string FormatDuration(Duration value);

class ConfigReader {
public:
    ConfigReader(ConfigNode node, std::vector<std::string> *errors) : node_(std::move(node)), errors_(errors) {}

    const ConfigNode &node() const { return node_; }
    bool Has(std::string_view key) const { return node_.Has(key); }

    ConfigReader Child(std::string_view key) { return ConfigReader(node_.Get(key), errors_); }

    std::string RequiredString(std::string_view key);
    std::string OptionalString(std::string_view key, std::string fallback);
    uint64_t RequiredUint(std::string_view key);
    uint64_t OptionalUint(std::string_view key, uint64_t fallback);
    double RequiredDouble(std::string_view key);
    double OptionalDouble(std::string_view key, double fallback);
    bool OptionalBool(std::string_view key, bool fallback);
    Duration RequiredDuration(std::string_view key);
    Duration OptionalDuration(std::string_view key, Duration fallback);

    IntSpec RequiredIntSpec(std::string_view key);
    IntSpec OptionalIntSpec(std::string_view key, IntSpec fallback);
    DurationSpec RequiredDurationSpec(std::string_view key);
    DurationSpec OptionalDurationSpec(std::string_view key, DurationSpec fallback);

    std::vector<ConfigNode> RequiredArray(std::string_view key);

    void Error(std::string message) { errors_->push_back(std::move(message)); }
    void ErrorAt(std::string_view key, const std::string &message);

    std::vector<std::string> *errors() { return errors_; }

private:
    std::string PathOf(std::string_view key) const;

    ConfigNode node_;
    std::vector<std::string> *errors_;
};

} // namespace kvcm_swarm
