#include "tools/kvcm_swarm/scenario/config_reader.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>

namespace kvcm_swarm {

bool ParseDuration(const std::string &text, Duration *out, std::string *error) {
    if (text.empty()) {
        *error = "duration is empty";
        return false;
    }
    size_t index = 0;
    while (index < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[index])) != 0 || text[index] == '.' || text[index] == '+')) {
        ++index;
    }
    if (index == 0) {
        *error = "duration must start with a number: '" + text + "'";
        return false;
    }
    const std::string number = text.substr(0, index);
    const std::string unit = text.substr(index);
    if (unit.empty()) {
        *error = "duration must carry an explicit unit (ns/us/ms/s/m/h): '" + text + "'";
        return false;
    }
    char *end = nullptr;
    const double magnitude = std::strtod(number.c_str(), &end);
    if (end == nullptr || *end != '\0' || magnitude < 0.0) {
        *error = "duration has an invalid magnitude: '" + text + "'";
        return false;
    }
    double nanos = 0.0;
    if (unit == "ns") {
        nanos = magnitude;
    } else if (unit == "us") {
        nanos = magnitude * 1e3;
    } else if (unit == "ms") {
        nanos = magnitude * 1e6;
    } else if (unit == "s") {
        nanos = magnitude * 1e9;
    } else if (unit == "m") {
        nanos = magnitude * 6e10;
    } else if (unit == "h") {
        nanos = magnitude * 3.6e12;
    } else {
        *error = "duration has an unsupported unit '" + unit + "' in '" + text + "'";
        return false;
    }
    *out = Duration(static_cast<int64_t>(nanos));
    return true;
}

std::string FormatDuration(Duration value) {
    char buffer[64];
    const double ms = ToMillis(value);
    std::snprintf(buffer, sizeof(buffer), "%.6gms", ms);
    return buffer;
}

std::string ConfigReader::PathOf(std::string_view key) const {
    const std::string &base = node_.path();
    return base.empty() ? std::string(key) : base + "." + std::string(key);
}

void ConfigReader::ErrorAt(std::string_view key, const std::string &message) {
    errors_->push_back(PathOf(key) + ": " + message);
}

std::string ConfigReader::RequiredString(std::string_view key) {
    ConfigNode child = node_.Get(key);
    std::string value;
    if (!child.AsString(&value)) {
        ErrorAt(key, "required string field is missing or not a string");
        return {};
    }
    return value;
}

std::string ConfigReader::OptionalString(std::string_view key, std::string fallback) {
    if (!node_.Has(key)) {
        node_.Get(key);
        return fallback;
    }
    return RequiredString(key);
}

uint64_t ConfigReader::RequiredUint(std::string_view key) {
    ConfigNode child = node_.Get(key);
    int64_t value = 0;
    if (!child.AsInt(&value)) {
        ErrorAt(key, "required integer field is missing or not an integer");
        return 0;
    }
    if (value < 0) {
        ErrorAt(key, "must not be negative");
        return 0;
    }
    return static_cast<uint64_t>(value);
}

uint64_t ConfigReader::OptionalUint(std::string_view key, uint64_t fallback) {
    if (!node_.Has(key)) {
        node_.Get(key);
        return fallback;
    }
    return RequiredUint(key);
}

double ConfigReader::RequiredDouble(std::string_view key) {
    ConfigNode child = node_.Get(key);
    double value = 0.0;
    if (!child.AsDouble(&value)) {
        ErrorAt(key, "required number field is missing or not a number");
        return 0.0;
    }
    return value;
}

double ConfigReader::OptionalDouble(std::string_view key, double fallback) {
    if (!node_.Has(key)) {
        node_.Get(key);
        return fallback;
    }
    return RequiredDouble(key);
}

bool ConfigReader::OptionalBool(std::string_view key, bool fallback) {
    if (!node_.Has(key)) {
        node_.Get(key);
        return fallback;
    }
    ConfigNode child = node_.Get(key);
    bool value = false;
    if (!child.AsBool(&value)) {
        ErrorAt(key, "must be a boolean");
        return fallback;
    }
    return value;
}

Duration ConfigReader::RequiredDuration(std::string_view key) {
    ConfigNode child = node_.Get(key);
    std::string text;
    if (!child.AsString(&text)) {
        ErrorAt(key, "required duration field must be a string such as \"10ms\"");
        return Duration::zero();
    }
    Duration value{};
    std::string error;
    if (!ParseDuration(text, &value, &error)) {
        ErrorAt(key, error);
        return Duration::zero();
    }
    return value;
}

Duration ConfigReader::OptionalDuration(std::string_view key, Duration fallback) {
    if (!node_.Has(key)) {
        node_.Get(key);
        return fallback;
    }
    return RequiredDuration(key);
}

IntSpec ConfigReader::RequiredIntSpec(std::string_view key) {
    ConfigNode child = node_.Get(key);
    int64_t scalar = 0;
    if (child.AsInt(&scalar)) {
        if (scalar < 0) {
            ErrorAt(key, "must not be negative");
            return IntSpec(0);
        }
        return IntSpec(static_cast<uint64_t>(scalar));
    }
    if (!child.IsObject()) {
        ErrorAt(key, "must be an integer or {\"min\":..,\"max\":..}");
        return IntSpec(0);
    }
    ConfigReader range(child, errors_);
    const uint64_t min_value = range.RequiredUint("min");
    const uint64_t max_value = range.RequiredUint("max");
    std::vector<std::string> unknown;
    for (const auto &child_key : child.Keys()) {
        if (child_key != "min" && child_key != "max") {
            unknown.push_back(child_key);
        }
    }
    for (const auto &bad : unknown) {
        ErrorAt(key, "unknown field '" + bad + "' in range object");
    }
    if (max_value < min_value) {
        ErrorAt(key, "max must be >= min");
        return IntSpec(min_value);
    }
    return IntSpec(min_value, max_value);
}

IntSpec ConfigReader::OptionalIntSpec(std::string_view key, IntSpec fallback) {
    if (!node_.Has(key)) {
        node_.Get(key);
        return fallback;
    }
    return RequiredIntSpec(key);
}

DurationSpec ConfigReader::RequiredDurationSpec(std::string_view key) {
    ConfigNode child = node_.Get(key);
    std::string text;
    if (child.AsString(&text)) {
        Duration value{};
        std::string error;
        if (!ParseDuration(text, &value, &error)) {
            ErrorAt(key, error);
            return DurationSpec(Duration::zero());
        }
        return DurationSpec(value);
    }
    if (!child.IsObject()) {
        ErrorAt(key, "must be a duration string or {\"min\":\"..\",\"max\":\"..\"}");
        return DurationSpec(Duration::zero());
    }
    ConfigReader range(child, errors_);
    const Duration min_value = range.RequiredDuration("min");
    const Duration max_value = range.RequiredDuration("max");
    for (const auto &child_key : child.Keys()) {
        if (child_key != "min" && child_key != "max") {
            ErrorAt(key, "unknown field '" + child_key + "' in range object");
        }
    }
    if (max_value < min_value) {
        ErrorAt(key, "max must be >= min");
        return DurationSpec(min_value);
    }
    return DurationSpec(min_value, max_value);
}

DurationSpec ConfigReader::OptionalDurationSpec(std::string_view key, DurationSpec fallback) {
    if (!node_.Has(key)) {
        node_.Get(key);
        return fallback;
    }
    return RequiredDurationSpec(key);
}

std::vector<ConfigNode> ConfigReader::RequiredArray(std::string_view key) {
    ConfigNode child = node_.Get(key);
    if (!child.IsArray()) {
        ErrorAt(key, "required array field is missing or not an array");
        return {};
    }
    return child.Items();
}

} // namespace kvcm_swarm
