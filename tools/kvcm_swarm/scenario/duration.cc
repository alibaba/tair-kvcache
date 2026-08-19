#include "tools/kvcm_swarm/scenario/duration.h"

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

} // namespace kvcm_swarm
