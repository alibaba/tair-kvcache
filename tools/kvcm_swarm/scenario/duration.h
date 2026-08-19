#pragma once

#include <string>

#include "tools/kvcm_swarm/runtime/clock.h"

namespace kvcm_swarm {

// Parses "10ms", "1.5s", "2m", "500us" and "3h". Bare numbers are
// rejected so run configuration never depends on an implicit unit.
bool ParseDuration(const std::string &text, Duration *out, std::string *error);
std::string FormatDuration(Duration value);

} // namespace kvcm_swarm
