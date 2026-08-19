#include "tools/kvcm_swarm/runtime/clock.h"

#include <ctime>

namespace kvcm_swarm {

std::string FormatWallClock(int64_t wall_ms) {
    const std::time_t secs = static_cast<std::time_t>(wall_ms / 1000);
    std::tm tm_value{};
    ::gmtime_r(&secs, &tm_value);
    char buffer[64];
    const size_t written = std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &tm_value);
    if (written == 0) {
        return {};
    }
    std::string out(buffer, written);
    const int millis = static_cast<int>(wall_ms % 1000);
    char suffix[8];
    std::snprintf(suffix, sizeof(suffix), ".%03dZ", millis);
    out.append(suffix);
    return out;
}

} // namespace kvcm_swarm
