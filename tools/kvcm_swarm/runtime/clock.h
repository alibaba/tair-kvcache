// Monotonic clock and duration helpers shared by every Swarm component.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace kvcm_swarm {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::nanoseconds;

inline TimePoint Now() { return Clock::now(); }

inline double ToMillis(Duration d) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(d).count();
}

inline double ToSeconds(Duration d) { return std::chrono::duration_cast<std::chrono::duration<double>>(d).count(); }

inline Duration FromMillis(double ms) {
    return std::chrono::duration_cast<Duration>(std::chrono::duration<double, std::milli>(ms));
}

// Wall clock timestamp in milliseconds, used only for report metadata.
inline int64_t WallClockMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string FormatWallClock(int64_t wall_ms);

} // namespace kvcm_swarm
