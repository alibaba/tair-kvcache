#pragma once

#include <chrono>
#include <string>
namespace kv_cache_manager {

class TimestampUtil {
public:
    static int64_t GetCurrentTimeUs() { return GetCurrentTime<std::chrono::microseconds>(); }

    static int64_t GetCurrentTimeMs() { return GetCurrentTime<std::chrono::milliseconds>(); }

    static int64_t GetCurrentTimeSec() { return GetCurrentTime<std::chrono::seconds>(); }

    static int64_t GetSteadyTimeUs() { return GetSteadyTime<std::chrono::microseconds>(); }

    static int64_t GetSteadyTimeMs() { return GetSteadyTime<std::chrono::milliseconds>(); }

    static int64_t GetSteadyTimeSec() { return GetSteadyTime<std::chrono::seconds>(); }

    static std::string FormatTimestampUs(int64_t timestamp_us);

private:
    template <typename T>
    static int64_t GetCurrentTime() {
        return std::chrono::duration_cast<T>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    template <typename T>
    static int64_t GetSteadyTime() {
        return std::chrono::duration_cast<T>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};
} // namespace kv_cache_manager
