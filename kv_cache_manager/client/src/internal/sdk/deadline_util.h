#pragma once

#include <chrono>
#include <optional>

namespace kv_cache_manager {

// 后端内部 deadline 的时间域：绝对时间点（steady_clock 毫秒）。
// 各后端从自身任务起点起算：deadline = SteadyClockMs() + 注入的静态预算。
// 0 = 无 deadline（仅内部管道使用，如 usrbio 的无超时路径）。
inline int64_t SteadyClockMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// 有 deadline 且已过期。
inline bool DeadlineExpired(int64_t deadline_ms) { return deadline_ms > 0 && SteadyClockMs() >= deadline_ms; }

// 剩余毫秒；无 deadline 或已过期时返回 nullopt（需区分两者时配合 DeadlineExpired）。
inline std::optional<int64_t> RemainingMs(int64_t deadline_ms) {
    if (deadline_ms <= 0) {
        return std::nullopt;
    }
    const int64_t now_ms = SteadyClockMs();
    if (now_ms >= deadline_ms) {
        return std::nullopt;
    }
    return deadline_ms - now_ms;
}

} // namespace kv_cache_manager
