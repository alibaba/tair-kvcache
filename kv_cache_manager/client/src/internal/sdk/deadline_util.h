#pragma once

#include <chrono>

namespace kv_cache_manager {

// deadline 语义：绝对时间点（steady_clock / CLOCK_MONOTONIC 毫秒）。
// 0 表示「无 deadline / 不限制」。
// 时钟一致性（已实测）：Python time.monotonic_ns()//1_000_000 与 C++ steady_clock
// 是同一个内核时钟，跨语言直接数值比较，无需校正。
inline int64_t SteadyClockMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// 准入检查：deadline_ms > 0 且已过期 → true；无 deadline 或未过期 → false。
inline bool DeadlineExpired(int64_t deadline_ms) { return deadline_ms > 0 && SteadyClockMs() >= deadline_ms; }

// 尝试获取剩余毫秒。返回 true 表示有剩余（不存在 deadline 或不存在未过期时返回 false）。
// 调用方仅在返回 true 时使用 out 值（剩余毫秒，>0）。
inline bool TryRemainingMs(int64_t deadline_ms, int64_t &out) {
    if (deadline_ms <= 0) {
        return false; // 无 deadline
    }
    int64_t now_ms = SteadyClockMs();
    if (now_ms >= deadline_ms) {
        return false; // 已过期
    }
    out = deadline_ms - now_ms;
    return true;
}

} // namespace kv_cache_manager
