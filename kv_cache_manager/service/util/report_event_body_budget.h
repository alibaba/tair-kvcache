#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace kv_cache_manager {

class ReportEventBodyBudget {
public:
    static constexpr size_t kMaxBytes = 1024 * 1024 * 1024;
    static constexpr int64_t kSampleIntervalUs = 200 * 1000;

    explicit ReportEventBodyBudget(size_t configured_bytes);

    // Called by the sampler for automatic budgets.
    void Refresh();
    bool TryReserve(size_t bytes);
    void Release(size_t bytes);

    size_t Limit() const;
    size_t InUse() const { return in_use_.load(std::memory_order_relaxed); }
    uint64_t TakeRejected() { return rejected_.exchange(0, std::memory_order_relaxed); }

private:
    void RefreshFromMemInfo(const char *meminfo_path, uint64_t total_bytes);

    const bool automatic_;
    std::atomic<size_t> limit_;
    std::atomic<int64_t> sampled_at_ms_{0};
    std::atomic<size_t> in_use_{0};
    std::atomic<uint64_t> rejected_{0};
};

} // namespace kv_cache_manager
