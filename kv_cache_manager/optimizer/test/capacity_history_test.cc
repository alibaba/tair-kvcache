#include <limits>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/metrics/capacity_history.h"

namespace kv_cache_manager {
class CapacityHistoryTest : public TESTBASE {};

TEST_F(CapacityHistoryTest, BackfillMaxAndShortReuse) {
    CapacityHistory history;
    constexpr int64_t t = 1000 * CapacityHistory::kStepMs;
    history.Update(t, 20ULL * 60 * 1000000000, 100, true);
    history.Update(t + 10 * 60 * 1000, 15ULL * 60 * 1000000000, 120, true);
    for (int bucket = 996; bucket <= 1002; ++bucket) {
        const auto &p = history.points()[bucket % CapacityHistory::kSlots];
        EXPECT_EQ(bucket * CapacityHistory::kStepMs, p.timestamp_ms);
        EXPECT_EQ(bucket >= 999 ? 120 : 100, p.capacity_bytes);
    }
    history.Update(t + 10 * 60 * 1000 + 100, 0, 1, true);
    EXPECT_EQ(120, history.points()[1002 % CapacityHistory::kSlots].capacity_bytes);
    history.Update(t + 15 * 60 * 1000 + 100, 1, 7, true);
    EXPECT_EQ(7, history.points()[1003 % CapacityHistory::kSlots].capacity_bytes);
}

TEST_F(CapacityHistoryTest, BoundedExpiryResetAndNoSample) {
    CapacityHistory history;
    constexpr int64_t t = 1000 * CapacityHistory::kStepMs;
    history.Update(t, std::numeric_limits<uint64_t>::max(), 100, true);
    size_t populated = 0;
    for (const auto &p : history.points()) {
        populated += p.timestamp_ms > 0;
        EXPECT_GE(p.timestamp_ms, (1000 - 287) * CapacityHistory::kStepMs);
    }
    EXPECT_EQ(288, populated);
    history.Update(t + 48 * 60 * 60 * 1000, 0, 0, false);
    for (const auto &p : history.points()) {
        EXPECT_EQ(0, p.timestamp_ms);
    }
    history.Update(t, 0, 1000, true);
    for (const auto &p : history.points()) {
        EXPECT_EQ(0, p.timestamp_ms);
    }
    history.Reset();
    history.Update(t, 0, 0, true);
    EXPECT_EQ(t, history.points()[1000 % CapacityHistory::kSlots].timestamp_ms);
    EXPECT_EQ(0, history.points()[1000 % CapacityHistory::kSlots].capacity_bytes);
}
} // namespace kv_cache_manager
