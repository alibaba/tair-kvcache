#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/util/report_event_body_budget.h"

namespace kv_cache_manager {
namespace {

constexpr size_t kMiB = 1024 * 1024;
constexpr uint64_t kGiB = uint64_t{1} << 30;

class ReportEventBodyBudgetTest : public TESTBASE {
protected:
    void SetUp() override {
        std::string pattern = (std::filesystem::temp_directory_path() / "kvcm-body-budget-XXXXXX").string();
        const char *path = mkdtemp(pattern.data());
        ASSERT_NE(nullptr, path);
        root_ = path;
    }

    void TearDown() override {
        if (!root_.empty()) {
            std::error_code error;
            std::filesystem::remove_all(root_, error);
            EXPECT_FALSE(error);
        }
    }

    void WriteMemInfo(std::string_view contents) {
        std::ofstream out(root_ / "meminfo");
        out << contents << '\n';
        out.close();
        ASSERT_TRUE(out);
    }

    void Refresh(ReportEventBodyBudget &budget) { budget.RefreshFromMemInfo((root_ / "meminfo").c_str()); }

    std::filesystem::path root_;
};

TEST_F(ReportEventBodyBudgetTest, UsesAvailableMemoryAndReservesSpaceBasedOnTotalMemory) {
    ReportEventBodyBudget budget(0);
    ASSERT_NO_FATAL_FAILURE(WriteMemInfo("MemTotal: 4194304 kB\nMemFree: 1048576 kB\nMemAvailable: 4194304 kB"));
    Refresh(budget);
    EXPECT_EQ((4 * kGiB - 512 * kMiB) / 8, budget.Limit());

    ASSERT_NO_FATAL_FAILURE(WriteMemInfo("MemTotal: 104857600 kB\nMemAvailable: 4194304 kB"));
    Refresh(budget);
    EXPECT_EQ((4 * kGiB - kGiB) / 8, budget.Limit());

    ASSERT_NO_FATAL_FAILURE(WriteMemInfo("MemTotal: 16777216 kB\nMemAvailable: 16777216 kB"));
    Refresh(budget);
    EXPECT_EQ(kGiB, budget.Limit());
}

TEST_F(ReportEventBodyBudgetTest, InvalidMemorySamplesUseFixedFallbackAndRecover) {
    ReportEventBodyBudget budget(0);
    Refresh(budget);
    EXPECT_EQ(64 * kMiB, budget.Limit());

    const std::string_view invalid[] = {
        "MemTotal: 4194304 kB",
        "MemFree: 4194304 kB",
        "MemTotal: 0 kB\nMemAvailable: 4194304 kB",
        "MemTotal: 4194304 kB\nMemAvailable: 4194305 kB",
        "MemTotal: 4194304 MB\nMemAvailable: 4194304 kB",
        "MemTotal: 4194304x kB\nMemAvailable: 4194304 kB",
        "MemTotal: -1 kB\nMemAvailable: 4194304 kB",
        "MemTotal: 18446744073709551615 kB\nMemAvailable: 4194304 kB",
        "MemTotal: 4194304 kB\nMemAvailable: 4194304 MB",
        "MemTotal: 4194304 kB\nMemAvailable: 4194304x kB",
        "MemTotal: 4194304 kB\nMemAvailable: -1 kB",
        "MemTotal: 4194304 kB\nMemAvailable: 18446744073709551615 kB",
    };
    for (std::string_view contents : invalid) {
        SCOPED_TRACE(contents);
        ASSERT_NO_FATAL_FAILURE(WriteMemInfo("MemTotal: 4194304 kB\nMemAvailable: 4194304 kB"));
        Refresh(budget);
        ASSERT_GT(budget.Limit(), 64 * kMiB);

        ASSERT_NO_FATAL_FAILURE(WriteMemInfo(contents));
        Refresh(budget);
        EXPECT_EQ(64 * kMiB, budget.Limit());
    }
}

TEST_F(ReportEventBodyBudgetTest, ContractionPreservesExistingReservationsAndBlocksNewBodies) {
    ReportEventBodyBudget budget(0);
    ASSERT_NO_FATAL_FAILURE(WriteMemInfo("MemTotal: 4194304 kB\nMemAvailable: 4194304 kB"));
    Refresh(budget);
    ASSERT_TRUE(budget.TryReserve(128 * kMiB));

    ASSERT_NO_FATAL_FAILURE(WriteMemInfo("MemTotal: 4194304 kB\nMemAvailable: 917504 kB"));
    Refresh(budget);
    EXPECT_FALSE(budget.TryReserve(1));
    EXPECT_EQ(128 * kMiB, budget.InUse());
    budget.Release(128 * kMiB);
    ASSERT_TRUE(budget.TryReserve(64 * kMiB));
    EXPECT_FALSE(budget.TryReserve(1));
    budget.Release(64 * kMiB);
    EXPECT_EQ(0u, budget.InUse());
    EXPECT_EQ(2u, budget.TakeRejected());
    EXPECT_EQ(0u, budget.TakeRejected());
}

TEST_F(ReportEventBodyBudgetTest, StaleSampleCapsBudgetUntilNextRefresh) {
    ReportEventBodyBudget budget(0);
    ASSERT_NO_FATAL_FAILURE(WriteMemInfo("MemTotal: 16777216 kB\nMemAvailable: 16777216 kB"));
    Refresh(budget);
    ASSERT_EQ(kGiB, budget.Limit());
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    EXPECT_EQ(64 * kMiB, budget.Limit());
    Refresh(budget);
    EXPECT_EQ(kGiB, budget.Limit());
}

TEST_F(ReportEventBodyBudgetTest, FixedBudgetCapsConcurrentReservations) {
    ReportEventBodyBudget budget(16);
    budget.Refresh();
    EXPECT_EQ(16u, budget.Limit());

    std::array<size_t, 8> admitted{};
    std::vector<std::thread> threads;
    std::promise<void> start;
    const std::shared_future<void> ready = start.get_future().share();
    for (size_t i = 0; i < admitted.size(); ++i) {
        threads.emplace_back([i, &admitted, &budget, &ready]() {
            ready.wait();
            for (int attempt = 0; attempt < 8; ++attempt) {
                if (budget.TryReserve(3)) {
                    ++admitted[i];
                }
            }
        });
    }
    start.set_value();
    for (std::thread &thread : threads) {
        thread.join();
    }

    EXPECT_EQ(5u, std::accumulate(admitted.begin(), admitted.end(), size_t{0}));
    EXPECT_EQ(15u, budget.InUse());
    EXPECT_TRUE(budget.TryReserve(1));
    EXPECT_FALSE(budget.TryReserve(1));
    budget.Release(16);
    EXPECT_EQ(0u, budget.InUse());
    EXPECT_EQ(60u, budget.TakeRejected());
}

} // namespace
} // namespace kv_cache_manager
