#include "kv_cache_manager/service/util/report_event_body_budget.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "autil/StringUtil.h"
#include "kv_cache_manager/common/timestamp_util.h"

namespace kv_cache_manager {
namespace {

constexpr size_t kMinAutoBytes = 64 * 1024 * 1024;
constexpr uint64_t kMinReserveBytes = 512 * 1024 * 1024;
constexpr int64_t kMaxSampleAgeMs = 1000;
constexpr uint64_t kBodyMemoryMultiplier = 8;
constexpr uint64_t kBytesPerKiB = 1024;

bool ReadMemoryInfo(const char *meminfo_path, uint64_t &total_bytes, uint64_t &available_bytes) {
    std::ifstream input(meminfo_path);
    bool found_total = false;
    bool found_available = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string name;
        if (!(fields >> name) || (name != "MemTotal:" && name != "MemAvailable:")) {
            continue;
        }

        std::string number;
        std::string unit;
        std::string extra;
        uint64_t kib = 0;
        if (!(fields >> number >> unit) || unit != "kB" || (fields >> extra) ||
            number.find_first_not_of("0123456789") != std::string::npos ||
            !autil::StringUtil::strToUInt64(number.c_str(), kib) ||
            kib > std::numeric_limits<uint64_t>::max() / kBytesPerKiB) {
            return false;
        }
        if (name == "MemTotal:") {
            total_bytes = kib * kBytesPerKiB;
            found_total = true;
        } else {
            available_bytes = kib * kBytesPerKiB;
            found_available = true;
        }
    }
    return found_total && found_available;
}

} // namespace

ReportEventBodyBudget::ReportEventBodyBudget(size_t configured_bytes)
    : automatic_(configured_bytes == 0), limit_(automatic_ ? kMinAutoBytes : std::min(configured_bytes, kMaxBytes)) {
    Refresh();
}

void ReportEventBodyBudget::Refresh() {
    if (automatic_) {
        RefreshFromMemInfo("/proc/meminfo");
    }
}

void ReportEventBodyBudget::RefreshFromMemInfo(const char *meminfo_path) {
    const int64_t now_ms = TimestampUtil::GetSteadyTimeMs();
    uint64_t total_bytes = 0;
    uint64_t available_bytes = 0;
    size_t bytes = kMinAutoBytes;
    if (ReadMemoryInfo(meminfo_path, total_bytes, available_bytes) && total_bytes > 0 &&
        available_bytes <= total_bytes) {
        const uint64_t reserve = std::max(kMinReserveBytes, total_bytes / 100);
        const uint64_t headroom = available_bytes > reserve ? available_bytes - reserve : 0;
        bytes = std::clamp<uint64_t>(headroom / kBodyMemoryMultiplier, kMinAutoBytes, kMaxBytes);
        sampled_at_ms_.store(now_ms, std::memory_order_relaxed);
    }
    limit_.store(bytes, std::memory_order_release);
}

size_t ReportEventBodyBudget::Limit() const {
    const size_t bytes = limit_.load(std::memory_order_acquire);
    if (automatic_ && bytes > kMinAutoBytes &&
        TimestampUtil::GetSteadyTimeMs() - sampled_at_ms_.load(std::memory_order_relaxed) > kMaxSampleAgeMs) {
        return kMinAutoBytes;
    }
    return bytes;
}

bool ReportEventBodyBudget::TryReserve(size_t bytes) {
    size_t in_use = in_use_.load(std::memory_order_relaxed);
    do {
        const size_t limit = Limit();
        if (bytes > limit || in_use > limit - bytes) {
            rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    } while (!in_use_.compare_exchange_weak(in_use, in_use + bytes, std::memory_order_relaxed));
    return true;
}

void ReportEventBodyBudget::Release(size_t bytes) { in_use_.fetch_sub(bytes, std::memory_order_relaxed); }

} // namespace kv_cache_manager
