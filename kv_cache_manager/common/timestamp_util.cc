#include "kv_cache_manager/common/timestamp_util.h"

#include <iomanip>
#include <sstream>
namespace kv_cache_manager {

std::string TimestampUtil::FormatTimestampUs(int64_t timestamp_us) {
    // 将微秒时间戳分解为秒和微秒部分
    time_t seconds = static_cast<time_t>(timestamp_us / 1000000);
    auto microseconds = static_cast<int>(timestamp_us % 1000000);

    // 转换为本地时间
    struct tm time_buf;
    struct tm *timeinfo = localtime_r(&seconds, &time_buf);

    // 格式化时间，包含微秒
    std::ostringstream oss;
    oss << std::put_time(timeinfo, "%Y.%m.%d,%H:%M:%S") << "." << std::setfill('0') << std::setw(6) << microseconds;

    return oss.str();
}

} // namespace kv_cache_manager