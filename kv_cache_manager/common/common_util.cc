#include "kv_cache_manager/common/common_util.h"

#include <cmath>
#include <limits>

namespace kv_cache_manager {

bool CommonUtil::IsZeroDouble(const double value) { return std::abs(value) < std::numeric_limits<double>::epsilon(); }

} // namespace kv_cache_manager
