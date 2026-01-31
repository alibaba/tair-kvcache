#include "kv_cache_manager/common/env_util.h"

#include "autil/StringUtil.h"

namespace kv_cache_manager {

template <typename T>
T EnvUtil::GetEnv(const std::string &key, const T &defaultValue) {
    const char *str = std::getenv(key.c_str());
    if (!str) {
        return defaultValue;
    }
    T ret = T();
    auto success = autil::StringUtil::fromString(str, ret);
    return success ? ret : defaultValue;
}

std::string EnvUtil::GetEnv(const std::string &key, const std::string &defaultValue) {
    const char *str = std::getenv(key.c_str());
    return str != nullptr ? str : defaultValue;
}

template int32_t EnvUtil::GetEnv<int32_t>(const std::string &, const int32_t &);
template int64_t EnvUtil::GetEnv<int64_t>(const std::string &, const int64_t &);
template bool EnvUtil::GetEnv<bool>(const std::string &, const bool &);

} // namespace kv_cache_manager
