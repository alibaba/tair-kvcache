#pragma once

#include <cstdint>
#include <string>

namespace kv_cache_manager {

class EnvUtil {
public:
    template <typename T>
    static T GetEnv(const std::string &key, const T &defaultValue);
    static std::string GetEnv(const std::string &key, const std::string &defaultValue);
};

struct ScopedEnv {
    ScopedEnv(const char *name, const char *value) : name_(name) { setenv(name, value, 1); }
    ~ScopedEnv() { unsetenv(name_); }
    const char *name_;
};

extern template int32_t EnvUtil::GetEnv<int32_t>(const std::string &, const int32_t &);
extern template int64_t EnvUtil::GetEnv<int64_t>(const std::string &, const int64_t &);
extern template bool EnvUtil::GetEnv<bool>(const std::string &, const bool &);

} // namespace kv_cache_manager
