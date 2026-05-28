#pragma once

#include <map>
#include <string>

namespace kv_cache_manager {

struct ServiceDiscoveryUrl {
    std::string scheme;
    std::string body;
    std::map<std::string, std::string> params;

    int GetIntParam(const std::string &key, int default_value) const;

    static bool Parse(const std::string &url, ServiceDiscoveryUrl &out);
};

} // namespace kv_cache_manager
