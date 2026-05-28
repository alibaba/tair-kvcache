#include "kv_cache_manager/common/service_discovery_url.h"

#include <cstdlib>
#include <sstream>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

int ServiceDiscoveryUrl::GetIntParam(const std::string &key, int default_value) const {
    auto it = params.find(key);
    if (it == params.end()) {
        return default_value;
    }
    char *end = nullptr;
    long val = std::strtol(it->second.c_str(), &end, 10);
    if (end == it->second.c_str() || *end != '\0') {
        return default_value;
    }
    return static_cast<int>(val);
}

bool ServiceDiscoveryUrl::Parse(const std::string &url, ServiceDiscoveryUrl &out) {
    // 1) 查找 ://
    const size_t sep = url.find("://");
    if (sep == std::string::npos) {
        KVCM_LOG_WARN("invalid service discovery url, no scheme separator: %s", url.c_str());
        return false;
    }

    out.scheme = url.substr(0, sep);
    if (out.scheme.empty()) {
        KVCM_LOG_WARN("empty scheme in service discovery url: %s", url.c_str());
        return false;
    }

    // 2) 提取 body + query params
    std::string remaining = url.substr(sep + 3);
    const size_t qmark = remaining.find('?');
    if (qmark != std::string::npos) {
        out.body = remaining.substr(0, qmark);
        std::string query = remaining.substr(qmark + 1);

        // 解析 key=value&key=value
        std::istringstream query_stream(query);
        std::string pair;
        while (std::getline(query_stream, pair, '&')) {
            const size_t eq = pair.find('=');
            if (eq == std::string::npos) {
                KVCM_LOG_WARN("malformed query param in service discovery url: %s", url.c_str());
                return false;
            }
            std::string key = pair.substr(0, eq);
            std::string value = pair.substr(eq + 1);
            if (key.empty()) {
                KVCM_LOG_WARN("empty param key in service discovery url: %s", url.c_str());
                return false;
            }
            out.params[key] = value;
        }
    } else {
        out.body = remaining;
    }

    return true;
}

} // namespace kv_cache_manager
