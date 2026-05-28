#include "kv_cache_manager/common/static_service_discovery.h"

#include <cctype>
#include <cstdlib>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {

bool ParseSinglePort(const std::string &port_str, int &port_out) {
    if (port_str.empty()) {
        return false;
    }
    for (char c : port_str) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    long port = std::strtol(port_str.c_str(), nullptr, 10);
    if (port <= 0 || port > 65535) {
        return false;
    }
    port_out = static_cast<int>(port);
    return true;
}

} // namespace

bool StaticServiceDiscovery::ParseHostPortList(const std::string &host_list, std::vector<ServiceEndpoint> &out) {
    out.clear();
    if (host_list.empty()) {
        return false;
    }
    size_t start = 0;
    while (start <= host_list.size()) {
        size_t end = host_list.find(',', start);
        if (end == std::string::npos) {
            end = host_list.size();
        }
        if (end == start) {
            // 连续逗号或末尾逗号 → 跳过空段
            start = end + 1;
            continue;
        }
        const std::string token = host_list.substr(start, end - start);
        const size_t colon = token.find(':');
        if (colon == std::string::npos || colon == 0 || colon == token.size() - 1) {
            KVCM_LOG_ERROR("Static endpoint missing host:port format, token=[%s]", token.c_str());
            out.clear();
            return false;
        }
        int port = 0;
        if (!ParseSinglePort(token.substr(colon + 1), port)) {
            KVCM_LOG_ERROR("Static endpoint port invalid, token=[%s]", token.c_str());
            out.clear();
            return false;
        }
        out.emplace_back(token.substr(0, colon), port);
        start = end + 1;
    }
    if (out.empty()) {
        KVCM_LOG_ERROR("Static endpoint list is empty after parsing, raw=[%s]", host_list.c_str());
        return false;
    }
    return true;
}

bool StaticServiceDiscovery::Init(const std::string &service_address) {
    std::vector<ServiceEndpoint> parsed;
    if (!ParseHostPortList(service_address, parsed)) {
        return false;
    }
    return InitWithEndpoints(std::move(parsed));
}

bool StaticServiceDiscovery::InitWithEndpoints(std::vector<ServiceEndpoint> endpoints) {
    if (endpoints.empty()) {
        KVCM_LOG_ERROR("StaticServiceDiscovery init with empty endpoints");
        return false;
    }
    endpoints_ = std::move(endpoints);
    rr_counter_.store(0, std::memory_order_relaxed);
    return true;
}

bool StaticServiceDiscovery::GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) {
    if (endpoints_.empty()) {
        endpoints.clear();
        return false;
    }
    endpoints = endpoints_;
    return true;
}

bool StaticServiceDiscovery::GetOneEndpoint(ServiceEndpoint &endpoint) {
    if (endpoints_.empty()) {
        return false;
    }
    const size_t idx = rr_counter_.fetch_add(1, std::memory_order_relaxed) % endpoints_.size();
    endpoint = endpoints_[idx];
    return true;
}

} // namespace kv_cache_manager
