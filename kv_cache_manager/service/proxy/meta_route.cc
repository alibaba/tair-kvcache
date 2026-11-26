#include "kv_cache_manager/service/proxy/meta_route.h"

#include <algorithm>
#include <arpa/inet.h>
#include <charconv>

namespace kv_cache_manager {
namespace {
std::string ReporterIp(std::string_view endpoint) {
    const auto colon = endpoint.rfind(':');
    if (colon == std::string_view::npos) {
        return {};
    }
    const auto port = endpoint.substr(colon + 1);
    unsigned value = 0;
    const auto result = std::from_chars(port.data(), port.data() + port.size(), value);
    if (result.ec != std::errc{} || result.ptr != port.data() + port.size() || !value || value > 65535) {
        return {};
    }
    std::string host(endpoint.substr(0, colon));
    const bool ipv6 = host.size() >= 2 && host.front() == '[' && host.back() == ']';
    if (ipv6) {
        host = host.substr(1, host.size() - 2);
    }
    if (host.find('\0') != std::string::npos) {
        return {};
    }
    const int family = ipv6 ? AF_INET6 : AF_INET;
    in6_addr addr{};
    char canonical[INET6_ADDRSTRLEN];
    if (inet_pton(family, host.c_str(), &addr) != 1 || !inet_ntop(family, &addr, canonical, sizeof(canonical))) {
        return {};
    }
    return canonical;
}
} // namespace

bool IsGlobalReporterEvent(const proto::meta::EventItem &event) {
    return event.event_type() == proto::meta::EVENT_NODE_REGISTER ||
           event.event_type() == proto::meta::EVENT_HOST_DOWN || event.event_type() == proto::meta::EVENT_HEARTBEAT;
}

RouteKey MetaRoute(const proto::meta::ReportEventRequest &request) {
    const bool lifecycle_only = !request.events().empty() &&
                                std::all_of(request.events().begin(), request.events().end(), IsGlobalReporterEvent);
    return {Placement::Hash, ReporterIp(request.host_ip_port()), lifecycle_only};
}
} // namespace kv_cache_manager
