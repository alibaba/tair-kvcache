#include "kv_cache_manager/common/net_util.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <string.h>

namespace kv_cache_manager {

std::string NetUtil::GetLocalIp() {
    struct ifaddrs *ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) {
        return "127.0.0.1";
    }

    std::string result = "127.0.0.1";
    for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }
        // 只关注 IPv4
        if (ifa->ifa_addr->sa_family != AF_INET) {
            continue;
        }
        // 跳过 loopback
        if (ifa->ifa_flags & IFF_LOOPBACK) {
            continue;
        }
        // 只选择 UP 状态的接口
        if (!(ifa->ifa_flags & IFF_UP)) {
            continue;
        }

        char buf[INET_ADDRSTRLEN];
        struct sockaddr_in *addr = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
        if (inet_ntop(AF_INET, &addr->sin_addr, buf, sizeof(buf)) != nullptr) {
            result = buf;
            break;
        }
    }

    freeifaddrs(ifaddr);
    return result;
}

} // namespace kv_cache_manager
