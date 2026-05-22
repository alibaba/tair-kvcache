#include "kv_cache_manager/common/net_util.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

namespace kv_cache_manager {

std::string NetUtil::GetLocalIp() {
    std::string ip = GetIpByDefaultRoute();
    if (ip.empty()) {
        ip = GetIpByIfaddrs();
    }
    return ip.empty() ? "127.0.0.1" : ip;
}

// UDP socket connect 不会真正发包，仅利用内核路由表选路来确定出口 IP
std::string NetUtil::GetIpByDefaultRoute() {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        return "";
    }

    struct sockaddr_in remote;
    memset(&remote, 0, sizeof(remote));
    remote.sin_family = AF_INET;
    remote.sin_port = htons(80);
    inet_pton(AF_INET, "8.8.8.8", &remote.sin_addr);

    if (connect(fd, reinterpret_cast<struct sockaddr *>(&remote), sizeof(remote)) != 0) {
        close(fd);
        return "";
    }

    struct sockaddr_in local;
    socklen_t len = sizeof(local);
    if (getsockname(fd, reinterpret_cast<struct sockaddr *>(&local), &len) != 0) {
        close(fd);
        return "";
    }
    close(fd);

    char buf[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf)) == nullptr) {
        return "";
    }

    // Reject loopback — fallback to ifaddrs instead
    if (strncmp(buf, "127.", 4) == 0) {
        return "";
    }
    return buf;
}

std::string NetUtil::GetIpByIfaddrs() {
    struct ifaddrs *ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) {
        return "";
    }

    std::string result;
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
