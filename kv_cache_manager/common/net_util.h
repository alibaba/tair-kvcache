#pragma once

#include <string>

namespace kv_cache_manager {

class NetUtil {
public:
    // 获取本机默认路由出口的 IPv4 地址
    // 失败时返回 "127.0.0.1"
    static std::string GetLocalIp();

private:
    static std::string GetIpByDefaultRoute();
    static std::string GetIpByIfaddrs();
};

} // namespace kv_cache_manager
