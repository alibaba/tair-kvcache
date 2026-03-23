#pragma once

#include <string>

namespace kv_cache_manager {

class NetUtil {
public:
    // 获取本机第一个非 loopback 的 IPv4 地址
    // 失败时返回 "127.0.0.1"
    static std::string GetLocalIp();
};

} // namespace kv_cache_manager
