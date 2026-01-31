#include <string>

namespace kv_cache_manager {

std::string ExtractIpFromPeer(const std::string &peer) {
    constexpr size_t IPV4_PREFIX_LEN = 5; // strlen("ipv4:")
    constexpr size_t IPV6_PREFIX_LEN = 5; // strlen("ipv6:")

    if (peer.rfind("ipv4:", 0) == 0) {
        // 格式: ipv4:192.168.1.1:50051
        size_t second_colon = peer.find(':', IPV4_PREFIX_LEN); // IP 和端口之间的冒号
        if (second_colon != std::string::npos) {
            return peer.substr(IPV4_PREFIX_LEN, second_colon - IPV4_PREFIX_LEN);
        }
        // 如果没有找到第二个冒号，可能没有端口号
        return peer.substr(IPV4_PREFIX_LEN);
    } else if (peer.rfind("ipv6:", 0) == 0) {
        // 格式: ipv6:[2001:db8::1]:50051 或 ipv6:2001:db8::1
        auto start = peer.find('[');
        auto end = peer.find(']');
        if (start != std::string::npos && end != std::string::npos && end > start) {
            return peer.substr(start + 1, end - start - 1);
        }
        // 如果没有方括号，可能是不带端口的 IPv6 地址
        return peer.substr(IPV6_PREFIX_LEN);
    }
    return peer;
}

} // namespace kv_cache_manager
