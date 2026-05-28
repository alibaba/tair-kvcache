#pragma once
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace kv_cache_manager {

/**
 * 统一的服务节点数据结构。
 * 不同服务发现实现（VIPServer / Spectrum 等）都使用此结构暴露端点。
 */
struct ServiceEndpoint {
    std::string ip;
    int port;
    std::string host; // ip:port，方便直接当字符串地址使用
    int weight;
    bool healthy;

    ServiceEndpoint() : port(0), weight(100), healthy(true) {}

    ServiceEndpoint(const std::string &ip, int port, int weight = 100)
        : ip(ip), port(port), host(ip + ":" + std::to_string(port)), weight(weight), healthy(true) {}
};

/**
 * 服务发现抽象基类。
 * 为不同的服务发现机制（VIPServer、EAS 等）提供统一接口。
 */
class ServiceDiscovery {
public:
    virtual ~ServiceDiscovery() = default;

    /**
     * 初始化服务发现。
     * @param service_address 服务地址（domain 或 URL，由具体实现决定语义）
     */
    virtual bool Init(const std::string &service_address) = 0;

    /** 获取所有可用服务节点。 */
    virtual bool GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) = 0;

    /** 获取单个服务节点（由实现自行决定负载均衡策略）。 */
    virtual bool GetOneEndpoint(ServiceEndpoint &endpoint) = 0;

    /** 强制刷新底层数据（具体语义由实现决定）。 */
    virtual bool Refresh() = 0;

    /** 服务发现类型名称（如 "VIPServer"、"EAS"）。 */
    virtual std::string GetType() const = 0;

protected:
    ServiceDiscovery() = default;
};

/**
 * 带本地 TTL 缓存的服务发现基类。
 * 适用于"按需拉取 + 短期缓存"型的服务发现实现（如 EAS 走 HTTP 拉端点列表）。
 *
 * 子类只需实现 FetchEndpoints；缓存生命周期、并发安全、负载均衡由本基类承担。
 *
 * 不适用于本身已带订阅式缓存的实现（如 VIPServer SDK），那类实现应直接继承
 * ServiceDiscovery。
 */
class CachedServiceDiscovery : public ServiceDiscovery {
public:
    bool GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) override;
    bool GetOneEndpoint(ServiceEndpoint &endpoint) override;
    bool Refresh() override;

protected:
    explicit CachedServiceDiscovery(int cache_ttl_seconds = 30);
    ~CachedServiceDiscovery() override = default;

    /**
     * 从服务源拉取端点列表（子类必须实现）。
     * 实现可以做 HTTP 请求等慢 IO，本基类保证调用此函数时不持有内部锁。
     */
    virtual bool FetchEndpoints(std::vector<ServiceEndpoint> &endpoints) = 0;

    /**
     * 运行时调整缓存 TTL。供工厂层根据 URL 中的 cache_time 参数注入。
     * 仅会影响后续的过期判断，不会主动失效当前缓存。
     */
    void SetCacheTtlSeconds(int cache_ttl_seconds) { cache_ttl_seconds_ = cache_ttl_seconds; }

private:
    bool IsCacheExpiredLocked() const;

    struct CacheEntry {
        std::vector<ServiceEndpoint> endpoints;
        std::chrono::steady_clock::time_point update_time;
        bool valid;
    };

    std::mutex cache_mutex_;
    CacheEntry cache_;
    int cache_ttl_seconds_;
};

} // namespace kv_cache_manager
