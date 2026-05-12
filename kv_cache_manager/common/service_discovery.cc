#include "kv_cache_manager/common/service_discovery.h"

#include <random>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

CachedServiceDiscovery::CachedServiceDiscovery(int cache_ttl_seconds) : cache_ttl_seconds_(cache_ttl_seconds) {
    cache_.valid = false;
}

bool CachedServiceDiscovery::GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) {
    // 1) 命中缓存的快路径：只在锁内读缓存。
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_.valid && !IsCacheExpiredLocked() && !cache_.endpoints.empty()) {
            endpoints = cache_.endpoints;
            return true;
        }
    }

    // 2) 缓存缺失/过期：在锁外做潜在慢 IO 的 FetchEndpoints，
    //    避免在持锁状态下进行网络调用。
    std::vector<ServiceEndpoint> new_endpoints;
    bool fetched = FetchEndpoints(new_endpoints);

    // 3) 写回缓存并取出结果，统一在一次加锁中完成。
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (fetched) {
        cache_.endpoints = std::move(new_endpoints);
        cache_.update_time = std::chrono::steady_clock::now();
        cache_.valid = true;
    }
    if (!cache_.valid || cache_.endpoints.empty()) {
        return false;
    }
    endpoints = cache_.endpoints;
    return true;
}

bool CachedServiceDiscovery::GetOneEndpoint(ServiceEndpoint &endpoint) {
    std::vector<ServiceEndpoint> endpoints;
    if (!GetAllEndpoints(endpoints) || endpoints.empty()) {
        return false;
    }

    // 简单随机负载均衡。后续可在 EndpointSelector 抽象上做加权/轮询/一致性哈希。
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_int_distribution<size_t> dis(0, endpoints.size() - 1);
    endpoint = endpoints[dis(gen)];
    return true;
}

bool CachedServiceDiscovery::Refresh() {
    std::vector<ServiceEndpoint> new_endpoints;
    if (!FetchEndpoints(new_endpoints)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.endpoints = std::move(new_endpoints);
    cache_.update_time = std::chrono::steady_clock::now();
    cache_.valid = true;
    return true;
}

bool CachedServiceDiscovery::IsCacheExpiredLocked() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - cache_.update_time).count();
    return elapsed >= cache_ttl_seconds_;
}

} // namespace kv_cache_manager
