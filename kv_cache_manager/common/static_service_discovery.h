#pragma once
#include <atomic>
#include <string>
#include <vector>

#include "kv_cache_manager/common/service_discovery.h"

namespace kv_cache_manager {

/**
 * 静态服务发现：把一组事先已知的 ip:port 当成"服务发现结果"返回。
 *
 * 适用场景：
 *   - 测试 / 单机部署直接写死 endpoint 列表
 *   - 不需要动态订阅但又想统一走 ServiceDiscovery 抽象的下游
 *
 * 行为：
 *   - GetAllEndpoints 返回构造时的全部端点
 *   - GetOneEndpoint 走 round-robin（atomic 计数器，线程安全）
 *   - Refresh 是 no-op，端点不会变更
 *
 * URL 形式（由工厂解析后注入）：
 *   static://11.22.33.44:8080,33.55.66.77:8080
 */
class StaticServiceDiscovery : public ServiceDiscovery {
public:
    StaticServiceDiscovery() = default;
    ~StaticServiceDiscovery() override = default;

    /**
     * 通用接口：service_address 形如 "ip1:port1,ip2:port2,..."。
     * 任一端点解析失败（缺 port / port 非数）都会让 Init 返回 false。
     */
    bool Init(const std::string &service_address) override;

    /** 直接用已经解析好的端点列表初始化；常用于工厂注入。 */
    bool InitWithEndpoints(std::vector<ServiceEndpoint> endpoints);

    bool GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) override;
    bool GetOneEndpoint(ServiceEndpoint &endpoint) override;

    /** Static 实现没有外部数据源，Refresh 永远成功且无副作用。 */
    bool Refresh() override { return !endpoints_.empty(); }

    std::string GetType() const override { return "Static"; }

    /**
     * 工具函数：把 "ip1:port1,ip2:port2" 这种字符串解析成 ServiceEndpoint 列表。
     * 任一段无效返回 false 并清空 out。
     */
    static bool ParseHostPortList(const std::string &host_list, std::vector<ServiceEndpoint> &out);

private:
    std::vector<ServiceEndpoint> endpoints_;
    std::atomic<size_t> rr_counter_{0};
};

} // namespace kv_cache_manager
