#pragma once
#include <string>
#include <vector>

#include "kv_cache_manager/common/service_discovery.h"

namespace kv_cache_manager {

/**
 * 开源构建下的 VIPServerServiceDiscovery 占位实现。
 * 接口形态与 internal 端完全一致（继承 ServiceDiscovery），仅运行期所有方法都返回失败，
 * 以便在没有 VIP SDK 的开源环境下保持链接通过 + 运行时清晰报错。
 */
class VIPServerServiceDiscovery : public ServiceDiscovery {
public:
    VIPServerServiceDiscovery() = default;
    ~VIPServerServiceDiscovery() override = default;

    bool Init(const std::string &service_address) override;
    bool GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) override;
    bool GetOneEndpoint(ServiceEndpoint &endpoint) override;
    bool Refresh() override;
    std::string GetType() const override { return "VIPServer"; }

    // 占位 setter：开源构建保持 ABI 与 internal 一致。
    void SetQueryTimeoutMs(int /*timeout_ms*/) {}
};

} // namespace kv_cache_manager
