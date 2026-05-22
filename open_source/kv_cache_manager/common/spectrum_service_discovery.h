#pragma once
#include <string>

#include "kv_cache_manager/common/service_discovery.h"

namespace kv_cache_manager {

/**
 * 开源构建下的 SpectrumServiceDiscovery 占位实现。
 * 接口形态与 internal 端完全一致（继承 CachedServiceDiscovery）；运行期所有
 * 方法都返回失败，以便没有 Spectrum 网关的开源环境下保持链接通过 + 运行时清晰报错。
 */
class SpectrumServiceDiscovery : public CachedServiceDiscovery {
public:
    SpectrumServiceDiscovery();
    ~SpectrumServiceDiscovery() override;

    bool Init(const std::string &virtual_service_id) override;
    std::string GetType() const override { return "Spectrum"; }

    // 占位 setter：开源构建保持 ABI 与 internal 一致。
    using CachedServiceDiscovery::SetCacheTtlSeconds;
    void SetRequestTimeoutMs(int /*timeout_ms*/) {}
    void SetRetryCount(int /*retry_count*/) {}
    void SetCustomPort(int /*custom_port*/) {}

protected:
    bool FetchEndpoints(std::vector<ServiceEndpoint> &endpoints) override;
};

} // namespace kv_cache_manager
