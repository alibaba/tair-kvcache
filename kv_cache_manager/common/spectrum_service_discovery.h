#pragma once
#include <string>

#include "kv_cache_manager/common/service_discovery.h"

namespace kv_cache_manager {

/**
 * Spectrum 服务发现实现
 *
 * 通过 HTTP GET 请求 Spectrum 网关固定接口获取虚拟服务的实例列表：
 *   http://127.0.0.1:8880/api/v1/discovery/virtual-services/{id}/instances
 *
 * 调用方只需传入 virtual_service_id（如 "v-ad2d143d"），不再传入完整 URL。
 */
class SpectrumServiceDiscovery : public CachedServiceDiscovery {
public:
    SpectrumServiceDiscovery();
    ~SpectrumServiceDiscovery() override;

    /**
     * 初始化 Spectrum 服务发现。
     * @param virtual_service_id 虚拟服务 ID，例如 "v-ad2d143d"
     * @return 初始化是否成功（会触发首次拉取）
     */
    bool Init(const std::string &virtual_service_id) override;

    /** 服务发现类型名称。 */
    std::string GetType() const override { return "Spectrum"; }

    // 工厂层根据 URL 参数注入用的运行时配置。所有 setter 均需在 Init 之前调用。
    using CachedServiceDiscovery::SetCacheTtlSeconds;
    void SetRequestTimeoutMs(int timeout_ms) { request_timeout_ms_ = timeout_ms; }
    void SetRetryCount(int retry_count) { retry_count_ = retry_count; }

protected:
    /** 从 Spectrum 网关获取实例列表（实现父类纯虚函数）。 */
    bool FetchEndpoints(std::vector<ServiceEndpoint> &endpoints) override;

    // 可被子类 / 测试覆盖的端点配置。
    virtual std::string GetSpectrumHost() const { return SPECTRUM_HOST; }
    virtual int GetSpectrumPort() const { return SPECTRUM_PORT; }
    virtual int GetRequestTimeoutMs() const { return request_timeout_ms_; }

private:
    bool DoFetchOnce(std::vector<ServiceEndpoint> &endpoints);

    std::string virtual_service_id_;
    int request_timeout_ms_ = REQUEST_TIMEOUT_MS;
    int retry_count_ = 0; // 0 表示仅尝试一次

    static constexpr const char *SPECTRUM_HOST = "127.0.0.1";
    static constexpr int SPECTRUM_PORT = 8880;
    static constexpr int REQUEST_TIMEOUT_MS = 5000;
};

} // namespace kv_cache_manager
