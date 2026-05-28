#include "kv_cache_manager/common/service_discovery_factory.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/service_discovery_url.h"
#include "stub_source/kv_cache_manager/common/spectrum_service_discovery.h"
#include "kv_cache_manager/common/static_service_discovery.h"
#include "stub_source/kv_cache_manager/common/vipserver_service_discovery.h"

namespace kv_cache_manager {

namespace {

constexpr const char *kSchemeVipserver = "vipserver";
constexpr const char *kSchemeSpectrum = "spectrum";
constexpr const char *kSchemeStatic = "static";

std::unique_ptr<ServiceDiscovery> CreateVipserver(const ServiceDiscoveryUrl &url_info) {
    if (url_info.body.empty()) {
        KVCM_LOG_WARN("vipserver domain is empty");
        return nullptr;
    }
    auto discovery = std::make_unique<VIPServerServiceDiscovery>();
    if (const int timeout_sec = url_info.GetIntParam("timeout", 0); timeout_sec > 0) {
        discovery->SetQueryTimeoutMs(timeout_sec * 1000);
    }
    if (!discovery->Init(url_info.body)) {
        KVCM_LOG_WARN("vipserver init fail, domain=[%s]", url_info.body.c_str());
        return nullptr;
    }
    return discovery;
}

std::unique_ptr<ServiceDiscovery> CreateSpectrum(const ServiceDiscoveryUrl &url_info) {
    auto discovery = std::make_unique<SpectrumServiceDiscovery>();
    if (const int cache_time_sec = url_info.GetIntParam("cache_time", 0); cache_time_sec > 0) {
        discovery->SetCacheTtlSeconds(cache_time_sec);
    }
    if (const int timeout_ms = url_info.GetIntParam("timeout", 0); timeout_ms > 0) {
        discovery->SetRequestTimeoutMs(timeout_ms);
    }
    if (const int retry_time = url_info.GetIntParam("retry_time", 0); retry_time > 0) {
        discovery->SetRetryCount(retry_time);
    }
    // port 表示自定义端口号；配置后会强制覆盖 Spectrum 网关返回的端口。
    if (const int custom_port = url_info.GetIntParam("port", 0); custom_port > 0) {
        discovery->SetCustomPort(custom_port);
    }
    if (!discovery->Init(url_info.body)) {
        KVCM_LOG_WARN("spectrum init fail, virtual_service_id=[%s]", url_info.body.c_str());
        return nullptr;
    }
    return discovery;
}

std::unique_ptr<ServiceDiscovery> CreateStatic(const ServiceDiscoveryUrl &url_info) {
    auto discovery = std::make_unique<StaticServiceDiscovery>();
    if (!discovery->Init(url_info.body)) {
        KVCM_LOG_WARN("static service discovery init fail, host_list=[%s]", url_info.body.c_str());
        return nullptr;
    }
    return discovery;
}

} // namespace

std::unique_ptr<ServiceDiscovery> ServiceDiscoveryFactory::CreateServiceDiscovery(const std::string &url) {
    if (url.empty()) {
        return nullptr;
    }
    ServiceDiscoveryUrl url_info;
    if (!ServiceDiscoveryUrl::Parse(url, url_info)) {
        return nullptr;
    }
    if (url_info.scheme == kSchemeVipserver) {
        return CreateVipserver(url_info);
    }
    if (url_info.scheme == kSchemeSpectrum) {
        return CreateSpectrum(url_info);
    }
    if (url_info.scheme == kSchemeStatic) {
        return CreateStatic(url_info);
    }
    KVCM_LOG_WARN("unsupported service discovery scheme=[%s], url=[%s]", url_info.scheme.c_str(), url.c_str());
    return nullptr;
}

} // namespace kv_cache_manager
