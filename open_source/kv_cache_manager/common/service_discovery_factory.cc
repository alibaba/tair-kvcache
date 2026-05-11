#include "stub_source/kv_cache_manager/common/service_discovery_factory.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/service_discovery_url.h"
#include "kv_cache_manager/common/static_service_discovery.h"

namespace kv_cache_manager {

namespace {

constexpr const char *kSchemeVipserver = "vipserver";
constexpr const char *kSchemeSpectrum = "spectrum";
constexpr const char *kSchemeStatic = "static";

} // namespace

std::unique_ptr<ServiceDiscovery> ServiceDiscoveryFactory::CreateServiceDiscovery(const std::string &url) {
    if (url.empty()) {
        return nullptr;
    }
    ServiceDiscoveryUrl url_info;
    if (!ServiceDiscoveryUrl::Parse(url, url_info)) {
        return nullptr;
    }
    if (url_info.scheme == kSchemeStatic) {
        auto discovery = std::make_unique<StaticServiceDiscovery>();
        if (!discovery->Init(url_info.body)) {
            KVCM_LOG_ERROR("static service discovery init fail, host_list=[%s]", url_info.body.c_str());
            return nullptr;
        }
        return discovery;
    }
    if (url_info.scheme == kSchemeVipserver || url_info.scheme == kSchemeSpectrum) {
        KVCM_LOG_ERROR("no implementation for CreateServiceDiscovery in opensource build, scheme=[%s], url=[%s]",
                       url_info.scheme.c_str(),
                       url.c_str());
        return nullptr;
    }
    KVCM_LOG_ERROR("unsupported service discovery scheme=[%s], url=[%s]", url_info.scheme.c_str(), url.c_str());
    return nullptr;
}

} // namespace kv_cache_manager
