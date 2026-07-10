#include "kv_cache_manager/common/service_discovery_factory.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/service_discovery_factory_extension.h"
#include "kv_cache_manager/common/service_discovery_url.h"
#include "kv_cache_manager/common/static_service_discovery.h"

namespace kv_cache_manager {

namespace {

constexpr const char *kSchemeStatic = "static";

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
    if (url_info.scheme == kSchemeStatic) {
        return CreateStatic(url_info);
    }
    auto discovery = CreateServiceDiscoveryExtension(url_info);
    if (discovery != nullptr) {
        return discovery;
    }
    KVCM_LOG_WARN("unsupported service discovery scheme=[%s], url=[%s]", url_info.scheme.c_str(), url.c_str());
    return nullptr;
}

} // namespace kv_cache_manager
