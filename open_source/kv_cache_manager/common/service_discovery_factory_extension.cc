#include "kv_cache_manager/common/service_discovery_factory_extension.h"

namespace kv_cache_manager {

std::unique_ptr<ServiceDiscovery> CreateServiceDiscoveryExtension(const ServiceDiscoveryUrl & /*url_info*/) {
    return nullptr;
}

} // namespace kv_cache_manager
