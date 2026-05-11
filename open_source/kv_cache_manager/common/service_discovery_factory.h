#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/common/service_discovery.h"

namespace kv_cache_manager {

class ServiceDiscoveryFactory {
public:
    static std::unique_ptr<ServiceDiscovery> CreateServiceDiscovery(const std::string &url);
};

} // namespace kv_cache_manager
