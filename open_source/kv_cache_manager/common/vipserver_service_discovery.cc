#include "stub_source/kv_cache_manager/common/vipserver_service_discovery.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

bool VIPServerServiceDiscovery::Init(const std::string &service_address) {
    KVCM_LOG_ERROR("no implementation for VIPServerServiceDiscovery::Init, service_address=%s", service_address.c_str());
    return false;
}

bool VIPServerServiceDiscovery::GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) {
    KVCM_LOG_ERROR("no implementation for VIPServerServiceDiscovery::GetAllEndpoints");
    endpoints.clear();
    return false;
}

bool VIPServerServiceDiscovery::GetOneEndpoint(ServiceEndpoint &endpoint) {
    KVCM_LOG_ERROR("no implementation for VIPServerServiceDiscovery::GetOneEndpoint");
    return false;
}

bool VIPServerServiceDiscovery::Refresh() {
    KVCM_LOG_ERROR("no implementation for VIPServerServiceDiscovery::Refresh");
    return false;
}

} // namespace kv_cache_manager
