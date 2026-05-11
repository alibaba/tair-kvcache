#include "stub_source/kv_cache_manager/common/vipserver_subscriber.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

bool VIPServerSubscriber::Init(const std::string &service_address) {
    KVCM_LOG_ERROR("no implementation for VIPServerSubscriber::Init, service_address=%s", service_address.c_str());
    return false;
}

bool VIPServerSubscriber::GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) {
    KVCM_LOG_ERROR("no implementation for VIPServerSubscriber::GetAllEndpoints");
    endpoints.clear();
    return false;
}

bool VIPServerSubscriber::GetOneEndpoint(ServiceEndpoint &endpoint) {
    KVCM_LOG_ERROR("no implementation for VIPServerSubscriber::GetOneEndpoint");
    return false;
}

bool VIPServerSubscriber::Refresh() {
    KVCM_LOG_ERROR("no implementation for VIPServerSubscriber::Refresh");
    return false;
}

} // namespace kv_cache_manager
