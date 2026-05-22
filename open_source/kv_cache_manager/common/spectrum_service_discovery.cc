#include "stub_source/kv_cache_manager/common/spectrum_service_discovery.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

SpectrumServiceDiscovery::SpectrumServiceDiscovery() : CachedServiceDiscovery(30) {}

SpectrumServiceDiscovery::~SpectrumServiceDiscovery() {}

bool SpectrumServiceDiscovery::Init(const std::string &virtual_service_id) {
    KVCM_LOG_ERROR("no implementation for SpectrumServiceDiscovery::Init, virtual_service_id=%s",
                   virtual_service_id.c_str());
    return false;
}

bool SpectrumServiceDiscovery::FetchEndpoints(std::vector<ServiceEndpoint> &endpoints) {
    KVCM_LOG_ERROR("no implementation for SpectrumServiceDiscovery::FetchEndpoints");
    endpoints.clear();
    return false;
}

} // namespace kv_cache_manager
