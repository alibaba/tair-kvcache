#include "stub_source/kv_cache_manager/common/vipserver_subscriber.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

bool VIPServerSubscriber::init(const std::string &domain) {
    KVCM_LOG_ERROR("no implementation for VIPServerSubscriber");
    return false;
}

bool VIPServerSubscriber::getOneAddress(std::string &address) const {
    KVCM_LOG_ERROR("no implementation for VIPServerSubscriber");
    return false;
}

bool VIPServerSubscriber::getAllAddresses(std::vector<std::string> &addresses) const {
    KVCM_LOG_ERROR("no implementation for VIPServerSubscriber");
    return false;
}

} // namespace kv_cache_manager