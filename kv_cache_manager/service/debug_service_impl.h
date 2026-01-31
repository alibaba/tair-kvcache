#pragma once

#include <memory>

#include "kv_cache_manager/protocol/protobuf/debug_service.pb.h"

namespace kv_cache_manager {

class CacheManager;

class DebugServiceImpl {
public:
    explicit DebugServiceImpl(std::shared_ptr<CacheManager> cache_manager) : cache_manager_(std::move(cache_manager)) {}
    ~DebugServiceImpl() = default;

    void InjectFault(const proto::debug::InjectFaultRequest *request, proto::debug::CommonResponse *response);

    void RemoveFault(const proto::debug::RemoveFaultRequest *request, proto::debug::CommonResponse *response);

    void ClearFaults(const proto::debug::ClearFaultsRequest *request, proto::debug::CommonResponse *response);

private:
    std::shared_ptr<CacheManager> cache_manager_;
};

} // namespace kv_cache_manager
