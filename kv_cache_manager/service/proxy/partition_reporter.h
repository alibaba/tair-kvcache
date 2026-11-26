#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "kv_cache_manager/common/service_discovery.h"
#include "kv_cache_manager/protocol/protobuf/proxy_control_service.grpc.pb.h"
#include "kv_cache_manager/routing/routing_table.h"

namespace kv_cache_manager {
class PartitionReporter {
public:
    bool Init(const std::string &target_file,
              const std::string &discovery_uri,
              std::string cluster,
              uint32_t version,
              uint32_t columns);
    void Start(std::string endpoint, std::function<bool()> serving);
    void Stop();
    ~PartitionReporter() { Stop(); }
    const proto::proxy::PartitionTarget &target() const { return target_; }
    bool Accepts(const RouteKey &route) const;

private:
    proto::proxy::PartitionTarget target_;
    std::string cluster_;
    uint32_t version_ = 0;
    uint32_t columns_ = 0;
    std::unique_ptr<ServiceDiscovery> discovery_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
};
} // namespace kv_cache_manager
