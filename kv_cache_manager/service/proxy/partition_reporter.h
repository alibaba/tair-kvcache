#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>

#include "kv_cache_manager/common/service_discovery.h"
#include "kv_cache_manager/protocol/protobuf/proxy_control_service.grpc.pb.h"

namespace kv_cache_manager {
class PartitionReporter {
public:
    bool Init(const std::string &target_file, const std::string &discovery_uri);
    void Start(std::string endpoint, std::function<bool()> serving);
    void Stop();
    ~PartitionReporter() { Stop(); }
    const proto::proxy::PartitionTarget &target() const { return target_; }
    bool OwnsInstance(std::string_view instance) const;

private:
    proto::proxy::PartitionTarget target_;
    std::unique_ptr<ServiceDiscovery> discovery_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
};
} // namespace kv_cache_manager
