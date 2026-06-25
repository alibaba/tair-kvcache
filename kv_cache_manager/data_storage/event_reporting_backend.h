#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class EventReportingBackend {
public:
    using CleanupCallback =
        std::function<void(const std::string &instance_id, const std::string &host_ip_port, uint64_t generation)>;

    virtual ~EventReportingBackend() = default;

    virtual ErrorCode RegisterNode(const std::string &instance_id,
                                   const std::string &host_ip_port,
                                   const std::vector<std::string> &mediums) = 0;
    virtual ErrorCode UnregisterNode(const std::string &instance_id, const std::string &host_ip_port) = 0;
    virtual ErrorCode OnHeartbeat(const std::string &instance_id,
                                  const std::string &host_ip_port,
                                  const std::map<std::string, std::string> &system_status) = 0;
    virtual void SetNodeUnavailable(const std::string &instance_id, const std::string &host_ip_port) = 0;
    virtual uint64_t GetNodeGeneration(const std::string &instance_id, const std::string &host_ip_port) const = 0;

    virtual void SetCleanupCallback(CleanupCallback cb) = 0;
    virtual bool IsCleanupCallbackSet() const = 0;

    virtual std::string BuildLocationId(const std::string &medium, const std::string &host_ip_port) const = 0;
    virtual std::string HostSuffix(const std::string &host_ip_port) const = 0;

    virtual DataStorageType GetStorageType() const = 0;

    virtual std::string GetProtocol() const = 0;
};

} // namespace kv_cache_manager
