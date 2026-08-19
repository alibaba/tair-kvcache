// The set of KVCM APIs the Swarm generator is allowed to call.
//
// Deployment-management operations (AddStorage, CreateInstanceGroup,
// UpdateInstanceGroup, ...) are intentionally absent: creating and destroying
// test resources belongs to integration_test/swarm, never to the generator.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace google {
namespace protobuf {
class Message;
} // namespace protobuf
} // namespace google

namespace kv_cache_manager::async_rpc {

enum class ServiceEndpoint {
    kMeta,
    kAdmin
};

const char *ServiceEndpointName(ServiceEndpoint endpoint);

enum class Api {
    kRegisterInstance,
    kGetCacheLocationsByBackend,
    kStartWriteCache,
    kFinishWriteCache,
    kReportEvent,
    kGetClusterInfo,
    kRemoveCache,
    kCheckHealth,
};

struct ApiInfo {
    Api api;
    std::string_view name;        // report/metric name
    std::string_view http_path;   // e.g. /api/registerInstance
    std::string_view grpc_method; // fully qualified gRPC method path
    ServiceEndpoint endpoint;
};

const ApiInfo &GetApiInfo(Api api);
const std::vector<ApiInfo> &AllApis();
std::string_view ApiName(Api api);

// Reads `header.status.code` from any KVCM response through reflection.
// Returns 0 when the message has no such field.
int ExtractServiceStatus(const google::protobuf::Message &response);

// Reads `header.status.message`; empty when absent.
std::string ExtractServiceMessage(const google::protobuf::Message &response);

// proto::meta::ErrorCode / proto::admin::ErrorCode values used by behaviors.
constexpr int kStatusOk = 1;
constexpr int kStatusServerNotLeader = 9;

} // namespace kv_cache_manager::async_rpc
