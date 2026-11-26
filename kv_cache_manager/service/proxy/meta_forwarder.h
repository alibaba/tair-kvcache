#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/routing/routing_table.h"

namespace kv_cache_manager {

proto::meta::Status *InitProxyStatus(const std::string &trace_id, proto::meta::CommonResponseHeader *header);

// Both protocol adapters retain the inbound request until Done runs. A dedicated
// client completion queue drives downstream calls after the ingress worker returns.
class MetaForwarder {
public:
    using Done = std::function<void(grpc::Status)>;
    using Deadline = std::chrono::system_clock::time_point;
    MetaForwarder(RoutingTable &routes,
                  proto::meta::MetaNodeEndpoint self,
                  std::chrono::milliseconds lease,
                  size_t max_inflight);
    ~MetaForwarder();
    void Close();
    MetaForwarder(const MetaForwarder &) = delete;
    MetaForwarder &operator=(const MetaForwarder &) = delete;

    void Forward(const proto::meta::RegisterInstanceRequest &request,
                 proto::meta::RegisterInstanceResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::GetInstanceInfoRequest &request,
                 proto::meta::GetInstanceInfoResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::GetCacheMetaRequest &request,
                 proto::meta::GetCacheMetaResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::GetCacheLocationRequest &request,
                 proto::meta::GetCacheLocationResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::GetCacheLocationsByBackendRequest &request,
                 proto::meta::GetCacheLocationsByBackendResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::GetCacheLocationLenRequest &request,
                 proto::meta::GetCacheLocationLenResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::StartWriteCacheRequest &request,
                 proto::meta::StartWriteCacheResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::FinishWriteCacheRequest &request,
                 proto::meta::CommonResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::RemoveCacheRequest &request,
                 proto::meta::CommonResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::TrimCacheRequest &request,
                 proto::meta::CommonResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::ReportEventRequest &request,
                 proto::meta::ReportEventResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::GetHostCacheStateRequest &request,
                 proto::meta::GetHostCacheStateResponse &response,
                 Deadline deadline,
                 Done done);
    void Forward(const proto::meta::GetClusterInfoRequest &request,
                 proto::meta::GetClusterInfoResponse &response,
                 Deadline deadline,
                 Done done);

private:
    struct Completion {
        virtual ~Completion() = default;
        virtual void Finish(bool ok) = 0;
    };
    template <typename Response>
    struct Call;
    template <typename Request, typename Response, typename Start>
    void Send(const Request &request, Response &response, Deadline deadline, Done done, Start start);

    template <typename Request, typename Response, typename Start>
    void
    SendOne(const Route &route, const Request &request, Response &response, Deadline deadline, Done done, Start start);

    RoutingTable &routes_;
    const proto::meta::MetaNodeEndpoint self_;
    const std::chrono::milliseconds lease_;
    const size_t max_inflight_;
    std::mutex mutex_;
    struct Peer {
        std::string endpoint;
        std::shared_ptr<proto::meta::MetaService::Stub> stub;
    };
    std::unordered_map<PartitionId, Peer> peers_;
    bool closed_ = false;
    std::unordered_set<grpc::ClientContext *> inflight_;
    grpc::CompletionQueue queue_;
    std::thread poller_;
};
} // namespace kv_cache_manager
