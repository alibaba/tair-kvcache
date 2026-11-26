#include "kv_cache_manager/service/proxy/meta_forwarder.h"

#include <grpcpp/grpcpp.h>

#include "kv_cache_manager/common/timestamp_util.h"

namespace kv_cache_manager {

proto::meta::Status *InitProxyStatus(const std::string &trace_id, proto::meta::CommonResponseHeader *header) {
    header->set_request_id(trace_id + "_" + std::to_string(TimestampUtil::GetCurrentTimeUs()));
    return header->mutable_status();
}

MetaForwarder::MetaForwarder(RoutingTable &routes,
                             proto::meta::MetaNodeEndpoint self,
                             std::chrono::milliseconds lease,
                             size_t max_inflight)
    : routes_(routes), self_(std::move(self)), lease_(lease), max_inflight_(max_inflight), poller_([this] {
        void *tag;
        bool ok;
        while (queue_.Next(&tag, &ok)) {
            std::unique_ptr<Completion> completion(static_cast<Completion *>(tag));
            completion->Finish(ok);
        }
    }) {}

MetaForwarder::~MetaForwarder() {
    queue_.Shutdown();
    poller_.join();
}

void MetaForwarder::Close() {
    std::lock_guard lock(mutex_);
    closed_ = true;
    for (auto *context : inflight_) {
        context->TryCancel();
    }
}

template <typename Response>
struct MetaForwarder::Call final : Completion {
    Call(MetaForwarder &owner, Done done) : owner(owner), done(std::move(done)) {}
    void Finish(bool ok) override {
        {
            std::lock_guard lock(owner.mutex_);
            owner.inflight_.erase(&context);
        }
        done(ok ? status : grpc::Status(grpc::StatusCode::UNAVAILABLE, "downstream RPC interrupted"));
    }

    MetaForwarder &owner;
    Done done;
    grpc::ClientContext context;
    grpc::Status status;
    std::shared_ptr<proto::meta::MetaService::Stub> stub;
    std::unique_ptr<grpc::ClientAsyncResponseReader<Response>> reader;
};

template <typename Request, typename Response, typename Start>
void MetaForwarder::Send(
    const char *method, const Request &request, Response &response, Deadline deadline, Done done, Start start) {
    const auto route = routes_.Resolve({method, request.instance_id()}, RoutingTable::Clock::now(), lease_);
    if (!route) {
        auto *status = InitProxyStatus(request.trace_id(), response.mutable_header());
        status->set_code(request.instance_id().empty() ? proto::meta::INVALID_ARGUMENT
                                                       : proto::meta::SERVICE_NOT_READY);
        status->set_message("no serving route for request");
        done(grpc::Status::OK);
        return;
    }
    auto call = std::make_unique<Call<Response>>(*this, std::move(done));
    if (deadline == Deadline::max()) {
        deadline = std::chrono::system_clock::now() + std::chrono::seconds(30);
    }
    call->context.set_deadline(deadline);
    call->context.AddMetadata("kvcm-route-version", std::to_string(route->version));
    call->context.AddMetadata("kvcm-partition", std::to_string(route->partition));
    std::unique_lock lock(mutex_);
    if (closed_ || inflight_.size() >= max_inflight_) {
        const auto code = closed_ ? proto::meta::SERVICE_NOT_READY : proto::meta::SERVICE_BUSY;
        lock.unlock();
        InitProxyStatus(request.trace_id(), response.mutable_header())->set_code(code);
        call->done(grpc::Status::OK);
        return;
    }
    auto &peer = peers_[route->partition];
    if (peer.endpoint != route->endpoint) {
        peer.stub =
            proto::meta::MetaService::NewStub(grpc::CreateChannel(route->endpoint, grpc::InsecureChannelCredentials()));
        peer.endpoint = route->endpoint;
    }
    call->stub = peer.stub;
    call->reader = start(*call->stub, &call->context, request, &queue_);
    inflight_.insert(&call->context);
    auto *tag = call.release();
    tag->reader->Finish(&response, &tag->status, static_cast<Completion *>(tag));
}

#define FORWARD_RPC(method, response_type)                                                                             \
    void MetaForwarder::Forward(const proto::meta::method##Request &request,                                           \
                                proto::meta::response_type &response,                                                  \
                                Deadline deadline,                                                                     \
                                Done done) {                                                                           \
        Send(#method,                                                                                                  \
             request,                                                                                                  \
             response,                                                                                                 \
             deadline,                                                                                                 \
             std::move(done),                                                                                          \
             [](auto &stub, auto *context, const auto &req, auto *queue) {                                             \
                 return stub.Async##method(context, req, queue);                                                       \
             });                                                                                                       \
    }
FORWARD_RPC(RegisterInstance, RegisterInstanceResponse)
FORWARD_RPC(GetInstanceInfo, GetInstanceInfoResponse)
FORWARD_RPC(GetCacheMeta, GetCacheMetaResponse)
FORWARD_RPC(GetCacheLocation, GetCacheLocationResponse)
FORWARD_RPC(GetCacheLocationsByBackend, GetCacheLocationsByBackendResponse)
FORWARD_RPC(GetCacheLocationLen, GetCacheLocationLenResponse)
FORWARD_RPC(StartWriteCache, StartWriteCacheResponse)
FORWARD_RPC(FinishWriteCache, CommonResponse)
FORWARD_RPC(RemoveCache, CommonResponse)
FORWARD_RPC(TrimCache, CommonResponse)
FORWARD_RPC(ReportEvent, ReportEventResponse)
FORWARD_RPC(GetHostCacheState, GetHostCacheStateResponse)
#undef FORWARD_RPC

void MetaForwarder::Forward(const proto::meta::GetClusterInfoRequest &request,
                            proto::meta::GetClusterInfoResponse &response,
                            Deadline,
                            Done done) {
    InitProxyStatus(request.trace_id(), response.mutable_header())->set_code(proto::meta::OK);
    response.set_self_node_id(self_.node_id());
    response.set_leader_node_id(self_.node_id());
    *response.mutable_leader_endpoint() = self_;
    done(grpc::Status::OK);
}
} // namespace kv_cache_manager
