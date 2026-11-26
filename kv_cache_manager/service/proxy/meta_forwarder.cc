#include "kv_cache_manager/service/proxy/meta_forwarder.h"

#include <algorithm>
#include <atomic>
#include <grpcpp/grpcpp.h>

#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/service/proxy/broadcast_result.h"
#include "kv_cache_manager/service/proxy/meta_route.h"

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

namespace {

template <typename Response>
struct Broadcast {
    explicit Broadcast(size_t count) : remaining(count), replies(count), statuses(count) {}
    std::atomic<size_t> remaining;
    std::vector<Response> replies;
    std::vector<grpc::Status> statuses;
};
} // namespace

template <typename Request, typename Response, typename Start>
void MetaForwarder::Send(const Request &request, Response &response, Deadline deadline, Done done, Start start) {
    if (deadline == Deadline::max()) {
        deadline = std::chrono::system_clock::now() + std::chrono::seconds(30);
    }
    const auto key = MetaRoute(request);
    auto reject = [&](auto code, const char *message) {
        auto *error = InitProxyStatus(request.trace_id(), response.mutable_header());
        error->set_code(code);
        error->set_message(message);
        done(grpc::Status::OK);
    };
    if (key.placement == Placement::Hash && key.key.empty()) {
        reject(proto::meta::INVALID_ARGUMENT, "missing or invalid request routing key");
        return;
    }
    auto now = RoutingTable::Clock::now();
    auto routes = routes_.Resolve(key, now, lease_);
    if (routes.empty()) {
        reject(proto::meta::SERVICE_NOT_READY, "no ready leader for required partition");
        return;
    }
    if (routes.size() == 1) {
        SendOne(routes.front(), request, response, deadline, std::move(done), start);
        return;
    }
    if constexpr (std::is_same_v<Request, proto::meta::GetHostCacheStateRequest>) {
        if (request.top_k_host_count() > 0 && request.backend_types_size()) {
            reject(proto::meta::UNSUPPORTED, "cross-partition remote prefix evidence is required for top-k");
            return;
        }
    }
    if (routes.size() > max_inflight_) {
        reject(proto::meta::SERVICE_BUSY, "broadcast exceeds downstream concurrency limit");
        return;
    }
    auto group = std::make_shared<Broadcast<Response>>(routes.size());
    for (size_t i = 0; i < routes.size(); ++i) {
        auto complete = [group, i, &response, done](grpc::Status status) {
            group->statuses[i] = std::move(status);
            if (group->remaining.fetch_sub(1) != 1) {
                return;
            }
            for (size_t j = 0; j < group->replies.size(); ++j) {
                if (!group->statuses[j].ok() || group->replies[j].header().status().code() != proto::meta::OK) {
                    response.Swap(&group->replies[j]);
                    done(group->statuses[j]);
                    return;
                }
            }
            if (!MergeReplies(group->replies, response)) {
                auto *header = response.mutable_header();
                *header = group->replies.front().header();
                header->mutable_status()->set_code(proto::meta::INVALID_ARGUMENT);
                header->mutable_status()->set_message("incompatible broadcast responses");
            }
            done(grpc::Status::OK);
        };
        SendOne(routes[i], request, group->replies[i], deadline, std::move(complete), start);
    }
}

template <typename Request, typename Response, typename Start>
void MetaForwarder::SendOne(
    const Route &route, const Request &request, Response &response, Deadline deadline, Done done, Start start) {
    auto call = std::make_unique<Call<Response>>(*this, std::move(done));
    call->context.set_deadline(deadline);
    call->context.AddMetadata("kvcm-route-version", std::to_string(route.version));
    call->context.AddMetadata("kvcm-partition", std::to_string(route.partition));
    std::unique_lock lock(mutex_);
    if (closed_ || inflight_.size() >= max_inflight_) {
        const auto code = closed_ ? proto::meta::SERVICE_NOT_READY : proto::meta::SERVICE_BUSY;
        lock.unlock();
        InitProxyStatus(request.trace_id(), response.mutable_header())->set_code(code);
        call->done(grpc::Status::OK);
        return;
    }
    auto &peer = peers_[route.partition];
    if (peer.endpoint != route.endpoint) {
        peer.stub =
            proto::meta::MetaService::NewStub(grpc::CreateChannel(route.endpoint, grpc::InsecureChannelCredentials()));
        peer.endpoint = route.endpoint;
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
        Send(request,                                                                                                  \
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
FORWARD_RPC(GetHostCacheState, GetHostCacheStateResponse)
#undef FORWARD_RPC

void MetaForwarder::Forward(const proto::meta::ReportEventRequest &request,
                            proto::meta::ReportEventResponse &response,
                            Deadline deadline,
                            Done done) {
    auto start = [](auto &stub, auto *context, const auto &req, auto *queue) {
        return stub.AsyncReportEvent(context, req, queue);
    };
    const auto has_lifecycle = std::any_of(request.events().begin(), request.events().end(), IsGlobalReporterEvent);
    if (!has_lifecycle) {
        Send(request, response, deadline, std::move(done), start);
        return;
    }
    if (deadline == Deadline::max()) {
        deadline = std::chrono::system_clock::now() + std::chrono::seconds(30);
    }
    auto key = MetaRoute(request);
    auto now = RoutingTable::Clock::now();
    auto owners = routes_.Resolve(key, now, lease_);
    auto global = routes_.Resolve({Placement::RoleZero, {}}, now, lease_);
    if (owners.empty() || global.empty()) {
        auto *status = InitProxyStatus(request.trace_id(), response.mutable_header());
        status->set_code(key.key.empty() ? proto::meta::INVALID_ARGUMENT : proto::meta::SERVICE_NOT_READY);
        status->set_message("reporter lifecycle requires a valid IP, role zero and its data partition");
        done(grpc::Status::OK);
        return;
    }
    if (owners.front().partition == 0) {
        SendOne(owners.front(), request, response, deadline, std::move(done), start);
        return;
    }
    // The authority sees only lifecycle events. The owner receives the complete,
    // ordered request after the authority acknowledges it, including mixed batches.
    auto lifecycle = std::make_shared<proto::meta::ReportEventRequest>();
    lifecycle->set_instance_id(request.instance_id());
    lifecycle->set_host_ip_port(request.host_ip_port());
    lifecycle->set_storage_type(request.storage_type());
    lifecycle->set_trace_id(request.trace_id());
    for (const auto &event : request.events()) {
        if (IsGlobalReporterEvent(event)) {
            *lifecycle->add_events() = event;
        }
    }
    auto ack = std::make_shared<proto::meta::ReportEventResponse>();
    auto continuation = [this, lifecycle, ack, owner = owners.front(), &request, &response, deadline, done, start](
                            grpc::Status transport) {
        if (!transport.ok() || ack->header().status().code() != proto::meta::OK || ack->item_results_size()) {
            *response.mutable_header() = ack->header();
            if (ack->item_results_size() == lifecycle->events_size()) {
                int sub = 0;
                for (const auto &event : request.events()) {
                    const auto result =
                        IsGlobalReporterEvent(event) ? ack->item_results(sub++) : proto::meta::SERVICE_NOT_READY;
                    response.add_item_results(result == proto::meta::OK ? proto::meta::SERVICE_NOT_READY : result);
                }
            }
            if (transport.ok() && response.header().status().code() == proto::meta::OK) {
                response.mutable_header()->mutable_status()->set_code(proto::meta::SERVICE_NOT_READY);
            }
            done(transport);
            return;
        }
        SendOne(owner, request, response, deadline, done, start);
    };
    SendOne(global.front(), *lifecycle, *ack, deadline, std::move(continuation), start);
}

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
