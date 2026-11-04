#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"

#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/service_request_executor.h"
#include "kv_cache_manager/service/util/common.h"

namespace kv_cache_manager {

// The CQ only accepts calls and observes completion. Business code always runs
// on the shared executor; in particular, this thread must keep polling when
// synchronous Admin/Debug/Optimizer services share the server.
class MetaServiceGRpc::AsyncCalls {
public:
    AsyncCalls(MetaServiceGRpc *service, std::shared_ptr<ServiceRequestExecutor> executor)
        : executor_(std::move(executor)) {
        assert(executor_);
#define ADD_METHOD(method, request, response)                                                                          \
    AddMethod<proto::meta::request, proto::meta::response>(                                                            \
        [service](auto *context, auto *req, auto *writer, auto *cq, void *tag) {                                       \
            service->Request##method(context, req, writer, cq, cq, tag);                                               \
        },                                                                                                             \
        [service](auto *context, auto *req, auto *resp) { return service->Prepare##method(context, req, resp); },      \
        [service](auto *context, auto *req, auto *resp) { service->meta_service_impl_->method(context, req, resp); })
        ADD_METHOD(RegisterInstance, RegisterInstanceRequest, RegisterInstanceResponse);
        ADD_METHOD(GetInstanceInfo, GetInstanceInfoRequest, GetInstanceInfoResponse);
        ADD_METHOD(GetCacheMeta, GetCacheMetaRequest, GetCacheMetaResponse);
        ADD_METHOD(GetCacheLocation, GetCacheLocationRequest, GetCacheLocationResponse);
        ADD_METHOD(GetCacheLocationsByBackend, GetCacheLocationsByBackendRequest, GetCacheLocationsByBackendResponse);
        ADD_METHOD(GetCacheLocationLen, GetCacheLocationLenRequest, GetCacheLocationLenResponse);
        ADD_METHOD(StartWriteCache, StartWriteCacheRequest, StartWriteCacheResponse);
        ADD_METHOD(FinishWriteCache, FinishWriteCacheRequest, CommonResponse);
        ADD_METHOD(RemoveCache, RemoveCacheRequest, CommonResponse);
        ADD_METHOD(TrimCache, TrimCacheRequest, CommonResponse);
        ADD_METHOD(GetClusterInfo, GetClusterInfoRequest, GetClusterInfoResponse);
        ADD_METHOD(ReportEvent, ReportEventRequest, ReportEventResponse);
        ADD_METHOD(GetHostCacheState, GetHostCacheStateRequest, GetHostCacheStateResponse);
#undef ADD_METHOD
    }

    ~AsyncCalls() {
        // Server owns shutdown ordering. This also handles a failed startup,
        // where no application RequestX was posted.
        assert(live_calls_ == 0);
        ShutdownCompletionQueueAndJoin();
    }

    bool PrepareCompletionQueue(grpc::ServerBuilder &builder) {
        assert(!cq_);
        cq_ = builder.AddCompletionQueue();
        try {
            poller_ = std::thread([this] {
                void *raw_tag = nullptr;
                bool ok = false;
                while (cq_->Next(&raw_tag, &ok)) {
                    std::unique_ptr<Tag> tag(static_cast<Tag *>(raw_tag));
                    tag->Complete(ok);
                }
            });
        } catch (const std::exception &error) {
            KVCM_LOG_ERROR("start Meta gRPC completion queue poller failed: %s", error.what());
            cq_->Shutdown();
            void *tag = nullptr;
            bool ok = false;
            while (cq_->Next(&tag, &ok)) {
                delete static_cast<Tag *>(tag);
            }
            cq_.reset();
            return false;
        }
        return true;
    }

    void StartAccepting() {
        {
            std::lock_guard<std::mutex> lock(accept_mutex_);
            assert(cq_ && !accepting_);
            accepting_ = true;
        }
        for (const auto &method : methods_) {
            method->Post();
        }
    }

    void StopAccepting() {
        std::lock_guard<std::mutex> lock(accept_mutex_);
        accepting_ = false;
    }

    void WaitForCalls() {
        std::unique_lock<std::mutex> lock(calls_mutex_);
        calls_drained_.wait(lock, [this] { return live_calls_ == 0; });
    }

    void ShutdownCompletionQueueAndJoin() {
        if (cq_) {
            cq_->Shutdown();
        }
        if (poller_.joinable()) {
            poller_.join();
        }
        cq_.reset();
    }

private:
    struct Tag {
        virtual ~Tag() = default;
        virtual void Complete(bool ok) = 0;
    };

    struct MethodBase {
        virtual ~MethodBase() = default;
        virtual void Post() = 0;
    };

    // Declared first in Call, so it is destroyed after context/writer and after
    // both event tags and the executor task release their shared ownership.
    struct CallLease {
        explicit CallLease(AsyncCalls *owner) : owner(owner) {
            std::lock_guard<std::mutex> lock(owner->calls_mutex_);
            ++owner->live_calls_;
        }
        ~CallLease() {
            std::lock_guard<std::mutex> lock(owner->calls_mutex_);
            if (--owner->live_calls_ == 0) {
                owner->calls_drained_.notify_all();
            }
        }
        AsyncCalls *owner;
    };

    template <typename Request, typename Response>
    class Method final : public MethodBase {
    public:
        using RequestFn = std::function<void(grpc::ServerContext *,
                                             Request *,
                                             grpc::ServerAsyncResponseWriter<Response> *,
                                             grpc::ServerCompletionQueue *,
                                             void *)>;
        using PrepareFn = std::function<bool(RequestContext *, Request *, Response *)>;
        using RunFn = std::function<void(RequestContext *, const Request *, Response *)>;

        Method(AsyncCalls *owner, RequestFn request, PrepareFn prepare, RunFn run)
            : owner_(owner), request_(std::move(request)), prepare_(std::move(prepare)), run_(std::move(run)) {}

        void Post() override {
            // Serialize RequestX with StopAccepting, including registration of
            // the CallState in the shutdown count.
            std::lock_guard<std::mutex> lock(owner_->accept_mutex_);
            if (!owner_->accepting_) {
                return;
            }
            auto call = std::make_shared<Call>(owner_);
            auto tag = std::make_unique<CallTag>(this, call, true);
            request_(&call->context, &call->request, &call->writer, owner_->cq_.get(), tag.get());
            tag.release();
        }

    private:
        struct Call {
            explicit Call(AsyncCalls *owner) : lease(owner), writer(&context) {}
            CallLease lease;
            grpc::ServerContext context;
            Request request;
            Response response;
            grpc::ServerAsyncResponseWriter<Response> writer;
            std::shared_ptr<RequestContext> request_context;
        };

        struct CallTag final : Tag {
            CallTag(Method *method, std::shared_ptr<Call> call, bool accept)
                : method(method), call(std::move(call)), accept(accept) {}
            void Complete(bool ok) override {
                if (accept && ok) {
                    method->Accepted(call);
                }
                // accept/ok=false means no RPC was matched. A finish event,
                // including ok=false, always releases its ownership here.
            }
            Method *method;
            std::shared_ptr<Call> call;
            bool accept;
        };

        void Finish(const std::shared_ptr<Call> &call, const grpc::Status &status = grpc::Status::OK) {
            auto tag = std::make_unique<CallTag>(this, call, false);
            call->writer.Finish(call->response, status, tag.get());
            tag.release();
        }

        void Reject(const std::shared_ptr<Call> &call, const char *message) {
            auto *header = call->response.mutable_header();
            header->set_request_id(call->request_context->request_id());
            header->mutable_status()->set_code(proto::meta::SERVICE_NOT_READY);
            header->mutable_status()->set_message(message);
            Finish(call);
        }

        void Accepted(const std::shared_ptr<Call> &call) {
            grpc::Status error;
            const char *rejection = nullptr;
            try {
                Post();
                // Capture arrival timing and peer data on the CQ thread. The
                // collector lookup may wait for instance removal's lifecycle
                // lock, so all preparation that can block belongs on a worker.
                call->request_context = std::make_shared<RequestContext>(call->request.trace_id());
                call->request_context->set_client_ip(ExtractIpFromPeer(call->context.peer()));
                ServiceRequestExecutor::Task task;
                task.run = [this, call](ServiceRequestExecutor::CompletionToken) {
                    grpc::Status status;
                    try {
                        if (prepare_(call->request_context.get(), &call->request, &call->response)) {
                            run_(call->request_context.get(), &call->request, &call->response);
                        }
                    } catch (...) {
                        status = grpc::Status(grpc::StatusCode::UNKNOWN, "Unexpected error in RPC handling");
                    }
                    Finish(call, status);
                };
                task.on_discard = [this, call](ServiceRequestExecutor::CompletionToken) {
                    Reject(call, "service is stopping");
                };
                const auto result = owner_->executor_->TrySubmit(std::move(task));
                if (result == ServiceRequestExecutor::SubmitResult::kAccepted) {
                    return;
                }
                rejection = result == ServiceRequestExecutor::SubmitResult::kFull ? "service request queue is full"
                                                                                  : "service is stopping";
            } catch (...) {
                error = grpc::Status(grpc::StatusCode::UNKNOWN, "Unexpected error in RPC handling");
            }
            // Queue rejection never performs a collector lookup. Finish is
            // terminal, even on failure.
            if (rejection) {
                Reject(call, rejection);
            } else {
                Finish(call, error);
            }
        }

        AsyncCalls *owner_;
        RequestFn request_;
        PrepareFn prepare_;
        RunFn run_;
    };

    template <typename Request, typename Response, typename RequestFn, typename PrepareFn, typename RunFn>
    void AddMethod(RequestFn request, PrepareFn prepare, RunFn run) {
        methods_.push_back(std::make_unique<Method<Request, Response>>(this, request, prepare, run));
    }

    std::shared_ptr<ServiceRequestExecutor> executor_;
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    std::thread poller_;
    std::mutex accept_mutex_;
    bool accepting_ = false;
    std::mutex calls_mutex_;
    std::condition_variable calls_drained_;
    size_t live_calls_ = 0;
    std::vector<std::unique_ptr<MethodBase>> methods_;
};

MetaServiceGRpc::MetaServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                                 std::shared_ptr<MetaServiceImpl> meta_service_impl,
                                 std::shared_ptr<RegistryManager> registry_manager,
                                 std::shared_ptr<ServiceRequestExecutor> executor,
                                 std::shared_ptr<MetricsLifecycle> metrics_lifecycle)
    : MetaServiceMetricsBase(std::move(metrics_registry), registry_manager, std::move(metrics_lifecycle))
    , meta_service_impl_(std::move(meta_service_impl))
    , async_calls_(std::make_unique<AsyncCalls>(this, std::move(executor))) {}

MetaServiceGRpc::~MetaServiceGRpc() = default;

void MetaServiceGRpc::Init() { MetaServiceMetricsBase::InitMetrics(); }

bool MetaServiceGRpc::PrepareCompletionQueue(grpc::ServerBuilder &builder) {
    return async_calls_->PrepareCompletionQueue(builder);
}

void MetaServiceGRpc::StartAccepting() { async_calls_->StartAccepting(); }
void MetaServiceGRpc::StopAccepting() { async_calls_->StopAccepting(); }
void MetaServiceGRpc::WaitForCalls() { async_calls_->WaitForCalls(); }
void MetaServiceGRpc::ShutdownCompletionQueueAndJoin() { async_calls_->ShutdownCompletionQueueAndJoin(); }

bool MetaServiceGRpc::PrepareRegisterInstance(RequestContext *request_context,
                                              const proto::meta::RegisterInstanceRequest *request,
                                              proto::meta::RegisterInstanceResponse *response) {
    request_context->InitMetricsCollector(KVCM_METRICS_COLLECTOR_(RegisterInstance));
    return true;
}

bool MetaServiceGRpc::PrepareGetInstanceInfo(RequestContext *request_context,
                                             const proto::meta::GetInstanceInfoRequest *request,
                                             proto::meta::GetInstanceInfoResponse *response) {
    request_context->InitMetricsCollector(KVCM_METRICS_COLLECTOR_(GetInstanceInfo));
    return true;
}

bool MetaServiceGRpc::PrepareGetCacheMeta(RequestContext *request_context,
                                          const proto::meta::GetCacheMetaRequest *request,
                                          proto::meta::GetCacheMetaResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(GetCacheMeta, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}

bool MetaServiceGRpc::PrepareGetCacheLocation(RequestContext *request_context,
                                              const proto::meta::GetCacheLocationRequest *request,
                                              proto::meta::GetCacheLocationResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(GetCacheLocation, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}

bool MetaServiceGRpc::PrepareGetCacheLocationsByBackend(RequestContext *request_context,
                                                        const proto::meta::GetCacheLocationsByBackendRequest *request,
                                                        proto::meta::GetCacheLocationsByBackendResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(GetCacheLocationsByBackend, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}

bool MetaServiceGRpc::PrepareGetCacheLocationLen(RequestContext *request_context,
                                                 const proto::meta::GetCacheLocationLenRequest *request,
                                                 proto::meta::GetCacheLocationLenResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(GetCacheLocationLen, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}

bool MetaServiceGRpc::PrepareStartWriteCache(RequestContext *request_context,
                                             const proto::meta::StartWriteCacheRequest *request,
                                             proto::meta::StartWriteCacheResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(StartWriteCache, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}
bool MetaServiceGRpc::PrepareFinishWriteCache(RequestContext *request_context,
                                              const proto::meta::FinishWriteCacheRequest *request,
                                              proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(FinishWriteCache, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}
bool MetaServiceGRpc::PrepareRemoveCache(RequestContext *request_context,
                                         const proto::meta::RemoveCacheRequest *request,
                                         proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(RemoveCache, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}
bool MetaServiceGRpc::PrepareTrimCache(RequestContext *request_context,
                                       const proto::meta::TrimCacheRequest *request,
                                       proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(TrimCache, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}

bool MetaServiceGRpc::PrepareGetClusterInfo(RequestContext *request_context,
                                            const proto::meta::GetClusterInfoRequest *request,
                                            proto::meta::GetClusterInfoResponse *response) {
    // instance_id 可能尚未注册（如 RegisterInstance 之前），此时 fallback 到全局 collector
    auto metrics_collector = get_metrics_collector_from_map_for_GetClusterInfo(request->instance_id());
    if (metrics_collector == nullptr) {
        metrics_collector = KVCM_METRICS_COLLECTOR_(GetClusterInfo);
    }
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}

bool MetaServiceGRpc::PrepareReportEvent(RequestContext *request_context,
                                         const proto::meta::ReportEventRequest *request,
                                         proto::meta::ReportEventResponse *response) {
    std::string metrics_type;
    auto metrics_collector = ResolveReportEventMetricsCollector(*request, metrics_type);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    AttachReportEventTypeMetricsCollectors(*request, metrics_type, request_context);
    return true;
}

bool MetaServiceGRpc::PrepareGetHostCacheState(RequestContext *request_context,
                                               const proto::meta::GetHostCacheStateRequest *request,
                                               proto::meta::GetHostCacheStateResponse *response) {
    API_CONTEXT_GET_AND_INIT_COLLECTOR(GetHostCacheState, false);
    request_context->InitMetricsCollector(std::move(metrics_collector));
    return true;
}

} // namespace kv_cache_manager
