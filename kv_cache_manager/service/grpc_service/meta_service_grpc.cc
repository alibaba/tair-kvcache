#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#endif

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/service_request_executor.h"
#include "kv_cache_manager/service/util/common.h"

namespace kv_cache_manager {

// Next/AsyncNext also deserialize typed protobuf requests and drive network IO;
// the last Call reference may release request/response storage on this thread.
// Business code and potentially blocking preparation run on the shared executor.
class MetaServiceGRpc::AsyncCalls {
public:
    AsyncCalls(MetaServiceGRpc *service, std::shared_ptr<ServiceRequestExecutor> executor)
        : service_(service), executor_(std::move(executor)) {
        assert(executor_);
    }

    ~AsyncCalls() {
        // Server owns shutdown ordering. This also handles a failed startup,
        // where no application RequestX was posted.
        assert(live_calls_ == 0);
        ShutdownCompletionQueueAndJoin();
    }

    bool PrepareCompletionQueue(grpc::ServerBuilder &builder, std::size_t poller_count) {
        assert(!cq_ && poller_count > 0);
        cq_ = builder.AddCompletionQueue();
        pollers_.reserve(poller_count);
        for (std::size_t i = 0; i < poller_count; ++i) {
            RegisterMethods(cq_.get());
        }
        for (std::size_t i = 0; i < poller_count; ++i) {
            try {
                pollers_.emplace_back([this] { Poll(); });
            } catch (const std::system_error &error) {
                KVCM_LOG_ERROR("create Meta gRPC CQ poller failed: %s", error.what());
                ShutdownCompletionQueueAndJoin();
                return false;
            }
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
        for (auto &poller : pollers_) {
            if (poller.joinable()) {
                poller.join();
            }
        }
        pollers_.clear();
        methods_.clear();
        cq_.reset();
    }

private:
    void Poll() {
#ifdef __linux__
        pthread_setname_np(pthread_self(), "kvcm_meta_cq");
#endif
        void *raw_tag = nullptr;
        bool ok = false;
        while (cq_->Next(&raw_tag, &ok)) {
            std::unique_ptr<Tag> tag(static_cast<Tag *>(raw_tag));
            tag->Complete(ok);
        }
    }

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
        using PrepareFn = std::function<bool(RequestContext *, const Request *, Response *)>;
        using RunFn = std::function<void(RequestContext *, const Request *, Response *)>;

        Method(AsyncCalls *owner,
               grpc::ServerCompletionQueue *cq,
               const char *name,
               RequestFn request,
               PrepareFn prepare,
               RunFn run)
            : owner_(owner)
            , cq_(cq)
            , name_(name)
            , request_(std::move(request))
            , prepare_(std::move(prepare))
            , run_(std::move(run)) {}

        void Post() noexcept override {
            // Register the lease before closing admission. Issue RequestX outside
            // this lock so transport work cannot serialize all CQ pollers.
            std::unique_lock<std::mutex> lock(owner_->accept_mutex_);
            if (!owner_->accepting_) {
                return;
            }
            auto call = std::make_shared<Call>(owner_);
            auto tag = std::make_unique<CallTag>(this, call, true);
            auto *done = new DoneTag(call);
            tag->unmatched_done = done;
            call->context.AsyncNotifyWhenDone(done);
            lock.unlock();
            request_(&call->context, &call->request, &call->writer, cq_, tag.get());
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
            std::atomic<bool> cancelled{false};
        };

        struct DoneTag final : Tag {
            explicit DoneTag(std::shared_ptr<Call> call) : call(std::move(call)) {}
            void Complete(bool) override {
                // IsCancelled is safe for an async ServerContext only after this event.
                call->cancelled.store(call->context.IsCancelled(), std::memory_order_relaxed);
            }
            std::shared_ptr<Call> call;
        };

        struct CallTag final : Tag {
            CallTag(Method *method, std::shared_ptr<Call> call, bool accept)
                : method(method), call(std::move(call)), accept(accept) {}
            void Complete(bool ok) override {
                if (accept) {
                    if (ok) {
                        method->Accepted(call);
                    } else {
                        // An unmatched receive never starts gRPC's completion op.
                        delete unmatched_done;
                    }
                }
            }
            Method *method;
            std::shared_ptr<Call> call;
            bool accept;
            DoneTag *unmatched_done = nullptr;
        };

        void Finish(const std::shared_ptr<Call> &call, const grpc::Status &status = grpc::Status::OK) noexcept {
            auto tag = std::make_unique<CallTag>(this, call, false);
            call->writer.Finish(call->response, status, tag.get());
            tag.release();
        }

        void Reject(const std::shared_ptr<Call> &call, ServiceRequestExecutor::SubmitResult reason) noexcept {
            auto *header = call->response.mutable_header();
            if (call->request_context) {
                header->set_request_id(call->request_context->request_id());
            } else {
                header->set_request_id(call->request.trace_id() + "_" +
                                       std::to_string(TimestampUtil::GetCurrentTimeUs()));
            }
            const bool full = reason == ServiceRequestExecutor::SubmitResult::kFull;
            // Unexecuted requests can use the client's existing leader failover.
            header->mutable_status()->set_code(full ? proto::meta::SERVICE_BUSY : proto::meta::SERVER_NOT_LEADER);
            header->mutable_status()->set_message(full ? "service request queue is full" : "service is stopping");
            Finish(call);
        }

        void Accepted(const std::shared_ptr<Call> &call) {
            try {
                Dispatch(call);
            } catch (...) { FinishUnexpected(call, "ingress"); }
            // Resolve admission before potentially blocking in RequestX.
            // Keep reposting outside the catch to avoid finishing a call twice.
            Post();
        }

        void FinishUnexpected(const std::shared_ptr<Call> &call, const char *phase) noexcept {
            KVCM_LOG_ERROR("Meta gRPC exception: method=%s phase=%s trace_id=%s request_id=%s",
                           name_,
                           phase,
                           call->request.trace_id().c_str(),
                           call->request_context ? call->request_context->request_id().c_str() : "");
            Finish(call, unexpected_error_);
        }

        const grpc::Status *StopReason(const Call &call) const {
            if (call.context.deadline() <= std::chrono::system_clock::now()) {
                return &deadline_exceeded_;
            }
            return call.cancelled.load(std::memory_order_relaxed) ? &cancelled_ : nullptr;
        }

        void Dispatch(const std::shared_ptr<Call> &call) {
            if (call->context.deadline() <= std::chrono::system_clock::now()) {
                Finish(call, deadline_exceeded_);
                return;
            }
            if (const auto rejection = owner_->executor_->RejectIfUnavailable()) {
                Reject(call, *rejection);
                return;
            }
            // Capture arrival timing and peer data on the CQ thread. The
            // collector lookup may wait for instance removal's lifecycle
            // lock, so all preparation that can block belongs on a worker.
            call->request_context = std::make_shared<RequestContext>(call->request.trace_id());
            call->request_context->set_client_ip(ExtractIpFromPeer(call->context.peer()));
            ServiceRequestExecutor::Request task;
            task.run = [this, call](ServiceRequestExecutor::CompletionToken) {
                // Expired queued calls have not started business work yet.
                // Avoid spending worker capacity on a result the caller cannot receive.
                if (const auto *reason = StopReason(*call)) {
                    Finish(call, *reason);
                    return;
                }
                auto *context = call->request_context.get();
                const grpc::Status *reason = nullptr;
                try {
                    const bool prepared = prepare_(context, &call->request, &call->response);
                    reason = StopReason(*call);
                    if (prepared && !reason) {
                        run_(context, &call->request, &call->response);
                    }
                } catch (...) {
                    FinishUnexpected(call, "worker");
                    return;
                }
                Finish(call, reason ? *reason : grpc::Status::OK);
            };
            task.on_discard = [this, call](ServiceRequestExecutor::CompletionToken) {
                Reject(call, ServiceRequestExecutor::SubmitResult::kClosed);
            };
            const auto result = owner_->executor_->TrySubmit(std::move(task));
            if (result != ServiceRequestExecutor::SubmitResult::kAccepted) {
                Reject(call, result);
            }
        }

        AsyncCalls *owner_;
        grpc::ServerCompletionQueue *const cq_;
        const char *const name_;
        RequestFn request_;
        PrepareFn prepare_;
        RunFn run_;
        const grpc::Status unexpected_error_{grpc::StatusCode::UNKNOWN, "Unexpected error in RPC handling"};
        const grpc::Status deadline_exceeded_{grpc::StatusCode::DEADLINE_EXCEEDED, "Request expired before execution"};
        const grpc::Status cancelled_{grpc::StatusCode::CANCELLED, "Request cancelled before execution"};
    };

    void RegisterMethods(grpc::ServerCompletionQueue *cq) {
        auto *service = service_;
#define ADD_METHOD(method, request, response)                                                                          \
    AddMethod<proto::meta::request, proto::meta::response>(                                                            \
        cq,                                                                                                            \
        #method,                                                                                                       \
        [service](auto *context, auto *req, auto *writer, auto *cq, void *tag) {                                       \
            service->Request##method(context, req, writer, cq, cq, tag);                                               \
        },                                                                                                             \
        [service](auto *context, const auto *req, auto *resp) {                                                        \
            return service->Prepare##method(context, req, resp);                                                       \
        },                                                                                                             \
        [service](auto *context, const auto *req, auto *resp) {                                                        \
            service->meta_service_impl_->method(context, req, resp);                                                   \
        })
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

    template <typename Request, typename Response, typename RequestFn, typename PrepareFn, typename RunFn>
    void AddMethod(grpc::ServerCompletionQueue *cq, const char *name, RequestFn request, PrepareFn prepare, RunFn run) {
        methods_.push_back(std::make_unique<Method<Request, Response>>(this, cq, name, request, prepare, run));
    }

    MetaServiceGRpc *const service_;
    std::shared_ptr<ServiceRequestExecutor> executor_;
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    std::vector<std::thread> pollers_;
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

bool MetaServiceGRpc::PrepareCompletionQueue(grpc::ServerBuilder &builder, std::size_t poller_count) {
    return async_calls_->PrepareCompletionQueue(builder, poller_count);
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
