#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#endif

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
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
        : executor_(std::move(executor)) {
        assert(executor_);
#define ADD_METHOD(method, request, response)                                                                          \
    AddMethod<proto::meta::request, proto::meta::response>(                                                            \
        [service](auto *context, auto *req, auto *writer, auto *cq, void *tag) {                                       \
            service->Request##method(context, req, writer, cq, cq, tag);                                               \
        },                                                                                                             \
        [service](auto *context, const auto *req, auto *resp) {                                                        \
            if (service->Prepare##method(context, req, resp)) {                                                        \
                service->meta_service_impl_->method(context, req, resp);                                               \
            }                                                                                                          \
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

    ~AsyncCalls() {
        // Server owns shutdown ordering. This also handles a failed startup,
        // where no application RequestX was posted.
        assert(live_calls_ == 0);
        ShutdownCompletionQueueAndJoin();
    }

    bool PrepareCompletionQueue(grpc::ServerBuilder &builder, std::size_t poller_count) {
        assert(!cq_ && poller_count > 0);
        cq_ = builder.AddCompletionQueue();
        try {
            pollers_.reserve(poller_count);
            for (std::size_t i = 0; i < poller_count; ++i) {
                pollers_.emplace_back([this] {
#ifdef __linux__
                    pthread_setname_np(pthread_self(), "kvcm_meta_cq");
#endif
                    try {
                        auto retry_at = std::chrono::system_clock::now() + std::chrono::milliseconds(100);
                        for (;;) {
                            void *raw_tag = nullptr;
                            bool ok = false;
                            const auto event = cq_->AsyncNext(&raw_tag, &ok, retry_at);
                            if (event == grpc::CompletionQueue::SHUTDOWN) {
                                break;
                            }
                            if (event == grpc::CompletionQueue::GOT_EVENT) {
                                std::unique_ptr<Tag> tag(static_cast<Tag *>(raw_tag));
                                tag->Complete(ok);
                            }
                            if (std::chrono::system_clock::now() >= retry_at) {
                                if (retry_pending_.exchange(false)) {
                                    for (const auto &method : methods_) {
                                        method->Post(true);
                                    }
                                }
                                retry_at = std::chrono::system_clock::now() + std::chrono::milliseconds(100);
                            }
                        }
                    } catch (...) {
                        // Request exceptions are handled below. An exception from
                        // gRPC Next/Finish may leave an operation partially issued;
                        // continuing would risk lost tags, hung shutdown or UAF.
                        FatalCompletionFailure();
                    }
                });
            }
        } catch (const std::exception &error) {
            KVCM_LOG_ERROR("start Meta gRPC completion queue poller failed: %s", error.what());
            // No receive slots have been posted before BuildAndStart. Join
            // any pollers already created before releasing their empty CQ.
            ShutdownCompletionQueueAndJoin();
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
        for (auto &poller : pollers_) {
            if (poller.joinable()) {
                poller.join();
            }
        }
        pollers_.clear();
        cq_.reset();
    }

private:
    [[noreturn]] static void FatalCompletionFailure() noexcept {
        // Avoid allocating another error/status object when memory is exhausted.
        std::fputs("fatal: Meta gRPC completion failed; cannot safely continue polling or finishing calls\n", stderr);
        std::abort();
    }

    struct Tag {
        virtual ~Tag() = default;
        virtual void Complete(bool ok) = 0;
    };

    struct MethodBase {
        virtual ~MethodBase() = default;
        virtual void Post(bool retry_only = false) = 0;
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
        using RunFn = std::function<void(RequestContext *, const Request *, Response *)>;

        Method(AsyncCalls *owner, RequestFn request, RunFn run)
            : owner_(owner), request_(std::move(request)), run_(std::move(run)) {}

        void Post(bool retry_only = false) override {
            // Serialize RequestX with StopAccepting, including registration of
            // the CallState in the shutdown count.
            std::lock_guard<std::mutex> lock(owner_->accept_mutex_);
            if (!owner_->accepting_ || (retry_only && !retry_post_)) {
                return;
            }
            if (retry_only && std::chrono::steady_clock::now() < retry_after_) {
                owner_->retry_pending_.store(true);
                return;
            }
            try {
                auto call = std::make_shared<Call>(owner_);
                auto tag = std::make_unique<CallTag>(this, call, true);
                // In pinned gRPC 1.19.1, RequestX's C++ allocations precede
                // IssueRequest (a C API call); no C++ work follows a posted call.
                request_(&call->context, &call->request, &call->writer, owner_->cq_.get(), tag.get());
                tag.release();
                retry_post_ = false;
            } catch (const std::bad_alloc &) {
                if (!retry_post_) {
                    std::fputs("Meta gRPC receive allocation failed; will retry posting the receive slot\n", stderr);
                }
                retry_post_ = true;
                retry_after_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
                owner_->retry_pending_.store(true);
            } catch (...) {
                // Only pre-post allocation failure has a verified retry path.
                FatalCompletionFailure();
            }
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

        void Finish(const std::shared_ptr<Call> &call, const grpc::Status &status = grpc::Status::OK) noexcept {
            try {
                auto tag = std::make_unique<CallTag>(this, call, false);
                call->writer.Finish(call->response, status, tag.get());
                tag.release();
            } catch (...) {
                // Do not let WorkerLoop swallow this exception and strand an
                // accepted RPC, or retry an operation that might already exist.
                FatalCompletionFailure();
            }
        }

        void Reject(const std::shared_ptr<Call> &call, const char *message) noexcept {
            try {
                auto *header = call->response.mutable_header();
                header->set_request_id(call->request_context->request_id());
                header->mutable_status()->set_code(proto::meta::SERVICE_NOT_READY);
                header->mutable_status()->set_message(message);
            } catch (...) {
                Finish(call, unexpected_error_);
                return;
            }
            Finish(call);
        }

        void Accepted(const std::shared_ptr<Call> &call) {
            try {
                Post();
                // Capture arrival timing and peer data on the CQ thread. The
                // collector lookup may wait for instance removal's lifecycle
                // lock, so all preparation that can block belongs on a worker.
                call->request_context = std::make_shared<RequestContext>(call->request.trace_id());
                call->request_context->set_client_ip(ExtractIpFromPeer(call->context.peer()));
                ServiceRequestExecutor::Request task;
                task.run = [this, call](ServiceRequestExecutor::CompletionToken) {
                    // Expired queued calls have not started business work yet.
                    // Avoid spending worker capacity on a result the caller cannot receive.
                    if (call->context.deadline() <= std::chrono::system_clock::now()) {
                        Finish(call, deadline_exceeded_);
                        return;
                    }
                    try {
                        run_(call->request_context.get(), &call->request, &call->response);
                    } catch (...) {
                        Finish(call, unexpected_error_);
                        return;
                    }
                    Finish(call);
                };
                task.on_discard = [this, call](ServiceRequestExecutor::CompletionToken) {
                    Reject(call, "service is stopping");
                };
                const auto result = owner_->executor_->TrySubmit(std::move(task));
                if (result != ServiceRequestExecutor::SubmitResult::kAccepted) {
                    Reject(call,
                           result == ServiceRequestExecutor::SubmitResult::kFull ? "service request queue is full"
                                                                                 : "service is stopping");
                }
            } catch (...) { Finish(call, unexpected_error_); }
        }

        AsyncCalls *owner_;
        RequestFn request_;
        RunFn run_;
        const grpc::Status unexpected_error_{grpc::StatusCode::UNKNOWN, "Unexpected error in RPC handling"};
        const grpc::Status deadline_exceeded_{grpc::StatusCode::DEADLINE_EXCEEDED, "Request expired in service queue"};
        bool retry_post_ = false; // Protected by accept_mutex_.
        std::chrono::steady_clock::time_point retry_after_;
    };

    template <typename Request, typename Response, typename RequestFn, typename RunFn>
    void AddMethod(RequestFn request, RunFn run) {
        methods_.push_back(std::make_unique<Method<Request, Response>>(this, request, run));
    }

    std::shared_ptr<ServiceRequestExecutor> executor_;
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    std::vector<std::thread> pollers_;
    std::mutex accept_mutex_;
    bool accepting_ = false;
    std::atomic<bool> retry_pending_{false};
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
