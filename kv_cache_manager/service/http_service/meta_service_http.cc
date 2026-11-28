#include "kv_cache_manager/service/http_service/meta_service_http.h"

#include <cassert>
#include <charconv>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/util/common.h"
#include "kv_cache_manager/service/util/report_event_body_budget.h"
#include "kv_cache_manager/service/util/report_event_json_parser.h"
#include "rapidjson/document.h"

namespace kv_cache_manager {

CoroHttpService::CachedJsonResponse MetaServiceHttp::TryRejectRequest() { return TryRejectRequest(false); }

CoroHttpService::CachedJsonResponse MetaServiceHttp::TryRejectRequest(bool over_body_budget) {
    const auto rejection = executor_->RejectIfUnavailable(over_body_budget);
    if (!rejection) {
        return std::nullopt;
    }
    // No request JSON is decoded on this overload path. Use a server-generated
    // request ID rather than spending IO capacity parsing the caller's trace ID.
    proto::meta::CommonResponse response;
    response.mutable_header()->set_request_id("_" + std::to_string(TimestampUtil::GetCurrentTimeUs()));
    auto *status = response.mutable_header()->mutable_status();
    status->set_code(*rejection == ServiceRequestExecutor::SubmitResult::kFull ? proto::meta::SERVICE_BUSY
                                                                               : proto::meta::SERVER_NOT_LEADER);
    status->set_message(*rejection == ServiceRequestExecutor::SubmitResult::kClosed ? "service is stopping"
                        : over_body_budget ? "ReportEvent body budget is exhausted"
                                           : "service request queue is full");
    std::string json;
    if (!ProtoMessageJsonUtil::ToJson(&response, json)) {
        KVCM_LOG_ERROR("serialize overloaded Meta response failed");
        return R"({"header":{"status":{"code":"INTERNAL_ERROR","message":"response serialization failed"}}})";
    }
    return json;
}

CoroHttpService::HandlerType MetaServiceHttp::LimitReportEventBodyBytes(HandlerType handler) {
    return [this, handler = std::move(handler)](coro_http::coro_http_request &req,
                                                coro_http::coro_http_response &res) -> async_simple::coro::Lazy<void> {
        constexpr size_t kMinBudgetedBodyBytes = 8 * 1024 * 1024;
        const size_t bytes = req.get_body().size();
        if (bytes < kMinBudgetedBodyBytes) {
            co_await handler(req, res);
            co_return;
        }
        if (!report_event_body_budget_->TryReserve(bytes)) {
            if (report_body_budget_rejection_) {
                report_body_budget_rejection_();
            }
            CachedJsonResponse rejection = TryRejectRequest(true);
            assert(rejection);
            res.add_header("Content-Type", "application/json");
            res.set_status_and_content(coro_http::status_type::ok, std::move(*rejection));
            co_return;
        }
        // Keep the reservation across parsing, queueing, and response generation.
        const auto release = [bytes](ReportEventBodyBudget *budget) { budget->Release(bytes); };
        const std::unique_ptr<ReportEventBodyBudget, decltype(release)> reservation(report_event_body_budget_.get(),
                                                                                    release);
        co_await handler(req, res);
    };
}

namespace {
// Generic protobuf JSON parsing ignores unknown enum names. Keep that behavior
// for other fields, but never silently narrow a replica requirement's scope.
const char *ValidateReplicaTypeJson(std::string_view json) {
    rapidjson::Document document;
    document.Parse(json.data(), json.size());
    if (document.HasParseError() || !document.IsObject()) {
        return "invalid JSON for storage_replica_requirement";
    }
    const auto valid_type_name = [](const rapidjson::Value &value) {
        if (!value.IsString()) {
            // Non-string values are preserved or rejected by protobuf parsing;
            // their enum validity is checked by the existing Manager path.
            return true;
        }
        proto::meta::StorageType type;
        if (proto::meta::StorageType_Parse(std::string(value.GetString(), value.GetStringLength()), &type)) {
            return true;
        }
        int number = 0;
        const char *begin = value.GetString();
        const char *end = begin + value.GetStringLength();
        const auto parsed = std::from_chars(begin, end, number);
        return parsed.ec == std::errc() && parsed.ptr == end && proto::meta::StorageType_IsValid(number);
    };
    for (const auto &member : document.GetObject()) {
        const std::string_view name(member.name.GetString(), member.name.GetStringLength());
        if ((name != "storage_replica_requirement" && name != "storageReplicaRequirement") ||
            !member.value.IsObject()) {
            continue;
        }
        for (const auto &field : member.value.GetObject()) {
            const std::string_view field_name(field.name.GetString(), field.name.GetStringLength());
            if (field.value.IsNull()) {
                continue;
            }
            if (field_name == "storage_type" || field_name == "storageType") {
                if (!valid_type_name(field.value)) {
                    return "storage_replica_requirement.storage_type must be a known StorageType name or number";
                }
            } else if (field_name == "excluded_storage_types" || field_name == "excludedStorageTypes") {
                if (!field.value.IsArray()) {
                    return "storage_replica_requirement.excluded_storage_types must be an array";
                }
                for (const auto &value : field.value.GetArray()) {
                    if (!valid_type_name(value)) {
                        return "excluded_storage_types contains an unknown StorageType name or number";
                    }
                }
            }
        }
    }
    return nullptr;
}
} // namespace

MetaServiceHttp::MetaServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                                 std::shared_ptr<MetaServiceImpl> meta_service_impl,
                                 std::shared_ptr<RegistryManager> registry_manager,
                                 std::shared_ptr<ServiceRequestExecutor> executor,
                                 std::shared_ptr<MetricsLifecycle> metrics_lifecycle,
                                 std::shared_ptr<ReportEventBodyBudget> report_event_body_budget,
                                 std::function<void()> report_body_budget_rejection)
    : MetaServiceMetricsBase(std::move(metrics_registry), registry_manager, std::move(metrics_lifecycle))
    , meta_service_impl_(std::move(meta_service_impl))
    , executor_(std::move(executor))
    , report_event_body_budget_(std::move(report_event_body_budget))
    , report_body_budget_rejection_(std::move(report_body_budget_rejection)) {
    assert(executor_);
    assert(report_event_body_budget_);
}

template <typename Request, typename Response, typename Callback>
async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult> MetaServiceHttp::DispatchRequest(
    coro_http::coro_http_connection *http_conn, Request *request, Response *response, Callback prepare_and_run) {
    // Capture only arrival timing and peer information on IO. Collector lookup,
    // logging and business execution belong to the worker. The outer HTTP
    // coroutine retains the protobuf messages, including their arena.
    auto context = std::make_shared<RequestContext>(request->trace_id());
    context->set_client_ip(GetHttpClientIp(http_conn));
    bool closing = false;
    auto visit = meta_service_impl_->TrackIdleInstanceRequest(
        request->instance_id(), std::is_same_v<Request, proto::meta::GetClusterInfoRequest> ? nullptr : &closing);
    return DispatchHttpRequest(
        *executor_,
        [context, run = std::move(prepare_and_run), visit = std::move(visit), closing, response] {
            static_cast<void>(visit);
            if (closing) {
                auto *header = response->mutable_header();
                header->set_request_id(context->request_id());
                header->mutable_status()->set_code(proto::meta::INSTANCE_NOT_EXIST);
                header->mutable_status()->set_message("instance removal in progress");
                return CachedJsonResponse{};
            }
            return run(context.get());
        },
        [context, response](ServiceRequestExecutor::SubmitResult reason, const char *message) {
            auto *header = response->mutable_header();
            header->set_request_id(context->request_id());
            auto *status = header->mutable_status();
            status->set_code(reason == ServiceRequestExecutor::SubmitResult::kFull ? proto::meta::SERVICE_BUSY
                                                                                   : proto::meta::SERVER_NOT_LEADER);
            status->set_message(message);
        },
        request->GetDescriptor()->name().c_str(),
        context);
}

void MetaServiceHttp::Init() { MetaServiceMetricsBase::InitMetrics(); }

void MetaServiceHttp::RegisterHandler() {
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, registerInstance, RegisterInstance, RegisterInstance, RegisterInstance);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, getInstanceInfo, GetInstanceInfo, GetInstanceInfo, GetInstanceInfo);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, getCacheMeta, GetCacheMeta, GetCacheMeta, GetCacheMeta);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, getCacheLocation, GetCacheLocation, GetCacheLocation, GetCacheLocation);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, getCacheLocationLen, GetCacheLocationLen, GetCacheLocationLen, GetCacheLocationLen);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post,
                                           getCacheLocationsByBackend,
                                           GetCacheLocationsByBackend,
                                           GetCacheLocationsByBackend,
                                           GetCacheLocationsByBackend);
    RegisterPostHandler("/api/startWriteCache", [this](auto &req, auto &res) -> async_simple::coro::Lazy<void> {
        auto handler =
            GetHandler<MetaServiceHttp, proto::meta::StartWriteCacheRequest, proto::meta::StartWriteCacheResponse>(
                [&req](MetaServiceHttp *service, auto *connection, auto *request, auto *response) {
                    return service->StartWriteCache(connection, request, response, req.get_body());
                });
        co_await handler(req, res);
    });
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, finishWriteCache, FinishWriteCache, Common, FinishWriteCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, removeCache, RemoveCache, Common, RemoveCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, trimCache, TrimCache, Common, TrimCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, getClusterInfo, GetClusterInfo, GetClusterInfo, GetClusterInfo);
    RegisterPostHandler(
        "/api/reportEvent",
        LimitReportEventBodyBytes(
            GetArenaHandler<MetaServiceHttp, proto::meta::ReportEventRequest, proto::meta::ReportEventResponse>(
                &MetaServiceHttp::ReportEvent, &ReportEventJsonParser::FromMutableNullTerminatedJson)));
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, getHostCacheState, GetHostCacheState, GetHostCacheState, GetHostCacheState);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::RegisterInstance(coro_http::coro_http_connection *http_conn,
                                  proto::meta::RegisterInstanceRequest *request,
                                  proto::meta::RegisterInstanceResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            request_context->InitMetricsCollector(KVCM_METRICS_COLLECTOR_(RegisterInstance));
            KVCM_LOG_INFO("[traceId: %s] RegisterInstance called with instance id: %s, instance group: %s",
                          request->trace_id().c_str(),
                          request->instance_id().c_str(),
                          request->instance_group().c_str());
            KVCM_LOG_DEBUG("[traceId: %s] RegisterInstance request details: %s",
                           request->trace_id().c_str(),
                           request->model_deployment().ShortDebugString().c_str());
            meta_service_impl_->RegisterInstance(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                                 proto::meta::GetInstanceInfoRequest *request,
                                 proto::meta::GetInstanceInfoResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            request_context->InitMetricsCollector(KVCM_METRICS_COLLECTOR_(GetInstanceInfo));
            KVCM_LOG_INFO("[traceId: %s] GetInstanceInfo called with instance id: %s",
                          request->trace_id().c_str(),
                          request->instance_id().c_str());
            meta_service_impl_->GetInstanceInfo(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetCacheLocation(coro_http::coro_http_connection *http_conn,
                                  proto::meta::GetCacheLocationRequest *request,
                                  proto::meta::GetCacheLocationResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(GetCacheLocation, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            meta_service_impl_->GetCacheLocation(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetCacheLocationLen(coro_http::coro_http_connection *http_conn,
                                     proto::meta::GetCacheLocationLenRequest *request,
                                     proto::meta::GetCacheLocationLenResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(GetCacheLocationLen, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            meta_service_impl_->GetCacheLocationLen(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetCacheLocationsByBackend(coro_http::coro_http_connection *http_conn,
                                            proto::meta::GetCacheLocationsByBackendRequest *request,
                                            proto::meta::GetCacheLocationsByBackendResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(GetCacheLocationsByBackend, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            meta_service_impl_->GetCacheLocationsByBackend(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetCacheMeta(coro_http::coro_http_connection *http_conn,
                              proto::meta::GetCacheMetaRequest *request,
                              proto::meta::GetCacheMetaResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(GetCacheMeta, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            KVCM_LOG_INFO("[traceId: %s] GetCacheMeta called with instance id: %s, block keys count: %d, "
                          "token ids count: %d, detail level: %d",
                          request->trace_id().c_str(),
                          request->instance_id().c_str(),
                          request->block_keys_size(),
                          request->token_ids_size(),
                          request->detail_level());
            KVCM_LOG_DEBUG("[traceId: %s] GetCacheMeta request details: %s",
                           request->trace_id().c_str(),
                           request->ShortDebugString().c_str());
            meta_service_impl_->GetCacheMeta(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::StartWriteCache(coro_http::coro_http_connection *http_conn,
                                proto::meta::StartWriteCacheRequest *request,
                                proto::meta::StartWriteCacheResponse *response,
                                std::string_view json_body) {
    // The raw JSON is backed by the connection. Pass only the validation result to the worker.
    const auto *validation_error =
        request->has_storage_replica_requirement() ? ValidateReplicaTypeJson(json_body) : nullptr;
    return DispatchRequest(
        http_conn,
        request,
        response,
        [this, request, response, validation_error](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(StartWriteCache, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            if (validation_error) {
                auto *status = response->mutable_header()->mutable_status();
                status->set_code(proto::meta::INVALID_ARGUMENT);
                status->set_message(validation_error);
                request_context->set_status_code(status->code());
                return std::nullopt;
            }
            KVCM_LOG_DEBUG("[traceId: %s] StartWriteCache request details: %s",
                           request->trace_id().c_str(),
                           request->ShortDebugString().c_str());
            meta_service_impl_->StartWriteCache(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::FinishWriteCache(coro_http::coro_http_connection *http_conn,
                                  proto::meta::FinishWriteCacheRequest *request,
                                  proto::meta::CommonResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(FinishWriteCache, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            KVCM_LOG_INFO("[traceId: %s] FinishWriteCache called with instance id: %s, write session id: %s",
                          request->trace_id().c_str(),
                          request->instance_id().c_str(),
                          request->write_session_id().c_str());
            KVCM_LOG_DEBUG("[traceId: %s] FinishWriteCache request details: %s",
                           request->trace_id().c_str(),
                           request->ShortDebugString().c_str());
            meta_service_impl_->FinishWriteCache(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::RemoveCache(coro_http::coro_http_connection *http_conn,
                             proto::meta::RemoveCacheRequest *request,
                             proto::meta::CommonResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(RemoveCache, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            KVCM_LOG_INFO("[traceId: %s] RemoveCache called with instance id: %s, block keys count: %d, "
                          "token ids count: %d",
                          request->trace_id().c_str(),
                          request->instance_id().c_str(),
                          request->block_keys_size(),
                          request->token_ids_size());
            KVCM_LOG_DEBUG("[traceId: %s] RemoveCache request details: %s",
                           request->trace_id().c_str(),
                           request->ShortDebugString().c_str());
            meta_service_impl_->RemoveCache(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::TrimCache(coro_http::coro_http_connection *http_conn,
                           proto::meta::TrimCacheRequest *request,
                           proto::meta::CommonResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(TrimCache, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            KVCM_LOG_DEBUG("[traceId: %s] TrimCache request details: %s",
                           request->trace_id().c_str(),
                           request->ShortDebugString().c_str());
            meta_service_impl_->TrimCache(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetClusterInfo(coro_http::coro_http_connection *http_conn,
                                proto::meta::GetClusterInfoRequest *request,
                                proto::meta::GetClusterInfoResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            // instance_id 可能尚未注册（如 RegisterInstance 之前），此时 fallback 到全局 collector
            auto metrics_collector = get_metrics_collector_from_map_for_GetClusterInfo(request->instance_id());
            if (metrics_collector == nullptr) {
                metrics_collector = KVCM_METRICS_COLLECTOR_(GetClusterInfo);
            }
            request_context->InitMetricsCollector(std::move(metrics_collector));
            KVCM_LOG_DEBUG("[traceId: %s] GetClusterInfo request details: %s",
                           request->trace_id().c_str(),
                           request->ShortDebugString().c_str());
            meta_service_impl_->GetClusterInfo(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::ReportEvent(coro_http::coro_http_connection *http_conn,
                             proto::meta::ReportEventRequest *request,
                             proto::meta::ReportEventResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            std::string metrics_type;
            auto metrics_collector = ResolveReportEventMetricsCollector(*request, metrics_type);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            std::string first_event_type = "N/A";
            std::string first_block_key = "N/A";
            if (request->events_size() > 0) {
                const auto &first_event = request->events(0);
                first_event_type = proto::meta::ReportEventType_Name(first_event.event_type());
                if (first_event.has_block_add()) {
                    first_block_key = first_event.block_add().block_key();
                } else if (first_event.has_block_delete()) {
                    first_block_key = first_event.block_delete().block_key();
                } else if (first_event.has_block_snapshot()) {
                    if (first_event.block_snapshot().blocks_size() > 0) {
                        first_block_key = first_event.block_snapshot().blocks(0).block_key();
                    }
                }
            }
            KVCM_LOG_DEBUG("[traceId: %s] ReportEvent called, instance_id: %s, host_ip_port: %s, event_count: %d, "
                           "first_event_type: %s, first_block_key: %s",
                           request->trace_id().c_str(),
                           request->instance_id().c_str(),
                           request->host_ip_port().c_str(),
                           request->events_size(),
                           first_event_type.c_str(),
                           first_block_key.c_str());
            AttachReportEventTypeMetricsCollectors(*request, metrics_type, request_context);
            meta_service_impl_->ReportEvent(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetHostCacheState(coro_http::coro_http_connection *http_conn,
                                   proto::meta::GetHostCacheStateRequest *request,
                                   proto::meta::GetHostCacheStateResponse *response) {
    return DispatchRequest(
        http_conn, request, response, [this, request, response](RequestContext *request_context) -> CachedJsonResponse {
            API_CONTEXT_GET_AND_INIT_COLLECTOR(GetHostCacheState, std::nullopt);
            request_context->InitMetricsCollector(std::move(metrics_collector));
            KVCM_LOG_DEBUG("[traceId: %s] GetHostCacheState called, instance_id: %s, block_cache_keys_count: %d",
                           request->trace_id().c_str(),
                           request->instance_id().c_str(),
                           request->block_cache_keys_size());
            meta_service_impl_->GetHostCacheState(request_context, request, response);
            return request_context->TakeReusableResponseJson();
        });
}

} // namespace kv_cache_manager
