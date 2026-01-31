#include "kv_cache_manager/service/meta_service_impl.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/manager/cache_location_view.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/util/fault_injector.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"
#include "kv_cache_manager/service/util/proto_message_json_util.h"
#include "kv_cache_manager/service/util/service_call_guard.h"

// TODO(rui): move into common.h
#define API_CALL_GUARD(api_name, is_leader_only)                                                                       \
    request_context->set_api_name(api_name);                                                                           \
    response->mutable_header()->set_request_id(request_context->request_id());                                         \
    std::string request_debug;                                                                                         \
    ProtoMessageJsonUtil::ToJson(request, request_debug);                                                              \
    request_context->set_request_debug(request_debug);                                                                 \
    if (!CheckAndIncrementRequestCount(is_leader_only)) {                                                              \
        auto *header = response->mutable_header();                                                                     \
        auto *status = header->mutable_status();                                                                       \
        status->set_code(proto::meta::SERVER_NOT_LEADER);                                                              \
        status->set_message("Server is not leader"); /* TODO: return current leader info */                            \
        request_context->set_status_code(status->code());                                                              \
        KVCM_LOG_INFO("[traceId: %s] %s rejected: service not ready", request->trace_id().c_str(), api_name);          \
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);                                                                \
        return;                                                                                                        \
    }                                                                                                                  \
    ServiceCallGuard service_call_guard(                                                                               \
        cache_manager_.get(), request_context, metrics_reporter_.get(), [request_context, response, this]() {          \
            std::string response_debug;                                                                                \
            ProtoMessageJsonUtil::ToJson(response, response_debug);                                                    \
            request_context->set_response_debug(response_debug);                                                       \
            DecrementRequestCount(is_leader_only);                                                                     \
        });

#define SET_SPAN_TRACER_STR_IN_HEADER(request_context_pointer)                                                         \
    if (request_context_pointer->need_span_tracer()) {                                                                 \
        span_tracer.reset();                                                                                           \
        span_tracer_helper.reset();                                                                                    \
        std::string span_tracer_result = request_context_pointer->EndAndGetSpanTracerDebugStr();                       \
        KVCM_LOG_INFO("trace_id [%s], span_tracer_result [%s]]",                                                       \
                      request_context_pointer->trace_id().c_str(),                                                     \
                      span_tracer_result.c_str());                                                                     \
        header->set_tracer_result(span_tracer_result);                                                                 \
    }

// 这里的字段检测不包含任何基本数据类型，例如int32、int64、bool等
#define CHECK_REQUIRED_FIELDS_VALIDATION(api_name, manager_req, single_field)                                          \
    do {                                                                                                               \
        if ((single_field)) {                                                                                          \
            invalid_fields += "{" api_name ": {" manager_req "}}";                                                     \
        } else {                                                                                                       \
            /* invalid_fields 已在ValidateRequiredFields构造完成 */                                              \
        }                                                                                                              \
        status->set_code(proto::meta::INVALID_ARGUMENT);                                                               \
        status->set_message(invalid_fields);                                                                           \
        request_context->set_status_code(status->code());                                                              \
        KVCM_LOG_DEBUG("[traceId: %s] %s failed due to invalid %s fields: %s",                                         \
                       request->trace_id().c_str(),                                                                    \
                       api_name,                                                                                       \
                       manager_req,                                                                                    \
                       invalid_fields.c_str());                                                                        \
    } while (0)

#define CHECK_FAULT_INJECTION(api_name)                                                                                \
    do {                                                                                                               \
        auto fault_opt = kv_cache_manager::FaultInjector::GetInstance().GetFault(api_name, request->trace_id());       \
        if (fault_opt.has_value()) {                                                                                   \
            /* 当前method在当前trace_id下会触发故障 */                                                      \
            const auto &cfg = fault_opt.value();                                                                       \
            status->set_code(kv_cache_manager::proto::meta::INTERNAL_ERROR);                                           \
            status->set_message(api_name " fail due to fault injection");                                              \
            request_context->set_status_code(status->code());                                                          \
            KVCM_LOG_WARN("[traceId: %s] %s fail, fault trigger strategy: %s",                                         \
                          request_context->trace_id().c_str(),                                                         \
                          api_name,                                                                                    \
                          ToString(cfg.fault_trigger_strategy).c_str());                                               \
            SET_SPAN_TRACER_STR_IN_HEADER(request_context);                                                            \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

namespace kv_cache_manager {

namespace {

inline proto::meta::ErrorCode ToPbError(ErrorCode internal_error) {
    static const std::unordered_map<ErrorCode, proto::meta::ErrorCode> error_map{
        {EC_UNIMPLEMENTED, proto::meta::UNSUPPORTED},
        {EC_BADARGS, proto::meta::INVALID_ARGUMENT},
        {EC_DUPLICATE_ENTITY, proto::meta::DUPLICATE_ENTITY},
        {EC_INSTANCE_NOT_EXIST, proto::meta::INSTANCE_NOT_EXIST},
        {EC_SERVICE_NOT_LEADER, proto::meta::SERVER_NOT_LEADER}};
    if (auto iter = error_map.find(internal_error); iter != error_map.end()) {
        return iter->second;
    }
    return proto::meta::INTERNAL_ERROR;
}

} // namespace

MetaServiceImpl::MetaServiceImpl(std::shared_ptr<CacheManager> cache_manager,
                                 std::shared_ptr<MetricsReporter> metrics_reporter)
    : ServiceImplBase(), cache_manager_(std::move(cache_manager)), metrics_reporter_(std::move(metrics_reporter)) {}

void MetaServiceImpl::RegisterInstance(RequestContext *request_context,
                                       const proto::meta::RegisterInstanceRequest *request,
                                       proto::meta::RegisterInstanceResponse *response) {
    SPAN_TRACER(request_context);
    API_CALL_GUARD("RegisterInstance", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_group().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("RegisterInstance", "instance_group", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("RegisterInstance", "instance_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 参数验证
    if (!request->has_model_deployment()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("RegisterInstance", "model_deployment", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 调用Manager层获取注册实例的结果
    ModelDeployment model_deployment_req;
    ProtoConvert::ModelDeploymentFromProto(&request->model_deployment(), model_deployment_req);
    if (!model_deployment_req.ValidateRequiredFields(invalid_fields)) {
        CHECK_REQUIRED_FIELDS_VALIDATION("RegisterInstance", "ModelDeployment", false);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }

    if (request->location_spec_infos().size() == 0) {
        CHECK_REQUIRED_FIELDS_VALIDATION("RegisterInstance", "location_spec_infos", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    std::vector<LocationSpecInfo> location_spec_infos;
    ProtoConvert::LocationSpecInfosFromProto(request->location_spec_infos(), location_spec_infos);
    {
        const std::string original_invalid_fields = invalid_fields;
        for (size_t i = 0; i < location_spec_infos.size(); ++i) {
            std::string local_invalid_fields = "index " + std::to_string(i) + ": ";
            if (!location_spec_infos[i].ValidateRequiredFields(local_invalid_fields)) {
                invalid_fields += local_invalid_fields;
            }
        }
        if (invalid_fields != original_invalid_fields) {
            CHECK_REQUIRED_FIELDS_VALIDATION("RegisterInstance", "LocationSpecInfo", false);
            SET_SPAN_TRACER_STR_IN_HEADER(request_context);
            return;
        }
    }
    std::vector<LocationSpecGroup> location_spec_groups;
    ProtoConvert::LocationSpecGroupsFromProto(request->location_spec_groups(), location_spec_groups);
    auto [ec_info, storage_configs] = cache_manager_->RegisterInstance(request_context,
                                                                       request->instance_group(),
                                                                       request->instance_id(),
                                                                       request->block_size(),
                                                                       location_spec_infos,
                                                                       model_deployment_req,
                                                                       location_spec_groups);

    if (ec_info != EC_OK) {
        status->set_code(ToPbError(ec_info));
        request_context->set_status_code(status->code());
        status->set_message("Failed to register instance : " + request_context->error_tracer()->ToJsonString());
        KVCM_LOG_ERROR("[traceId: %s] RegisterInstance failed", request->trace_id().c_str());
    } else {
        status->set_code(proto::meta::OK);
        request_context->set_status_code(status->code());
        status->set_message("Instance registered successfully");
        response->set_storage_configs(storage_configs);
        KVCM_LOG_INFO("[traceId: %s] RegisterInstance succeeded", request->trace_id().c_str());
    }
    SET_SPAN_TRACER_STR_IN_HEADER(request_context);
}

void MetaServiceImpl::GetInstanceInfo(RequestContext *request_context,
                                      const proto::meta::GetInstanceInfoRequest *request,
                                      proto::meta::GetInstanceInfoResponse *response) {
    SPAN_TRACER(request_context);
    API_CALL_GUARD("GetInstanceInfo", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("GetInstanceInfo", "instance_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 调用Manager层获取实例信息的结果
    std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>> get_instance_info =
        cache_manager_->GetInstanceInfo(request_context, request->instance_id());
    ErrorCode ec_info = get_instance_info.first;
    std::shared_ptr<const InstanceInfo> instance_info_ptr = get_instance_info.second;

    if (ec_info == ErrorCode::EC_INSTANCE_NOT_EXIST) {
        status->set_code(ToPbError(ec_info));
        status->set_message(request_context->error_tracer()->ToJsonString());
        request_context->set_status_code(status->code());
        KVCM_LOG_INFO(
            "[traceId: %s] GetInstanceInfo %s not found", request->trace_id().c_str(), request->instance_id().c_str());
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    if (ec_info != ErrorCode::EC_OK) {
        status->set_code(ToPbError(ec_info));
        status->set_message(request_context->error_tracer()->ToJsonString());
        request_context->set_status_code(status->code());
        KVCM_LOG_ERROR("[traceId: %s] GetInstanceInfo failed", request->trace_id().c_str());
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }

    status->set_code(proto::meta::OK);
    request_context->set_status_code(status->code());
    response->set_instance_group(instance_info_ptr->instance_group_name());
    ProtoConvert::InstanceInfoToProto(*instance_info_ptr, response->mutable_instance_info());
    KVCM_LOG_INFO("[traceId: %s] GetInstanceInfo succeeded", request->trace_id().c_str());
    SET_SPAN_TRACER_STR_IN_HEADER(request_context);
}

void MetaServiceImpl::GetCacheLocation(RequestContext *request_context,
                                       const proto::meta::GetCacheLocationRequest *request,
                                       proto::meta::GetCacheLocationResponse *response) {
    SPAN_TRACER(request_context);
    API_CALL_GUARD("GetCacheLocation", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    CHECK_FAULT_INJECTION("GetCacheLocation");
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("GetCacheLocation", "instance_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    if (request->block_keys().empty() && request->token_ids().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("GetCacheLocation", "block_keys and token_ids", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 调用Manager层获取缓存位置信息
    BlockMask block_mask_req;
    ProtoConvert::BlockMaskFromProto(&request->block_mask(), block_mask_req);
    // 处理location_spec_names参数
    std::vector<std::string> location_spec_names;
    location_spec_names.reserve(request->location_spec_names_size());
    for (const auto &name : request->location_spec_names()) {
        location_spec_names.push_back(name);
    }

    std::pair<ErrorCode, CacheLocationViewVecWrapper> get_cache_meta = cache_manager_->GetCacheLocation(
        request_context,
        request->instance_id(),
        static_cast<CacheManager::QueryType>(request->query_type()),
        std::vector<int64_t>(request->block_keys().begin(), request->block_keys().end()),
        std::vector<int64_t>(request->token_ids().begin(), request->token_ids().end()),
        block_mask_req,
        request->sw_size(),
        location_spec_names);
    ErrorCode ec_info = get_cache_meta.first;
    CacheLocationViewVecWrapper cache_location_view_vec_wrapper(std::move(get_cache_meta.second));
    CacheLocationViewVec cache_locations_res = cache_location_view_vec_wrapper.cache_locations_view();
    if (ec_info != EC_OK) {
        status->set_code(ToPbError(ec_info));
        request_context->set_status_code(status->code());
        status->set_message("Failed to get cache locations : " + request_context->error_tracer()->ToJsonString());
        KVCM_LOG_ERROR("[traceId: %s] GetCacheLocation failed, ec: %d", request->trace_id().c_str(), ec_info);
    } else {
        for (const auto &cache_location : cache_locations_res) {
            auto *location_meta = response->add_locations();
            ProtoConvert::CacheLocationViewToProto(cache_location, location_meta);
        }
        status->set_code(proto::meta::OK);
        request_context->set_status_code(status->code());
        status->set_message("Cache locations retrieved successfully");
        KVCM_LOG_INFO("[traceId: %s] GetCacheLocation succeeded, returned %d locations",
                      request->trace_id().c_str(),
                      response->locations_size());
    }
    SET_SPAN_TRACER_STR_IN_HEADER(request_context);
}

void MetaServiceImpl::GetCacheMeta(RequestContext *request_context,
                                   const proto::meta::GetCacheMetaRequest *request,
                                   proto::meta::GetCacheMetaResponse *response) {
    SPAN_TRACER(request_context);
    API_CALL_GUARD("GetCacheMeta", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("GetCacheMeta", "instance_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    if (request->block_keys().empty() && request->token_ids().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("GetCacheMeta", "block_keys and token_ids", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 调用Manager层获取缓存元数据
    BlockMask block_mask_req;
    ProtoConvert::BlockMaskFromProto(&request->block_mask(), block_mask_req);
    std::pair<ErrorCode, CacheMetaVecWrapper> get_cache_meta =
        cache_manager_->GetCacheMeta(request_context,
                                     request->instance_id(),
                                     std::vector<int64_t>(request->block_keys().begin(), request->block_keys().end()),
                                     std::vector<int64_t>(request->token_ids().begin(), request->token_ids().end()),
                                     block_mask_req,
                                     request->detail_level());
    ErrorCode ec_info = get_cache_meta.first;
    CacheMetaVecWrapper cache_meta_vec_wrapper(std::move(get_cache_meta.second));
    CacheLocationViewVec cache_locations_res = cache_meta_vec_wrapper.cache_locations_view();
    std::vector<std::string> metas_res = cache_meta_vec_wrapper.metas();

    if (ec_info != EC_OK) {
        status->set_code(ToPbError(ec_info));
        request_context->set_status_code(status->code());
        status->set_message("Failed to get cache metadata : " + request_context->error_tracer()->ToJsonString());
        KVCM_LOG_ERROR("[traceId: %s] GetCacheMeta failed", request->trace_id().c_str());
    } else {
        for (const auto &cache_location : cache_locations_res) {
            auto *location_meta = response->add_locations();
            ProtoConvert::CacheLocationViewToProto(cache_location, location_meta);
        }
        for (const auto &meta : metas_res) {
            response->add_metas(meta);
        }
        status->set_code(proto::meta::OK);
        request_context->set_status_code(status->code());
        status->set_message("Cache metadata retrieved successfully");
        KVCM_LOG_INFO("[traceId: %s] GetCacheMeta succeeded, returned %d locations and %d metas",
                      request->trace_id().c_str(),
                      response->locations_size(),
                      response->metas_size());
    }
    SET_SPAN_TRACER_STR_IN_HEADER(request_context);
}

void MetaServiceImpl::StartWriteCache(RequestContext *request_context,
                                      const proto::meta::StartWriteCacheRequest *request,
                                      proto::meta::StartWriteCacheResponse *response) {
    SPAN_TRACER(request_context);
    API_CALL_GUARD("StartWriteCache", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    CHECK_FAULT_INJECTION("StartWriteCache");
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("StartWriteCache", "instance_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    if (request->block_keys().empty() && request->token_ids().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("StartWriteCache", "block_keys and token_ids", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 调用Manager层开始写入缓存
    // 处理location_spec_group_names参数
    std::vector<std::string> location_spec_group_names;
    location_spec_group_names.reserve(request->location_spec_group_names_size());
    for (const auto &name : request->location_spec_group_names()) {
        location_spec_group_names.push_back(name);
    }

    std::pair<ErrorCode, StartWriteCacheInfo> start_write_cache = cache_manager_->StartWriteCache(
        request_context,
        request->instance_id(),
        std::vector<int64_t>(request->block_keys().begin(), request->block_keys().end()),
        std::vector<int64_t>(request->token_ids().begin(), request->token_ids().end()),
        location_spec_group_names,
        request->write_timeout_seconds());
    ErrorCode ec_info = start_write_cache.first;
    std::string write_session_id_res = start_write_cache.second.write_session_id();
    BlockMask block_mask_res = start_write_cache.second.block_mask();
    CacheLocationViewVecWrapper cache_locations_res(std::move(start_write_cache.second.locations_mut()));

    if (ec_info != EC_OK) {
        status->set_code(ToPbError(ec_info));
        request_context->set_status_code(status->code());
        status->set_message("Failed to start write cache : " + request_context->error_tracer()->ToJsonString());
        KVCM_LOG_ERROR("[traceId: %s] StartWriteCache failed", request->trace_id().c_str());
    } else {
        response->set_write_session_id(write_session_id_res);
        auto *block_mask_meta = response->mutable_block_mask();
        ProtoConvert::BlockMaskToProto(block_mask_res, block_mask_meta);
        for (const auto &cache_location : cache_locations_res.cache_locations_view()) {
            auto *location_meta = response->add_locations();
            ProtoConvert::CacheLocationViewToProto(cache_location, location_meta);
        }
        status->set_code(proto::meta::OK);
        request_context->set_status_code(status->code());
        status->set_message("Write cache session started successfully");
        KVCM_LOG_INFO("[traceId: %s] StartWriteCache succeeded, writeSessionId: %s, returned %d locations",
                      request->trace_id().c_str(),
                      write_session_id_res.c_str(),
                      response->locations_size());
    }
    SET_SPAN_TRACER_STR_IN_HEADER(request_context);
}

void MetaServiceImpl::FinishWriteCache(RequestContext *request_context,
                                       const proto::meta::FinishWriteCacheRequest *request,
                                       proto::meta::CommonResponse *response) {
    SPAN_TRACER(request_context);
    API_CALL_GUARD("FinishWriteCache", true);
    // 设置响应头
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    CHECK_FAULT_INJECTION("FinishWriteCache");
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("FinishWriteCache", "instance_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    if (request->write_session_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("FinishWriteCache", "write_session_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    if (!request->has_success_blocks()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("FinishWriteCache", "success_blocks", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 调用Manager层完成写入缓存
    BlockMask success_blocks_req;
    ProtoConvert::BlockMaskFromProto(&request->success_blocks(), success_blocks_req);
    ErrorCode ec_info = cache_manager_->FinishWriteCache(
        request_context, request->instance_id(), request->write_session_id(), success_blocks_req);

    if (ec_info != EC_OK) {
        status->set_code(ToPbError(ec_info));
        request_context->set_status_code(status->code());
        status->set_message("Failed to finish write cache : " + request_context->error_tracer()->ToJsonString());
        KVCM_LOG_ERROR("[traceId: %s] FinishWriteCache failed", request->trace_id().c_str());
    } else {
        status->set_code(proto::meta::OK);
        request_context->set_status_code(status->code());
        status->set_message("Write cache session finished successfully");
        KVCM_LOG_INFO("[traceId: %s] FinishWriteCache succeeded, writeSessionId: %s",
                      request->trace_id().c_str(),
                      request->write_session_id().c_str());
    }
    SET_SPAN_TRACER_STR_IN_HEADER(request_context);
}

void MetaServiceImpl::RemoveCache(RequestContext *request_context,
                                  const proto::meta::RemoveCacheRequest *request,
                                  proto::meta::CommonResponse *response) {
    SPAN_TRACER(request_context);
    API_CALL_GUARD("RemoveCache", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("RemoveCache", "instance_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    if (request->block_keys().empty() && request->token_ids().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("RemoveCache", "block_keys and token_ids", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 调用Manager层删除缓存
    BlockMask block_mask_req;
    ProtoConvert::BlockMaskFromProto(&request->block_mask(), block_mask_req);
    ErrorCode ec_info =
        cache_manager_->RemoveCache(request_context,
                                    request->instance_id(),
                                    std::vector<int64_t>(request->block_keys().begin(), request->block_keys().end()),
                                    std::vector<int64_t>(request->token_ids().begin(), request->token_ids().end()),
                                    block_mask_req);
    if (ec_info != EC_OK) {
        status->set_code(ToPbError(ec_info));
        request_context->set_status_code(status->code());
        status->set_message("Failed to remove cache : " + request_context->error_tracer()->ToJsonString());
        KVCM_LOG_ERROR("[traceId: %s] RemoveCache failed", request->trace_id().c_str());
    } else {
        status->set_code(proto::meta::OK);
        request_context->set_status_code(status->code());
        status->set_message("Cache removed successfully");
        KVCM_LOG_INFO("[traceId: %s] RemoveCache succeeded", request->trace_id().c_str());
    }
    SET_SPAN_TRACER_STR_IN_HEADER(request_context);
}

void MetaServiceImpl::TrimCache(RequestContext *request_context,
                                const proto::meta::TrimCacheRequest *request,
                                proto::meta::CommonResponse *response) {
    SPAN_TRACER(request_context);
    API_CALL_GUARD("TrimCache", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION("TrimCache", "instance_id", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 对于TS_TIMESTAMP策略，需要验证时间戳参数
    if (request->strategy() == proto::meta::TS_UNSPECIFIED) {
        CHECK_REQUIRED_FIELDS_VALIDATION("TrimCache", "strategy", true);
        SET_SPAN_TRACER_STR_IN_HEADER(request_context);
        return;
    }
    // 调用Manager层
    ErrorCode ec_info = cache_manager_->TrimCache(request_context,
                                                  request->instance_id(),
                                                  request->strategy(),
                                                  request->begin_timestamp(),
                                                  request->end_timestamp());
    if (ec_info != EC_OK) {
        status->set_code(ToPbError(ec_info));
        request_context->set_status_code(status->code());
        status->set_message("Failed to trim cache : " + request_context->error_tracer()->ToJsonString());
        KVCM_LOG_ERROR("[traceId: %s] TrimCache failed", request->trace_id().c_str());
    } else {
        status->set_code(proto::meta::OK);
        request_context->set_status_code(status->code());
        status->set_message("Cache trimmed successfully");
        KVCM_LOG_INFO("[traceId: %s] TrimCache succeeded", request->trace_id().c_str());
    }
    SET_SPAN_TRACER_STR_IN_HEADER(request_context);
}

} // namespace kv_cache_manager
