#include "kv_cache_manager/service/admin_service_impl.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
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
        status->set_code(proto::admin::SERVER_NOT_LEADER);                                                             \
        status->set_message("Server is not leader"); /* TODO: return current leader info */                            \
        request_context->set_status_code(status->code());                                                              \
        KVCM_LOG_INFO("[traceId: %s] %s rejected: service not ready", request->trace_id().c_str(), api_name);          \
        return;                                                                                                        \
    }                                                                                                                  \
    ServiceCallGuard service_call_guard(                                                                               \
        cache_manager_.get(), request_context, metrics_reporter_.get(), [request_context, response, this]() {          \
            std::string response_debug;                                                                                \
            ProtoMessageJsonUtil::ToJson(response, response_debug);                                                    \
            request_context->set_response_debug(response_debug);                                                       \
            DecrementRequestCount(is_leader_only);                                                                     \
        });

// 这里的字段检测不包含任何基本数据类型，例如int32、int64、bool等
#define CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN(api_name, manager_req, single_field)                               \
    do {                                                                                                               \
        if ((single_field)) {                                                                                          \
            invalid_fields += "{" api_name ": {" manager_req "}}";                                                     \
        } else {                                                                                                       \
            /* invalid_fields 已在ValidateRequiredFields构造完成 */                                              \
        }                                                                                                              \
        status->set_code(proto::admin::INVALID_ARGUMENT);                                                              \
        status->set_message(invalid_fields);                                                                           \
        request_context->set_status_code(status->code());                                                              \
        KVCM_LOG_DEBUG("[traceId: %s] %s failed due to invalid %s fields: %s",                                         \
                       request->trace_id().c_str(),                                                                    \
                       api_name,                                                                                       \
                       manager_req,                                                                                    \
                       invalid_fields.c_str());                                                                        \
        return;                                                                                                        \
    } while (0)

namespace kv_cache_manager {

AdminServiceImpl::AdminServiceImpl(std::shared_ptr<CacheManager> cache_manager,
                                   std::shared_ptr<MetricsReporter> metrics_reporter,
                                   std::shared_ptr<MetricsRegistry> metrics_registry,
                                   std::shared_ptr<RegistryManager> registry_manager,
                                   std::shared_ptr<LeaderElector> leader_elector)
    : cache_manager_(std::move(cache_manager))
    , metrics_reporter_(std::move(metrics_reporter))
    , metrics_registry_(std::move(metrics_registry))
    , registry_manager_(std::move(registry_manager))
    , leader_elector_(std::move(leader_elector)) {}

// Storage相关接口实现
void AdminServiceImpl::AddStorage(RequestContext *request_context,
                                  const proto::admin::AddStorageRequest *request,
                                  proto::admin::CommonResponse *response) {
    API_CALL_GUARD("AddStorage", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (!request->has_storage()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("AddStorage", "storage", true);
    }
    StorageConfig storage_req;
    ProtoConvert::StorageFromProto(&request->storage(), storage_req);
    if (!storage_req.ValidateRequiredFields(invalid_fields)) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("AddStorage", "StorageConfig", false);
    }
    // KVCM_LOG_INFO("storage config: %s", storage_req.global_unique_name().c_str());
    ErrorCode ec_info = registry_manager_->AddStorage(request_context, storage_req);

    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        status->set_message("Failed to add storage");
        request_context->set_status_code(status->code());
        KVCM_LOG_ERROR("[traceId: %s] AddStorage failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Storage added successfully");
        KVCM_LOG_INFO("[traceId: %s] AddStorage succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::EnableStorage(RequestContext *request_context,
                                     const proto::admin::EnableStorageRequest *request,
                                     proto::admin::CommonResponse *response) {
    API_CALL_GUARD("EnableStorage", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->storage_unique_name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("EnableStorage", "storage_unique_name", true);
    }
    ErrorCode ec_info = registry_manager_->EnableStorage(request_context, request->storage_unique_name());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to enable storage");
        KVCM_LOG_ERROR("[traceId: %s] EnableStorage failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Storage enabled successfully");
        KVCM_LOG_INFO("[traceId: %s] EnableStorage succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::DisableStorage(RequestContext *request_context,
                                      const proto::admin::DisableStorageRequest *request,
                                      proto::admin::CommonResponse *response) {
    API_CALL_GUARD("DisableStorage", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->storage_unique_name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("DisableStorage", "storage_unique_name", true);
    }
    ErrorCode ec_info = registry_manager_->DisableStorage(request_context, request->storage_unique_name());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to disable storage");
        KVCM_LOG_ERROR("[traceId: %s] DisableStorage failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Storage disabled successfully");
        KVCM_LOG_INFO("[traceId: %s] DisableStorage succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::RemoveStorage(RequestContext *request_context,
                                     const proto::admin::RemoveStorageRequest *request,
                                     proto::admin::CommonResponse *response) {
    API_CALL_GUARD("RemoveStorage", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->storage_unique_name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RemoveStorage", "storage_unique_name", true);
    }
    ErrorCode ec_info = registry_manager_->RemoveStorage(request_context, request->storage_unique_name());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to remove storage");
        KVCM_LOG_ERROR("[traceId: %s] RemoveStorage failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Storage removed successfully");
        KVCM_LOG_INFO("[traceId: %s] RemoveStorage succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::UpdateStorage(RequestContext *request_context,
                                     const proto::admin::UpdateStorageRequest *request,
                                     proto::admin::CommonResponse *response) {
    API_CALL_GUARD("UpdateStorage", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    StorageConfig storage_req;
    std::string invalid_fields = "missing or invalid fields: ";
    if (!request->has_storage()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("UpdateStorage", "storage", true);
    }
    ProtoConvert::StorageFromProto(&request->storage(), storage_req);
    if (!storage_req.ValidateRequiredFields(invalid_fields)) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("AddStorage", "StorageConfig", false);
        return;
    }
    ErrorCode ec_info = registry_manager_->UpdateStorage(request_context, storage_req, request->force_update());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to update storage");
        KVCM_LOG_ERROR("[traceId: %s] UpdateStorage failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Storage updated successfully");
        KVCM_LOG_INFO("[traceId: %s] UpdateStorage succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::ListStorage(RequestContext *request_context,
                                   const proto::admin::ListStorageRequest *request,
                                   proto::admin::ListStorageResponse *response) {
    API_CALL_GUARD("ListStorage", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::pair<ErrorCode, std::vector<StorageConfig>> list_storage = registry_manager_->ListStorage(request_context);
    ErrorCode ec_info = list_storage.first;
    std::vector<StorageConfig> list_storage_res = list_storage.second;
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to list storage");
        KVCM_LOG_ERROR("[traceId: %s] ListStorage failed", request->trace_id().c_str());
        return;
    } else {
        for (const auto &storage_config : list_storage_res) {
            auto *storage_admin = response->add_storage();
            ProtoConvert::StorageConfigToProto(storage_config, storage_admin);
        }
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Storage listed successfully");
        KVCM_LOG_INFO("[traceId: %s] ListStorage succeeded", request->trace_id().c_str());
    }
};

// InstanceGroup相关接口实现
void AdminServiceImpl::CreateInstanceGroup(RequestContext *request_context,
                                           const proto::admin::CreateInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response) {
    API_CALL_GUARD("CreateInstanceGroup", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();

    InstanceGroup instance_group_req;
    std::string invalid_fields = "missing or invalid fields: ";
    if (!request->has_instance_group()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("CreateInstanceGroup", "instance_group", true);
    }
    ProtoConvert::InstanceGroupFromProto(&request->instance_group(), instance_group_req);
    if (!instance_group_req.ValidateRequiredFields(invalid_fields)) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("CreateInstanceGroup", "InstanceGroup", false);
        return;
    }
    ErrorCode ec_info = registry_manager_->CreateInstanceGroup(request_context, instance_group_req);
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to create instance group");
        KVCM_LOG_ERROR("[traceId: %s] CreateInstanceGroup failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Instance group created successfully");
        KVCM_LOG_INFO("[traceId: %s] CreateInstanceGroup succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::UpdateInstanceGroup(RequestContext *request_context,
                                           const proto::admin::UpdateInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response) {
    API_CALL_GUARD("UpdateInstanceGroup", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();

    InstanceGroup instance_group_req;
    std::string invalid_fields = "missing or invalid fields: ";
    if (!request->has_instance_group()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("UpdateInstanceGroup", "instance_group", true);
    }
    ProtoConvert::InstanceGroupFromProto(&request->instance_group(), instance_group_req);
    if (!instance_group_req.ValidateRequiredFields(invalid_fields)) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("UpdateInstanceGroup", "InstanceGroup", false);
    }
    ErrorCode ec_info =
        registry_manager_->UpdateInstanceGroup(request_context, instance_group_req, request->current_version());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to update instance group");
        KVCM_LOG_ERROR("[traceId: %s] UpdateInstanceGroup failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Instance group updated successfully");
        KVCM_LOG_INFO("[traceId: %s] UpdateInstanceGroup succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::RemoveInstanceGroup(RequestContext *request_context,
                                           const proto::admin::RemoveInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response) {
    API_CALL_GUARD("RemoveInstanceGroup", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RemoveInstanceGroup", "name", true);
    }
    ErrorCode ec_info = registry_manager_->RemoveInstanceGroup(request_context, request->name());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to remove instance group");
        KVCM_LOG_ERROR("[traceId: %s] RemoveInstanceGroup failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Instance group removed successfully");
        KVCM_LOG_INFO("[traceId: %s] RemoveInstanceGroup succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::GetInstanceGroup(RequestContext *request_context,
                                        const proto::admin::GetInstanceGroupRequest *request,
                                        proto::admin::GetInstanceGroupResponse *response) {
    API_CALL_GUARD("GetInstanceGroup", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("GetInstanceGroup", "name", true);
    }
    InstanceGroup instance_group_req;
    std::pair<ErrorCode, std::shared_ptr<const InstanceGroup>> get_instance_group =
        registry_manager_->GetInstanceGroup(request_context, request->name());
    ErrorCode ec_info = get_instance_group.first;
    std::shared_ptr<const InstanceGroup> instance_group_res = get_instance_group.second;
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to get instance group");
        KVCM_LOG_ERROR("[traceId: %s] GetInstanceGroup failed", request->trace_id().c_str());
        return;
    } else {
        auto *instance_group_config = response->mutable_instance_group();
        ProtoConvert::InstanceGroupToProto(*instance_group_res, instance_group_config);
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Instance group retrieved successfully");
        KVCM_LOG_INFO("[traceId: %s] GetInstanceGroup succeeded", request->trace_id().c_str());
    }
}

// Cache相关接口实现
void AdminServiceImpl::GetCacheMeta(RequestContext *request_context,
                                    const proto::admin::GetCacheMetaRequest *request,
                                    proto::admin::GetCacheMetaResponse *response) {
    API_CALL_GUARD("GetCacheMeta", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("GetCacheMeta", "instance_id", true);
    }
    if (request->block_keys().empty() && request->token_ids().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("GetCacheMeta", "block_keys and token_ids", true);
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
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to get cache metadata");
        KVCM_LOG_ERROR("[traceId: %s] GetCacheMeta failed", request->trace_id().c_str());
        return;
    } else {
        for (const auto &cache_location : cache_locations_res) {
            auto *location_meta = response->add_locations();
            ProtoConvert::CacheLocationViewToProto(cache_location, location_meta);
        }
        for (const auto &meta : metas_res) {
            response->add_metas(meta);
        }
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Cache metadata retrieved successfully");
        KVCM_LOG_INFO("[traceId: %s] GetCacheMeta succeeded, returned %d locations and %d metas",
                      request->trace_id().c_str(),
                      response->locations_size(),
                      response->metas_size());
    }
}

void AdminServiceImpl::RemoveCache(RequestContext *request_context,
                                   const proto::admin::RemoveCacheRequest *request,
                                   proto::admin::CommonResponse *response) {
    API_CALL_GUARD("RemoveCache", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RemoveCache", "instance_id", true);
    }
    if (request->block_keys().empty() && request->token_ids().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RemoveCache", "block_keys and token_ids", true);
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
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to remove cache");
        KVCM_LOG_ERROR("[traceId: %s] RemoveCache failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Cache removed successfully");
        KVCM_LOG_INFO("[traceId: %s] RemoveCache succeeded", request->trace_id().c_str());
    }
};

// Instance相关接口实现
void AdminServiceImpl::RegisterInstance(RequestContext *request_context,
                                        const proto::admin::RegisterInstanceRequest *request,
                                        proto::admin::CommonResponse *response) {
    API_CALL_GUARD("RegisterInstance", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_group().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RegisterInstance", "instance_group", true);
    }
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RegisterInstance", "instance_id", true);
    }
    if (!request->has_model_deployment()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RegisterInstance", "model_deployment", true);
    }
    ModelDeployment model_deployment_req;
    ProtoConvert::ModelDeploymentFromProto(&request->model_deployment(), model_deployment_req);
    if (!model_deployment_req.ValidateRequiredFields(invalid_fields)) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RegisterInstance", "ModelDeployment", false);
    }
    if (request->location_spec_infos().size() == 0) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RegisterInstance", "location_spec_infos", true);
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
            CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RegisterInstance", "LocationSpecInfo", false);
        }
    }
    std::vector<LocationSpecGroup> location_spec_groups;
    ProtoConvert::LocationSpecGroupsFromProto(request->location_spec_groups(), location_spec_groups);
    auto [ec_info, _] = cache_manager_->RegisterInstance(request_context,
                                                         request->instance_group(),
                                                         request->instance_id(),
                                                         request->block_size(),
                                                         location_spec_infos,
                                                         model_deployment_req,
                                                         location_spec_groups);
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to register instance");
        KVCM_LOG_ERROR("[traceId: %s] RegisterInstance failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Instance registered successfully");
        KVCM_LOG_INFO("[traceId: %s] RegisterInstance succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::RemoveInstance(RequestContext *request_context,
                                      const proto::admin::RemoveInstanceRequest *request,
                                      proto::admin::CommonResponse *response) {
    API_CALL_GUARD("RemoveInstance", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RemoveInstance", "instance_id", true);
    }
    ErrorCode ec_info =
        cache_manager_->RemoveInstance(request_context, request->instance_group(), request->instance_id());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to remove instance");
        KVCM_LOG_ERROR("[traceId: %s] RemoveInstance failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Instance removed successfully");
        KVCM_LOG_INFO("[traceId: %s] RemoveInstance succeeded", request->trace_id().c_str());
    }
}

void AdminServiceImpl::GetInstanceInfo(RequestContext *request_context,
                                       const proto::admin::GetInstanceInfoRequest *request,
                                       proto::admin::GetInstanceInfoResponse *response) {
    API_CALL_GUARD("GetInstanceInfo", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_id().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("GetInstanceInfo", "instance_id", true);
    }
    std::shared_ptr<const InstanceInfo> get_instance_info_res =
        registry_manager_->GetInstanceInfo(request_context, request->instance_id());
    if (!get_instance_info_res) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to get instance info");
        KVCM_LOG_ERROR("[traceId: %s] GetInstanceInfo failed", request->trace_id().c_str());
        return;
    } else {
        ProtoConvert::InstanceInfoToProto(*get_instance_info_res, response->mutable_instance_info());
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        KVCM_LOG_INFO("[traceId: %s] GetInstanceInfo succeeded", request->trace_id().c_str());
    }
};
void AdminServiceImpl::ListInstanceInfo(RequestContext *request_context,
                                        const proto::admin::ListInstanceInfoRequest *request,
                                        proto::admin::ListInstanceInfoResponse *response) {
    API_CALL_GUARD("ListInstanceInfo", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->instance_group_name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("ListInstanceInfo", "instance_group_name", true);
    }
    std::pair<ErrorCode, std::vector<std::shared_ptr<const InstanceInfo>>> list_instance_info =
        registry_manager_->ListInstanceInfo(request_context, request->instance_group_name());
    ErrorCode ec_info = list_instance_info.first;
    std::vector<std::shared_ptr<const InstanceInfo>> list_instance_info_res = list_instance_info.second;
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        KVCM_LOG_ERROR("[traceId: %s] ListInstanceInfo failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        for (const auto &instance_info : list_instance_info_res) {
            auto *instance_info_config = response->add_instance_info();
            ProtoConvert::InstanceInfoToProto(*instance_info, instance_info_config);
        }
        KVCM_LOG_INFO("[traceId: %s] ListInstanceInfo succeeded, returned %d instances",
                      request->trace_id().c_str(),
                      response->instance_info_size());
    }
};

// Account相关接口实现
void AdminServiceImpl::AddAccount(RequestContext *request_context,
                                  const proto::admin::AddAccountRequest *request,
                                  proto::admin::CommonResponse *response) {
    API_CALL_GUARD("AddAccount", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->user_name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("AddAccount", "user_name", true);
    }
    proto::admin::AccountRole role = request->role();
    AccountRole role_req;
    switch (role) {
    case proto::admin::ROLE_USER:
        role_req = AccountRole::ROLE_USER;
        break;
    case proto::admin::ROLE_ADMIN:
        role_req = AccountRole::ROLE_ADMIN;
        break;
    default:
        role_req = AccountRole::ROLE_USER; // 默认值
        break;
    }
    ErrorCode ec_info =
        registry_manager_->AddAccount(request_context, request->user_name(), request->password(), role_req);

    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to add account");
        KVCM_LOG_ERROR("[traceId: %s] AddAccount failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Account added successfully");
        KVCM_LOG_INFO("[traceId: %s] AddAccount succeeded", request->trace_id().c_str());
    }
};
void AdminServiceImpl::DeleteAccount(RequestContext *request_context,
                                     const proto::admin::DeleteAccountRequest *request,
                                     proto::admin::CommonResponse *response) {
    API_CALL_GUARD("DeleteAccount", true);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->user_name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("DeleteAccount", "user_name", true);
    }
    ErrorCode ec_info = registry_manager_->DeleteAccount(request_context, request->user_name());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to delete account");
        KVCM_LOG_ERROR("[traceId: %s] DeleteAccount failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Account deleted successfully");
        KVCM_LOG_INFO("[traceId: %s] DeleteAccount succeeded", request->trace_id().c_str());
    }
};
void AdminServiceImpl::ListAccount(RequestContext *request_context,
                                   const proto::admin::ListAccountRequest *request,
                                   proto::admin::ListAccountResponse *response) {
    API_CALL_GUARD("ListAccount", true);
    std::pair<ErrorCode, std::vector<std::shared_ptr<const Account>>> list_account =
        registry_manager_->ListAccount(request_context);
    ErrorCode ec_info = list_account.first;
    std::vector<std::shared_ptr<const Account>> list_account_res = list_account.second;
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to list account");
        KVCM_LOG_ERROR("[traceId: %s] ListAccount failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("list account successfully");
        for (const auto &account : list_account_res) {
            auto *account_config = response->add_accounts();
            ProtoConvert::AccountToProto(*account, account_config);
        }
        KVCM_LOG_INFO("[traceId: %s] ListAccount succeeded, returned %d accounts",
                      request->trace_id().c_str(),
                      response->accounts_size());
    }
};
// ConfigSnapshot相关接口实现
void AdminServiceImpl::GenConfigSnapshot(RequestContext *request_context,
                                         const proto::admin::GenConfigSnapshotRequest *request,
                                         proto::admin::ConfigSnapShotResponse *response) {
    API_CALL_GUARD("GenConfigSnapshot", true);

    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    // std::string config_snapshot;
    std::pair<ErrorCode, std::string> gen_config_snapshot = registry_manager_->GenConfigSnapshot(request_context);
    ErrorCode ec_info = gen_config_snapshot.first;
    std::string gen_config_snapshot_res = gen_config_snapshot.second;
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to generate config snapshot");
        KVCM_LOG_ERROR("[traceId: %s] GenConfigSnapshot failed", request->trace_id().c_str());
        return;
    } else {
        response->set_config_snapshot(gen_config_snapshot_res);
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Config snapshot generated successfully");
        KVCM_LOG_INFO("[traceId: %s] GenConfigSnapshot succeeded", request->trace_id().c_str());
    }
};

void AdminServiceImpl::LoadConfigSnapshot(RequestContext *request_context,
                                          const proto::admin::LoadConfigSnapshotRequest *request,
                                          proto::admin::CommonResponse *response) {
    API_CALL_GUARD("LoadConfigSnapshot", true);

    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    ErrorCode ec_info = registry_manager_->LoadConfigSnapshot(request_context, request->config_snapshot());
    if (ec_info != EC_OK) {
        status->set_code(proto::admin::INTERNAL_ERROR);
        request_context->set_status_code(status->code());
        status->set_message("Failed to load config snapshot");
        KVCM_LOG_ERROR("[traceId: %s] LoadConfigSnapshot failed", request->trace_id().c_str());
        return;
    } else {
        status->set_code(proto::admin::OK);
        request_context->set_status_code(status->code());
        status->set_message("Config snapshot loaded successfully");
        KVCM_LOG_INFO("[traceId: %s] LoadConfigSnapshot succeeded", request->trace_id().c_str());
    }
}

void AdminServiceImpl::GetMetrics(RequestContext *request_context,
                                  const proto::admin::GetMetricsRequest *request,
                                  proto::admin::GetMetricsResponse *response) {
    API_CALL_GUARD("GetMetrics", false);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();

    std::vector<MetricsRegistry::metrics_tuple_t> all_metrics;
    metrics_registry_->GetAllMetrics(all_metrics);

    for (const auto &[name, tags, val] : all_metrics) {
        auto *pb_data = response->add_metrics();

        // name
        pb_data->set_metric_name(name);

        // tags
        for (const auto &[tag_name, tag_value] : tags) {
            auto *tag = pb_data->add_metric_tags();
            tag->set_tag_key(tag_name);
            tag->set_tag_value(tag_value);
        }

        // value
        if (std::holds_alternative<CounterValue>(*val)) {
            pb_data->mutable_metric_value()->set_int_value(
                static_cast<std::int64_t>(std::get<CounterValue>(*val).load()));
        } else if (std::holds_alternative<GaugeValue>(*val)) {
            pb_data->mutable_metric_value()->set_float_value(std::get<GaugeValue>(*val).load());
        }
    }

    response->set_timestamp(TimestampUtil::GetCurrentTimeMs());
    status->set_code(proto::admin::OK);
    request_context->set_status_code(status->code());
    status->set_message("Metrics retrieved successfully");
    KVCM_LOG_INFO("[traceId: %s] GetMetrics succeeded", request->trace_id().c_str());
}

// 高可用相关接口实现
void AdminServiceImpl::CheckHealth(RequestContext *request_context,
                                   const proto::admin::CheckHealthRequest *request,
                                   proto::admin::CheckHealthResponse *response) {
    API_CALL_GUARD("CheckHealth", false);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();

    response->set_is_leader(leader_elector_->IsLeader());
    response->set_is_health(true); // TODO: 实现健康检查逻辑，如IOHang探测、elector长时间不loop等
    response->set_elector_last_loop_time_ms(leader_elector_->GetLastLoopTimeUs() / 1000);

    status->set_code(proto::admin::OK);
    request_context->set_status_code(status->code());
    status->set_message("Health check completed");
    KVCM_LOG_INFO("[traceId: %s] CheckHealth succeeded", request->trace_id().c_str());
}

void AdminServiceImpl::GetManagerClusterInfo(RequestContext *request_context,
                                             const proto::admin::GetManagerClusterInfoRequest *request,
                                             proto::admin::GetManagerClusterInfoResponse *response) {
    API_CALL_GUARD("GetManagerClusterInfo", false);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();

    response->set_self_leader_expiration_time(leader_elector_->GetLeaseExpirationTime());
    response->set_leader_node_id(leader_elector_->GetLeaderNodeID());
    response->set_self_node_id(leader_elector_->GetSelfNodeID());
    // TODO: 实现实际的集群信息获取逻辑
    response->set_info_updated_time(0);

    status->set_code(proto::admin::OK);
    request_context->set_status_code(status->code());
    status->set_message("Manager cluster info retrieved");
    KVCM_LOG_INFO("[traceId: %s] GetManagerClusterInfo succeeded", request->trace_id().c_str());
}

void AdminServiceImpl::LeaderDemote(RequestContext *request_context,
                                    const proto::admin::LeaderDemoteRequest *request,
                                    proto::admin::CommonResponse *response) {
    API_CALL_GUARD("LeaderDemote", false); // 不标记为leader请求，避免demote和等待所有leader请求结束互相死锁
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();

    if (!leader_elector_->IsLeader()) {
        status->set_code(proto::admin::SERVER_NOT_LEADER);
        request_context->set_status_code(status->code());
        status->set_message("Server is not leader");
        return;
    }

    leader_elector_->Demote();

    status->set_code(proto::admin::OK);
    request_context->set_status_code(status->code());
    status->set_message("Leader demote request accepted");
    KVCM_LOG_INFO("[traceId: %s] LeaderDemote succeeded", request->trace_id().c_str());
}

void AdminServiceImpl::GetLeaderElectorConfig(RequestContext *request_context,
                                              const proto::admin::GetLeaderElectorConfigRequest *request,
                                              proto::admin::GetLeaderElectorConfigResponse *response) {
    API_CALL_GUARD("GetLeaderElectorConfig", false);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();

    response->set_campaign_delay_time_ms(leader_elector_->GetForbidCampaignLeaderTimeMs());

    status->set_code(proto::admin::OK);
    request_context->set_status_code(status->code());
    status->set_message("Leader elector config retrieved");
    KVCM_LOG_INFO("[traceId: %s] GetLeaderElectorConfig succeeded", request->trace_id().c_str());
}

void AdminServiceImpl::UpdateLeaderElectorConfig(RequestContext *request_context,
                                                 const proto::admin::UpdateLeaderElectorConfigRequest *request,
                                                 proto::admin::CommonResponse *response) {
    API_CALL_GUARD("UpdateLeaderElectorConfig", false);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();

    leader_elector_->SetForbidCampaignLeaderTimeMs(request->campaign_delay_time_ms());

    status->set_code(proto::admin::OK);
    request_context->set_status_code(status->code());
    status->set_message("Leader elector config updated");
    KVCM_LOG_INFO("[traceId: %s] UpdateLeaderElectorConfig succeeded", request->trace_id().c_str());
}

void AdminServiceImpl::UpdateLogger(RequestContext *request_context,
                                    const proto::admin::UpdateLoggerRequest *request,
                                    proto::admin::CommonResponse *response) {
    API_CALL_GUARD("UpdateLogger", false);
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    uint32_t log_level = Logger::StringToLevel(request->log_level());
    if (log_level == Logger::LEVEL_UNSET) {
        status->set_code(proto::admin::INVALID_ARGUMENT);
        request_context->set_status_code(status->code());
        status->set_message("Invalid log level string");
        return;
    }
    LoggerBroker::SetLogLevel(log_level);
    status->set_code(proto::admin::OK);
    request_context->set_status_code(status->code());
    status->set_message("Update logger success");
    KVCM_LOG_INFO("[traceId: %s] Update logger success", request->trace_id().c_str());
}

} // namespace kv_cache_manager
