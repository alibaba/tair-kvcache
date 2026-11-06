#include "kv_cache_manager/service/debug_service_impl.h"

#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/protocol/protobuf/debug_service.pb.h"
#include "kv_cache_manager/service/util/fault_injector.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"

// 这里的字段检测不包含任何基本数据类型，例如int32、int64、bool等
#define CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN(api_name, manager_req, single_field)                               \
    do {                                                                                                               \
        if ((single_field)) {                                                                                          \
            invalid_fields += "{" api_name ": {" manager_req "}}";                                                     \
        } else {                                                                                                       \
            /* invalid_fields 已在ValidateRequiredFields构造完成 */                                              \
        }                                                                                                              \
        status->set_code(proto::debug::INVALID_ARGUMENT);                                                              \
        status->set_message(invalid_fields);                                                                           \
        KVCM_LOG_DEBUG("%s failed due to invalid %s fields: %s", api_name, manager_req, invalid_fields.c_str());       \
        return;                                                                                                        \
    } while (0)

namespace kv_cache_manager {

void DebugServiceImpl::GetCacheMetaDetail(const proto::meta::GetCacheMetaDetailRequest *request,
                                          proto::meta::GetCacheMetaDetailResponse *response) {
    RequestContext request_context(request->trace_id());
    auto *header = response->mutable_header();
    header->set_request_id(request_context.request_id());
    auto *status = header->mutable_status();
    if (request->instance_id().empty() || (request->block_keys().empty() && request->token_ids().empty())) {
        status->set_code(proto::meta::INVALID_ARGUMENT);
        status->set_message("instance_id and block_keys or token_ids are required");
        return;
    }

    BlockMask block_mask;
    ProtoConvert::BlockMaskFromProto(&request->block_mask(), block_mask);
    auto [ec, details] = cache_manager_->GetCacheMetaDetail(
        &request_context,
        request->instance_id(),
        std::vector<int64_t>(request->block_keys().begin(), request->block_keys().end()),
        std::vector<int64_t>(request->token_ids().begin(), request->token_ids().end()),
        block_mask,
        request->detail_level());

    size_t item_error_count = 0;
    for (const auto &detail : details) {
        item_error_count += detail.error_code != EC_OK;
        ProtoConvert::CacheKeyMetaDetailToProto(detail, response->add_items());
    }
    if (ec != EC_OK) {
        status->set_code(ToPbError<proto::meta::ErrorCode>(ec));
        status->set_message("Failed to get cache metadata detail: " + request_context.error_tracer()->ToJsonString());
        KVCM_LOG_ERROR("[traceId: %s] GetCacheMetaDetail failed, ec: %d, returned %d items",
                       request->trace_id().c_str(),
                       ec,
                       response->items_size());
        return;
    }
    status->set_code(proto::meta::OK);
    status->set_message(item_error_count == 0 ? "Cache metadata detail retrieved successfully"
                                              : "Cache metadata detail retrieved with " +
                                                    std::to_string(item_error_count) + " item errors");
}

void DebugServiceImpl::InjectFault(const proto::debug::InjectFaultRequest *request,
                                   proto::debug::CommonResponse *response) {
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->api_name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("InjectFault", "api_name", true);
    }
    if (!kv_cache_manager::proto::debug::FaultType_IsValid(static_cast<int>(request->fault_type()))) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("InjectFault", "fault_type", true);
    }
    if (!kv_cache_manager::proto::debug::FaultTriggerStrategy_IsValid(
            static_cast<int>(request->fault_trigger_strategy()))) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("InjectFault", "fault_trigger_strategy", true);
    }
    MethodFaultConfig cfg;
    cfg.fault_type = static_cast<FaultType>(request->fault_type());
    cfg.fault_trigger_strategy = static_cast<FaultTriggerStrategy>(request->fault_trigger_strategy());
    cfg.trigger_at_call = request->trigger_at_call();
    FaultInjector::GetInstance().InjectFault(request->api_name(), std::move(cfg));

    auto fault_type = static_cast<FaultType>(request->fault_type());
    auto fault_trigger_strategy = static_cast<FaultTriggerStrategy>(request->fault_trigger_strategy());
    status->set_code(proto::debug::OK);
    status->set_message("Inject fault successfully, api_name: " + request->api_name() + ", fault_type: " +
                        ToString(fault_type) + ", fault_trigger_strategy: " + ToString(fault_trigger_strategy));
    KVCM_LOG_INFO("Inject fault successfully, api_name: %s, fault_type: %s, fault_trigger_strategy: %s",
                  request->api_name().c_str(),
                  ToString(fault_type).c_str(),
                  ToString(fault_trigger_strategy).c_str());
    return;
}

void DebugServiceImpl::RemoveFault(const proto::debug::RemoveFaultRequest *request,
                                   proto::debug::CommonResponse *response) {
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    std::string invalid_fields = "missing or invalid fields: ";
    if (request->api_name().empty()) {
        CHECK_REQUIRED_FIELDS_VALIDATION_AND_RETURN("RemoveFault", "api_name", true);
    }
    bool ret = FaultInjector::GetInstance().RemoveFault(request->api_name());
    if (ret) {
        status->set_code(proto::debug::OK);
        status->set_message("Remove fault successfully, api_name: " + request->api_name());
        KVCM_LOG_INFO("Remove fault successfully, api_name: %s", request->api_name().c_str());
    } else {
        status->set_code(proto::debug::INVALID_ARGUMENT);
        status->set_message("Remove fault fail, api_name:" + request->api_name());
        KVCM_LOG_WARN("Remove fault fail, api_name: %s", request->api_name().c_str());
    }
    return;
}

void DebugServiceImpl::ClearFaults(const proto::debug::ClearFaultsRequest *request,
                                   proto::debug::CommonResponse *response) {
    auto *header = response->mutable_header();
    auto *status = header->mutable_status();
    FaultInjector::GetInstance().ClearFaults();
    status->set_code(proto::debug::OK);
    status->set_message("Clear faults successfully");
    KVCM_LOG_INFO("Clear faults successfully");
    return;
}

} // namespace kv_cache_manager
