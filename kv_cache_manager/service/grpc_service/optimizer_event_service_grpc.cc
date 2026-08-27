#include "kv_cache_manager/service/grpc_service/optimizer_event_service_grpc.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/optimizer_stream/subscription_event_sink.h"
#include "kv_cache_manager/manager/cache_manager.h"

namespace kv_cache_manager {

namespace {

ErrorCode DecodeOptimizerEventTokens(const proto::optimizer::TraceQueryRequest &request,
                                     RequestContext *request_context,
                                     CacheManager::TokenIdsVector *tokens) {
    if (request.token_ids_size() != 0 && !request.token_ids_le64().empty()) {
        request_context->error_tracer()->AddErrorMsg("token_ids and token_ids_le64 are mutually exclusive");
        return EC_BADARGS;
    }
    if (request.token_ids_le64().empty()) {
        tokens->assign(request.token_ids().begin(), request.token_ids().end());
        return EC_OK;
    }
    if (request.token_ids_le64().size() % sizeof(int64_t) != 0) {
        request_context->error_tracer()->AddErrorMsg("token_ids_le64 size must be a multiple of 8");
        return EC_BADARGS;
    }

    const auto token_count = request.token_ids_le64().size() / sizeof(int64_t);
    tokens->resize(token_count);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(tokens->data(), request.token_ids_le64().data(), request.token_ids_le64().size());
#else
    const auto *data = reinterpret_cast<const unsigned char *>(request.token_ids_le64().data());
    for (size_t token_index = 0; token_index < token_count; ++token_index) {
        uint64_t value = 0;
        for (size_t byte_index = 0; byte_index < sizeof(int64_t); ++byte_index) {
            value |= static_cast<uint64_t>(data[token_index * sizeof(int64_t) + byte_index]) << (byte_index * 8);
        }
        (*tokens)[token_index] = static_cast<int64_t>(value);
    }
#endif
    return EC_OK;
}

ErrorCode PublishOptimizerEvent(CacheManager *cache_manager,
                                RequestContext *request_context,
                                const proto::optimizer::TraceQueryRequest &request) {
    CacheManager::TokenIdsVector tokens;
    const auto decode_ec = DecodeOptimizerEventTokens(request, request_context, &tokens);
    if (decode_ec != EC_OK) {
        return decode_ec;
    }
    return cache_manager->ReportOptimizerEvent(
        request_context,
        request.instance_id(),
        {request.block_keys().begin(), request.block_keys().end()},
        tokens,
        request.input_token_len(),
        request.timestamp_ns(),
        {request.location_spec_names().begin(), request.location_spec_names().end()});
}

void SetOptimizerEventStatus(ErrorCode ec, RequestContext &request_context, proto::optimizer::Status *status) {
    if (ec == EC_OK) {
        status->set_code(proto::optimizer::OK);
    } else if (ec == EC_BADARGS) {
        status->set_code(proto::optimizer::INVALID_ARGUMENT);
        status->set_message(request_context.error_tracer()->ToJsonString());
    } else if (ec == EC_INSTANCE_NOT_EXIST) {
        status->set_code(proto::optimizer::INSTANCE_NOT_EXIST);
        status->set_message(request_context.error_tracer()->ToJsonString());
    } else {
        status->set_code(proto::optimizer::SERVICE_NOT_READY);
        status->set_message(request_context.error_tracer()->ToJsonString());
    }
}

} // namespace

OptimizerEventServiceGRpc::OptimizerEventServiceGRpc(std::shared_ptr<SubscriptionEventSink> sink,
                                                     std::shared_ptr<RegistryManager> registry_manager,
                                                     std::shared_ptr<LeaderElector> leader_elector,
                                                     std::shared_ptr<CacheManager> cache_manager)
    : sink_(std::move(sink))
    , registry_manager_(std::move(registry_manager))
    , leader_elector_(std::move(leader_elector))
    , cache_manager_(std::move(cache_manager)) {
    DisableSubscriptions();
}

void OptimizerEventServiceGRpc::EnableSubscriptions() {
    if (sink_) {
        sink_->EnableSubscriptions();
    }
}

void OptimizerEventServiceGRpc::DisableSubscriptions() {
    if (sink_) {
        sink_->DisableSubscriptions();
    }
}

bool OptimizerEventServiceGRpc::IsAvailable() const {
    return leader_elector_ && leader_elector_->GetRoleState() == RoleState::LEADER &&
           leader_elector_->IsStableState() && registry_manager_ && registry_manager_->IsRecoverComplete() && sink_ &&
           sink_->accepting_subscriptions() && !sink_->stopped();
}

grpc::Status OptimizerEventServiceGRpc::GetConfiguration(grpc::ServerContext *,
                                                         const proto::optimizer::KvcmConfigurationRequest *request,
                                                         proto::optimizer::KvcmConfigurationResponse *response) {
    auto *status = response->mutable_header()->mutable_status();
    if (!IsAvailable()) {
        status->set_code(proto::optimizer::SERVICE_NOT_READY);
        status->set_message("KVCM is unavailable");
        return grpc::Status::OK;
    }

    RequestContext request_context(request->trace_id());
    const auto [group_ec, instance_groups] = registry_manager_->ListInstanceGroup(&request_context);
    if (group_ec != EC_OK) {
        status->set_code(proto::optimizer::INTERNAL_ERROR);
        status->set_message("Failed to list KVCM instance groups");
        return grpc::Status::OK;
    }

    for (const auto &instance_group : instance_groups) {
        auto *group = response->add_instance_groups();
        group->set_name(instance_group->name());
        group->set_capacity_bytes(instance_group->quota().capacity());

        const auto [instance_ec, instances] =
            registry_manager_->ListInstanceInfo(&request_context, instance_group->name());
        if (instance_ec != EC_OK) {
            response->clear_instance_groups();
            response->clear_instances();
            status->set_code(proto::optimizer::INTERNAL_ERROR);
            status->set_message("Failed to list KVCM instances");
            return grpc::Status::OK;
        }
        for (const auto &instance_info : instances) {
            auto *instance = response->add_instances();
            instance->set_instance_group_name(instance_info->instance_group_name());
            instance->set_instance_id(instance_info->instance_id());
            instance->set_block_size(instance_info->block_size());
            for (const auto &spec_info : instance_info->location_spec_infos()) {
                auto *spec = instance->add_location_spec_infos();
                spec->set_name(spec_info.name());
                spec->set_size(spec_info.size());
            }
            for (const auto &spec_group : instance_info->location_spec_groups()) {
                auto *group_config = instance->add_location_spec_groups();
                group_config->set_name(spec_group.name());
                for (const auto &spec_name : spec_group.spec_names()) {
                    group_config->add_spec_names(spec_name);
                }
            }
        }
    }

    if (!IsAvailable()) {
        response->clear_instance_groups();
        response->clear_instances();
        status->set_code(proto::optimizer::SERVICE_NOT_READY);
        status->set_message("KVCM is unavailable");
        return grpc::Status::OK;
    }

    status->set_code(proto::optimizer::OK);
    return grpc::Status::OK;
}

grpc::Status OptimizerEventServiceGRpc::ReportOptimizerEvent(grpc::ServerContext *,
                                                             const proto::optimizer::TraceQueryRequest *request,
                                                             proto::optimizer::CommonResponse *response) {
    auto *status = response->mutable_header()->mutable_status();
    if (!IsAvailable() || !cache_manager_) {
        status->set_code(proto::optimizer::SERVICE_NOT_READY);
        status->set_message("KVCM is unavailable");
        return grpc::Status::OK;
    }

    RequestContext request_context(request->trace_id());
    const auto ec = PublishOptimizerEvent(cache_manager_.get(), &request_context, *request);
    SetOptimizerEventStatus(ec, request_context, status);
    return grpc::Status::OK;
}

grpc::Status
OptimizerEventServiceGRpc::SubscribeEvents(grpc::ServerContext *context,
                                           const proto::optimizer::OptimizerEventSubscriptionRequest *request,
                                           grpc::ServerWriter<proto::optimizer::TraceQueryRequest> *writer) {
    if (!IsAvailable()) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "KVCM is unavailable");
    }
    auto subscription = sink_->Subscribe(request->consumer_id());
    if (!subscription) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "KVCM is unavailable");
    }

    KVCM_LOG_INFO("OptimizerEventServiceGRpc: stream opened, consumer_id=%s, peer=%s",
                  subscription->consumer_id().c_str(),
                  context->peer().c_str());
    bool subscription_closed = false;
    while (!context->IsCancelled()) {
        proto::optimizer::TraceQueryRequest event;
        const auto result = subscription->WaitNext(&event, std::chrono::milliseconds(100));
        if (result == SubscriptionEventSink::Subscription::WaitResult::kTimeout) {
            continue;
        }
        if (result == SubscriptionEventSink::Subscription::WaitResult::kClosed) {
            subscription_closed = true;
            break;
        }
        if (!writer->Write(event)) {
            break;
        }
    }
    sink_->Unsubscribe(subscription);
    KVCM_LOG_INFO("OptimizerEventServiceGRpc: stream closed, consumer_id=%s", subscription->consumer_id().c_str());
    if (subscription_closed) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "KVCM is unavailable");
    }
    return grpc::Status::OK;
}

} // namespace kv_cache_manager
