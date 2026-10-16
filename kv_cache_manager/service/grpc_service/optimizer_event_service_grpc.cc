#include "kv_cache_manager/service/grpc_service/optimizer_event_service_grpc.h"

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/optimizer_stream/subscription_event_sink.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/service/util/optimizer_event_proto_util.h"

namespace kv_cache_manager {

namespace {

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
    std::vector<int64_t> tokens;
    auto ec = DecodeOptimizerEventTokens(*request, &request_context, &tokens);
    if (ec == EC_OK) {
        ec = cache_manager_->ReportOptimizerEvent(
            &request_context,
            request->instance_id(),
            {request->block_keys().begin(), request->block_keys().end()},
            tokens,
            request->input_token_len(),
            request->timestamp_ns(),
            {request->location_spec_names().begin(), request->location_spec_names().end()});
    }
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
