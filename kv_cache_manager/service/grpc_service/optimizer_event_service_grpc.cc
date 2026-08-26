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
#include "kv_cache_manager/manager/hash_util.h"

namespace kv_cache_manager {

namespace {

constexpr int kMaxTraceObservationBatchSize = 256;

std::vector<std::int64_t> GenTraceBlockKeys(const google::protobuf::RepeatedField<std::int64_t> &tokens,
                                            std::int32_t block_size) {
    std::vector<std::int64_t> block_keys;
    if (block_size <= 0) {
        return block_keys;
    }
    const auto total_blocks = tokens.size() / block_size;
    block_keys.reserve(total_blocks);
    std::int64_t hash = 0;
    for (int index = 0; index < total_blocks; ++index) {
        const auto offset = index * block_size;
        hash = hashInt64Array(hash, tokens.data() + offset, tokens.data() + offset + block_size);
        block_keys.push_back(hash);
    }
    return block_keys;
}

} // namespace

OptimizerEventServiceGRpc::OptimizerEventServiceGRpc(std::shared_ptr<SubscriptionEventSink> sink,
                                                     std::shared_ptr<RegistryManager> registry_manager,
                                                     std::shared_ptr<LeaderElector> leader_elector)
    : sink_(std::move(sink))
    , registry_manager_(std::move(registry_manager))
    , leader_elector_(std::move(leader_elector)) {
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

void OptimizerEventServiceGRpc::ReportTraceBatch(
    const proto::optimizer::TraceObservationBatchRequest *request,
    proto::optimizer::TraceObservationBatchResponse *response) {
    auto *status = response->mutable_header()->mutable_status();
    if (!IsAvailable()) {
        status->set_code(proto::optimizer::SERVICE_NOT_READY);
        status->set_message("KVCM is unavailable");
        return;
    }
    if (request->producer_id().empty() || request->observations_size() == 0 ||
        request->observations_size() > kMaxTraceObservationBatchSize) {
        status->set_code(proto::optimizer::INVALID_ARGUMENT);
        status->set_message("producer_id and 1..256 observations are required");
        return;
    }

    std::uint64_t previous_sequence = 0;
    bool has_previous_sequence = false;
    for (const auto &observation : request->observations()) {
        if (observation.trace_id().empty() || observation.instance_id().empty() ||
            observation.token_ids().empty() ||
            (has_previous_sequence && observation.sequence() <= previous_sequence)) {
            status->set_code(proto::optimizer::INVALID_ARGUMENT);
            status->set_message("invalid observation or non-increasing sequence");
            return;
        }
        previous_sequence = observation.sequence();
        has_previous_sequence = true;
    }

    std::vector<proto::optimizer::TraceQueryRequest> events;
    events.reserve(request->observations_size());
    for (const auto &observation : request->observations()) {
        RequestContext request_context(observation.trace_id());
        const auto instance_info = registry_manager_->GetInstanceInfo(&request_context, observation.instance_id());
        if (!instance_info || instance_info->block_size() <= 0) {
            status->set_code(proto::optimizer::INSTANCE_NOT_EXIST);
            status->set_message("instance metadata is unavailable");
            return;
        }
        auto &event = events.emplace_back();
        event.set_trace_id(observation.trace_id());
        event.set_instance_id(observation.instance_id());
        event.set_input_token_len(observation.token_ids_size());
        event.set_timestamp_ns(observation.timestamp_ns());
        for (const auto key : GenTraceBlockKeys(observation.token_ids(), instance_info->block_size())) {
            event.add_block_keys(key);
        }
        for (const auto &name : observation.location_spec_names()) {
            event.add_location_spec_names(name);
        }
    }

    std::lock_guard<std::mutex> report_lock(report_trace_mutex_);
    if (!sink_->SendBatch(events)) {
        status->set_code(proto::optimizer::SERVICE_NOT_READY);
        status->set_message("optimizer subscriber queue is unavailable or full");
        return;
    }
    response->set_accepted_count(events.size());
    response->set_last_accepted_sequence(request->observations(request->observations_size() - 1).sequence());
    status->set_code(proto::optimizer::OK);
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
