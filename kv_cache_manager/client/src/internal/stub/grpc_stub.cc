#include "kv_cache_manager/client/src/internal/stub/grpc_stub.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <grpcpp/grpcpp.h>
#include <type_traits>

#include "kv_cache_manager/client/src/internal/util/debug_string_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"

#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a##b

#define BASE_NARGS(...) BASE_NARGS_IMPL(__VA_ARGS__, 1, 0)
#define BASE_NARGS_IMPL(_1, N, ...) N

#define BASE_RETURN_0(ec, ...) ec
#define BASE_RETURN_1(ec, v1)                                                                                          \
    { ec, v1 }
#define BASE_RETURN(ec, ...) CAT(BASE_RETURN_, BASE_NARGS(__VA_ARGS__))(ec, __VA_ARGS__)

#define PREFIX_LOG(LEVEL, format, args...)                                                                             \
    KVCM_LOG_##LEVEL("trace_id [%s] instance [%s] | " format, trace_id.c_str(), instance_id.c_str(), ##args);

#define CHECK_GRPC_STATUS_BASE(grpc_status, ...)                                                                       \
    if (!grpc_status.ok()) {                                                                                           \
        PREFIX_LOG(WARN, "grpc error [%s], code [%d]", grpc_status.error_message().c_str(), grpc_status.error_code()); \
        return BASE_RETURN(ER_INVALID_GRPCSTATUS, __VA_ARGS__);                                                        \
    }
#define CHECK_GRPC_STATUS(grpc_status) CHECK_GRPC_STATUS_BASE(grpc_status)
#define CHECK_GRPC_STATUS_WITH_TYPE(grpc_status) CHECK_GRPC_STATUS_BASE(grpc_status, {})

#define CHECK_COMMON_HEADER_BASE(response, ...)                                                                        \
    if (!response.has_header() || !response.header().has_status()) {                                                   \
        PREFIX_LOG(WARN, "common response invalid");                                                                   \
        return BASE_RETURN(ER_SERVICE_NO_STATUS, __VA_ARGS__);                                                         \
    }                                                                                                                  \
    if (response.header().status().code() != proto::meta::ErrorCode::OK) {                                             \
        const auto &status = response.header().status();                                                               \
        ClientErrorCode client_ec = ToClientError(status.code());                                                      \
        PREFIX_LOG(WARN,                                                                                               \
                   "request_id [%s], service_err[%d], msg[%s], client_err [%d]",                                       \
                   response.header().request_id().c_str(),                                                             \
                   status.code(),                                                                                      \
                   status.message().c_str(),                                                                           \
                   client_ec);                                                                                         \
        return BASE_RETURN(client_ec, __VA_ARGS__);                                                                    \
    }
#define CHECK_COMMON_HEADER(response) CHECK_COMMON_HEADER_BASE(response)
#define CHECK_COMMON_HEADER_WITH_TYPE(response) CHECK_COMMON_HEADER_BASE(response, {})

#define GET_AND_CHECK_STUB_BASE(...)                                                                                   \
    GetStub();                                                                                                         \
    if (stub == nullptr) {                                                                                             \
        PREFIX_LOG(ERROR, "no valid stub");                                                                            \
        return BASE_RETURN(ER_INVALID_STUB, __VA_ARGS__);                                                              \
    }
#define GET_AND_CHECK_STUB() GET_AND_CHECK_STUB_BASE()
#define GET_AND_CHECK_STUB_WITH_TYPE() GET_AND_CHECK_STUB_BASE({})

namespace {

kv_cache_manager::Locations GenLocations(
    const google::protobuf::RepeatedPtrField<::kv_cache_manager::proto::meta::CacheLocation> &proto_locations) {
    kv_cache_manager::Locations locations;
    locations.reserve(proto_locations.size());
    for (const auto &proto_location : proto_locations) {
        const auto &location_specs = proto_location.location_specs();
        locations.push_back({});
        locations.back().reserve(location_specs.size());
        for (const auto &location_spec : location_specs) {
            locations.back().push_back({location_spec.name(), location_spec.uri()});
        }
    }
    return locations;
}

kv_cache_manager::ClientErrorCode
GenCacheLocation(const kv_cache_manager::Locations &locations,
                 google::protobuf::RepeatedPtrField<::kv_cache_manager::proto::meta::CacheLocation> *proto_locations) {
    if (locations.empty()) {
        return kv_cache_manager::ClientErrorCode::ER_OK;
    }
    for (const auto &location : locations) {
        auto *cache_location = proto_locations->Add();
        cache_location->set_type(::kv_cache_manager::proto::meta::StorageType::ST_UNSPECIFIED);
        cache_location->set_spec_size(-1);
        for (const auto &location_spec : location) {
            auto *spec = cache_location->add_location_specs();
            spec->set_name(location_spec.spec_name);
            spec->set_uri(location_spec.uri);
        }
    }
    return kv_cache_manager::ClientErrorCode::ER_OK;
}

template <typename ProtoMessage>
inline std::enable_if_t<std::is_base_of_v<google::protobuf::Message, ProtoMessage>>
SetCommonInfo(ProtoMessage &proto_message, const std::string &trace_id, const std::string &instance_id) {
    proto_message.set_trace_id(trace_id);
    proto_message.set_instance_id(instance_id);
}

template <typename ProtoMessage>
inline std::enable_if_t<std::is_base_of_v<google::protobuf::Message, ProtoMessage>>
SetKeysAndTokens(ProtoMessage &proto_message,
                 const std::string &trace_id,
                 const std::string &instance_id,
                 const kv_cache_manager::Stub::KeyVector &keys,
                 const kv_cache_manager::Stub::TokenIdsVector &tokens) {
    SetCommonInfo(proto_message, trace_id, instance_id);
    std::for_each(keys.begin(), keys.end(), [&proto_message](kv_cache_manager::Stub::KeyType key) {
        proto_message.add_block_keys(key);
    });
    std::for_each(tokens.begin(), tokens.end(), [&proto_message](kv_cache_manager::Stub::KeyType token) {
        proto_message.add_token_ids(token);
    });
}

kv_cache_manager::ClientErrorCode ToClientError(kv_cache_manager::proto::meta::ErrorCode service_error) {
    using namespace kv_cache_manager;
    static const std::unordered_map<kv_cache_manager::proto::meta::ErrorCode, kv_cache_manager::ClientErrorCode>
        error_map{{proto::meta::UNSUPPORTED, ER_SERVICE_UNSUPPORTED},
                  {proto::meta::INVALID_ARGUMENT, ER_SERVICE_INVALID_ARGUMENT},
                  {proto::meta::DUPLICATE_ENTITY, ER_SERVICE_DUPLICATE_ENTITY},
                  {proto::meta::INSTANCE_NOT_EXIST, ER_SERVICE_INSTANCE_NOT_EXIST},
                  {proto::meta::SERVER_NOT_LEADER, ER_SERVICE_NOT_LEADER}};
    if (auto iter = error_map.find(service_error); iter != error_map.end()) {
        return iter->second;
    }
    return ER_SERVICE_INTERNAL_ERROR;
}

const static std::string kGrpcNoRetryPolicyFormat = R"(
    {
        "methodConfig":
        [
            {
                "name":
                [
                    {
                        "service": "kv_cache_manager.proto.meta.MetaService"
                    }
                ],
                "waitForReady": true,
                "timeout": "%.3fs"
            }
        ]
    })";

const static std::string kGrpcRetryPolicyFormat = R"(
{
    "methodConfig":
    [
        {
            "name":
            [
                {
                    "service": "kv_cache_manager.proto.meta.MetaService"
                }
            ],
            "waitForReady": true,
            "timeout": "%.3fs",
            "retryPolicy":
            {
                "maxAttempts": %u,
                "initialBackoff": "%.3fs",
                "maxBackoff": "%.3fs",
                "backoffMultiplier": 1.2,
                "retryableStatusCodes":
                [
                    "UNAVAILABLE"
                ]
            }
        }
    ]
})";

} // namespace

namespace kv_cache_manager {

GrpcStub::GrpcStub(uint32_t retry_time, uint32_t call_timeout)
    : retry_time_(std::max(retry_time, static_cast<uint32_t>(1)))
    , call_timeout_(std::max(call_timeout, static_cast<uint32_t>(1))) {}

ClientErrorCode GrpcStub::AddConnection(const std::string &address, uint32_t connection_timeout) {
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
    args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
    args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    std::array<char, 8192> buffer;
    float timeout = call_timeout_ / 1000.0f;
    int n = 0;
    if (retry_time_ > 1) {
        float initial_backoff = timeout / (retry_time_ + 1);
        float max_backoff = initial_backoff * 2;
        n = std::snprintf(buffer.data(),
                          buffer.size(),
                          kGrpcRetryPolicyFormat.c_str(),
                          timeout,
                          retry_time_,
                          initial_backoff,
                          max_backoff);
    } else {
        n = std::snprintf(buffer.data(), buffer.size(), kGrpcNoRetryPolicyFormat.c_str(), timeout);
    }
    std::string policy_str(buffer.data(), n);
    KVCM_LOG_INFO("grpc channel policy [%s]", policy_str.c_str());
    args.SetServiceConfigJSON(policy_str);
    uint32_t one_cnt_timeout = std::max(static_cast<uint32_t>(10), connection_timeout / retry_time_);
    uint32_t next_cnt_time = one_cnt_timeout;
    std::shared_ptr<grpc::Channel> channel;
    for (int i = 0; i < retry_time_; ++i) {
        channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args);
        if (channel->WaitForConnected(
                gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC), gpr_time_from_millis(next_cnt_time, GPR_TIMESPAN)))) {
            break;
        }
        next_cnt_time += one_cnt_timeout;
        KVCM_LOG_ERROR(
            "try [%d] connection to [%s] timeout, next timeout [%u ms]", i + 1, address.c_str(), next_cnt_time);
    }
    if (channel && channel->GetState(false) == GRPC_CHANNEL_READY) {
        KVCM_LOG_INFO("create connection to [%s] success", address.c_str());
        auto stub = std::make_shared<proto::meta::MetaService::Stub>(channel);
        {
            std::scoped_lock write_guard(stubs_mutex_);
            stubs_.push_back(stub);
        }
        return ER_OK;
    }
    return ER_CONNECT_FAIL;
}

void GrpcStub::RemoveAllConnections() {
    std::scoped_lock write_guard(stubs_mutex_);
    stubs_.clear();
}

std::pair<ClientErrorCode, std::string> GrpcStub::RegisterInstance(const std::string &trace_id,
                                                                   const std::string instance_group,
                                                                   const std::string &instance_id,
                                                                   int32_t block_size,
                                                                   const LocationSpecInfoMap &location_spec_infos,
                                                                   const ModelDeployment &model_deployment,
                                                                   const LocationSpecGroups &location_spec_groups) {
    auto stub = GET_AND_CHECK_STUB_WITH_TYPE();
    proto::meta::RegisterInstanceRequest request;
    SetCommonInfo(request, trace_id, instance_id);
    request.set_instance_group(instance_group);
    request.set_block_size(block_size);
    std::vector<LocationSpecInfo> info_vec;
    info_vec.reserve(location_spec_infos.size());
    std::for_each(location_spec_infos.begin(), location_spec_infos.end(), [&info_vec](const auto &info_pair) {
        info_vec.emplace_back(info_pair.first, info_pair.second);
    });
    ProtoConvert::LocationSpecInfosToProto(info_vec, request.mutable_location_spec_infos());
    ProtoConvert::ModelDeploymentToProto(model_deployment, request.mutable_model_deployment());
    std::vector<kv_cache_manager::LocationSpecGroup> kvcm_group_vec;
    kvcm_group_vec.reserve(location_spec_groups.size());
    std::for_each(location_spec_groups.begin(), location_spec_groups.end(), [&kvcm_group_vec](const auto &group_pair) {
        kvcm_group_vec.emplace_back(group_pair.first, group_pair.second);
    });
    ProtoConvert::LocationSpecGroupsToProto(kvcm_group_vec, request.mutable_location_spec_groups());
    grpc::ClientContext context;
    proto::meta::RegisterInstanceResponse response;
    auto grpc_status = stub->RegisterInstance(&context, request, &response);
    CHECK_GRPC_STATUS_WITH_TYPE(grpc_status);
    CHECK_COMMON_HEADER_WITH_TYPE(response);
    KVCM_LOG_DEBUG("register instance success, storage_config: %s", response.storage_configs().c_str());
    return {ER_OK, response.storage_configs()};
}

std::pair<ClientErrorCode, InstanceInfo> GrpcStub::GetInstanceInfo(const std::string &trace_id,
                                                                   const std::string &instance_id) {
    auto stub = GET_AND_CHECK_STUB_WITH_TYPE();
    proto::meta::GetInstanceInfoRequest request;
    SetCommonInfo(request, trace_id, instance_id);
    grpc::ClientContext context;
    proto::meta::GetInstanceInfoResponse response;
    auto grpc_status = stub->GetInstanceInfo(&context, request, &response);
    CHECK_GRPC_STATUS_WITH_TYPE(grpc_status);
    CHECK_COMMON_HEADER_WITH_TYPE(response);
    InstanceInfo instance_info;
    ProtoConvert::InstanceInfoFromProto(&response.instance_info(), instance_info);
    KVCM_LOG_DEBUG("get instance info success, instance_info: %s", instance_info.ToString().c_str());
    return {ER_OK, instance_info};
}

std::pair<ClientErrorCode, Metas> GrpcStub::GetCacheMeta(const std::string &trace_id,
                                                         const std::string &instance_id,
                                                         const KeyVector &keys,
                                                         const TokenIdsVector &tokens,
                                                         const BlockMask &block_mask,
                                                         int32_t detail_level) {
    auto stub = GET_AND_CHECK_STUB_WITH_TYPE();
    proto::meta::GetCacheMetaRequest request;
    SetKeysAndTokens(request, trace_id, instance_id, keys, tokens);
    ProtoConvert::BlockMaskToProto(block_mask, request.mutable_block_mask());
    request.set_detail_level(detail_level);
    grpc::ClientContext context;
    proto::meta::GetCacheMetaResponse response;
    auto grpc_status = stub->GetCacheMeta(&context, request, &response);
    CHECK_GRPC_STATUS_WITH_TYPE(grpc_status);
    CHECK_COMMON_HEADER_WITH_TYPE(response);
    auto locations = GenLocations(response.locations());
    std::vector<std::string> metas;
    std::for_each(
        response.metas().begin(), response.metas().end(), [&metas](const std::string &meta) { metas.push_back(meta); });
    KVCM_LOG_DEBUG("get cache meta success, locations: %s, metas: %s",
                   DebugStringUtil::ToString(locations).c_str(),
                   DebugStringUtil::ToString(metas).c_str());
    return {ER_OK, {locations, metas}};
}

std::pair<ClientErrorCode, Locations> GrpcStub::GetCacheLocation(const std::string &trace_id,
                                                                 const std::string &instance_id,
                                                                 QueryType query_type,
                                                                 const KeyVector &keys,
                                                                 const TokenIdsVector &tokens,
                                                                 const BlockMask &block_mask,
                                                                 int32_t sw_size,
                                                                 const std::vector<std::string> &location_spec_names) {
    auto stub = GET_AND_CHECK_STUB_WITH_TYPE();
    proto::meta::GetCacheLocationRequest request;
    request.set_query_type(static_cast<proto::meta::QueryType>(query_type));
    request.set_sw_size(sw_size);
    SetKeysAndTokens(request, trace_id, instance_id, keys, tokens);
    // 添加location_spec_names参数
    for (const auto &name : location_spec_names) {
        request.add_location_spec_names(name);
    }

    ProtoConvert::BlockMaskToProto(block_mask, request.mutable_block_mask());
    grpc::ClientContext context;
    proto::meta::GetCacheLocationResponse response;
    auto grpc_status = stub->GetCacheLocation(&context, request, &response);
    CHECK_GRPC_STATUS_WITH_TYPE(grpc_status);
    CHECK_COMMON_HEADER_WITH_TYPE(response);
    auto locations = GenLocations(response.locations());
    KVCM_LOG_DEBUG("get cache location success, locations: %s", DebugStringUtil::ToString(locations).c_str());
    return {ER_OK, locations};
}

std::pair<ClientErrorCode, WriteLocation>
GrpcStub::StartWriteCache(const std::string &trace_id,
                          const std::string &instance_id,
                          const KeyVector &keys,
                          const TokenIdsVector &tokens,
                          const std::vector<std::string> &location_spec_group_names,
                          int64_t write_timeout_seconds) {
    auto stub = GET_AND_CHECK_STUB_WITH_TYPE();
    proto::meta::StartWriteCacheRequest request;
    SetKeysAndTokens(request, trace_id, instance_id, keys, tokens);
    // 添加location_spec_group_names参数
    for (const auto &name : location_spec_group_names) {
        request.add_location_spec_group_names(name);
    }
    request.set_write_timeout_seconds(write_timeout_seconds);
    grpc::ClientContext context;
    proto::meta::StartWriteCacheResponse response;
    auto grpc_status = stub->StartWriteCache(&context, request, &response);
    CHECK_GRPC_STATUS_WITH_TYPE(grpc_status);
    CHECK_COMMON_HEADER_WITH_TYPE(response);
    std::string write_session_id = response.write_session_id();
    BlockMask block_mask;
    ProtoConvert::BlockMaskFromProto(&response.block_mask(), block_mask);
    auto locations = GenLocations(response.locations());
    KVCM_LOG_DEBUG("start write cache success, write_session_id: %s, block_mask: %s, locations: %s",
                   write_session_id.c_str(),
                   DebugStringUtil::ToString(block_mask).c_str(),
                   DebugStringUtil::ToString(locations).c_str());
    return {ER_OK, {write_session_id, block_mask, locations}};
}

ClientErrorCode GrpcStub::FinishWriteCache(const std::string &trace_id,
                                           const std::string &instance_id,
                                           const std::string write_session_id,
                                           const BlockMask &success_block,
                                           const Locations &locations) {
    auto stub = GET_AND_CHECK_STUB();
    proto::meta::FinishWriteCacheRequest request;
    SetCommonInfo(request, trace_id, instance_id);
    request.set_write_session_id(write_session_id);
    auto proto_locations = request.mutable_locations();
    auto ec = GenCacheLocation(locations, proto_locations);
    if (ec != ER_OK) {
        KVCM_LOG_DEBUG("finish write cache failed, write_session_id: %s, block_mask: %s, locations: %s",
                       write_session_id.c_str(),
                       DebugStringUtil::ToString(success_block).c_str(),
                       DebugStringUtil::ToString(locations).c_str());
        return ec;
    }
    ProtoConvert::BlockMaskToProto(success_block, request.mutable_success_blocks());
    grpc::ClientContext context;
    proto::meta::CommonResponse response;
    auto grpc_status = stub->FinishWriteCache(&context, request, &response);
    CHECK_GRPC_STATUS(grpc_status);
    CHECK_COMMON_HEADER(response);
    KVCM_LOG_DEBUG("finish write cache success, write_session_id: %s, success_block: %s, locations: %s",
                   write_session_id.c_str(),
                   DebugStringUtil::ToString(success_block).c_str(),
                   DebugStringUtil::ToString(locations).c_str());
    return ER_OK;
}

ClientErrorCode GrpcStub::RemoveCache(const std::string &trace_id,
                                      const std::string &instance_id,
                                      const KeyVector &keys,
                                      const TokenIdsVector &tokens,
                                      const BlockMask &block_mask) {
    auto stub = GET_AND_CHECK_STUB();
    proto::meta::RemoveCacheRequest request;
    SetKeysAndTokens(request, trace_id, instance_id, keys, tokens);
    ProtoConvert::BlockMaskToProto(block_mask, request.mutable_block_mask());
    grpc::ClientContext context;
    proto::meta::CommonResponse response;
    auto grpc_status = stub->RemoveCache(&context, request, &response);
    CHECK_GRPC_STATUS(grpc_status);
    CHECK_COMMON_HEADER(response);
    KVCM_LOG_DEBUG("remove cache success");
    return ER_OK;
}

bool GrpcStub::TrimCache() {
    // TODO
    return false;
}

std::shared_ptr<proto::meta::MetaService::Stub> GrpcStub::GetStub() const {
    std::shared_lock read_guard(stubs_mutex_);
    if (stubs_.empty()) {
        return nullptr;
    }
    size_t current_stub = (next_stub_++) % stubs_.size();
    return stubs_[current_stub];
}

} // namespace kv_cache_manager