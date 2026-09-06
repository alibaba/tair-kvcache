#include "kv_cache_manager/client/src/kv_meta_client_impl.h"

#include <chrono>
#include <grpcpp/grpcpp.h>
#include <string>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {
namespace {

ClientErrorCode ToClientError(proto::kv_meta::ErrorCode error) {
    switch (error) {
    case proto::kv_meta::OK:
        return ER_OK;
    case proto::kv_meta::UNSUPPORTED:
        return ER_SERVICE_UNSUPPORTED;
    case proto::kv_meta::INTERNAL_ERROR:
        return ER_SERVICE_INTERNAL_ERROR;
    case proto::kv_meta::SERVICE_NOT_READY:
        return ER_SERVICE_NOT_READY;
    case proto::kv_meta::INVALID_ARGUMENT:
        return ER_SERVICE_INVALID_ARGUMENT;
    case proto::kv_meta::DUPLICATE_ENTITY:
        return ER_SERVICE_DUPLICATE_ENTITY;
    case proto::kv_meta::REACH_MAX_ENTITY_CAPACITY:
        return ER_SERVICE_REACH_MAX_ENTITY_CAPACITY;
    case proto::kv_meta::INSTANCE_NOT_EXIST:
        return ER_SERVICE_INSTANCE_NOT_EXIST;
    case proto::kv_meta::SERVER_NOT_LEADER:
        return ER_SERVICE_NOT_LEADER;
    case proto::kv_meta::RESOURCE_EXHAUSTED:
        return ER_SERVICE_RESOURCE_EXHAUSTED;
    case proto::kv_meta::NOT_FOUND:
        return ER_SERVICE_NOT_FOUND;
    case proto::kv_meta::WRITE_IN_PROGRESS:
        return ER_SERVICE_WRITE_IN_PROGRESS;
    case proto::kv_meta::SESSION_NOT_FOUND:
        return ER_SERVICE_SESSION_NOT_FOUND;
    case proto::kv_meta::SIZE_MISMATCH:
        return ER_SERVICE_SIZE_MISMATCH;
    case proto::kv_meta::IO_ERROR:
        return ER_SERVICE_IO_ERROR;
    case proto::kv_meta::UNSPECIFIED:
    case proto::kv_meta::UNKNOWN_ERROR:
    case proto::kv_meta::ERROR_MAX:
    default:
        return ER_SERVICE_INTERNAL_ERROR;
    }
}

bool IsRetryable(ClientErrorCode error) {
    return error == ER_SERVICE_NOT_LEADER || error == ER_SERVICE_NOT_READY;
}

bool ToPublicStorageType(proto::kv_meta::StorageType source, KvMetaStorageType &target) {
    switch (source) {
    case proto::kv_meta::ST_3FS:
        target = KvMetaStorageType::HF3FS;
        return true;
    case proto::kv_meta::ST_MOONCAKE:
        target = KvMetaStorageType::MOONCAKE;
        return true;
    case proto::kv_meta::ST_TAIRMEMPOOL:
        target = KvMetaStorageType::TAIR_MEMPOOL;
        return true;
    case proto::kv_meta::ST_NFS:
        target = KvMetaStorageType::NFS;
        return true;
    case proto::kv_meta::ST_VCNS_3FS:
        target = KvMetaStorageType::VCNS_HF3FS;
        return true;
    case proto::kv_meta::ST_DUMMY:
        target = KvMetaStorageType::DUMMY;
        return true;
    case proto::kv_meta::ST_EVENT_REPORT_L1P5:
        target = KvMetaStorageType::EVENT_REPORT_L1P5;
        return true;
    case proto::kv_meta::ST_EVENT_REPORT_L2:
        target = KvMetaStorageType::EVENT_REPORT_L2;
        return true;
    case proto::kv_meta::ST_TAIRMEMPOOL_SSD:
        target = KvMetaStorageType::TAIR_MEMPOOL_SSD;
        return true;
    case proto::kv_meta::ST_UNSPECIFIED:
    default:
        return false;
    }
}

bool ToPublicLocation(const proto::kv_meta::ValueLocation &source, KvMetaValueLocation &target) {
    if (source.value_size() == 0 || source.spec_size() != 1 || source.location_specs_size() != 1 ||
        !ToPublicStorageType(source.type(), target.type)) {
        return false;
    }
    const auto &source_spec = source.location_specs(0);
    const DataStorageUri uri(source_spec.uri());
    std::uint64_t uri_size = 0;
    uri.GetParamAs<std::uint64_t>("size", uri_size);
    const auto scheme_matches = [&]() {
        switch (target.type) {
        case KvMetaStorageType::HF3FS:
            return uri.GetProtocol() == "hf3fs";
        case KvMetaStorageType::VCNS_HF3FS:
            return uri.GetProtocol() == "hf3fs" || uri.GetProtocol() == "vcns_hf3fs";
        case KvMetaStorageType::MOONCAKE:
            return uri.GetProtocol() == "mooncake";
        case KvMetaStorageType::TAIR_MEMPOOL:
        case KvMetaStorageType::TAIR_MEMPOOL_SSD:
            return uri.GetProtocol() == "pace";
        case KvMetaStorageType::NFS:
            return uri.GetProtocol() == "file";
        case KvMetaStorageType::DUMMY:
            return uri.GetProtocol() == "dummy";
        case KvMetaStorageType::EVENT_REPORT_L1P5:
            return uri.GetProtocol() == "event_report_l1p5";
        case KvMetaStorageType::EVENT_REPORT_L2:
            return uri.GetProtocol() == "event_report_l2";
        case KvMetaStorageType::UNSPECIFIED:
        default:
            return false;
        }
    };
    if (source_spec.name() != "value" || !uri.Valid() || uri.GetHostName().empty() ||
        uri_size != source.value_size() || !scheme_matches()) {
        target = {};
        return false;
    }
    target.value_size = source.value_size();
    target.location_specs.clear();
    target.location_specs.push_back({source_spec.name(), source_spec.uri()});
    return true;
}

bool ValidateKeys(const std::vector<std::string> &keys) {
    if (keys.empty()) {
        return false;
    }
    std::unordered_set<std::string> unique;
    unique.reserve(keys.size());
    for (const auto &key : keys) {
        if (key.empty() || !unique.insert(key).second) {
            return false;
        }
    }
    return true;
}

template <typename Request>
void SetCommonRequestFields(Request &request, const std::string &trace_id, const std::string &instance_id) {
    request.set_trace_id(trace_id);
    request.set_instance_id(instance_id);
}

} // namespace

ClientErrorCode KvMetaClientImpl::Init(const KvMetaClientConfig &config) {
    if (config.instance_id.empty() || config.call_timeout_ms == 0 || config.call_timeout_ms > 600000 ||
        config.addresses.empty() || config.addresses.size() > 64) {
        return ER_INVALID_CLIENT_CONFIG;
    }

    config_ = config;
    config_.addresses.clear();
    std::unordered_set<std::string> unique_addresses;
    for (const auto &address : config.addresses) {
        if (address.empty() || address.size() > 1024) {
            return ER_INVALID_ADDRESS;
        }
        if (!unique_addresses.insert(address).second) {
            continue;
        }
        grpc::ChannelArguments arguments;
        constexpr int kMaxKvMetaMessageBytes = 16 * 1024 * 1024;
        arguments.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, kMaxKvMetaMessageBytes);
        arguments.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, kMaxKvMetaMessageBytes);
        arguments.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
        arguments.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
        arguments.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
        auto channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), arguments);
        if (!channel) {
            return ER_CONNECT_FAIL;
        }
        auto stub = proto::kv_meta::MetaService::NewStub(std::move(channel));
        if (!stub) {
            return ER_INVALID_STUB;
        }
        config_.addresses.push_back(address);
        stubs_.push_back(std::move(stub));
    }
    return stubs_.empty() ? ER_INVALID_CLIENT_CONFIG : ER_OK;
}

template <typename Response, typename Rpc>
ClientErrorCode KvMetaClientImpl::Call(Response *response, Rpc &&rpc) {
    if (!response || stubs_.empty()) {
        return ER_INVALID_STUB;
    }
    const std::size_t start = preferred_stub_.load(std::memory_order_relaxed) % stubs_.size();
    ClientErrorCode last_error = ER_INVALID_GRPCSTATUS;
    const auto deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(config_.call_timeout_ms);
    for (std::size_t attempt = 0; attempt < stubs_.size(); ++attempt) {
        if (std::chrono::system_clock::now() >= deadline) {
            break;
        }
        const std::size_t index = (start + attempt) % stubs_.size();
        response->Clear();
        grpc::ClientContext context;
        context.set_deadline(deadline);
        const grpc::Status grpc_status = rpc(*stubs_[index], &context, response);
        if (!grpc_status.ok()) {
            last_error = ER_INVALID_GRPCSTATUS;
            KVCM_LOG_WARN("KVMeta RPC endpoint[%zu] failed with gRPC code[%d]",
                          index,
                          static_cast<int>(grpc_status.error_code()));
            continue;
        }
        if (!response->has_header() || !response->header().has_status()) {
            return ER_SERVICE_NO_STATUS;
        }
        const ClientErrorCode service_error = ToClientError(response->header().status().code());
        if (service_error == ER_OK) {
            preferred_stub_.store(index, std::memory_order_relaxed);
            return ER_OK;
        }
        last_error = service_error;
        if (!IsRetryable(service_error)) {
            return service_error;
        }
    }
    return last_error;
}

std::pair<ClientErrorCode, std::string>
KvMetaClientImpl::RegisterInstance(const std::string &trace_id,
                                   const std::string &instance_group,
                                   const std::string &user_data) {
    if (instance_group.empty()) {
        return {ER_INVALID_PARAMS, {}};
    }
    proto::kv_meta::RegisterInstanceRequest request;
    request.set_trace_id(trace_id);
    request.set_instance_group(instance_group);
    request.set_instance_id(config_.instance_id);
    request.set_user_data(user_data);
    proto::kv_meta::RegisterInstanceResponse response;
    const ClientErrorCode ec = Call(&response, [&request](auto &stub, auto *context, auto *output) {
        return stub.RegisterInstance(context, request, output);
    });
    if (ec != ER_OK) {
        return {ec, {}};
    }
    if (response.storage_configs().empty()) {
        return {ER_SERVICE_INTERNAL_ERROR, {}};
    }
    return {ER_OK, response.storage_configs()};
}

std::pair<ClientErrorCode, KvMetaInstanceInfo>
KvMetaClientImpl::GetInstanceInfo(const std::string &trace_id) {
    proto::kv_meta::GetInstanceInfoRequest request;
    SetCommonRequestFields(request, trace_id, config_.instance_id);
    proto::kv_meta::GetInstanceInfoResponse response;
    const ClientErrorCode ec = Call(&response, [&request](auto &stub, auto *context, auto *output) {
        return stub.GetInstanceInfo(context, request, output);
    });
    if (ec != ER_OK) {
        return {ec, {}};
    }
    const auto &source = response.instance_info();
    if (!response.has_instance_info() || source.instance_id() != config_.instance_id ||
        source.instance_group_name().empty() || response.instance_group() != source.instance_group_name()) {
        return {ER_SERVICE_INTERNAL_ERROR, {}};
    }
    return {ER_OK,
            {source.quota_group_name(), source.instance_group_name(), source.instance_id()}};
}

std::pair<ClientErrorCode, KvMetaGetResult>
KvMetaClientImpl::Get(const std::string &trace_id, const std::vector<std::string> &keys) {
    if (!ValidateKeys(keys)) {
        return {ER_INVALID_PARAMS, {}};
    }
    proto::kv_meta::GetRequest request;
    SetCommonRequestFields(request, trace_id, config_.instance_id);
    request.set_query_type(proto::kv_meta::QT_BATCH_GET);
    for (const auto &key : keys) {
        request.add_keys(key);
    }
    proto::kv_meta::GetResponse response;
    const ClientErrorCode ec = Call(&response, [&request](auto &stub, auto *context, auto *output) {
        return stub.Get(context, request, output);
    });
    if (ec != ER_OK) {
        return {ec, {}};
    }
    if (!response.has_hit_mask() || response.hit_mask().values_size() != static_cast<int>(keys.size()) ||
        response.locations_size() != static_cast<int>(keys.size())) {
        return {ER_SERVICE_INTERNAL_ERROR, {}};
    }

    KvMetaGetResult result;
    result.hit_mask.reserve(keys.size());
    result.locations.resize(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const bool hit = response.hit_mask().values(static_cast<int>(i));
        result.hit_mask.push_back(hit);
        if (hit && !ToPublicLocation(response.locations(static_cast<int>(i)), result.locations[i])) {
            return {ER_SERVICE_INTERNAL_ERROR, {}};
        }
        if (!hit) {
            const auto &miss = response.locations(static_cast<int>(i));
            if (miss.type() != proto::kv_meta::ST_UNSPECIFIED || miss.spec_size() != 0 || miss.value_size() != 0 ||
                miss.location_specs_size() != 0) {
                return {ER_SERVICE_INTERNAL_ERROR, {}};
            }
        }
    }
    return {ER_OK, std::move(result)};
}

std::pair<ClientErrorCode, KvMetaStartWriteResult>
KvMetaClientImpl::StartWrite(const std::string &trace_id,
                             const std::vector<std::string> &keys,
                             const std::vector<std::uint64_t> &value_sizes,
                             std::int32_t write_timeout_seconds) {
    if (!ValidateKeys(keys) || value_sizes.size() != keys.size() || write_timeout_seconds <= 0) {
        return {ER_INVALID_PARAMS, {}};
    }
    for (const std::uint64_t size : value_sizes) {
        if (size == 0) {
            return {ER_INVALID_PARAMS, {}};
        }
    }

    proto::kv_meta::PutStartRequest request;
    SetCommonRequestFields(request, trace_id, config_.instance_id);
    request.set_write_timeout_seconds(write_timeout_seconds);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        request.add_keys(keys[i]);
        request.add_value_sizes(value_sizes[i]);
    }
    proto::kv_meta::PutStartResponse response;
    const ClientErrorCode ec = Call(&response, [&request](auto &stub, auto *context, auto *output) {
        return stub.PutStart(context, request, output);
    });
    if (ec != ER_OK) {
        return {ec, {}};
    }

    const auto abort_malformed_session = [&]() {
        if (response.write_session_id().empty()) {
            return;
        }
        std::size_t count = 0;
        if (response.has_key_mask() &&
            response.key_mask().values_size() == static_cast<int>(keys.size())) {
            for (const bool masked : response.key_mask().values()) {
                count += masked ? 0 : 1;
            }
        } else {
            count = static_cast<std::size_t>(response.locations_size());
        }
        if (count != 0) {
            const ClientErrorCode abort_ec =
                FinishWrite(trace_id, response.write_session_id(), std::vector<bool>(count, false));
            if (abort_ec != ER_OK) {
                KVCM_LOG_WARN("failed to abort malformed KVMeta write session, ec[%d]", static_cast<int>(abort_ec));
            }
        }
    };

    if (!response.has_key_mask() || response.key_mask().values_size() != static_cast<int>(keys.size())) {
        abort_malformed_session();
        return {ER_SERVICE_INTERNAL_ERROR, {}};
    }
    std::size_t expected_locations = 0;
    for (const bool masked : response.key_mask().values()) {
        expected_locations += masked ? 0 : 1;
    }
    if (response.locations_size() != static_cast<int>(expected_locations) ||
        (expected_locations == 0) != response.write_session_id().empty()) {
        abort_malformed_session();
        return {ER_SERVICE_INTERNAL_ERROR, {}};
    }

    KvMetaStartWriteResult result;
    result.write_session_id = response.write_session_id();
    result.key_mask.reserve(keys.size());
    result.locations.reserve(expected_locations);
    std::size_t location_index = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const bool masked = response.key_mask().values(static_cast<int>(i));
        result.key_mask.push_back(masked);
        if (masked) {
            continue;
        }
        KvMetaValueLocation location;
        if (!ToPublicLocation(response.locations(static_cast<int>(location_index)), location)) {
            abort_malformed_session();
            return {ER_SERVICE_INTERNAL_ERROR, {}};
        }
        if (location.value_size != value_sizes[i]) {
            abort_malformed_session();
            return {ER_SERVICE_SIZE_MISMATCH, {}};
        }
        result.locations.push_back(std::move(location));
        ++location_index;
    }
    return {ER_OK, std::move(result)};
}

ClientErrorCode KvMetaClientImpl::FinishWrite(const std::string &trace_id,
                                               const std::string &write_session_id,
                                               const std::vector<bool> &success_keys) {
    if (write_session_id.empty() || success_keys.empty()) {
        return ER_INVALID_PARAMS;
    }
    proto::kv_meta::PutFinishRequest request;
    SetCommonRequestFields(request, trace_id, config_.instance_id);
    request.set_write_session_id(write_session_id);
    for (const bool success : success_keys) {
        request.mutable_success_keys()->add_values(success);
    }
    proto::kv_meta::CommonResponse response;
    return Call(&response, [&request](auto &stub, auto *context, auto *output) {
        return stub.PutFinish(context, request, output);
    });
}

ClientErrorCode KvMetaClientImpl::Remove(const std::string &trace_id, const std::vector<std::string> &keys) {
    if (!ValidateKeys(keys)) {
        return ER_INVALID_PARAMS;
    }
    proto::kv_meta::RemoveRequest request;
    SetCommonRequestFields(request, trace_id, config_.instance_id);
    for (const auto &key : keys) {
        request.add_keys(key);
    }
    proto::kv_meta::CommonResponse response;
    return Call(&response, [&request](auto &stub, auto *context, auto *output) {
        return stub.Remove(context, request, output);
    });
}

ClientErrorCode KvMetaClientImpl::TrimAll(const std::string &trace_id, bool metadata_only) {
    proto::kv_meta::TrimRequest request;
    SetCommonRequestFields(request, trace_id, config_.instance_id);
    request.set_strategy(metadata_only ? proto::kv_meta::TS_REMOVE_ALL_META : proto::kv_meta::TS_REMOVE_ALL_CACHE);
    proto::kv_meta::CommonResponse response;
    return Call(&response, [&request](auto &stub, auto *context, auto *output) {
        return stub.Trim(context, request, output);
    });
}

std::unique_ptr<KvMetaClient> KvMetaClient::Create(const KvMetaClientConfig &config) {
    LoggerBroker::InitLoggerForClientOnce();
    auto client = std::make_unique<KvMetaClientImpl>();
    const ClientErrorCode ec = client->Init(config);
    if (ec != ER_OK) {
        KVCM_LOG_ERROR("create KVMeta client failed, ec[%d]", static_cast<int>(ec));
        return nullptr;
    }
    return client;
}

} // namespace kv_cache_manager
