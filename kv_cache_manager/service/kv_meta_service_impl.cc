#include "kv_cache_manager/service/kv_meta_service_impl.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/kv_meta_manager.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"
#include "kv_cache_manager/service/util/service_call_guard.h"

namespace kv_cache_manager {
namespace {

using PbError = proto::kv_meta::ErrorCode;

PbError ToKvMetaPbError(ErrorCode ec, bool session_lookup = false) {
    switch (ec) {
    case EC_OK:
        return proto::kv_meta::OK;
    case EC_UNIMPLEMENTED:
        return proto::kv_meta::UNSUPPORTED;
    case EC_BADARGS:
    case EC_OUT_OF_RANGE:
        return proto::kv_meta::INVALID_ARGUMENT;
    case EC_DUPLICATE_ENTITY:
    case EC_EXIST:
        return proto::kv_meta::DUPLICATE_ENTITY;
    case EC_INSTANCE_NOT_EXIST:
        return proto::kv_meta::INSTANCE_NOT_EXIST;
    case EC_SERVICE_NOT_LEADER:
        return proto::kv_meta::SERVER_NOT_LEADER;
    case EC_NOSPC:
        return proto::kv_meta::RESOURCE_EXHAUSTED;
    case EC_OUT_OF_LIMIT:
        return proto::kv_meta::REACH_MAX_ENTITY_CAPACITY;
    case EC_NOENT:
        return session_lookup ? proto::kv_meta::SESSION_NOT_FOUND : proto::kv_meta::NOT_FOUND;
    case EC_MISMATCH:
        return proto::kv_meta::SIZE_MISMATCH;
    case EC_IO_ERROR:
    case EC_TIMEOUT:
        return proto::kv_meta::IO_ERROR;
    case EC_UNKNOWN:
        return proto::kv_meta::UNKNOWN_ERROR;
    default:
        return proto::kv_meta::INTERNAL_ERROR;
    }
}

proto::kv_meta::StorageType ToKvMetaStorageType(DataStorageType type) {
    switch (type) {
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
        return proto::kv_meta::ST_3FS;
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
        return proto::kv_meta::ST_VCNS_3FS;
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
        return proto::kv_meta::ST_MOONCAKE;
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
        return proto::kv_meta::ST_TAIRMEMPOOL;
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD:
        return proto::kv_meta::ST_TAIRMEMPOOL_SSD;
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
        return proto::kv_meta::ST_NFS;
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY:
        return proto::kv_meta::ST_DUMMY;
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
        return proto::kv_meta::ST_EVENT_REPORT_L1P5;
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
        return proto::kv_meta::ST_EVENT_REPORT_L2;
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
    default:
        return proto::kv_meta::ST_UNSPECIFIED;
    }
}

bool FillLocation(const KvMetaManager::ValueLocation &source, proto::kv_meta::ValueLocation *target) {
    if (!target || source.type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN || source.value_size == 0 ||
        source.specs.size() != 1 || source.specs.front().first != "value") {
        return false;
    }
    const DataStorageUri uri(source.specs.front().second);
    std::uint64_t uri_size = 0;
    uri.GetParamAs<std::uint64_t>("size", uri_size);
    const DataStorageType uri_type = ToDataStorageType(uri.GetProtocol());
    const bool scheme_matches =
        IsTairMempoolStorageType(source.type)
            ? uri.GetProtocol() == kTairMempoolUriScheme
            : uri_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN &&
                  ToBaseType(uri_type) == ToBaseType(source.type);
    if (!uri.Valid() || uri.GetHostName().empty() || uri_size != source.value_size || !scheme_matches) {
        return false;
    }
    target->Clear();
    target->set_type(ToKvMetaStorageType(source.type));
    target->set_spec_size(static_cast<std::int32_t>(source.specs.size()));
    target->set_value_size(source.value_size);
    for (const auto &[name, uri] : source.specs) {
        auto *spec = target->add_location_specs();
        spec->set_name(name);
        spec->set_uri(uri);
    }
    return true;
}

std::string ErrorMessage(const char *operation, ErrorCode ec, RequestContext *request_context) {
    std::string message = std::string(operation) + " failed, ec=" + std::to_string(static_cast<int>(ec));
    if (request_context && request_context->error_tracer()) {
        const std::string detail = request_context->error_tracer()->ToJsonString();
        if (!detail.empty() && detail != "[]") {
            message += ", detail=" + detail;
        }
    }
    return message;
}

void SetResult(RequestContext *request_context,
               proto::kv_meta::Status *status,
               ErrorCode ec,
               const char *operation,
               bool session_lookup = false) {
    const PbError pb_error = ToKvMetaPbError(ec, session_lookup);
    status->set_code(pb_error);
    status->set_message(ec == EC_OK ? std::string(operation) + " succeeded"
                                    : ErrorMessage(operation, ec, request_context));
    request_context->set_status_code(static_cast<int>(pb_error));
}

void SetDirectError(RequestContext *request_context,
                    proto::kv_meta::Status *status,
                    PbError error,
                    const std::string &message) {
    status->set_code(error);
    status->set_message(message);
    request_context->set_status_code(static_cast<int>(error));
}

class TracerResultGuard {
public:
    TracerResultGuard(RequestContext *request_context, proto::kv_meta::CommonResponseHeader *header)
        : request_context_(request_context), header_(header) {}

    ~TracerResultGuard() {
        if (request_context_ && header_ && request_context_->need_span_tracer()) {
            header_->set_tracer_result(request_context_->EndAndGetSpanTracerDebugStr());
        }
    }

private:
    RequestContext *request_context_;
    proto::kv_meta::CommonResponseHeader *header_;
};

} // namespace

#define KV_META_API_CALL_GUARD(api_name)                                                                              \
    request_context->set_api_name(api_name);                                                                           \
    auto *header = response->mutable_header();                                                                          \
    header->set_request_id(request_context->request_id());                                                              \
    if (!CheckAndIncrementRequestCount(true)) {                                                                         \
        SetDirectError(request_context, header->mutable_status(), proto::kv_meta::SERVER_NOT_LEADER,                   \
                       "Server is not the active KVMeta leader");                                                      \
        if (request_context->need_span_tracer()) {                                                                      \
            header->set_tracer_result(request_context->EndAndGetSpanTracerDebugStr());                                 \
        }                                                                                                               \
        return;                                                                                                         \
    }                                                                                                                   \
    ServiceCallGuard service_call_guard(cache_manager_.get(), request_context, metrics_reporter_.get(), [this]() {     \
        DecrementRequestCount(true);                                                                                    \
    });                                                                                                                 \
    TracerResultGuard tracer_result_guard(request_context, header);                                                     \
    auto *status = header->mutable_status()

KvMetaServiceImpl::KvMetaServiceImpl(std::shared_ptr<CacheManager> cache_manager,
                                     std::shared_ptr<KvMetaManager> kv_meta_manager,
                                     std::shared_ptr<MetricsReporter> metrics_reporter)
    : cache_manager_(std::move(cache_manager))
    , kv_meta_manager_(std::move(kv_meta_manager))
    , metrics_reporter_(std::move(metrics_reporter)) {}

void KvMetaServiceImpl::RegisterInstance(RequestContext *request_context,
                                         const proto::kv_meta::RegisterInstanceRequest *request,
                                         proto::kv_meta::RegisterInstanceResponse *response) {
    KV_META_API_CALL_GUARD("KvMeta.RegisterInstance");
    if (!kv_meta_manager_) {
        SetDirectError(request_context, status, proto::kv_meta::SERVICE_NOT_READY, "KVMeta manager is not initialized");
        return;
    }
    const auto [ec, storage_configs] = kv_meta_manager_->RegisterInstance(
        request_context, request->instance_group(), request->instance_id(), request->user_data());
    SetResult(request_context, status, ec, "RegisterInstance");
    if (ec == EC_OK) {
        response->set_storage_configs(storage_configs);
    }
}

void KvMetaServiceImpl::GetInstanceInfo(RequestContext *request_context,
                                        const proto::kv_meta::GetInstanceInfoRequest *request,
                                        proto::kv_meta::GetInstanceInfoResponse *response) {
    KV_META_API_CALL_GUARD("KvMeta.GetInstanceInfo");
    if (!kv_meta_manager_) {
        SetDirectError(request_context, status, proto::kv_meta::SERVICE_NOT_READY, "KVMeta manager is not initialized");
        return;
    }
    const auto [ec, info] = kv_meta_manager_->GetInstanceInfo(request_context, request->instance_id());
    SetResult(request_context, status, ec, "GetInstanceInfo");
    if (ec != EC_OK || !info) {
        return;
    }
    response->set_instance_group(info->instance_group_name());
    auto *output = response->mutable_instance_info();
    output->set_quota_group_name(info->quota_group_name());
    output->set_instance_group_name(info->instance_group_name());
    output->set_instance_id(info->instance_id());
}

void KvMetaServiceImpl::Get(RequestContext *request_context,
                            const proto::kv_meta::GetRequest *request,
                            proto::kv_meta::GetResponse *response) {
    KV_META_API_CALL_GUARD("KvMeta.Get");
    if (!kv_meta_manager_) {
        SetDirectError(request_context, status, proto::kv_meta::SERVICE_NOT_READY, "KVMeta manager is not initialized");
        return;
    }
    if (request->query_type() != proto::kv_meta::QT_UNSPECIFIED &&
        request->query_type() != proto::kv_meta::QT_BATCH_GET) {
        SetDirectError(request_context, status, proto::kv_meta::UNSUPPORTED, "Unsupported KVMeta query_type");
        return;
    }
    if (!request->metas().empty()) {
        SetDirectError(request_context, status, proto::kv_meta::UNSUPPORTED, "KVMeta query metas are not implemented");
        return;
    }
    const std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    const auto [ec, values] = kv_meta_manager_->Get(request_context, request->instance_id(), keys);
    SetResult(request_context, status, ec, "Get");
    if (ec != EC_OK) {
        return;
    }
    if (values.size() != keys.size()) {
        SetDirectError(
            request_context, status, proto::kv_meta::INTERNAL_ERROR, "KVMeta Get returned a malformed batch");
        response->clear_locations();
        response->clear_hit_mask();
        return;
    }
    auto *hit_mask = response->mutable_hit_mask();
    for (const auto &value : values) {
        auto *location = response->add_locations();
        hit_mask->add_values(value.found);
        if (value.found && !FillLocation(value.location, location)) {
            SetDirectError(
                request_context, status, proto::kv_meta::INTERNAL_ERROR, "KVMeta Get returned an invalid location");
            response->clear_locations();
            response->clear_hit_mask();
            return;
        }
    }
}

void KvMetaServiceImpl::PutStart(RequestContext *request_context,
                                 const proto::kv_meta::PutStartRequest *request,
                                 proto::kv_meta::PutStartResponse *response) {
    KV_META_API_CALL_GUARD("KvMeta.PutStart");
    if (!kv_meta_manager_) {
        SetDirectError(request_context, status, proto::kv_meta::SERVICE_NOT_READY, "KVMeta manager is not initialized");
        return;
    }
    const std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    const std::vector<std::uint64_t> sizes(request->value_sizes().begin(), request->value_sizes().end());
    auto [ec, result] = kv_meta_manager_->StartWrite(
        request_context, request->instance_id(), keys, sizes, request->write_timeout_seconds());
    SetResult(request_context, status, ec, "PutStart");
    if (ec != EC_OK) {
        return;
    }
    const std::size_t write_count =
        static_cast<std::size_t>(std::count(result.key_mask.begin(), result.key_mask.end(), false));
    const auto abort_session = [&]() {
        if (result.write_session_id.empty()) {
            return;
        }
        // The request-aligned mask is the authoritative compact-session
        // cardinality. If even that shape is malformed, locations is the only
        // remaining bounded hint; a wrong count fails without consuming the
        // session and timeout recovery remains the final safety net.
        const std::size_t abort_count =
            result.key_mask.size() == keys.size() ? write_count : result.locations.size();
        if (abort_count == 0) {
            return;
        }
        const std::vector<bool> failed(abort_count, false);
        const ErrorCode abort_ec =
            kv_meta_manager_->FinishWrite(request_context, request->instance_id(), result.write_session_id, failed);
        if (abort_ec != EC_OK) {
            KVCM_LOG_ERROR("[traceId: %s] failed to abort malformed KVMeta PutStart result, ec[%d]",
                           request->trace_id().c_str(),
                           static_cast<int>(abort_ec));
        }
    };
    if (result.key_mask.size() != keys.size() || result.locations.size() != write_count ||
        (write_count == 0 && !result.write_session_id.empty()) ||
        (write_count != 0 && result.write_session_id.empty())) {
        abort_session();
        SetDirectError(
            request_context, status, proto::kv_meta::INTERNAL_ERROR, "KVMeta PutStart returned a malformed batch");
        return;
    }
    std::vector<proto::kv_meta::ValueLocation> converted_locations;
    converted_locations.reserve(result.locations.size());
    for (const auto &location : result.locations) {
        proto::kv_meta::ValueLocation converted;
        if (!FillLocation(location, &converted)) {
            abort_session();
            SetDirectError(request_context,
                           status,
                           proto::kv_meta::INTERNAL_ERROR,
                           "KVMeta PutStart returned an invalid location");
            return;
        }
        converted_locations.push_back(std::move(converted));
    }
    response->set_write_session_id(result.write_session_id);
    for (const bool value : result.key_mask) {
        response->mutable_key_mask()->add_values(value);
    }
    for (const auto &location : converted_locations) {
        *response->add_locations() = location;
    }
}

void KvMetaServiceImpl::PutFinish(RequestContext *request_context,
                                  const proto::kv_meta::PutFinishRequest *request,
                                  proto::kv_meta::CommonResponse *response) {
    KV_META_API_CALL_GUARD("KvMeta.PutFinish");
    if (!kv_meta_manager_) {
        SetDirectError(request_context, status, proto::kv_meta::SERVICE_NOT_READY, "KVMeta manager is not initialized");
        return;
    }
    if (!request->has_success_keys()) {
        SetDirectError(
            request_context, status, proto::kv_meta::INVALID_ARGUMENT, "KVMeta PutFinish requires success_keys");
        return;
    }
    std::vector<bool> successes;
    successes.reserve(request->success_keys().values_size());
    for (const bool value : request->success_keys().values()) {
        successes.push_back(value);
    }
    const ErrorCode ec = kv_meta_manager_->FinishWrite(
        request_context, request->instance_id(), request->write_session_id(), successes);
    SetResult(request_context, status, ec, "PutFinish", true);
}

void KvMetaServiceImpl::Remove(RequestContext *request_context,
                               const proto::kv_meta::RemoveRequest *request,
                               proto::kv_meta::CommonResponse *response) {
    KV_META_API_CALL_GUARD("KvMeta.Remove");
    if (!kv_meta_manager_) {
        SetDirectError(request_context, status, proto::kv_meta::SERVICE_NOT_READY, "KVMeta manager is not initialized");
        return;
    }
    const std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    const ErrorCode ec = kv_meta_manager_->Remove(request_context, request->instance_id(), keys);
    SetResult(request_context, status, ec, "Remove");
}

void KvMetaServiceImpl::Trim(RequestContext *request_context,
                             const proto::kv_meta::TrimRequest *request,
                             proto::kv_meta::CommonResponse *response) {
    KV_META_API_CALL_GUARD("KvMeta.Trim");
    if (!kv_meta_manager_) {
        SetDirectError(request_context, status, proto::kv_meta::SERVICE_NOT_READY, "KVMeta manager is not initialized");
        return;
    }
    bool metadata_only = false;
    switch (request->strategy()) {
    case proto::kv_meta::TS_REMOVE_ALL_CACHE:
        break;
    case proto::kv_meta::TS_REMOVE_ALL_META:
        metadata_only = true;
        break;
    case proto::kv_meta::TS_TIMESTAMP:
        SetDirectError(request_context,
                       status,
                       proto::kv_meta::UNSUPPORTED,
                       "Timestamp-based KVMeta trim is not implemented");
        return;
    case proto::kv_meta::TS_UNSPECIFIED:
    default:
        SetDirectError(request_context, status, proto::kv_meta::INVALID_ARGUMENT, "KVMeta trim strategy is invalid");
        return;
    }
    const ErrorCode ec = kv_meta_manager_->TrimAll(request_context, request->instance_id(), metadata_only);
    SetResult(request_context, status, ec, "Trim");
}

#undef KV_META_API_CALL_GUARD

} // namespace kv_cache_manager
