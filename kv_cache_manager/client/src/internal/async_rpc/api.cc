#include "kv_cache_manager/client/src/internal/async_rpc/api.h"

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <stdexcept>
#include <string>

namespace kv_cache_manager::async_rpc {
namespace {

const std::vector<ApiInfo> &Table() {
    static const std::vector<ApiInfo> table = {
        {Api::kRegisterInstance,
         "RegisterInstance",
         "/api/registerInstance",
         "/kv_cache_manager.proto.meta.MetaService/RegisterInstance",
         ServiceEndpoint::kMeta},
        {Api::kGetCacheLocationsByBackend,
         "GetCacheLocationsByBackend",
         "/api/getCacheLocationsByBackend",
         "/kv_cache_manager.proto.meta.MetaService/GetCacheLocationsByBackend",
         ServiceEndpoint::kMeta},
        {Api::kStartWriteCache,
         "StartWriteCache",
         "/api/startWriteCache",
         "/kv_cache_manager.proto.meta.MetaService/StartWriteCache",
         ServiceEndpoint::kMeta},
        {Api::kFinishWriteCache,
         "FinishWriteCache",
         "/api/finishWriteCache",
         "/kv_cache_manager.proto.meta.MetaService/FinishWriteCache",
         ServiceEndpoint::kMeta},
        {Api::kReportEvent,
         "ReportEvent",
         "/api/reportEvent",
         "/kv_cache_manager.proto.meta.MetaService/ReportEvent",
         ServiceEndpoint::kMeta},
        {Api::kGetClusterInfo,
         "GetClusterInfo",
         "/api/getClusterInfo",
         "/kv_cache_manager.proto.meta.MetaService/GetClusterInfo",
         ServiceEndpoint::kMeta},
        {Api::kRemoveCache,
         "RemoveCache",
         "/api/removeCache",
         "/kv_cache_manager.proto.meta.MetaService/RemoveCache",
         ServiceEndpoint::kMeta},
        {Api::kCheckHealth,
         "CheckHealth",
         "/api/checkHealth",
         "/kv_cache_manager.proto.admin.AdminService/CheckHealth",
         ServiceEndpoint::kAdmin},
    };
    return table;
}

} // namespace

const char *ServiceEndpointName(ServiceEndpoint endpoint) {
    switch (endpoint) {
    case ServiceEndpoint::kMeta:
        return "meta";
    case ServiceEndpoint::kAdmin:
        return "admin";
    }
    return "unknown";
}

const ApiInfo &GetApiInfo(Api api) {
    for (const auto &info : Table()) {
        if (info.api == api) {
            return info;
        }
    }
    throw std::logic_error("unknown Api value");
}

const std::vector<ApiInfo> &AllApis() { return Table(); }

std::string_view ApiName(Api api) { return GetApiInfo(api).name; }

namespace {

const google::protobuf::Message *NestedMessage(const google::protobuf::Message &message, const char *field_name) {
    const google::protobuf::Descriptor *descriptor = message.GetDescriptor();
    const google::protobuf::FieldDescriptor *field = descriptor->FindFieldByName(field_name);
    if (field == nullptr || field->type() != google::protobuf::FieldDescriptor::TYPE_MESSAGE || field->is_repeated()) {
        return nullptr;
    }
    return &message.GetReflection()->GetMessage(message, field);
}

} // namespace

int ExtractServiceStatus(const google::protobuf::Message &response) {
    const google::protobuf::Message *header = NestedMessage(response, "header");
    if (header == nullptr) {
        return 0;
    }
    const google::protobuf::Message *status = NestedMessage(*header, "status");
    if (status == nullptr) {
        return 0;
    }
    const google::protobuf::FieldDescriptor *code = status->GetDescriptor()->FindFieldByName("code");
    if (code == nullptr || code->type() != google::protobuf::FieldDescriptor::TYPE_ENUM) {
        return 0;
    }
    return status->GetReflection()->GetEnumValue(*status, code);
}

std::string ExtractServiceMessage(const google::protobuf::Message &response) {
    const google::protobuf::Message *header = NestedMessage(response, "header");
    if (header == nullptr) {
        return {};
    }
    const google::protobuf::Message *status = NestedMessage(*header, "status");
    if (status == nullptr) {
        return {};
    }
    const google::protobuf::FieldDescriptor *message_field = status->GetDescriptor()->FindFieldByName("message");
    if (message_field == nullptr || message_field->cpp_type() != google::protobuf::FieldDescriptor::CPPTYPE_STRING) {
        return {};
    }
    return status->GetReflection()->GetString(*status, message_field);
}

} // namespace kv_cache_manager::async_rpc
