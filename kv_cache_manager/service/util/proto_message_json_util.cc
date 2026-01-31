#include "proto_message_json_util.h"

#include <google/protobuf/util/json_util.h>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {
::google::protobuf::util::JsonPrintOptions CreateJsonPrintOption() {
    ::google::protobuf::util::JsonPrintOptions option;
    option.add_whitespace = false;
    option.always_print_primitive_fields = true;
    option.always_print_enums_as_ints = false;
    option.preserve_proto_field_names = true;
    return option;
}

::google::protobuf::util::JsonParseOptions CreateJsonParseOption() {
    ::google::protobuf::util::JsonParseOptions option;
    option.ignore_unknown_fields = true;
    option.case_insensitive_enum_parsing = false;
    return option;
}

} // namespace

bool ProtoMessageJsonUtil::ToJson(const ::google::protobuf::Message *message, std::string &json) {
    if (!message) {
        return false;
    }
    static ::google::protobuf::util::JsonPrintOptions option = CreateJsonPrintOption();
    auto status = google::protobuf::util::MessageToJsonString(*message, &json, option);
    return status.ok();
}

bool ProtoMessageJsonUtil::FromJson(const std::string &json, ::google::protobuf::Message *message) {
    if (!message) {
        return false;
    }
    static ::google::protobuf::util::JsonParseOptions option = CreateJsonParseOption();
    auto status = google::protobuf::util::JsonStringToMessage(json, message, option);
    if (!status.ok()) {
        // TODO: change to return in response
        KVCM_LOG_WARN("json parse error, message: %s", status.error_message().data());
    }
    return status.ok();
}

} // namespace kv_cache_manager