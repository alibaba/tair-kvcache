#include "kv_cache_manager/client/src/internal/async_rpc/json_codec.h"

#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>

namespace kv_cache_manager::async_rpc {
namespace {

const google::protobuf::util::JsonPrintOptions &PrintOptions() {
    static const google::protobuf::util::JsonPrintOptions options = []() {
        google::protobuf::util::JsonPrintOptions o;
        o.add_whitespace = false;
        o.always_print_primitive_fields = true;
        o.always_print_enums_as_ints = false;
        o.preserve_proto_field_names = true;
        return o;
    }();
    return options;
}

const google::protobuf::util::JsonParseOptions &ParseOptions() {
    static const google::protobuf::util::JsonParseOptions options = []() {
        google::protobuf::util::JsonParseOptions o;
        o.ignore_unknown_fields = true;
        o.case_insensitive_enum_parsing = false;
        return o;
    }();
    return options;
}

} // namespace

bool MessageToJson(const google::protobuf::Message &message, std::string *json) {
    json->clear();
    return google::protobuf::util::MessageToJsonString(message, json, PrintOptions()).ok();
}

bool JsonToMessage(const std::string &json, google::protobuf::Message *message, std::string *error) {
    const google::protobuf::StringPiece input(json.data(), static_cast<ptrdiff_t>(json.size()));
    const auto status = google::protobuf::util::JsonStringToMessage(input, message, ParseOptions());
    if (!status.ok()) {
        if (error != nullptr) {
            *error = std::string(status.error_message().data(), status.error_message().size());
        }
        return false;
    }
    return true;
}

} // namespace kv_cache_manager::async_rpc
