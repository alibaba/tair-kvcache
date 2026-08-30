#include "kv_cache_manager/service/util/access_log_writer.h"

#include <cstdint>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kv_cache_manager {
namespace {

using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

void WriteString(JsonWriter &writer, const std::string &value) {
    writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
}

void WriteTrustedObject(JsonWriter &writer, const kv_cache_manager::RequestContext::JsonFragment &fragment) {
    if (!fragment.valid || fragment.json.empty()) {
        writer.StartObject();
        writer.EndObject();
        return;
    }

    // RawValue deliberately skips validation. All production fragments come
    // from ProtoMessageJsonUtil or RapidJSON summary writers, and the valid bit
    // is set only after those producers complete successfully.
    writer.RawValue(
        fragment.json.data(), static_cast<rapidjson::SizeType>(fragment.json.size()), rapidjson::kObjectType);
}

} // namespace

std::string AccessLogWriter::Build(const RequestContext &request_context) {
    // Materialize lazy fragments before taking the end timestamp so access-log
    // request cost retains the existing inclusion of JSON serialization time.
    const RequestContext::JsonFragment &request_json = request_context.request_debug_json();
    const RequestContext::JsonFragment &response_json = request_context.response_debug_json();
    const int64_t cost_us = TimestampUtil::GetCurrentTimeUs() - request_context.request_begin_time_us();

    rapidjson::StringBuffer buffer;
    JsonWriter writer(buffer);
    writer.StartObject();

    writer.Key("request_begin_time");
    WriteString(writer, TimestampUtil::FormatTimestampUs(request_context.request_begin_time_us()));
    writer.Key("client_ip");
    WriteString(writer, request_context.client_ip());
    writer.Key("trace_id");
    WriteString(writer, request_context.trace_id());
    writer.Key("request_id");
    WriteString(writer, request_context.request_id());
    writer.Key("api_name");
    WriteString(writer, request_context.api_name());
    writer.Key("status_code");
    writer.Int(request_context.status_code());
    writer.Key("request_cost_time_us");
    writer.Int64(cost_us);

    writer.Key("request");
    WriteTrustedObject(writer, request_json);
    writer.Key("response");
    WriteTrustedObject(writer, response_json);

    writer.EndObject();
    return std::string(buffer.GetString(), buffer.GetSize());
}

} // namespace kv_cache_manager
