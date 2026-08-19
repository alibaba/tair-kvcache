// Protobuf <-> JSON codec for the plaintext HTTP transport.
//
// The print/parse options mirror the KVCM HTTP service exactly
// (snake_case field names, string enum names, unknown fields ignored) so the
// HTTP and gRPC paths carry the same request and response content.
#pragma once

#include <string>

namespace google {
namespace protobuf {
class Message;
} // namespace protobuf
} // namespace google

namespace kv_cache_manager::async_rpc {

bool MessageToJson(const google::protobuf::Message &message, std::string *json);
bool JsonToMessage(const std::string &json, google::protobuf::Message *message, std::string *error);

} // namespace kv_cache_manager::async_rpc
