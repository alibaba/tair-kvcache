#pragma once

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <string>
#include <string_view>

namespace kv_cache_manager {

// Direct protobuf reflection <-> RapidJSON codec for the protobuf JSON shapes
// used by KVCM. Callers must fall back to protobuf's JSON utility when the
// descriptor or input shape is unsupported.
class FastProtoJsonCodec {
public:
    static bool Supports(const ::google::protobuf::Descriptor *descriptor);
    static bool TryToJson(const ::google::protobuf::Message &message, std::string &json);
    static bool TryFromJson(std::string_view json, ::google::protobuf::Message *message);
};

} // namespace kv_cache_manager
