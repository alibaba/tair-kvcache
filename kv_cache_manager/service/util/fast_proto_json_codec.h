#pragma once

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <string>
#include <string_view>

namespace kv_cache_manager {

// Direct protobuf reflection <-> RapidJSON codec for the protobuf JSON shapes
// used by KVCM. Callers must fall back to protobuf's JSON utility when the
// descriptor or input shape is unsupported. The fast serializer preserves
// protobuf JSON semantics, but intentionally uses RapidJSON's normal string
// escaping rather than protobuf 3.8's byte-for-byte HTML-safe escaping. Its
// output must be consumed as JSON, not embedded directly into executable HTML.
class FastProtoJsonCodec {
public:
    static bool Supports(const ::google::protobuf::Descriptor *descriptor);
    static bool TryToJson(const ::google::protobuf::Message &message, std::string &json);
    static bool TryFromJson(std::string_view json, ::google::protobuf::Message *message);
};

} // namespace kv_cache_manager
