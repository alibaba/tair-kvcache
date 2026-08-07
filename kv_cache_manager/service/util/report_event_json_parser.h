#pragma once

#include <cstddef>
#include <string_view>

#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"

namespace kv_cache_manager {

// Allocation-bounded JSON decoder for the high-volume ReportEvent HTTP API.
// Canonical ReportEvent payloads are decoded directly into the destination
// protobuf (and therefore its Arena). Less common protobuf-JSON spellings are
// deliberately delegated to ProtoMessageJsonUtil so compatibility remains
// defined by protobuf rather than by this fast path.
class ReportEventJsonParser {
public:
    static bool FromJson(std::string_view json, proto::meta::ReportEventRequest *message);

    // HTTP-specific entry point for cinatra's mutable, NUL-terminated request
    // body. Large ASCII payloads are parsed in place, avoiding a second full
    // body copy. `json[size]` must be readable and equal to '\0'; callers that
    // cannot provide that contract must use FromJson instead.
    static bool FromMutableNullTerminatedJson(char *json, size_t size, proto::meta::ReportEventRequest *message);

    // Exposed for focused tests and benchmarks. False means "use the generic
    // protobuf parser"; it does not necessarily mean that the JSON is invalid.
    static bool TryFromJson(std::string_view json, proto::meta::ReportEventRequest *message);
};

} // namespace kv_cache_manager
