#pragma once

#include <optional>
#include <vector>

#include "kv_cache_manager/client/include/common.h"

namespace kv_cache_manager {

// Backends that understand only one host region (mooncake) take the caller's
// explicit span, or the sole segment of a one-element list -- a single-segment
// list is the legacy singular registration. Several segments must not be
// collapsed into one, so those backends see no registration instead of a wrong
// region that covers only the first segment.
//
// The result is a copy: callers hand it to C++ that runs without the GIL, where
// the source object is no longer safe to read.
inline std::optional<RegistSpan> SelectSingularRegistSpan(const RegistSpan *explicit_span,
                                                          const std::vector<const RegistSpan *> &host_spans) {
    if (explicit_span != nullptr) {
        return *explicit_span;
    }
    if (host_spans.size() == 1 && host_spans.front() != nullptr) {
        return *host_spans.front();
    }
    return std::nullopt;
}

} // namespace kv_cache_manager
