#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/util/host_registration.h"

namespace kv_cache_manager {

// The list channel: an unset entry is dropped (a None in the caller's list becomes
// one; C++ has no such value), a partially filled entry raises with its position --
// dropping it would hand the backend fewer segments than the caller passed.
inline std::vector<SharedMemoryRegistration>
NormalizeHostRegistrations(const std::vector<SharedMemoryRegistration> &entries) {
    std::vector<SharedMemoryRegistration> registrations;
    registrations.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto &entry = entries[i];
        const HostRegistrationState state = ClassifyHostRegistration(entry);
        if (state == HostRegistrationState::Disabled) {
            continue;
        }
        if (state == HostRegistrationState::Partial) {
            std::string missing_fields;
            if (entry.base == nullptr) {
                missing_fields += " base";
            }
            if (entry.size == 0) {
                missing_fields += " size";
            }
            if (entry.fd < 0) {
                missing_fields += " fd";
            }
            throw std::invalid_argument("regist_spans[" + std::to_string(i) + "] is incomplete, missing:" +
                                        missing_fields);
        }
        registrations.push_back(entry);
    }
    return registrations;
}

// Backends that understand only one host region (mooncake) take the caller's
// explicit span, or the sole registration left after normalization -- entries
// that register nothing are already gone, so they do not count as segments.
// Several segments must not be collapsed into one, so those backends see no
// registration instead of a wrong region that covers only the first segment.
//
// The result is a copy: callers hand it to C++ that runs without the GIL, where
// the source objects are no longer safe to read.
inline std::optional<RegistSpan> SelectSingularRegistSpan(const std::optional<RegistSpan> &explicit_span,
                                                          const std::vector<SharedMemoryRegistration> &registrations) {
    if (explicit_span) {
        return *explicit_span;
    }
    if (registrations.size() == 1) {
        const auto &only = registrations.front();
        return RegistSpan{only.base, only.size};
    }
    return std::nullopt;
}

} // namespace kv_cache_manager
