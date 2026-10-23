#pragma once

#include "kv_cache_manager/client/include/common.h"

namespace kv_cache_manager {

// A host registration is three-valued: left entirely at its defaults it registers
// nothing, filled in completely it registers the region, filled in part-way it is a
// caller bug.
enum class HostRegistrationState {
    Disabled,
    Partial,
    Complete,
};

inline HostRegistrationState ClassifyHostRegistration(const SharedMemoryRegistration &registration) {
    if (registration.base == nullptr && registration.size == 0 && registration.fd == -1) {
        return HostRegistrationState::Disabled;
    }
    if (registration.base == nullptr || registration.size == 0 || registration.fd < 0) {
        return HostRegistrationState::Partial;
    }
    return HostRegistrationState::Complete;
}

} // namespace kv_cache_manager
