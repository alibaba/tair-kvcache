// SampleSpec<T>: a scalar or a closed-interval uniform distribution.
//
// Scalars normalise to {min == max}. Integers include both endpoints;
// durations are sampled at the runtime clock resolution (nanoseconds).
#pragma once

#include <cstdint>
#include <string>

#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/runtime/rng.h"

namespace kvcm_swarm {

template <typename T>
struct SampleSpec {
    T min{};
    T max{};

    SampleSpec() = default;
    explicit SampleSpec(T value) : min(value), max(value) {}
    SampleSpec(T low, T high) : min(low), max(high) {}

    bool IsScalar() const { return min == max; }
};

using IntSpec = SampleSpec<uint64_t>;
using DurationSpec = SampleSpec<Duration>;

inline uint64_t Sample(const IntSpec &spec, Rng &rng) { return rng.NextInRange(spec.min, spec.max); }

inline Duration Sample(const DurationSpec &spec, Rng &rng) {
    const int64_t low = spec.min.count();
    const int64_t high = spec.max.count();
    if (high <= low) {
        return Duration(low);
    }
    return Duration(static_cast<int64_t>(rng.NextInRange(static_cast<uint64_t>(low), static_cast<uint64_t>(high))));
}

} // namespace kvcm_swarm
