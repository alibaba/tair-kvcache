//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <assert.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __BMI2__
#include <immintrin.h>
#endif

#include <cstdint>
#include <type_traits>
#include <utility>

namespace kv_cache_manager {

// Number of bits set to 1. Also known as "population count".
template <typename T>
inline int BitsSetToOne(T v) {
    static_assert(std::is_integral_v<T>, "non-integral type");
    static_assert(!std::is_reference_v<T>, "use std::remove_reference_t");

    static_assert(sizeof(T) <= sizeof(unsigned long long), "type too big");
    if (sizeof(T) < sizeof(unsigned int)) {
        // This bit mask is to avoid a compiler warning on unused path
        constexpr auto mm = 8 * sizeof(unsigned int) - 1;
        // This bit mask is to neutralize sign extension on small signed types
        constexpr unsigned int m = (1U << ((8 * sizeof(T)) & mm)) - 1;
        return __builtin_popcount(static_cast<unsigned int>(v) & m);
    } else if (sizeof(T) == sizeof(unsigned int)) {
        return __builtin_popcount(static_cast<unsigned int>(v));
    } else if (sizeof(T) <= sizeof(unsigned long)) {
        return __builtin_popcountl(static_cast<unsigned long>(v));
    } else {
        return __builtin_popcountll(static_cast<unsigned long long>(v));
    }
}

} // namespace kv_cache_manager
