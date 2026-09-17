#pragma once

#include <cstring>
#include <functional>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace kv_cache_manager {

class HashUtil {
public:
    template <typename Int>
    static inline int64_t HashIntFunc(const std::hash<Int> &hasher, int64_t hash, Int value) {
        // Jenkins hash function (modified for 64 bits). The historical code
        // evaluated shifts on signed int64_t, which made negative left shifts
        // undefined and right shifts implementation-defined. Spell out the
        // de-facto GCC behavior: modulo-2^64 arithmetic plus a 64-bit arithmetic
        // right shift. This preserves existing bits while making persisted
        // checksums reproducible across optimization levels and languages.
        const uint64_t hash_bits = static_cast<uint64_t>(hash);
        uint64_t shifted_right = hash_bits >> 32;
        if (hash < 0) {
            shifted_right |= UINT64_C(0xFFFFFFFF00000000);
        }
        // The utility is used with integer keys. Use their modulo-2^64 value
        // directly instead of relying on the implementation-defined stability
        // of std::hash across C++ standard libraries.
        const uint64_t value_hash = [&] {
            if constexpr (std::is_integral_v<Int>) {
                return static_cast<uint64_t>(value);
            }
            return static_cast<uint64_t>(hasher(value));
        }();
        const uint64_t mixed = value_hash + UINT64_C(0x9e3779b97f4a7c15) + (hash_bits << 12) + shifted_right;
        const uint64_t result_bits = hash_bits ^ mixed;
        int64_t result = 0;
        static_assert(sizeof(result) == sizeof(result_bits));
        std::memcpy(&result, &result_bits, sizeof(result));
        return result;
    }

    template <typename Int>
    static inline int64_t HashIntArray(Int *begin, Int *end, int64_t hash) {
        std::hash<Int> hasher;
        while (begin != end) {
            hash = HashIntFunc(hasher, hash, *begin);
            begin++;
        }
        return hash;
    }

    template <typename Int>
    static inline int64_t HashIntVector(const std::vector<Int> &vec, int64_t hash) {
        std::hash<Int> hasher;
        for (const auto &v : vec) {
            hash = HashIntFunc(hasher, hash, v);
        }
        return hash;
    }
};

} // namespace kv_cache_manager
