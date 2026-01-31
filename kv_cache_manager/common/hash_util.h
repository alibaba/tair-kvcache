#pragma once

#include <stdint.h>
#include <vector>

namespace kv_cache_manager {

class HashUtil {
public:
    template <typename Int>
    static inline int64_t HashIntFunc(const std::hash<Int> &hasher, int64_t hash, Int value) {
        // Jenkins hash function (modified for 64 bits)
        hash ^= hasher(value) + 0x9e3779b97f4a7c15 + (hash << 12) + (hash >> 32);
        return hash;
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