#pragma once

#include <functional>

// copy from RTP-LLM: rtp-llm/cpp/utils/HashUtil.h
namespace kv_cache_manager {

inline int64_t hashInt64Func(const std::hash<int64_t> &hasher, int64_t hash, int64_t value) {
    // Jenkins hash function (modified for 64 bits)
    hash ^= hasher(value) + 0x9e3779b97f4a7c15 + (hash << 12) + (hash >> 32);
    return hash;
}

inline int64_t hashInt64Array(int64_t hash, const int64_t *begin, const int64_t *end) {
    std::hash<int64_t> hasher;

    while (begin != end) {
        // Combine the hash of each element
        hash = hashInt64Func(hasher, hash, *begin);
        begin++;
    }

    return hash;
}

inline int64_t hashInt64Vector(int64_t hash, const std::vector<int64_t> &vec) {
    std::hash<int64_t> hasher;

    for (const auto &value : vec) {
        // Combine the hash of each element
        hash = hashInt64Func(hasher, hash, (int64_t)value);
    }

    return hash;
}
} // namespace kv_cache_manager
