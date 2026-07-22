#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"

#include <stdexcept>

namespace kv_cache_manager {

int64_t PrefixHashNext(int64_t previous_hash, int64_t raw_value) {
    const uint64_t hash = static_cast<uint64_t>(previous_hash);
    const uint64_t value = static_cast<uint64_t>(raw_value);
    constexpr uint64_t kGoldenRatio = 0x9e3779b97f4a7c15ULL;
    const uint64_t rhs = value + kGoldenRatio + (hash << 12) + (hash >> 32);
    return static_cast<int64_t>(hash ^ rhs);
}

std::vector<int64_t> ApplyPrefixHash(const std::vector<int64_t> &raw_keys) {
    std::vector<int64_t> prefix_keys;
    prefix_keys.reserve(raw_keys.size());
    int64_t hash = 0;
    for (int64_t raw_key : raw_keys) {
        hash = PrefixHashNext(hash, raw_key);
        prefix_keys.push_back(hash);
    }
    return prefix_keys;
}

NormalizedRequest NormalizeRequest(const std::vector<int64_t> &block_keys,
                                   int64_t input_token_len,
                                   uint64_t block_size_tokens,
                                   bool enable_prefix_hash) {
    if (block_size_tokens == 0) {
        throw std::invalid_argument("NormalizeRequest block_size_tokens must be positive");
    }
    if (input_token_len < 0) {
        throw std::invalid_argument("NormalizeRequest input_token_len must not be negative");
    }

    NormalizedRequest normalized;
    normalized.input_token_len = input_token_len > 0 ? static_cast<uint64_t>(input_token_len)
                                                     : static_cast<uint64_t>(block_keys.size()) * block_size_tokens;

    const uint64_t expected_complete_blocks = normalized.input_token_len / block_size_tokens;
    if (expected_complete_blocks != static_cast<uint64_t>(block_keys.size())) {
        throw std::invalid_argument(
            "NormalizeRequest block_keys must contain exactly floor(input_token_len / block_size_tokens) blocks");
    }

    normalized.block_keys = enable_prefix_hash ? ApplyPrefixHash(block_keys) : block_keys;
    return normalized;
}

} // namespace kv_cache_manager
