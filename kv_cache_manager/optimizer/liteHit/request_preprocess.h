#pragma once

#include <cstdint>
#include <vector>

namespace kv_cache_manager {

int64_t PrefixHashNext(int64_t previous_hash, int64_t raw_value);
std::vector<int64_t> ApplyPrefixHash(const std::vector<int64_t> &raw_keys);

struct NormalizedRequest {
    std::vector<int64_t> block_keys;
    uint64_t input_token_len = 0;
};

NormalizedRequest NormalizeRequest(const std::vector<int64_t> &block_keys,
                                   int64_t input_token_len,
                                   uint64_t block_size_tokens,
                                   bool enable_prefix_hash,
                                   uint64_t trace_block_size_tokens = 0);

} // namespace kv_cache_manager
