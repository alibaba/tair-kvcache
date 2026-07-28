#pragma once

#include <cstdint>
#include <vector>

namespace kv_cache_manager {

// Rolling prefix-hash step shared by online and offline preprocessing. It is
// bit-for-bit identical to the Python producer
// (optimizer/tools/trace_converter/utils/prefix_hash.py::hash_int64_func):
// the Jenkins 64-bit variant evaluated with explicit uint64 arithmetic
// (logical right shift), then reinterpreted as int64. Negative results are
// valid keys. Note this intentionally follows the trace producer, not the
// signed-shift HashUtil::HashIntFunc, which diverges for negative
// intermediate hashes.
int64_t PrefixHashNext(int64_t previous_hash, int64_t raw_value);

// Converts per-block raw hashes to rolling prefix-chained keys, starting
// from hash 0. Matches prefix_hash.py::apply_prefix_hash.
std::vector<int64_t> ApplyPrefixHash(const std::vector<int64_t> &raw_keys);

struct NormalizedRequest {
    std::vector<int64_t> block_keys;
    uint64_t input_token_len = 0;
};

// Shared stateless request preprocessing for online and offline paths.
//
// - An explicit positive input_token_len is the authoritative denominator;
//   a non-positive value is treated as missing and derived as
//   block_keys.size() * trace_block_size_tokens (token_ids fallbacks are
//   resolved by the caller before this function).
// - trace_block_size_tokens is the granularity the trace keys were produced
//   at; 0 means "same as block_size_tokens" (no re-blocking). When it is
//   smaller than block_size_tokens, block_size_tokens must be an exact
//   multiple k of it and the request is re-blocked to the coarser analysis
//   granularity by keeping every k-th prefix-chained key (an incomplete tail
//   is dropped). A prefix-chained key encodes its whole prefix, so sampling
//   preserves cross-request matching and the fork contract.
// - Validates block_keys.size() == floor(input_token_len /
//   trace_block_size_tokens). Zero-length input with empty keys is legal.
// - When enable_prefix_hash is true the input keys are interpreted as
//   per-block raw hashes and converted to rolling prefix-chained keys (at
//   trace granularity, before any re-blocking); otherwise they must already
//   be prefix-chained and are passed through.
//
// Throws std::invalid_argument on violations.
NormalizedRequest NormalizeRequest(const std::vector<int64_t> &block_keys,
                                   int64_t input_token_len,
                                   uint64_t block_size_tokens,
                                   bool enable_prefix_hash,
                                   uint64_t trace_block_size_tokens = 0);

} // namespace kv_cache_manager
