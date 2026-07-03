#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kv_cache_manager {

// Two-stage block checksum verification for the read path.
//
// Stage 1 (fast): XOR-aggregate expected and actual into one uint64 each and
// compare once. Each block's contribution is multiplied by a position-dependent
// odd constant before being XORed in, so a plain block swap (expected=[A,B] vs
// actual=[B,A]) changes the aggregate — the fast path is order-sensitive.
// Common case: a Load batch that is fully consistent returns in O(1) compares
// regardless of batch size, with no per-block branching.
//
// Stage 2 (slow, only on fast-path mismatch or when strict_mode=true): walk the
// batch and report the indices of every block whose checksum disagrees. The
// caller uses these indices to log per-block diagnostics or publish a per-block
// ChecksumMismatchEvent.
//
// Sentinels: expected[i] == 0 means "no checksum was stored for this block"
// (legacy data or legacy client). Such entries are skipped in both stages.
//
// strict_mode (typically driven by KVCM_CHECKSUM_STRICT_MODE) bypasses the fast
// aggregate and always compares per block. Kept as a diagnostic knob for
// on-call: when triaging a suspected data-integrity issue you may want per-block
// index output without relying on the fast-path fallback triggering.
struct ChecksumVerifyResult {
    bool mismatch = false;
    std::vector<std::size_t> faulty_indices; // populated only when mismatch == true
};

inline ChecksumVerifyResult VerifyBatchChecksums(const std::vector<std::int64_t> &expected,
                                                 const std::vector<std::int64_t> &actual,
                                                 bool strict_mode) {
    ChecksumVerifyResult result;
    if (expected.size() != actual.size()) {
        result.mismatch = true;
        return result;
    }
    if (strict_mode) {
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (expected[i] == 0) {
                continue;
            }
            if (expected[i] != actual[i]) {
                result.faulty_indices.push_back(i);
            }
        }
        result.mismatch = !result.faulty_indices.empty();
        return result;
    }
    // fast path: xor aggregate with a position-dependent odd multiplier so a
    // plain block swap changes the aggregate (integer multiplication is not
    // GF(2)-linear, so (A*m0)^(B*m1) != (B*m0)^(A*m1) in general). Multiplying
    // by an odd constant is bijective mod 2^64, preserving per-element entropy.
    // kIndexSalt is the 2^64/phi constant used by splitmix64/xxhash; any well-
    // dispersed odd constant works.
    constexpr std::uint64_t kIndexSalt = 0x9E3779B97F4A7C15ULL;
    std::uint64_t expected_xor = 0;
    std::uint64_t actual_xor = 0;
    bool any_compared = false;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (expected[i] == 0) {
            continue;
        }
        const std::uint64_t multiplier = (2ULL * i + 1ULL) * kIndexSalt;
        expected_xor ^= static_cast<std::uint64_t>(expected[i]) * multiplier;
        actual_xor ^= static_cast<std::uint64_t>(actual[i]) * multiplier;
        any_compared = true;
    }
    if (!any_compared || expected_xor == actual_xor) {
        return result; // all match (or nothing to check)
    }
    // fast path detected a mismatch; locate the offending block(s)
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (expected[i] == 0) {
            continue;
        }
        if (expected[i] != actual[i]) {
            result.faulty_indices.push_back(i);
        }
    }
    result.mismatch = true;
    return result;
}

} // namespace kv_cache_manager
