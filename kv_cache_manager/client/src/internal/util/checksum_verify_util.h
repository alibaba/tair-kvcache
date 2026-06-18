#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kv_cache_manager {

// Two-stage block checksum verification for the read path.
//
// Stage 1 (fast): XOR-aggregate expected and actual into one int64 each and compare
// once. This is the common case: a Load batch that is fully consistent returns in
// O(1) comparisons regardless of batch size, with no per-block branching.
//
// Stage 2 (slow, only on fast-path mismatch or when strict_mode=true): walk the
// batch and report the indices of every block whose checksum disagrees. The caller
// uses these indices to log per-block diagnostics or publish a per-block
// ChecksumMismatchEvent.
//
// Sentinels: expected[i] == 0 means "no checksum was stored for this block" (legacy
// data or legacy client). Such entries are skipped in both stages.
//
// strict_mode (typically driven by an env var like KVCM_CHECKSUM_STRICT_MODE) skips
// the fast aggregate and goes straight to per-block comparison. Useful for offline
// triage when investigating "complicated" mismatch scenarios where the fast XOR
// path could theoretically miss a paired cancellation (probability ~2^-64).
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
    // fast path: xor aggregate
    std::int64_t expected_xor = 0;
    std::int64_t actual_xor = 0;
    bool any_compared = false;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (expected[i] == 0) {
            continue;
        }
        expected_xor ^= expected[i];
        actual_xor ^= actual[i];
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
