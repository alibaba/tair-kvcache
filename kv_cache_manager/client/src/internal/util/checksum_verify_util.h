#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kv_cache_manager {

// Walk the batch and report the indices of every block whose checksum disagrees.
// The caller uses these indices to log per-block diagnostics.
//
// Sentinels: expected[i] == 0 means "no checksum was stored for this block"
// (legacy data or legacy client). Such entries are skipped in both stages.
//
struct ChecksumVerifyResult {
    bool mismatch = false;
    std::vector<std::size_t> faulty_indices; // populated only when mismatch == true
};

inline ChecksumVerifyResult VerifyBatchChecksums(const std::vector<std::int64_t> &expected,
                                                 const std::vector<std::int64_t> &actual) {
    ChecksumVerifyResult result;
    if (expected.size() != actual.size()) {
        result.mismatch = true;
        return result;
    }
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

} // namespace kv_cache_manager
