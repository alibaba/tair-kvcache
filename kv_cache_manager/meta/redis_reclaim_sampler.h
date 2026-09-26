#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

// Incrementally samples one Redis key prefix without issuing a full keyspace
// scan in a single reclaim round. The cursor and overflow buffer are shared by
// all callers for one backend, so successive rounds eventually cover sparse
// instances while each call has bounded Redis work.
class RedisReclaimSampler {
public:
    using ScanCallback = std::function<ErrorCode(const std::string &escaped_matching_prefix,
                                                 const std::string &cursor,
                                                 int64_t count,
                                                 std::string &out_next_cursor,
                                                 std::vector<std::string> &out_keys)>;

    // The callback runs after the sampler has serialized cursor access. Acquire
    // and release a pool client inside each callback invocation so callers
    // waiting for this sampler never reserve a scarce foreground connection.
    ErrorCode
    Sample(const ScanCallback &scan, const std::string &cache_key_prefix, int64_t count, KeyTypeVec &out_keys) noexcept;

    void Reset() noexcept;

private:
    static constexpr int64_t MIN_SCAN_COUNT_HINT = 128;
    static constexpr int64_t MAX_SCAN_COUNT_HINT = 4096;
    static constexpr int64_t MAX_SCAN_DURATION_MS = 50;
    static constexpr size_t MAX_SCAN_CALLS = 16;
    static constexpr size_t MAX_PENDING_KEYS = 4096;
    static constexpr size_t MAX_PROCESSED_KEYS_PER_SCAN_CALL = MAX_PENDING_KEYS + MAX_SCAN_COUNT_HINT;

    std::mutex mutex_;
    std::string cursor_{"0"};
    // Redis SCAN treats COUNT as a hint and can return a page much larger
    // than requested. Keep bounded, resumable progress within that page and
    // advance cursor_ only after the complete page has been inspected. A
    // replay must use the original COUNT hint: Redis is allowed to partition
    // the same cursor differently when COUNT changes.
    size_t page_offset_{0};
    int64_t page_scan_count_{0};
    std::deque<std::string> pending_keys_;
};

} // namespace kv_cache_manager
