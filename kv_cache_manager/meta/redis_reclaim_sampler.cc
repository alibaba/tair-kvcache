#include "kv_cache_manager/meta/redis_reclaim_sampler.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <exception>
#include <system_error>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace {

std::string EscapeRedisGlobLiteral(const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '\\':
        case '*':
        case '?':
        case '[':
        case ']':
            escaped.push_back('\\');
            break;
        default:
            break;
        }
        escaped.push_back(ch);
    }
    return escaped;
}

bool IsValidRedisScanCursor(const std::string &cursor) {
    if (cursor.empty()) {
        return false;
    }
    uint64_t parsed = 0;
    const auto result = std::from_chars(cursor.data(), cursor.data() + cursor.size(), parsed);
    return result.ec == std::errc{} && result.ptr == cursor.data() + cursor.size();
}

enum class CacheKeyParseResult {
    kValid,
    kMalformed,
    kUnexpectedPrefix
};

bool HasCanonicalInt64Syntax(const char *begin, const char *end) noexcept {
    if (begin == end) {
        return false;
    }
    const bool negative = *begin == '-';
    const char *const first_digit = negative ? begin + 1 : begin;
    if (first_digit == end) {
        return false;
    }
    if (*first_digit == '0') {
        return !negative && first_digit + 1 == end;
    }
    return *first_digit >= '1' && *first_digit <= '9';
}

CacheKeyParseResult ParseCacheKey(const std::string &key, const std::string &cache_key_prefix, KeyType &out_key) {
    if (key.size() < cache_key_prefix.size() || key.compare(0, cache_key_prefix.size(), cache_key_prefix) != 0) {
        return CacheKeyParseResult::kUnexpectedPrefix;
    }
    const char *const suffix_begin = key.data() + cache_key_prefix.size();
    const char *const suffix_end = key.data() + key.size();
    const auto parse_result = std::from_chars(suffix_begin, suffix_end, out_key);
    if (!HasCanonicalInt64Syntax(suffix_begin, suffix_end) || parse_result.ec != std::errc{} ||
        parse_result.ptr != suffix_end) {
        KVCM_INTERVAL_LOG_WARN(10,
                               "redis reclaim sampling skips malformed cache key[%s], prefix[%s]",
                               key.c_str(),
                               cache_key_prefix.c_str());
        return CacheKeyParseResult::kMalformed;
    }
    return CacheKeyParseResult::kValid;
}

class ReclaimSampleTransaction {
public:
    ReclaimSampleTransaction(std::string &cursor,
                             size_t &page_offset,
                             int64_t &page_scan_count,
                             std::deque<std::string> &pending_keys)
        : cursor_(cursor)
        , page_offset_(page_offset)
        , page_scan_count_(page_scan_count)
        , pending_keys_(pending_keys)
        , initial_cursor_(cursor)
        , initial_page_offset_(page_offset)
        , initial_page_scan_count_(page_scan_count)
        , initial_pending_keys_(pending_keys) {}

    ~ReclaimSampleTransaction() noexcept { Restore(); }

    ErrorCode Commit() noexcept {
        completed_ = true;
        return EC_OK;
    }

    ErrorCode Rollback(const ErrorCode ec, KeyTypeVec &out_keys) noexcept {
        Restore();
        out_keys.clear();
        return ec;
    }

private:
    void Restore() noexcept {
        if (completed_) {
            return;
        }
        cursor_.swap(initial_cursor_);
        page_offset_ = initial_page_offset_;
        page_scan_count_ = initial_page_scan_count_;
        pending_keys_.swap(initial_pending_keys_);
        completed_ = true;
    }

    std::string &cursor_;
    size_t &page_offset_;
    int64_t &page_scan_count_;
    std::deque<std::string> &pending_keys_;
    std::string initial_cursor_;
    size_t initial_page_offset_;
    int64_t initial_page_scan_count_;
    std::deque<std::string> initial_pending_keys_;
    bool completed_{false};
};

} // namespace

ErrorCode RedisReclaimSampler::Sample(const ScanCallback &scan,
                                      const std::string &cache_key_prefix,
                                      const int64_t count,
                                      KeyTypeVec &out_keys) noexcept {
    out_keys.clear();
    if (count <= 0) {
        return EC_OK;
    }
    if (!scan || cache_key_prefix.empty()) {
        return EC_BADARGS;
    }

    try {
        std::lock_guard<std::mutex> lock(mutex_);
        ReclaimSampleTransaction transaction(cursor_, page_offset_, page_scan_count_, pending_keys_);

        // Both sets remain bounded by the requested result and the hard-capped
        // overflow buffer. In particular, do not retain every key from a SCAN
        // response: Redis treats COUNT as a hint and may return a very large page.
        std::unordered_set<std::string> buffered_keys(pending_keys_.begin(), pending_keys_.end());
        std::unordered_set<KeyType> emitted_keys;

        while (!pending_keys_.empty() && out_keys.size() < static_cast<uint64_t>(count)) {
            std::string key = std::move(pending_keys_.front());
            pending_keys_.pop_front();
            KeyType parsed_key = 0;
            const CacheKeyParseResult parse_result = ParseCacheKey(key, cache_key_prefix, parsed_key);
            if (parse_result == CacheKeyParseResult::kUnexpectedPrefix) {
                return transaction.Rollback(EC_ERROR, out_keys);
            }
            if (parse_result == CacheKeyParseResult::kValid && emitted_keys.insert(parsed_key).second) {
                out_keys.emplace_back(parsed_key);
            }
        }
        if (out_keys.size() >= static_cast<uint64_t>(count)) {
            return transaction.Commit();
        }

        const std::string scan_prefix = EscapeRedisGlobLiteral(cache_key_prefix);
        const auto scan_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(MAX_SCAN_DURATION_MS);
        for (size_t scan_calls = 0; scan_calls < MAX_SCAN_CALLS; ++scan_calls) {
            const auto remaining = static_cast<int64_t>(static_cast<uint64_t>(count) - out_keys.size());
            const int64_t requested_scan_count = std::clamp(remaining, MIN_SCAN_COUNT_HINT, MAX_SCAN_COUNT_HINT);
            // SCAN's COUNT is part of the effective page shape even though it
            // is only a hint. Replaying a partially consumed cursor with a new
            // COUNT can return a different page, making the saved offset skip
            // candidates. Pin it until the original page is fully inspected.
            const int64_t scan_count = page_offset_ == 0 ? requested_scan_count : page_scan_count_;
            if (scan_count < MIN_SCAN_COUNT_HINT || scan_count > MAX_SCAN_COUNT_HINT) {
                return transaction.Rollback(EC_CORRUPTION, out_keys);
            }
            std::string next_cursor;
            std::vector<std::string> scanned_keys;
            const ErrorCode ec = scan(scan_prefix, cursor_, scan_count, next_cursor, scanned_keys);
            if (ec != EC_OK || !IsValidRedisScanCursor(next_cursor)) {
                return transaction.Rollback(ec == EC_OK ? EC_ERROR : ec, out_keys);
            }

            // A SCAN page is normally stable while the keyspace is unchanged,
            // but Redis explicitly permits concurrent mutations. If a replayed
            // page shrank below our saved offset, restart that page instead of
            // skipping it or getting permanently stuck. SCAN already permits
            // duplicates, so replay is the safe direction.
            if (page_offset_ > scanned_keys.size()) {
                KVCM_INTERVAL_LOG_WARN(10,
                                       "redis reclaim scan page shrank while resuming, cursor[%s], old offset[%zu], "
                                       "new size[%zu]",
                                       cursor_.c_str(),
                                       page_offset_,
                                       scanned_keys.size());
                page_offset_ = 0;
            }

            size_t processed_key_count = 0;
            bool page_is_partial = false;
            for (; page_offset_ < scanned_keys.size(); ++page_offset_) {
                if (processed_key_count++ >= MAX_PROCESSED_KEYS_PER_SCAN_CALL) {
                    page_is_partial = true;
                    break;
                }
                auto &key = scanned_keys[page_offset_];
                KeyType parsed_key = 0;
                const CacheKeyParseResult parse_result = ParseCacheKey(key, cache_key_prefix, parsed_key);
                if (parse_result == CacheKeyParseResult::kUnexpectedPrefix) {
                    return transaction.Rollback(EC_ERROR, out_keys);
                }
                if (parse_result == CacheKeyParseResult::kMalformed || emitted_keys.count(parsed_key) != 0) {
                    continue;
                }
                if (out_keys.size() < static_cast<uint64_t>(count)) {
                    emitted_keys.emplace(parsed_key);
                    out_keys.emplace_back(parsed_key);
                } else if (pending_keys_.size() < MAX_PENDING_KEYS && buffered_keys.emplace(key).second) {
                    pending_keys_.emplace_back(std::move(key));
                }
                if (out_keys.size() >= static_cast<uint64_t>(count) && pending_keys_.size() >= MAX_PENDING_KEYS) {
                    ++page_offset_;
                    page_is_partial = page_offset_ < scanned_keys.size();
                    break;
                }
            }

            if (page_is_partial) {
                page_scan_count_ = scan_count;
                break;
            }
            // The current response has been fully inspected. Only now is it
            // safe to move to Redis' next cursor; otherwise a bounded call
            // could permanently discard the tail of an oversized page.
            page_offset_ = 0;
            page_scan_count_ = 0;
            cursor_ = std::move(next_cursor);
            if (out_keys.size() >= static_cast<uint64_t>(count) || cursor_ == "0" ||
                std::chrono::steady_clock::now() >= scan_deadline) {
                break;
            }
        }
        return transaction.Commit();
    } catch (const std::exception &e) {
        out_keys.clear();
        KVCM_INTERVAL_LOG_ERROR(10, "redis reclaim sampling caught exception[%s]", e.what());
        return EC_ERROR;
    } catch (...) {
        out_keys.clear();
        KVCM_INTERVAL_LOG_ERROR(10, "redis reclaim sampling caught unknown exception");
        return EC_ERROR;
    }
}

void RedisReclaimSampler::Reset() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        cursor_ = "0";
        page_offset_ = 0;
        page_scan_count_ = 0;
        pending_keys_.clear();
    } catch (const std::exception &e) {
        KVCM_INTERVAL_LOG_ERROR(10, "reset redis reclaim sampler caught exception[%s]", e.what());
    } catch (...) { KVCM_INTERVAL_LOG_ERROR(10, "reset redis reclaim sampler caught unknown exception"); }
}

} // namespace kv_cache_manager
