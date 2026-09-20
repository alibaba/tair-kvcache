#include "kv_cache_manager/meta/redis_reclaim_sampler.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <system_error>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"

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

CacheKeyParseResult ParseCacheKey(const std::string &key, const std::string &cache_key_prefix, KeyType &out_key) {
    if (key.size() < cache_key_prefix.size() || key.compare(0, cache_key_prefix.size(), cache_key_prefix) != 0) {
        return CacheKeyParseResult::kUnexpectedPrefix;
    }
    const std::string key_suffix = key.substr(cache_key_prefix.size());
    if (!StringUtil::StrToInt64(key_suffix.c_str(), out_key)) {
        KVCM_INTERVAL_LOG_WARN(10,
                               "redis reclaim sampling skips malformed cache key[%s], prefix[%s]",
                               key.c_str(),
                               cache_key_prefix.c_str());
        return CacheKeyParseResult::kMalformed;
    }
    return CacheKeyParseResult::kValid;
}

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

    std::lock_guard<std::mutex> lock(mutex_);
    const std::string initial_cursor = cursor_;
    std::deque<std::string> initial_pending_keys = pending_keys_;
    std::unordered_set<std::string> known_keys(pending_keys_.begin(), pending_keys_.end());
    std::unordered_set<std::string> emitted_keys;

    const auto rollback = [&](const ErrorCode ec) {
        cursor_ = initial_cursor;
        pending_keys_ = std::move(initial_pending_keys);
        out_keys.clear();
        return ec;
    };

    while (!pending_keys_.empty() && out_keys.size() < static_cast<uint64_t>(count)) {
        std::string key = std::move(pending_keys_.front());
        pending_keys_.pop_front();
        if (!emitted_keys.insert(key).second) {
            continue;
        }
        KeyType parsed_key = 0;
        const CacheKeyParseResult parse_result = ParseCacheKey(key, cache_key_prefix, parsed_key);
        if (parse_result == CacheKeyParseResult::kUnexpectedPrefix) {
            return rollback(EC_ERROR);
        }
        if (parse_result == CacheKeyParseResult::kValid) {
            out_keys.emplace_back(parsed_key);
        }
    }
    if (out_keys.size() >= static_cast<uint64_t>(count)) {
        return EC_OK;
    }

    const std::string scan_prefix = EscapeRedisGlobLiteral(cache_key_prefix);
    const auto scan_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(MAX_SCAN_DURATION_MS);
    for (size_t scan_calls = 0; scan_calls < MAX_SCAN_CALLS; ++scan_calls) {
        const auto remaining = static_cast<int64_t>(static_cast<uint64_t>(count) - out_keys.size());
        const int64_t scan_count = std::clamp(remaining, MIN_SCAN_COUNT_HINT, MAX_SCAN_COUNT_HINT);
        std::string next_cursor;
        std::vector<std::string> scanned_keys;
        const ErrorCode ec = scan(scan_prefix, cursor_, scan_count, next_cursor, scanned_keys);
        if (ec != EC_OK || !IsValidRedisScanCursor(next_cursor)) {
            return rollback(ec == EC_OK ? EC_ERROR : ec);
        }

        cursor_ = std::move(next_cursor);
        for (auto &key : scanned_keys) {
            if (!known_keys.insert(key).second) {
                continue;
            }
            KeyType parsed_key = 0;
            const CacheKeyParseResult parse_result = ParseCacheKey(key, cache_key_prefix, parsed_key);
            if (parse_result == CacheKeyParseResult::kUnexpectedPrefix) {
                return rollback(EC_ERROR);
            }
            if (parse_result == CacheKeyParseResult::kMalformed) {
                continue;
            }
            if (out_keys.size() < static_cast<uint64_t>(count)) {
                out_keys.emplace_back(parsed_key);
            } else if (pending_keys_.size() < MAX_PENDING_KEYS) {
                pending_keys_.emplace_back(std::move(key));
            }
        }
        if (out_keys.size() >= static_cast<uint64_t>(count) || cursor_ == "0" ||
            std::chrono::steady_clock::now() >= scan_deadline) {
            break;
        }
    }
    return EC_OK;
}

void RedisReclaimSampler::Reset() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    cursor_ = "0";
    pending_keys_.clear();
}

} // namespace kv_cache_manager
