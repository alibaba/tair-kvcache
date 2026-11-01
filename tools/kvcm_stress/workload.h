#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <random>
#include <stdexcept>

#include "kv_cache_manager/client/include/meta_client.h"
#include "kv_cache_manager/client/include/transfer_client.h"

namespace kv_cache_manager::stress {
using Clock = std::chrono::steady_clock;
enum Phase {
    START,
    PUT,
    FINISH,
    QUERY,
    GET,
    WRITE,
    PHASE_COUNT
};
inline constexpr const char *kPhaseNames[] = {"start", "put", "finish", "query", "get", "write_e2e"};

struct Metric {
    std::atomic<uint64_t> calls{0}, errors{0}, micros{0};
    std::atomic<int> last_error{0};
    std::array<std::atomic<uint64_t>, 32> histogram{};
    void Record(Clock::time_point begin, ClientErrorCode ec) {
        auto us =
            std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin).count());
        unsigned bucket = 0;
        while (bucket < 31 && (uint64_t{1} << bucket) < static_cast<uint64_t>(us))
            ++bucket;
        histogram[bucket].fetch_add(1, std::memory_order_relaxed);
        micros.fetch_add(us, std::memory_order_relaxed);
        if (ec != ER_OK) {
            errors.fetch_add(1, std::memory_order_relaxed);
            last_error.store(ec, std::memory_order_relaxed);
        }
        calls.fetch_add(1, std::memory_order_relaxed);
    }
};
struct Stats {
    std::array<Metric, PHASE_COUNT> phases;
    std::atomic<uint64_t> queried{0}, hits{0}, written{0}, read{0}, skipped{0}, corrupt{0};
};
struct Options {
    size_t batch = 16, bytes = 65536, pool = 4096;
    int query_percent = 30, read_percent = 100, write_timeout = 60;
    bool verify = false;
    std::string spec;
};

// Each worker owns its buffers and bounded recent-key ring; only counters and
// the next key are shared. No per-block allocation in the payload path.
class Worker {
public:
    Worker(MetaClient &meta,
           TransferClient &transfer,
           const Options &options,
           Stats &stats,
           std::atomic<uint64_t> &next_key,
           uint64_t seed)
        : meta_(meta)
        , transfer_(transfer)
        , options_(options)
        , stats_(stats)
        , next_key_(next_key)
        , rng_(seed)
        , data_(options.batch * options.bytes, 'K') {
        pool_.reserve(options.pool);
    }

    bool Write() {
        const auto begin = Clock::now();
        auto ec = WriteBatch();
        stats_.phases[WRITE].Record(begin, ec);
        return ec == ER_OK;
    }
    bool HasKeys() const { return !pool_.empty(); }
    bool Step() {
        if (HasKeys() && rng_() % 100 < static_cast<unsigned>(options_.query_percent))
            return Read();
        return Write();
    }

private:
    BlockBuffer Buffer(size_t i) {
        return {{{MemoryType::CPU, data_.data() + i * options_.bytes, options_.bytes, false}}};
    }
    static unsigned char Pattern(uint64_t key, size_t offset) {
        // Repeat all key bytes, not merely its low byte, to detect wrong-block reads.
        return static_cast<unsigned char>((key >> ((offset % 8) * 8)) ^ (offset * 131));
    }
    void Fill(size_t i, uint64_t key) {
        for (size_t j = 0; j < options_.bytes; ++j)
            data_[i * options_.bytes + j] = Pattern(key, j);
    }
    std::string Trace() { return "stress-" + std::to_string(rng_()); }
    std::string Uri(const Location &location) const {
        for (const auto &spec : location)
            if (spec.spec_name == options_.spec)
                return spec.uri;
        return {};
    }
    void Remember(int64_t key) {
        if (pool_.size() < options_.pool)
            pool_.push_back(key);
        else {
            pool_[cursor_] = key;
            cursor_ = (cursor_ + 1) % options_.pool;
        }
    }
    ClientErrorCode WriteBatch() {
        const auto first = next_key_.fetch_add(options_.batch, std::memory_order_relaxed);
        if (first > static_cast<uint64_t>(INT64_MAX) - options_.batch)
            throw std::runtime_error("key range exhausted");
        std::vector<int64_t> keys(options_.batch);
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = first + i;
        const auto trace = Trace();
        auto begin = Clock::now();
        auto [ec, write] = meta_.StartWrite(trace, keys, {}, {}, options_.write_timeout);
        stats_.phases[START].Record(begin, ec);
        if (ec != ER_OK)
            return ec;
        if (write.write_session_id.empty())
            return ER_SERVICE_NO_STATUS;

        // StartWrite masks describe SKIPPED input keys. Locations and the
        // FinishWrite success mask are compacted to the remaining keys.
        std::vector<int64_t> selected;
        bool valid = true;
        if (const auto *mask = std::get_if<BlockMaskVector>(&write.block_mask)) {
            valid = mask->size() == keys.size();
            if (valid)
                for (size_t i = 0; i < keys.size(); ++i)
                    if (!(*mask)[i])
                        selected.push_back(keys[i]);
        } else {
            auto offset = std::get<BlockMaskOffset>(write.block_mask);
            valid = offset <= keys.size();
            if (valid)
                selected.assign(keys.begin() + offset, keys.end());
        }
        valid = valid && selected.size() == write.locations.size();
        UriStrVec uris;
        BlockBuffers buffers;
        if (valid) {
            for (size_t i = 0; i < selected.size(); ++i) {
                uris.push_back(Uri(write.locations[i]));
                valid = valid && !uris.back().empty();
                if (options_.verify)
                    Fill(i, selected[i]);
                buffers.push_back(Buffer(i));
            }
            stats_.skipped.fetch_add(keys.size() - selected.size(), std::memory_order_relaxed);
        }
        ec = valid ? ER_OK : ER_INVALID_PARAMS;
        if (valid && !uris.empty()) {
            begin = Clock::now();
            auto result = transfer_.SaveKvCaches(uris, buffers);
            ec = result.first;
            if (ec == ER_OK && result.second != uris) {
                // This server branch ignores FinishWrite.locations. Its PACE
                // addresses must be preallocated; never publish a different URI.
                ec = ER_INVALID_ADDRESS;
            }
            stats_.phases[PUT].Record(begin, ec);
        }
        const auto put_ec = ec;
        begin = Clock::now();
        // Even failed puts must close the session; never publish partial data.
        const BlockMask success =
            put_ec == ER_OK ? BlockMask{BlockMaskVector(selected.size(), true)} : BlockMask{BlockMaskOffset{0}};
        ec = meta_.FinishWrite(trace, write.write_session_id, success, put_ec == ER_OK ? write.locations : Locations{});
        stats_.phases[FINISH].Record(begin, ec);
        if (put_ec != ER_OK)
            return put_ec;
        if (ec != ER_OK)
            return ec;
        for (auto key : selected)
            Remember(key);
        stats_.written.fetch_add(selected.size(), std::memory_order_relaxed);
        return ER_OK;
    }
    bool Read() {
        std::vector<int64_t> keys(options_.batch);
        for (auto &key : keys)
            key = pool_[rng_() % pool_.size()];
        auto begin = Clock::now();
        auto [ec, locations] = meta_.MatchLocation(
            Trace(), QueryType::QT_BATCH_GET, keys, {}, BlockMaskOffset{0}, 0, std::vector<std::string>{options_.spec});
        if (ec == ER_OK && locations.size() != keys.size())
            ec = ER_SERVICE_NO_STATUS;
        stats_.phases[QUERY].Record(begin, ec);
        if (ec != ER_OK)
            return false;
        stats_.queried.fetch_add(keys.size(), std::memory_order_relaxed);
        UriStrVec uris;
        BlockBuffers buffers;
        std::vector<int64_t> hit_keys;
        for (size_t i = 0; i < locations.size(); ++i) {
            auto uri = Uri(locations[i]);
            if (uri.empty())
                continue;
            uris.push_back(std::move(uri));
            hit_keys.push_back(keys[i]);
            buffers.push_back(Buffer(buffers.size()));
        }
        stats_.hits.fetch_add(uris.size(), std::memory_order_relaxed);
        if (uris.empty() || rng_() % 100 >= static_cast<unsigned>(options_.read_percent))
            return true;
        if (options_.verify)
            std::fill(data_.begin(), data_.end(), 0);
        begin = Clock::now();
        ec = transfer_.LoadKvCaches(uris, buffers);
        if (ec == ER_OK && options_.verify) {
            for (size_t i = 0; i < hit_keys.size(); ++i) {
                for (size_t j = 0; j < options_.bytes; ++j) {
                    if (static_cast<unsigned char>(data_[i * options_.bytes + j]) != Pattern(hit_keys[i], j)) {
                        stats_.corrupt.fetch_add(1, std::memory_order_relaxed);
                        ec = ER_SDKREAD_ERROR;
                        break;
                    }
                }
            }
        }
        stats_.phases[GET].Record(begin, ec);
        if (ec == ER_OK)
            stats_.read.fetch_add(uris.size(), std::memory_order_relaxed);
        return ec == ER_OK;
    }
    MetaClient &meta_;
    TransferClient &transfer_;
    const Options &options_;
    Stats &stats_;
    std::atomic<uint64_t> &next_key_;
    std::mt19937_64 rng_;
    std::vector<char> data_;
    std::vector<int64_t> pool_;
    size_t cursor_ = 0;
};
} // namespace kv_cache_manager::stress
