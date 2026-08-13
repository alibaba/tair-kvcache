// Offline replay harness for the online MRC engine (kv_cache_manager/mrc).
//
// Purpose: validate the online engine against the formal offline LiteHit
// Fenwick results on the same CacheBoard trace window (P1c acceptance).
//
// Input (stdin), one line per request, rows must be in trace order:
//   <lane_id>\t<hex_key>,<hex_key>,...
// where lane_id is "deployment/pod" and hex keys are the 256-token block
// hashes straight from the enriched trace (parsed as uint64).
//
// Two exact engines run side by side per lane:
//   - exact:   LiteHitCore (request-level prefix hit + leaf-first commit,
//              replicating the offline LiteHit semantics) with per-grid-point
//              counters. Expected to be bit-exact vs the formal per-instance
//              `points`.
//   - online:  the bounded production MrcProfiler. It must match exact at
//              every capacity <= max_capacity_blocks.
//
// Output: single JSON document on --output path.
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/mrc/lite_hit_core.h"
#include "kv_cache_manager/mrc/mrc_profiler.h"

namespace {

struct Options {
    int64_t max_capacity_blocks = 4096;
    int64_t capacity_step_blocks = 16;
    std::string output = "mrc-replay.json";
};

struct Lane {
    kv_cache_manager::LiteHitCore exact{1 << 16};
    // exact_bucket[i] counts prefix-hit thresholds t with
    // grid[i-1] < t <= grid[i]; prefix sums give exact hit counts at each
    // grid capacity (a block hits capacity C iff C >= t).
    std::vector<int64_t> exact_bucket;
    std::unique_ptr<kv_cache_manager::MrcProfiler> online;
    int64_t total_blocks = 0;
};

// Prefix-chained block keys, replicating the offline LiteHit preprocessing
// (`PrefixHashNext`) and matching KVCM's chained GenKeyVector semantics: a
// block's identity embeds its whole prefix.
int64_t PrefixHashNext(int64_t previous_hash, int64_t raw_value) {
    const uint64_t hash = static_cast<uint64_t>(previous_hash);
    const uint64_t value = static_cast<uint64_t>(raw_value);
    constexpr uint64_t kGoldenRatio = 0x9e3779b97f4a7c15ULL;
    const uint64_t rhs = value + kGoldenRatio + (hash << 12) + (hash >> 32);
    return static_cast<int64_t>(hash ^ rhs);
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](int64_t &v) { v = (i + 1 < argc) ? strtoll(argv[++i], nullptr, 10) : v; };
        if (arg == "--max-capacity-blocks") {
            next(options.max_capacity_blocks);
        } else if (arg == "--capacity-step-blocks") {
            next(options.capacity_step_blocks);
        } else if (arg == "--output") {
            if (i + 1 < argc) {
                options.output = argv[++i];
            }
        } else {
            fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return 2;
        }
    }

    // Capacity grid: 0, step, 2*step, ..., max (matches the formal run).
    std::vector<int64_t> grid;
    for (int64_t c = 0; c <= options.max_capacity_blocks; c += options.capacity_step_blocks) {
        grid.push_back(c);
    }

    kv_cache_manager::MrcProfiler::Options online_options;
    online_options.max_tracked_blocks = options.max_capacity_blocks;
    online_options.window_seconds = 0;

    std::unordered_map<std::string, std::unique_ptr<Lane>> lanes;

    std::string line;
    std::vector<int64_t> keys;
    std::vector<uint64_t> thresholds;
    int64_t row_count = 0;
    while (std::getline(std::cin, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos || tab == 0) {
            continue;
        }
        auto &lane_ptr = lanes[line.substr(0, tab)];
        if (!lane_ptr) {
            lane_ptr = std::make_unique<Lane>();
            lane_ptr->exact_bucket.assign(grid.size() + 1, 0); // +1: threshold > max grid
            lane_ptr->online = std::make_unique<kv_cache_manager::MrcProfiler>(online_options);
        }
        Lane &lane = *lane_ptr;

        keys.clear();
        const char *p = line.c_str() + tab + 1;
        int64_t prefix = 0;
        while (*p != '\0') {
            char *end = nullptr;
            const uint64_t key = strtoull(p, &end, 16);
            if (end == p) {
                break;
            }
            prefix = PrefixHashNext(prefix, static_cast<int64_t>(key));
            keys.push_back(prefix);
            p = (*end == ',') ? end + 1 : end;
        }
        if (keys.empty()) {
            continue;
        }

        lane.total_blocks += static_cast<int64_t>(keys.size());
        thresholds.clear();
        lane.exact.ProcessRequest(keys, thresholds);
        for (const uint64_t threshold : thresholds) {
            // Hit at capacity C iff C >= threshold: first grid index with
            // grid[idx] >= threshold.
            const auto it =
                std::lower_bound(grid.begin(), grid.end(), static_cast<int64_t>(threshold));
            ++lane.exact_bucket[static_cast<size_t>(it - grid.begin())];
        }
        lane.online->Observe(keys, /*now_us=*/0);

        if (++row_count % 1000000 == 0) {
            fprintf(stderr, "processed %" PRId64 " rows, %zu lanes\n", row_count, lanes.size());
        }
    }

    FILE *out = fopen(options.output.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "cannot open output %s\n", options.output.c_str());
        return 1;
    }
    fprintf(out,
            "{\n\"schema_version\": 2,\n\"engine\": \"kvcm-online-mrc replay (formal exact + bounded online exact)\",\n");
    fprintf(out,
            "\"max_capacity_blocks\": %" PRId64 ",\n\"capacity_step_blocks\": %" PRId64 ",\n",
            options.max_capacity_blocks,
            options.capacity_step_blocks);
    fprintf(out, "\"capacities_blocks\": [");
    for (size_t i = 0; i < grid.size(); ++i) {
        fprintf(out, "%s%" PRId64, i ? "," : "", grid[i]);
    }
    fprintf(out, "],\n\"lanes\": [\n");

    std::vector<double> grid_double(grid.begin(), grid.end());
    bool first_lane = true;
    for (const auto &[lane_id, lane] : lanes) {
        if (!first_lane) {
            fprintf(out, ",\n");
        }
        first_lane = false;

        fprintf(out,
                "{\"instance_id\": \"%s\", \"total_blocks\": %" PRId64 ", \"unique_blocks\": %" PRId64 ",\n",
                lane_id.c_str(),
                lane->total_blocks,
                lane->exact.unique_blocks());

        // Exact hit counts per grid capacity: prefix sums of the buckets.
        // exact_bucket[i] holds thresholds in (grid[i-1], grid[i]]; capacity
        // grid[i] therefore hits everything in buckets 0..i.
        fprintf(out, " \"exact_hit_blocks\": [");
        int64_t cumulative = 0;
        for (size_t i = 0; i < grid.size(); ++i) {
            cumulative += lane->exact_bucket[i];
            fprintf(out, "%s%" PRId64, i ? "," : "", cumulative);
        }
        fprintf(out, "],\n");

        kv_cache_manager::MrcProfiler::Snapshot snapshot;
        lane->online->QueryCumulative(grid_double, snapshot);
        fprintf(out,
                " \"online_exact\": {\"tracked_blocks\": %" PRId64
                ", \"memory_bytes\": %" PRId64
                ", \"max_tracked_hit_rate\": %.17g, \"hit_rate\": [",
                lane->online->tracked_blocks(),
                lane->online->memory_usage_bytes(),
                snapshot.max_tracked_hit_rate);
        for (size_t i = 0; i < snapshot.hit_rates.size(); ++i) {
            // max_digits10 preserves the exact binary double across the JSON
            // round trip, so replay_compare does not confuse formatting loss
            // with an online/offline MRC mismatch.
            fprintf(out, "%s%.17g", i ? "," : "", snapshot.hit_rates[i]);
        }
        fprintf(out, "]}}");
    }
    fprintf(out, "\n]\n}\n");
    fclose(out);

    fprintf(stderr, "done: %" PRId64 " rows, %zu lanes -> %s\n", row_count, lanes.size(), options.output.c_str());
    return 0;
}
