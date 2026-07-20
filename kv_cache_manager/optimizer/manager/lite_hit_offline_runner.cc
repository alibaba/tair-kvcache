#include "kv_cache_manager/optimizer/manager/lite_hit_offline_runner.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"
#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"
#include "kv_cache_manager/optimizer/trace_loader/trace_util.h"

namespace kv_cache_manager {

namespace {

// Print a percentile from an ascending-sorted latency vector.
uint32_t Percentile(const std::vector<uint32_t> &sorted_ns, double p) {
    if (sorted_ns.empty()) {
        return 0;
    }
    size_t idx = static_cast<size_t>(p * (sorted_ns.size() - 1));
    return sorted_ns[idx];
}

// Report per-query TraceQuery latency: overall distribution, throughput, growth of
// mean latency across 10 progress buckets (cache size P grows monotonically for the
// infinite-capacity ceiling), and mean latency bucketed by request block count B.
void ReportQueryLatency(const std::vector<uint32_t> &lat_ns, const std::vector<uint32_t> &blocks) {
    const size_t n = lat_ns.size();
    unsigned long long total = 0;
    for (uint32_t v : lat_ns) {
        total += v;
    }
    const double mean = static_cast<double>(total) / static_cast<double>(n);

    std::vector<uint32_t> sorted(lat_ns);
    std::sort(sorted.begin(), sorted.end());

    fprintf(stderr, "\n===== TraceQuery latency benchmark =====\n");
    fprintf(stderr, "queries            : %zu\n", n);
    fprintf(stderr, "total query time   : %.3f s\n", static_cast<double>(total) / 1e9);
    fprintf(
        stderr, "throughput         : %.0f queries/s\n", static_cast<double>(n) / (static_cast<double>(total) / 1e9));
    fprintf(stderr, "mean per query     : %.0f ns\n", mean);
    fprintf(stderr,
            "min/p50/p90/p99    : %u / %u / %u / %u ns\n",
            sorted.front(),
            Percentile(sorted, 0.50),
            Percentile(sorted, 0.90),
            Percentile(sorted, 0.99));
    fprintf(stderr,
            "p999/p9999/max     : %u / %u / %u ns\n",
            Percentile(sorted, 0.999),
            Percentile(sorted, 0.9999),
            sorted.back());

    // 10 equal-count progress buckets: mean latency vs replay progress. Because the
    // ceiling never evicts, cache size P grows with progress, so a flat/log-shaped
    // trend here confirms sub-linear (O(log P)) per-query cost, not O(P).
    fprintf(stderr, "\nmean latency by progress bucket (P grows left->right):\n");
    const size_t nb = 10;
    for (size_t b = 0; b < nb; ++b) {
        const size_t lo = n * b / nb;
        const size_t hi = n * (b + 1) / nb;
        unsigned long long sum = 0;
        unsigned long long bsum = 0;
        for (size_t i = lo; i < hi; ++i) {
            sum += lat_ns[i];
            bsum += blocks[i];
        }
        const size_t cnt = hi - lo;
        fprintf(stderr,
                "  [%2zu%%-%3zu%%] mean=%6.0f ns  (mean_blocks=%.1f)\n",
                b * 10,
                (b + 1) * 10,
                cnt ? static_cast<double>(sum) / cnt : 0.0,
                cnt ? static_cast<double>(bsum) / cnt : 0.0);
    }

    // Mean latency bucketed by request block count B, to expose the O(B) factor.
    fprintf(stderr, "\nmean latency by request block count B:\n");
    const uint32_t edges[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, UINT32_MAX};
    const size_t ne = sizeof(edges) / sizeof(edges[0]);
    for (size_t e = 0; e + 1 < ne; ++e) {
        unsigned long long sum = 0;
        size_t cnt = 0;
        for (size_t i = 0; i < n; ++i) {
            if (blocks[i] >= edges[e] && blocks[i] < edges[e + 1]) {
                sum += lat_ns[i];
                ++cnt;
            }
        }
        if (cnt) {
            fprintf(stderr,
                    "  B in [%5u,%5u): mean=%6.0f ns  n=%zu\n",
                    edges[e],
                    edges[e + 1],
                    static_cast<double>(sum) / cnt,
                    cnt);
        }
    }
    fprintf(stderr, "========================================\n");
}

} // namespace

bool LiteHitOfflineRunner::Run() {
    // Empty registry_uri keeps everything in-memory. Init() would be a no-op for an
    // empty uri (it only logs and never creates a storage backend), and every registry
    // CRUD path guards on the null storage_ before persisting, so the offline driver
    // skips Init() entirely and needs no external coordination backend.
    auto registry = std::make_shared<OptimizerRegistryManager>("");
    OnlineOptimizerManager manager(registry);

    // Register instance groups and instances exactly like the online path.
    for (const auto &group : config_.instance_groups()) {
        ErrorCode ec = manager.CreateInstanceGroup(group);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("LiteHitOfflineRunner: CreateInstanceGroup[%s] failed, ec=%d",
                           group.name().c_str(),
                           static_cast<int>(ec));
            return false;
        }
    }
    for (const auto &instance : config_.instances()) {
        RegisterInstanceResult register_result;
        ErrorCode ec = manager.RegisterInstance(instance, register_result);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("LiteHitOfflineRunner: RegisterInstance[%s] failed, ec=%d",
                           instance.instance_id().c_str(),
                           static_cast<int>(ec));
            return false;
        }
    }

    const std::string &override_id = config_.override_instance_id();
    const uint64_t block_size =
        config_.instances().empty() ? 0 : static_cast<uint64_t>(config_.instances().front().block_size());
    const bool aggregate_only = config_.aggregate_only();

    std::ofstream out(config_.output_result_path());
    if (!out.is_open()) {
        KVCM_LOG_ERROR("LiteHitOfflineRunner: failed to open output file [%s]", config_.output_result_path().c_str());
        return false;
    }

    // Per-capacity aggregate accumulators (aggregate_only mode). agg_hit_count[i]
    // sums hit blocks over all requests at finite capacity i; agg_max_hit_count is
    // the theoretical (infinite-capacity) sum. This pools the whole trace into one
    // global result, which combined with override_instance_id gives a single
    // service-wide hit rate curve without emitting per-request rows.
    std::vector<double> agg_capacity_gb;
    std::vector<uint64_t> agg_hit_count;
    uint64_t agg_max_hit_count = 0;
    bool agg_has_inf = false;
    uint64_t agg_input_tokens = 0;

    // Per-request long table (default). One row per (request, capacity), keyed by
    // the trace's own trace_id. The per-instance aggregate is derivable by grouping
    // on (instance_id, capacity_gb). A capacity_gb of "inf" is the theoretical
    // (infinite-capacity) upper bound.
    if (!aggregate_only) {
        out << "trace_id,instance_id,timestamp_ns,capacity_gb,hit_count,hit_rate,input_tokens\n";
    }

    // One read request is replayed into TraceQuery. TraceQuery/LiteHit are incremental
    // and keep no trace history, so a request is handled, written out, and then dropped.
    uint64_t processed = 0;
    uint64_t skipped = 0;
    // Optional latency benchmark (LITEHIT_BENCH set): time only the TraceQuery call
    // (excludes trace IO/parse) and record per-query latency + request block count so
    // we can report the latency distribution and how it grows with cache size P.
    const bool bench = std::getenv("LITEHIT_BENCH") != nullptr;
    std::vector<uint32_t> bench_lat_ns;
    std::vector<uint32_t> bench_blocks;
    if (bench) {
        bench_lat_ns.reserve(8u << 20);
        bench_blocks.reserve(8u << 20);
    }
    auto replay_read = [&](const std::shared_ptr<OptimizerSchemaTrace> &trace) {
        // Only read events carry input_len and the full-block keys TraceQuery needs.
        const auto *read_trace = dynamic_cast<const GetLocationSchemaTrace *>(trace.get());
        if (read_trace == nullptr) {
            return;
        }
        // Attribute to a single global instance when override_instance_id is set,
        // otherwise keep the trace's own (per-pod) instance_id.
        const std::string &instance_id = override_id.empty() ? read_trace->instance_id() : override_id;
        TraceQueryResult query_result;
        std::chrono::steady_clock::time_point t_begin;
        if (bench) {
            t_begin = std::chrono::steady_clock::now();
        }
        ErrorCode ec = manager.TraceQuery(instance_id, read_trace->keys(), read_trace->input_len(), query_result);
        if (bench) {
            const auto elapsed = std::chrono::steady_clock::now() - t_begin;
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
            bench_lat_ns.push_back(ns > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(ns));
            bench_blocks.push_back(static_cast<uint32_t>(read_trace->keys().size()));
        }
        if (ec != EC_OK) {
            KVCM_LOG_WARN("LiteHitOfflineRunner: skip request of instance[%s] trace_id[%s], TraceQuery ec=%d",
                          instance_id.c_str(),
                          read_trace->trace_id().c_str(),
                          static_cast<int>(ec));
            skipped++;
            return;
        }

        if (aggregate_only) {
            if (agg_hit_count.empty() && !query_result.hit_count_per_capacity.empty()) {
                agg_capacity_gb = query_result.capacity_gb;
                agg_hit_count.assign(query_result.hit_count_per_capacity.size(), 0);
            }
            for (size_t i = 0; i < query_result.hit_count_per_capacity.size() && i < agg_hit_count.size(); ++i) {
                agg_hit_count[i] += static_cast<uint64_t>(query_result.hit_count_per_capacity[i]);
            }
            if (query_result.max_hit_count >= 0) {
                agg_has_inf = true;
                agg_max_hit_count += static_cast<uint64_t>(query_result.max_hit_count);
            }
            agg_input_tokens += static_cast<uint64_t>(query_result.input_token_len);
            processed++;
            return;
        }

        // trace_id is optional in the standard trace; synthesize a stable id from
        // instance_id + timestamp_ns when absent so every request row stays identifiable.
        std::string trace_id = read_trace->trace_id();
        if (trace_id.empty()) {
            trace_id = DefaultTraceId(*read_trace);
        }
        for (size_t i = 0; i < query_result.hit_count_per_capacity.size() && i < query_result.capacity_gb.size(); ++i) {
            out << trace_id << ',' << instance_id << ',' << read_trace->timestamp_ns() << ','
                << query_result.capacity_gb[i] << ',' << query_result.hit_count_per_capacity[i] << ','
                << query_result.hit_rate_per_capacity[i] << ',' << query_result.input_token_len << '\n';
        }
        // Emitted only when enable_theoretical_max_cache is on for the instance group.
        if (query_result.max_hit_count >= 0) {
            out << trace_id << ',' << instance_id << ',' << read_trace->timestamp_ns() << ',' << "inf" << ','
                << query_result.max_hit_count << ',' << query_result.max_hit_rate << ',' << query_result.input_token_len
                << '\n';
        }
        processed++;
    };

    try {
        if (config_.assume_time_sorted()) {
            // Trace is assumed ordered by timestamp_ns: replay in a single streaming
            // pass without materializing the whole file.
            StandardTraceLoader::StreamFromFile(config_.trace_file_path(), replay_read);
        } else {
            // Order is not guaranteed: load everything and sort by timestamp first,
            // since TraceQuery models LRU recency by request order.
            std::vector<std::shared_ptr<OptimizerSchemaTrace>> traces =
                StandardTraceLoader::LoadFromFile(config_.trace_file_path());
            TraceTimeSorter::SortTracesByTimestamp(traces);
            for (const auto &trace : traces) {
                replay_read(trace);
            }
        }
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR(
            "LiteHitOfflineRunner: failed to load/replay trace [%s]: %s", config_.trace_file_path().c_str(), e.what());
        return false;
    }

    if (aggregate_only) {
        // One row per capacity: hit blocks, hit tokens (= hit blocks * block_size),
        // input tokens, and the aggregate token hit rate. The trailing "inf" row is
        // the theoretical maximum (infinite capacity), the reference for X% targets.
        out << "capacity_gb,hit_count,hit_tokens,input_tokens,hit_rate\n";
        for (size_t i = 0; i < agg_hit_count.size(); ++i) {
            const uint64_t hit_tokens = agg_hit_count[i] * block_size;
            const double hit_rate =
                agg_input_tokens > 0 ? static_cast<double>(hit_tokens) / static_cast<double>(agg_input_tokens) : 0.0;
            const double cap = i < agg_capacity_gb.size() ? agg_capacity_gb[i] : 0.0;
            out << cap << ',' << agg_hit_count[i] << ',' << hit_tokens << ',' << agg_input_tokens << ',' << hit_rate
                << '\n';
        }
        if (agg_has_inf) {
            const uint64_t hit_tokens = agg_max_hit_count * block_size;
            const double hit_rate =
                agg_input_tokens > 0 ? static_cast<double>(hit_tokens) / static_cast<double>(agg_input_tokens) : 0.0;
            out << "inf" << ',' << agg_max_hit_count << ',' << hit_tokens << ',' << agg_input_tokens << ',' << hit_rate
                << '\n';
        }
    }

    out.close();

    if (bench && !bench_lat_ns.empty()) {
        ReportQueryLatency(bench_lat_ns, bench_blocks);
    }

    KVCM_LOG_INFO("LiteHitOfflineRunner: done. processed=%lu skipped=%lu, results written to %s",
                  static_cast<unsigned long>(processed),
                  static_cast<unsigned long>(skipped),
                  config_.output_result_path().c_str());
    return true;
}

} // namespace kv_cache_manager
