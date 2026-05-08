// bench_meta_service.cpp
//
// Benchmark tool for MetaService gRPC interface.
// Supports:
//   - write     : StartWriteCache + FinishWriteCache with random keys
//   - multi_loc : accumulate N locations per block then RemoveCache, for multi-location perf testing
//                 (requires server started with KVCM_BENCH_SKIP_WRITE_FILTER=1)
//
// Each worker thread creates its own gRPC channel to simulate multiple independent clients.
//
// Build:  bazel build //tools/bench_meta_service
// Usage:  ./bench_meta_service -u <grpc_uri> -i <instance_id>
//             [-T bench_type] [-t threads] [-k keys_per_request]
//             [-Q target_qps] [-d duration_seconds] [-w warmup_seconds]
//             [-L locations_per_key]  (multi_loc mode only)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <grpcpp/grpcpp.h>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "service/proto/meta_service.grpc.pb.h"

namespace proto = kv_cache_manager::proto::meta;
using SteadyClock = std::chrono::steady_clock;

// ─── global stop flag ───

static std::atomic<bool> g_stop_flag{false};

static void SignalHandler(int /*sig*/) { g_stop_flag.store(true, std::memory_order_relaxed); }

// ─── bench type ───

enum class BenchType {
    WRITE,
    MULTI_LOC,
};

static bool ParseBenchType(const char *str, BenchType &out) {
    if (strcmp(str, "write") == 0) {
        out = BenchType::WRITE;
        return true;
    }
    if (strcmp(str, "multi_loc") == 0) {
        out = BenchType::MULTI_LOC;
        return true;
    }
    return false;
}

static const char *BenchTypeName(BenchType t) {
    switch (t) {
    case BenchType::WRITE:
        return "write";
    case BenchType::MULTI_LOC:
        return "multi_loc";
    }
    return "unknown";
}

// ─── latency statistics ───

struct LatencyStats {
    double min_ms;
    double avg_ms;
    double p50_ms;
    double p99_ms;
    double p999_ms;
    double max_ms;
};

static LatencyStats ComputeStats(std::vector<double> &lats) {
    if (lats.empty()) {
        return {0, 0, 0, 0, 0, 0};
    }
    std::sort(lats.begin(), lats.end());
    const size_t n = lats.size();
    double sum = 0;
    for (double v : lats)
        sum += v;
    return LatencyStats{
        .min_ms = lats.front(),
        .avg_ms = sum / static_cast<double>(n),
        .p50_ms = lats[n / 2],
        .p99_ms = lats[std::min(static_cast<size_t>(n * 0.99), n - 1)],
        .p999_ms = lats[std::min(static_cast<size_t>(n * 0.999), n - 1)],
        .max_ms = lats.back(),
    };
}

static double ElapsedMs(SteadyClock::time_point start) {
    auto dur = SteadyClock::now() - start;
    return std::chrono::duration<double, std::milli>(dur).count();
}

// ─── per-thread metrics ───

struct ThreadMetrics {
    std::vector<double> pair_latencies_ms;
    std::vector<double> start_write_latencies_ms;
    std::vector<double> finish_write_latencies_ms;
    std::vector<double> remove_latencies_ms;
    int64_t success_count = 0;
    int64_t start_write_fail_count = 0;
    int64_t finish_write_fail_count = 0;
    int64_t remove_success_count = 0;
    int64_t remove_fail_count = 0;
};

// ─── gRPC channel creation ───

static std::shared_ptr<grpc::Channel> CreateChannel(const std::string &uri) {
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
    args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
    args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    return grpc::CreateCustomChannel(uri, grpc::InsecureChannelCredentials(), args);
}

// ─── write benchmark worker ───

struct BenchConfig {
    std::string uri;
    std::string instance_id;
    BenchType bench_type;
    int threads;
    int keys_per_request;
    int target_qps;
    int duration_seconds;
    int warmup_seconds;
    int64_t seed;
    int locations_per_key;
    int random_start_sec; // each thread sleeps random [0, random_start_sec) before starting
};

static void WriteBenchWorker(int thread_id, const BenchConfig &cfg, ThreadMetrics &metrics) {
    // Random start delay: each thread sleeps [0, random_start_sec) seconds
    if (cfg.random_start_sec > 0) {
        std::mt19937 delay_rng(cfg.seed + thread_id + 99999);
        std::uniform_int_distribution<int> delay_dist(0, cfg.random_start_sec * 1000 - 1);
        int delay_ms = delay_dist(delay_rng);
        fprintf(stdout, "[thread %d] random start delay: %d ms\n", thread_id, delay_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    // Create independent gRPC channel and stub for this thread
    auto channel = CreateChannel(cfg.uri);
    if (!channel->WaitForConnected(
            gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC), gpr_time_from_seconds(5, GPR_TIMESPAN)))) {
        fprintf(stderr, "[thread %d] Failed to connect to %s within 5s\n", thread_id, cfg.uri.c_str());
        return;
    }
    auto stub = proto::MetaService::NewStub(channel);

    // Rate limiting: interval per operation for this thread
    const double interval_us = (cfg.target_qps > 0) ? (static_cast<double>(cfg.threads) / cfg.target_qps) * 1e6 : 0;

    // Pre-allocate latency vectors
    const size_t estimated_ops =
        static_cast<size_t>((static_cast<double>(cfg.target_qps) / cfg.threads) * cfg.duration_seconds * 1.2) + 100;
    metrics.pair_latencies_ms.reserve(estimated_ops);
    metrics.start_write_latencies_ms.reserve(estimated_ops);
    metrics.finish_write_latencies_ms.reserve(estimated_ops);

    // RNG per thread: seed deterministically derived from global seed + thread_id
    std::mt19937_64 rng(cfg.seed + thread_id);
    std::uniform_int_distribution<int64_t> key_dist(1, INT64_MAX / 2);

    const auto bench_start = SteadyClock::now();
    const auto warmup_end = bench_start + std::chrono::seconds(cfg.warmup_seconds);
    auto next_send_time = bench_start;
    int64_t seq = 0;
    int consecutive_failures = 0;

    while (!g_stop_flag.load(std::memory_order_relaxed)) {
        // Rate limiting: sleep until next scheduled send time
        auto now = SteadyClock::now();
        if (interval_us > 0 && now < next_send_time) {
            std::this_thread::sleep_until(next_send_time);
        }
        next_send_time += std::chrono::microseconds(static_cast<int64_t>(interval_us));

        bool in_warmup = SteadyClock::now() < warmup_end;
        seq++;

        // Generate random block_keys and token_ids
        std::vector<int64_t> block_keys(cfg.keys_per_request);
        std::vector<int64_t> token_ids(cfg.keys_per_request);
        for (int j = 0; j < cfg.keys_per_request; ++j) {
            block_keys[j] = key_dist(rng);
            token_ids[j] = key_dist(rng);
        }

        // Build trace_id
        char trace_buf[64];
        snprintf(trace_buf, sizeof(trace_buf), "bench_%d_%ld", thread_id, seq);
        std::string trace_id(trace_buf);

        // ── StartWriteCache ──
        proto::StartWriteCacheRequest start_req;
        start_req.set_trace_id(trace_id);
        start_req.set_instance_id(cfg.instance_id);
        for (int j = 0; j < cfg.keys_per_request; ++j) {
            start_req.add_block_keys(block_keys[j]);
            start_req.add_token_ids(token_ids[j]);
        }
        start_req.set_write_timeout_seconds(60);

        proto::StartWriteCacheResponse start_resp;
        grpc::ClientContext start_ctx;
        start_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));

        auto pair_start = SteadyClock::now();
        auto sw_start = SteadyClock::now();
        grpc::Status start_status = stub->StartWriteCache(&start_ctx, start_req, &start_resp);
        double sw_ms = ElapsedMs(sw_start);

        bool start_ok = start_status.ok() && start_resp.header().status().code() == proto::OK;

        if (!start_ok) {
            if (!in_warmup) {
                metrics.start_write_fail_count++;
                metrics.start_write_latencies_ms.push_back(sw_ms);
            }
            // Print error details for first few failures
            if (consecutive_failures < 10) {
                if (!start_status.ok()) {
                    fprintf(stderr,
                            "[thread %d] StartWrite gRPC error: %s\n",
                            thread_id,
                            start_status.error_message().c_str());
                } else {
                    fprintf(stderr,
                            "[thread %d] StartWrite business error: code=%d msg=%s\n",
                            thread_id,
                            static_cast<int>(start_resp.header().status().code()),
                            start_resp.header().status().message().c_str());
                }
            }
            consecutive_failures++;
            continue;
        }
        consecutive_failures = 0;

        // ── FinishWriteCache ──
        proto::FinishWriteCacheRequest finish_req;
        finish_req.set_trace_id(trace_id);
        finish_req.set_instance_id(cfg.instance_id);
        finish_req.set_write_session_id(start_resp.write_session_id());
        // success_blocks: [0, offset) are marked as success
        // Use locations size to determine how many blocks actually need writing
        int success_count_blocks = start_resp.locations_size();
        finish_req.mutable_success_blocks()->set_offset(success_count_blocks);

        proto::CommonResponse finish_resp;
        grpc::ClientContext finish_ctx;
        finish_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));

        auto fw_start = SteadyClock::now();
        grpc::Status finish_status = stub->FinishWriteCache(&finish_ctx, finish_req, &finish_resp);
        double fw_ms = ElapsedMs(fw_start);
        double pair_ms = ElapsedMs(pair_start);

        bool finish_ok = finish_status.ok() && finish_resp.header().status().code() == proto::OK;

        if (!in_warmup) {
            metrics.start_write_latencies_ms.push_back(sw_ms);
            metrics.finish_write_latencies_ms.push_back(fw_ms);
            if (finish_ok) {
                metrics.success_count++;
                metrics.pair_latencies_ms.push_back(pair_ms);
            } else {
                metrics.finish_write_fail_count++;
                if (!finish_status.ok()) {
                    fprintf(stderr,
                            "[thread %d] FinishWrite gRPC error: %s\n",
                            thread_id,
                            finish_status.error_message().c_str());
                } else {
                    fprintf(stderr,
                            "[thread %d] FinishWrite business error: code=%d msg=%s\n",
                            thread_id,
                            static_cast<int>(finish_resp.header().status().code()),
                            finish_resp.header().status().message().c_str());
                }
            }
        }
    }
}

// ─── multi_loc benchmark worker ───
// Phase 1: Write the same fixed key set L times (locations_per_key), accumulating L locations per block.
//          Requires KVCM_BENCH_SKIP_WRITE_FILTER=1 on the server to bypass write deduplication.
// Phase 2: RemoveCache to delete all locations for these keys.
// Repeat until duration expires.

static void MultiLocBenchWorker(int thread_id, const BenchConfig &cfg, ThreadMetrics &metrics) {
    // Random start delay: each thread sleeps [0, random_start_sec) seconds
    if (cfg.random_start_sec > 0) {
        std::mt19937 delay_rng(cfg.seed + thread_id + 99999);
        std::uniform_int_distribution<int> delay_dist(0, cfg.random_start_sec * 1000 - 1);
        int delay_ms = delay_dist(delay_rng);
        fprintf(stdout, "[thread %d] random start delay: %d ms\n", thread_id, delay_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    auto channel = CreateChannel(cfg.uri);
    if (!channel->WaitForConnected(
            gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC), gpr_time_from_seconds(5, GPR_TIMESPAN)))) {
        fprintf(stderr, "[thread %d] Failed to connect to %s within 5s\n", thread_id, cfg.uri.c_str());
        return;
    }
    auto stub = proto::MetaService::NewStub(channel);

    const double interval_us = (cfg.target_qps > 0) ? (static_cast<double>(cfg.threads) / cfg.target_qps) * 1e6 : 0;

    const size_t estimated_ops =
        static_cast<size_t>((static_cast<double>(cfg.target_qps) / cfg.threads) * cfg.duration_seconds * 1.2) + 100;
    metrics.pair_latencies_ms.reserve(estimated_ops);
    metrics.start_write_latencies_ms.reserve(estimated_ops);
    metrics.finish_write_latencies_ms.reserve(estimated_ops);
    metrics.remove_latencies_ms.reserve(estimated_ops);

    std::mt19937_64 rng(cfg.seed + thread_id);
    std::uniform_int_distribution<int64_t> key_dist(1, INT64_MAX / 2);

    const int num_keys = cfg.keys_per_request;

    std::vector<int64_t> block_keys(num_keys);
    std::vector<int64_t> token_ids(num_keys);

    const auto bench_start = SteadyClock::now();
    const auto warmup_end = bench_start + std::chrono::seconds(cfg.warmup_seconds);
    auto next_send_time = bench_start;
    int64_t seq = 0;
    int consecutive_failures = 0;
    int64_t round_number = 0;

    while (!g_stop_flag.load(std::memory_order_relaxed)) {
        round_number++;

        // Re-randomize keys each round so every round operates on fresh keys
        for (int j = 0; j < num_keys; ++j) {
            block_keys[j] = key_dist(rng);
            token_ids[j] = key_dist(rng);
        }

        // Write L rounds to accumulate locations on the same keys
        for (int loc_round = 0; loc_round < cfg.locations_per_key; ++loc_round) {
            if (g_stop_flag.load(std::memory_order_relaxed))
                break;

            auto now = SteadyClock::now();
            if (interval_us > 0 && now < next_send_time) {
                std::this_thread::sleep_until(next_send_time);
            }
            next_send_time += std::chrono::microseconds(static_cast<int64_t>(interval_us));
            bool in_warmup = SteadyClock::now() < warmup_end;
            seq++;

            char trace_buf[64];
            snprintf(trace_buf, sizeof(trace_buf), "ml_%d_%ld", thread_id, seq);
            std::string trace_id(trace_buf);

            // StartWriteCache
            proto::StartWriteCacheRequest start_req;
            start_req.set_trace_id(trace_id);
            start_req.set_instance_id(cfg.instance_id);
            for (int j = 0; j < num_keys; ++j) {
                start_req.add_block_keys(block_keys[j]);
                start_req.add_token_ids(token_ids[j]);
            }
            start_req.set_write_timeout_seconds(60);

            proto::StartWriteCacheResponse start_resp;
            grpc::ClientContext start_ctx;
            start_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));

            auto pair_start = SteadyClock::now();
            grpc::Status start_status = stub->StartWriteCache(&start_ctx, start_req, &start_resp);
            double sw_ms = ElapsedMs(pair_start);

            bool start_ok = start_status.ok() && start_resp.header().status().code() == proto::OK;

            if (!start_ok) {
                if (!in_warmup) {
                    metrics.start_write_fail_count++;
                    metrics.start_write_latencies_ms.push_back(sw_ms);
                }
                if (consecutive_failures < 5) {
                    if (!start_status.ok()) {
                        fprintf(stderr,
                                "[thread %d] StartWrite gRPC error: %s\n",
                                thread_id,
                                start_status.error_message().c_str());
                    } else {
                        fprintf(stderr,
                                "[thread %d] StartWrite biz error: code=%d msg=%s\n",
                                thread_id,
                                static_cast<int>(start_resp.header().status().code()),
                                start_resp.header().status().message().c_str());
                    }
                }
                consecutive_failures++;
                continue;
            }
            consecutive_failures = 0;

            // FinishWriteCache — mark all blocks as success
            proto::FinishWriteCacheRequest finish_req;
            finish_req.set_trace_id(trace_id);
            finish_req.set_instance_id(cfg.instance_id);
            finish_req.set_write_session_id(start_resp.write_session_id());
            finish_req.mutable_success_blocks()->set_offset(start_resp.locations_size());

            proto::CommonResponse finish_resp;
            grpc::ClientContext finish_ctx;
            finish_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));

            auto fw_start = SteadyClock::now();
            grpc::Status finish_status = stub->FinishWriteCache(&finish_ctx, finish_req, &finish_resp);
            double fw_ms = ElapsedMs(fw_start);
            double pair_ms = ElapsedMs(pair_start);

            bool finish_ok = finish_status.ok() && finish_resp.header().status().code() == proto::OK;

            if (!in_warmup) {
                metrics.start_write_latencies_ms.push_back(sw_ms);
                metrics.finish_write_latencies_ms.push_back(fw_ms);
                if (finish_ok) {
                    metrics.success_count++;
                    metrics.pair_latencies_ms.push_back(pair_ms);
                } else {
                    metrics.finish_write_fail_count++;
                    if (!finish_status.ok()) {
                        fprintf(stderr,
                                "[thread %d] FinishWrite gRPC error: %s\n",
                                thread_id,
                                finish_status.error_message().c_str());
                    } else {
                        fprintf(stderr,
                                "[thread %d] FinishWrite biz error: code=%d msg=%s\n",
                                thread_id,
                                static_cast<int>(finish_resp.header().status().code()),
                                finish_resp.header().status().message().c_str());
                    }
                }
            }
        } // end loc_round loop

        if (g_stop_flag.load(std::memory_order_relaxed))
            break;

        // Log accumulation progress periodically
        bool in_warmup = SteadyClock::now() < warmup_end;
        if (!in_warmup && round_number % 10 == 1) {
            fprintf(stdout,
                    "[thread %d] round %ld: accumulated %d locations/key across %d keys\n",
                    thread_id,
                    round_number,
                    cfg.locations_per_key,
                    num_keys);
            fflush(stdout);
        }
    }
}

// ─── multi_loc benchmark worker ───
// Phase 1: Write the same fixed key set L times (locations_per_key), accumulating L locations per block.
//          Requires KVCM_BENCH_SKIP_WRITE_FILTER=1 on the server to bypass write deduplication.
// Phase 2: RemoveCache to delete all locations for these keys.
// Repeat until duration expires.

static void MultiLocBenchWorker(int thread_id, const BenchConfig &cfg, ThreadMetrics &metrics) {
    // Random start delay: each thread sleeps [0, random_start_sec) seconds
    if (cfg.random_start_sec > 0) {
        std::mt19937 delay_rng(cfg.seed + thread_id + 99999);
        std::uniform_int_distribution<int> delay_dist(0, cfg.random_start_sec * 1000 - 1);
        int delay_ms = delay_dist(delay_rng);
        fprintf(stdout, "[thread %d] random start delay: %d ms\n", thread_id, delay_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    auto channel = CreateChannel(cfg.uri);
    if (!channel->WaitForConnected(
            gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC), gpr_time_from_seconds(5, GPR_TIMESPAN)))) {
        fprintf(stderr, "[thread %d] Failed to connect to %s within 5s\n", thread_id, cfg.uri.c_str());
        return;
    }
    auto stub = proto::MetaService::NewStub(channel);

    const double interval_us = (cfg.target_qps > 0) ? (static_cast<double>(cfg.threads) / cfg.target_qps) * 1e6 : 0;

    const size_t estimated_ops =
        static_cast<size_t>((static_cast<double>(cfg.target_qps) / cfg.threads) * cfg.duration_seconds * 1.2) + 100;
    metrics.pair_latencies_ms.reserve(estimated_ops);
    metrics.start_write_latencies_ms.reserve(estimated_ops);
    metrics.finish_write_latencies_ms.reserve(estimated_ops);
    metrics.remove_latencies_ms.reserve(estimated_ops);

    std::mt19937_64 rng(cfg.seed + thread_id);
    std::uniform_int_distribution<int64_t> key_dist(1, INT64_MAX / 2);

    const int num_keys = cfg.keys_per_request;

    std::vector<int64_t> block_keys(num_keys);
    std::vector<int64_t> token_ids(num_keys);

    const auto bench_start = SteadyClock::now();
    const auto warmup_end = bench_start + std::chrono::seconds(cfg.warmup_seconds);
    auto next_send_time = bench_start;
    int64_t seq = 0;
    int consecutive_failures = 0;
    int64_t round_number = 0;

    while (!g_stop_flag.load(std::memory_order_relaxed)) {
        round_number++;

        // Re-randomize keys each round so every round operates on fresh keys
        for (int j = 0; j < num_keys; ++j) {
            block_keys[j] = key_dist(rng);
            token_ids[j] = key_dist(rng);
        }

        // Write L rounds to accumulate locations on the same keys
        for (int loc_round = 0; loc_round < cfg.locations_per_key; ++loc_round) {
            if (g_stop_flag.load(std::memory_order_relaxed))
                break;

            auto now = SteadyClock::now();
            if (interval_us > 0 && now < next_send_time) {
                std::this_thread::sleep_until(next_send_time);
            }
            next_send_time += std::chrono::microseconds(static_cast<int64_t>(interval_us));
            bool in_warmup = SteadyClock::now() < warmup_end;
            seq++;

            char trace_buf[64];
            snprintf(trace_buf, sizeof(trace_buf), "ml_%d_%ld", thread_id, seq);
            std::string trace_id(trace_buf);

            // StartWriteCache
            proto::StartWriteCacheRequest start_req;
            start_req.set_trace_id(trace_id);
            start_req.set_instance_id(cfg.instance_id);
            for (int j = 0; j < num_keys; ++j) {
                start_req.add_block_keys(block_keys[j]);
                start_req.add_token_ids(token_ids[j]);
            }
            start_req.set_write_timeout_seconds(60);

            proto::StartWriteCacheResponse start_resp;
            grpc::ClientContext start_ctx;
            start_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));

            auto pair_start = SteadyClock::now();
            grpc::Status start_status = stub->StartWriteCache(&start_ctx, start_req, &start_resp);
            double sw_ms = ElapsedMs(pair_start);

            bool start_ok = start_status.ok() && start_resp.header().status().code() == proto::OK;

            if (!start_ok) {
                if (!in_warmup) {
                    metrics.start_write_fail_count++;
                    metrics.start_write_latencies_ms.push_back(sw_ms);
                }
                if (consecutive_failures < 5) {
                    if (!start_status.ok()) {
                        fprintf(stderr,
                                "[thread %d] StartWrite gRPC error: %s\n",
                                thread_id,
                                start_status.error_message().c_str());
                    } else {
                        fprintf(stderr,
                                "[thread %d] StartWrite biz error: code=%d msg=%s\n",
                                thread_id,
                                static_cast<int>(start_resp.header().status().code()),
                                start_resp.header().status().message().c_str());
                    }
                }
                consecutive_failures++;
                continue;
            }
            consecutive_failures = 0;

            // FinishWriteCache — mark all blocks as success
            proto::FinishWriteCacheRequest finish_req;
            finish_req.set_trace_id(trace_id);
            finish_req.set_instance_id(cfg.instance_id);
            finish_req.set_write_session_id(start_resp.write_session_id());
            finish_req.mutable_success_blocks()->set_offset(start_resp.locations_size());

            proto::CommonResponse finish_resp;
            grpc::ClientContext finish_ctx;
            finish_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));

            auto fw_start = SteadyClock::now();
            grpc::Status finish_status = stub->FinishWriteCache(&finish_ctx, finish_req, &finish_resp);
            double fw_ms = ElapsedMs(fw_start);
            double pair_ms = ElapsedMs(pair_start);

            bool finish_ok = finish_status.ok() && finish_resp.header().status().code() == proto::OK;

            if (!in_warmup) {
                metrics.start_write_latencies_ms.push_back(sw_ms);
                metrics.finish_write_latencies_ms.push_back(fw_ms);
                if (finish_ok) {
                    metrics.success_count++;
                    metrics.pair_latencies_ms.push_back(pair_ms);
                } else {
                    metrics.finish_write_fail_count++;
                    if (!finish_status.ok()) {
                        fprintf(stderr,
                                "[thread %d] FinishWrite gRPC error: %s\n",
                                thread_id,
                                finish_status.error_message().c_str());
                    } else {
                        fprintf(stderr,
                                "[thread %d] FinishWrite biz error: code=%d msg=%s\n",
                                thread_id,
                                static_cast<int>(finish_resp.header().status().code()),
                                finish_resp.header().status().message().c_str());
                    }
                }
            }
        } // end loc_round loop

        if (g_stop_flag.load(std::memory_order_relaxed))
            break;

        // Log accumulation progress periodically
        bool in_warmup = SteadyClock::now() < warmup_end;
        if (!in_warmup && round_number % 10 == 1) {
            fprintf(stdout,
                    "[thread %d] round %ld: accumulated %d locations/key across %d keys\n",
                    thread_id,
                    round_number,
                    cfg.locations_per_key,
                    num_keys);
            fflush(stdout);
        }
    }
}

// ─── usage ───

static void PrintUsage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Required:\n"
            "  -u <uri>                gRPC server address, e.g. localhost:6381\n"
            "  -i <instance_id>        Instance ID for benchmark\n"
            "\n"
            "Optional:\n"
            "  -T <bench_type>         Benchmark type: write|multi_loc (default: write)\n"
            "  -t <threads>            Number of worker threads (default: 4)\n"
            "  -k <keys_per_request>   Number of block keys per request (default: 8)\n"
            "  -Q <target_qps>         Target total QPS across all threads (default: 100)\n"
            "  -d <duration_seconds>   Benchmark duration in seconds (default: 30)\n"
            "  -w <warmup_seconds>     Warmup time in seconds (default: 3)\n"
            "  -s <seed>              Random seed for key generation (default: current timestamp)\n"
            "  -L <locations_per_key>  multi_loc mode: locations to accumulate per key before removing (default: 100)\n"
            "  -R <random_start_sec>  Each thread randomly sleeps 0~R seconds before starting (default: 0)\n"
            "  -h                      Show this help message\n",
            prog);
}

// ─── main ───

int main(int argc, char *argv[]) {
    BenchConfig cfg;
    cfg.bench_type = BenchType::WRITE;
    cfg.threads = 4;
    cfg.keys_per_request = 8;
    cfg.target_qps = 100;
    cfg.duration_seconds = 30;
    cfg.warmup_seconds = 3;
    cfg.seed = std::chrono::steady_clock::now().time_since_epoch().count();
    cfg.locations_per_key = 100;
    cfg.random_start_sec = 0;

    int opt;
    while ((opt = getopt(argc, argv, "u:i:T:t:k:Q:d:w:s:L:R:h")) != -1) {
        switch (opt) {
        case 'u':
            cfg.uri = optarg;
            break;
        case 'i':
            cfg.instance_id = optarg;
            break;
        case 'T':
            if (!ParseBenchType(optarg, cfg.bench_type)) {
                fprintf(stderr, "Error: unsupported bench type '%s' (supported: write, multi_loc)\n", optarg);
                return 1;
            }
            break;
        case 't':
            cfg.threads = atoi(optarg);
            break;
        case 'k':
            cfg.keys_per_request = atoi(optarg);
            break;
        case 'Q':
            cfg.target_qps = atoi(optarg);
            break;
        case 'd':
            cfg.duration_seconds = atoi(optarg);
            break;
        case 'w':
            cfg.warmup_seconds = atoi(optarg);
            break;
        case 's':
            cfg.seed = atoll(optarg);
            break;
        case 'L':
            cfg.locations_per_key = atoi(optarg);
            break;
        case 'R':
            cfg.random_start_sec = atoi(optarg);
            break;
        case 'h':
            PrintUsage(argv[0]);
            return 0;
        default:
            PrintUsage(argv[0]);
            return 1;
        }
    }

    if (cfg.uri.empty() || cfg.instance_id.empty()) {
        fprintf(stderr, "Error: -u <uri> and -i <instance_id> are required\n\n");
        PrintUsage(argv[0]);
        return 1;
    }
    if (cfg.threads <= 0 || cfg.keys_per_request <= 0 || cfg.target_qps <= 0 || cfg.duration_seconds <= 0) {
        fprintf(stderr, "Error: threads, keys_per_request, target_qps, duration must be > 0\n");
        return 1;
    }

    // Register signal handler for graceful shutdown
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    fprintf(stdout,
            "=== MetaService Benchmark ===\n"
            "  uri:               %s\n"
            "  instance_id:       %s\n"
            "  bench_type:        %s\n"
            "  threads:           %d\n"
            "  keys_per_request:  %d\n"
            "  target_qps:        %d\n"
            "  duration:          %ds\n"
            "  warmup:            %ds\n"
            "  seed:              %ld\n\n",
            cfg.uri.c_str(),
            cfg.instance_id.c_str(),
            BenchTypeName(cfg.bench_type),
            cfg.threads,
            cfg.keys_per_request,
            cfg.target_qps,
            cfg.duration_seconds,
            cfg.warmup_seconds,
            cfg.seed);
    if (cfg.bench_type == BenchType::MULTI_LOC) {
        fprintf(stdout, "  locations_per_key: %d\n", cfg.locations_per_key);
    }
    fprintf(stdout, "\n");
    fflush(stdout);

    // Launch worker threads
    std::vector<ThreadMetrics> all_metrics(cfg.threads);
    std::vector<std::thread> workers;
    workers.reserve(cfg.threads);

    auto overall_start = SteadyClock::now();

    for (int i = 0; i < cfg.threads; ++i) {
        switch (cfg.bench_type) {
        case BenchType::WRITE:
            workers.emplace_back(WriteBenchWorker, i, std::cref(cfg), std::ref(all_metrics[i]));
            break;
        case BenchType::MULTI_LOC:
            workers.emplace_back(MultiLocBenchWorker, i, std::cref(cfg), std::ref(all_metrics[i]));
            break;
        }
    }

    // Wait for duration, checking stop_flag periodically
    auto deadline = overall_start + std::chrono::seconds(cfg.duration_seconds + cfg.warmup_seconds);
    while (!g_stop_flag.load(std::memory_order_relaxed) && SteadyClock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    g_stop_flag.store(true, std::memory_order_relaxed);

    // Join all workers
    for (auto &w : workers) {
        w.join();
    }

    double actual_duration_s = std::chrono::duration<double>(SteadyClock::now() - overall_start).count();
    // Subtract warmup from reported duration
    double effective_duration_s = actual_duration_s - cfg.warmup_seconds;
    if (effective_duration_s < 0.001)
        effective_duration_s = 0.001;

    // Merge metrics from all threads
    std::vector<double> merged_pair, merged_sw, merged_fw, merged_rm;
    int64_t total_success = 0, total_sw_fail = 0, total_fw_fail = 0;
    int64_t total_rm_success = 0, total_rm_fail = 0;

    for (auto &m : all_metrics) {
        merged_pair.insert(merged_pair.end(), m.pair_latencies_ms.begin(), m.pair_latencies_ms.end());
        merged_sw.insert(merged_sw.end(), m.start_write_latencies_ms.begin(), m.start_write_latencies_ms.end());
        merged_fw.insert(merged_fw.end(), m.finish_write_latencies_ms.begin(), m.finish_write_latencies_ms.end());
        merged_rm.insert(merged_rm.end(), m.remove_latencies_ms.begin(), m.remove_latencies_ms.end());
        total_success += m.success_count;
        total_sw_fail += m.start_write_fail_count;
        total_fw_fail += m.finish_write_fail_count;
        total_rm_success += m.remove_success_count;
        total_rm_fail += m.remove_fail_count;
    }

    int64_t total_ops = total_success + total_sw_fail + total_fw_fail + total_rm_success + total_rm_fail;
    double actual_qps = total_ops / effective_duration_s;

    auto pair_stats = ComputeStats(merged_pair);
    auto sw_stats = ComputeStats(merged_sw);
    auto fw_stats = ComputeStats(merged_fw);
    auto rm_stats = ComputeStats(merged_rm);

    // Print results
    fprintf(stdout,
            "=== Results (actual %.2fs, effective %.2fs excl. warmup) ===\n"
            "  Total operations:    %ld\n"
            "  Actual QPS:          %.1f\n"
            "  Write success:       %ld\n"
            "  StartWrite fail:     %ld\n"
            "  FinishWrite fail:    %ld\n",
            actual_duration_s,
            effective_duration_s,
            total_ops,
            actual_qps,
            total_success,
            total_sw_fail,
            total_fw_fail);

    if (total_rm_success > 0 || total_rm_fail > 0) {
        fprintf(stdout,
                "  Remove success:      %ld\n"
                "  Remove fail:         %ld\n",
                total_rm_success,
                total_rm_fail);
    }

    if (!merged_pair.empty()) {
        fprintf(stdout,
                "\n  Write Pair Latency (ms):\n"
                "    avg      p50      p99      p999     max\n"
                "    %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n",
                pair_stats.avg_ms,
                pair_stats.p50_ms,
                pair_stats.p99_ms,
                pair_stats.p999_ms,
                pair_stats.max_ms);
    }

    if (!merged_sw.empty()) {
        fprintf(stdout,
                "\n  StartWrite Latency (ms):\n"
                "    avg      p50      p99      p999     max\n"
                "    %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n",
                sw_stats.avg_ms,
                sw_stats.p50_ms,
                sw_stats.p99_ms,
                sw_stats.p999_ms,
                sw_stats.max_ms);
    }

    if (!merged_fw.empty()) {
        fprintf(stdout,
                "\n  FinishWrite Latency (ms):\n"
                "    avg      p50      p99      p999     max\n"
                "    %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n",
                fw_stats.avg_ms,
                fw_stats.p50_ms,
                fw_stats.p99_ms,
                fw_stats.p999_ms,
                fw_stats.max_ms);
    }

    if (!merged_rm.empty()) {
        fprintf(stdout,
                "\n  RemoveCache Latency (ms):\n"
                "    avg      p50      p99      p999     max\n"
                "    %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n",
                rm_stats.avg_ms,
                rm_stats.p50_ms,
                rm_stats.p99_ms,
                rm_stats.p999_ms,
                rm_stats.max_ms);
    }

    fprintf(stdout, "\n");
    return 0;
}
