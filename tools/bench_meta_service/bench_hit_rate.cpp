// bench_hit_rate.cpp
//
// Hit-rate benchmark tool for MetaService gRPC interface.
// Simulates multi-turn conversations: for each client, across multiple rounds,
// the tool calls GetCacheLocation + StartWriteCache + FinishWriteCache,
// measuring per-round cache hit rates.
//
// Workload generation logic references sglang bench_multiturn.py:
//   - Round 0: client sends initial token_ids (prompt), writes to cache.
//   - Round 1..N: client extends history with output_tokens + sub_question_tokens,
//     queries GetCacheLocation to measure prefix hit, then writes the full sequence.
//
// Build:  bazel build //tools/bench_meta_service:bench_hit_rate
// Usage:  ./bench_hit_rate -u <grpc_uri> -i <instance_id>
//             [-c num_clients] [-t threads] [-k tokens_per_request]
//             [-o output_tokens] [-K sub_question_tokens] [-n num_rounds]
//             [-R request_rate] [-D distribution] [-b block_size] [-B] [-s seed]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
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

// ─── config ───

struct HitRateConfig {
    std::string uri;
    std::string instance_id;
    int num_clients;           // number of simulated conversation sessions
    int threads;               // worker thread count
    int tokens_per_request;    // initial prompt length in token_ids
    int output_tokens;         // simulated output token_ids per round
    int sub_question_tokens;   // sub-question token_ids per round (0 = use tokens_per_request)
    int num_rounds;            // rounds per client
    double request_rate;       // requests per second (controls pacing)
    std::string distribution;  // "poisson" or "uniform"
    bool enable_round_barrier; // sync all clients between rounds
    int block_size;            // tokens per block (0 = auto-detect via GetInstanceInfo)
    int64_t seed;              // seed for input token generation (deterministic)
    int64_t output_seed;       // seed for output/sub-question token generation
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

// ─── per-client state ───

struct ClientState {
    int client_id;
    int current_round;
    std::vector<int64_t> token_ids; // accumulated token ids (grows each round)
};

// ─── per-round metrics (thread-local) ───

struct RoundMetrics {
    int64_t total_tokens_queried = 0;
    int64_t total_tokens_hit = 0;
    std::vector<double> get_location_latencies_ms;
    std::vector<double> start_write_latencies_ms;
    std::vector<double> finish_write_latencies_ms;
    std::vector<double> round_trip_latencies_ms;
    int64_t success_count = 0;
    int64_t fail_count = 0;
};

// ─── thread metrics ───

struct ThreadMetrics {
    std::vector<RoundMetrics> per_round; // indexed by round number
};

// ─── barrier for round sync ───

class Barrier {
public:
    explicit Barrier(int count) : threshold_(count), count_(count), generation_(0) {}

    // Wait at the barrier. Returns true if all threads arrived normally,
    // false if released early due to g_stop_flag.
    bool Wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        int gen = generation_;
        if (--count_ == 0) {
            generation_++;
            count_ = threshold_;
            cv_.notify_all();
            return true;
        }
        while (gen == generation_ && !g_stop_flag.load(std::memory_order_relaxed)) {
            cv_.wait_for(lock, std::chrono::milliseconds(200));
        }
        return gen != generation_;
    }

    // Called when a thread exits early without reaching Wait().
    // Permanently reduces the expected thread count so remaining threads
    // won't deadlock waiting for a thread that will never arrive.
    void Withdraw() {
        std::unique_lock<std::mutex> lock(mutex_);
        threshold_--;
        if (--count_ == 0) {
            generation_++;
            count_ = threshold_;
            cv_.notify_all();
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int threshold_;
    int count_;
    int generation_;
};

// ─── query block_size via GetInstanceInfo ───

static int FetchBlockSize(const std::string &uri, const std::string &instance_id) {
    auto channel = CreateChannel(uri);
    if (!channel->WaitForConnected(
            gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC), gpr_time_from_seconds(5, GPR_TIMESPAN)))) {
        fprintf(stderr, "FetchBlockSize: failed to connect to %s\n", uri.c_str());
        return 0;
    }
    auto stub = proto::MetaService::NewStub(channel);

    proto::GetInstanceInfoRequest req;
    req.set_trace_id("bench_hr_get_block_size");
    req.set_instance_id(instance_id);

    proto::GetInstanceInfoResponse resp;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    grpc::Status status = stub->GetInstanceInfo(&ctx, req, &resp);
    if (!status.ok()) {
        fprintf(stderr, "FetchBlockSize: gRPC error: %s\n", status.error_message().c_str());
        return 0;
    }
    if (resp.header().status().code() != proto::OK) {
        fprintf(stderr,
                "FetchBlockSize: service error: code=%d msg=%s\n",
                static_cast<int>(resp.header().status().code()),
                resp.header().status().message().c_str());
        return 0;
    }

    int bs = resp.instance_info().block_size();
    fprintf(stdout, "  Fetched block_size=%d from GetInstanceInfo\n", bs);
    return bs;
}

// ─── worker function ───

static void HitRateWorker(int thread_id,
                          const HitRateConfig &cfg,
                          ThreadMetrics &metrics,
                          const std::vector<int> &client_ids,
                          Barrier *barrier) {
    // Create independent gRPC channel and stub
    auto channel = CreateChannel(cfg.uri);
    if (!channel->WaitForConnected(
            gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC), gpr_time_from_seconds(5, GPR_TIMESPAN)))) {
        fprintf(stderr, "[thread %d] Failed to connect to %s within 5s\n", thread_id, cfg.uri.c_str());
        if (barrier) {
            barrier->Withdraw();
        }
        return;
    }
    auto stub = proto::MetaService::NewStub(channel);

    // RNG for input tokens: deterministic from seed
    std::mt19937_64 rng(cfg.seed + thread_id);
    std::uniform_int_distribution<int64_t> token_dist(1, INT64_MAX / 2);

    // RNG for output/sub-question tokens: separate seed (non-deterministic by default)
    std::mt19937_64 output_rng(cfg.output_seed + thread_id);
    std::uniform_int_distribution<int64_t> output_token_dist(1, INT64_MAX / 2);

    // RNG for rate-limiting: isolated so -R/-D flags don't affect token sequences
    std::mt19937_64 rate_rng(cfg.seed ^ (static_cast<int64_t>(0xC6A4A7935BD1E995ULL) + thread_id));

    int actual_sub_q_tokens = (cfg.sub_question_tokens > 0) ? cfg.sub_question_tokens : cfg.tokens_per_request;

    // Initialize per-round metrics
    metrics.per_round.resize(cfg.num_rounds);

    // Initialize client states with initial token_ids (simulating first prompt)
    std::vector<ClientState> clients;
    clients.reserve(client_ids.size());
    for (int cid : client_ids) {
        ClientState cs;
        cs.client_id = cid;
        cs.current_round = 0;
        cs.token_ids.resize(cfg.tokens_per_request);
        for (int j = 0; j < cfg.tokens_per_request; ++j) {
            cs.token_ids[j] = token_dist(rng);
        }
        clients.push_back(std::move(cs));
    }

    for (int round = 0; round < cfg.num_rounds && !g_stop_flag.load(std::memory_order_relaxed); ++round) {
        for (auto &cs : clients) {
            if (g_stop_flag.load(std::memory_order_relaxed))
                break;

            // ── Rate limiting ──
            // Per-thread rate = total target QPS / num_threads, so that the
            // aggregate rate across all threads equals cfg.request_rate.
            if (cfg.request_rate > 0) {
                double per_thread_rate = cfg.request_rate / cfg.threads;
                double sleep_s;
                if (cfg.distribution == "poisson") {
                    std::exponential_distribution<double> exp_dist(per_thread_rate);
                    sleep_s = exp_dist(rate_rng);
                } else {
                    double avg_interval = 1.0 / per_thread_rate;
                    std::uniform_real_distribution<double> uni_dist(0, 2 * avg_interval);
                    sleep_s = uni_dist(rate_rng);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(sleep_s * 1e6)));
            }

            char trace_buf[64];
            snprintf(trace_buf, sizeof(trace_buf), "bench_hr_%d_%d_%d", thread_id, cs.client_id, round);
            std::string trace_id(trace_buf);

            auto round_start = SteadyClock::now();

            // ── Step 1: GetCacheLocation (prefix match) ──
            proto::GetCacheLocationRequest get_req;
            get_req.set_trace_id(trace_id);
            get_req.set_instance_id(cfg.instance_id);
            get_req.set_query_type(proto::QT_PREFIX_MATCH);
            for (auto t : cs.token_ids)
                get_req.add_token_ids(t);

            proto::GetCacheLocationResponse get_resp;
            grpc::ClientContext get_ctx;
            get_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

            auto gl_start = SteadyClock::now();
            grpc::Status get_status = stub->GetCacheLocation(&get_ctx, get_req, &get_resp);
            double gl_ms = ElapsedMs(gl_start);

            bool get_ok = get_status.ok() && get_resp.header().status().code() == proto::OK;
            int64_t tokens_hit = 0;
            int64_t tokens_queried = static_cast<int64_t>(cs.token_ids.size());

            if (get_ok) {
                // locations_size() returns block count; multiply by block_size
                // to get token-level hit count
                tokens_hit = static_cast<int64_t>(get_resp.locations_size()) * cfg.block_size;
                // Cap to tokens_queried to avoid > 100% hit rate from rounding
                if (tokens_hit > tokens_queried) {
                    tokens_hit = tokens_queried;
                }
            } else {
                if (!get_status.ok()) {
                    fprintf(stderr,
                            "[thread %d] GetCacheLocation gRPC error: %s\n",
                            thread_id,
                            get_status.error_message().c_str());
                } else {
                    fprintf(stderr,
                            "[thread %d] GetCacheLocation error: code=%d msg=%s\n",
                            thread_id,
                            static_cast<int>(get_resp.header().status().code()),
                            get_resp.header().status().message().c_str());
                }
                metrics.per_round[round].fail_count++;
                continue;
            }

            // ── Step 2: StartWriteCache ──
            proto::StartWriteCacheRequest start_req;
            start_req.set_trace_id(trace_id);
            start_req.set_instance_id(cfg.instance_id);
            for (auto t : cs.token_ids)
                start_req.add_token_ids(t);
            start_req.set_write_timeout_seconds(60);

            proto::StartWriteCacheResponse start_resp;
            grpc::ClientContext start_ctx;
            start_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

            auto sw_start = SteadyClock::now();
            grpc::Status start_status = stub->StartWriteCache(&start_ctx, start_req, &start_resp);
            double sw_ms = ElapsedMs(sw_start);

            bool start_ok = start_status.ok() && start_resp.header().status().code() == proto::OK;

            if (!start_ok) {
                if (!start_status.ok()) {
                    fprintf(stderr,
                            "[thread %d] StartWriteCache gRPC error: %s\n",
                            thread_id,
                            start_status.error_message().c_str());
                } else {
                    fprintf(stderr,
                            "[thread %d] StartWriteCache error: code=%d msg=%s\n",
                            thread_id,
                            static_cast<int>(start_resp.header().status().code()),
                            start_resp.header().status().message().c_str());
                }
                metrics.per_round[round].fail_count++;
                continue;
            }

            // ── Step 3: FinishWriteCache ──
            proto::FinishWriteCacheRequest finish_req;
            finish_req.set_trace_id(trace_id);
            finish_req.set_instance_id(cfg.instance_id);
            finish_req.set_write_session_id(start_resp.write_session_id());
            // Mark all returned locations as successfully written
            int success_count_blocks = start_resp.locations_size();
            finish_req.mutable_success_blocks()->set_offset(success_count_blocks);

            proto::CommonResponse finish_resp;
            grpc::ClientContext finish_ctx;
            finish_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

            auto fw_start = SteadyClock::now();
            grpc::Status finish_status = stub->FinishWriteCache(&finish_ctx, finish_req, &finish_resp);
            double fw_ms = ElapsedMs(fw_start);
            double round_trip_ms = ElapsedMs(round_start);

            bool finish_ok = finish_status.ok() && finish_resp.header().status().code() == proto::OK;

            if (!finish_ok) {
                if (!finish_status.ok()) {
                    fprintf(stderr,
                            "[thread %d] FinishWriteCache gRPC error: %s\n",
                            thread_id,
                            finish_status.error_message().c_str());
                } else {
                    fprintf(stderr,
                            "[thread %d] FinishWriteCache error: code=%d msg=%s\n",
                            thread_id,
                            static_cast<int>(finish_resp.header().status().code()),
                            finish_resp.header().status().message().c_str());
                }
                metrics.per_round[round].fail_count++;
                continue;
            }

            // ── Record metrics ──
            auto &rm = metrics.per_round[round];
            rm.total_tokens_queried += tokens_queried;
            rm.total_tokens_hit += tokens_hit;
            rm.get_location_latencies_ms.push_back(gl_ms);
            rm.start_write_latencies_ms.push_back(sw_ms);
            rm.finish_write_latencies_ms.push_back(fw_ms);
            rm.round_trip_latencies_ms.push_back(round_trip_ms);
            rm.success_count++;

            // ── Extend history for next round ──
            // Append simulated "output" tokens (model-generated, output_rng)
            // then "sub-question" tokens (user's next-turn input, input rng)
            if (round < cfg.num_rounds - 1) {
                for (int j = 0; j < cfg.output_tokens; ++j) {
                    cs.token_ids.push_back(output_token_dist(output_rng));
                }
                for (int j = 0; j < actual_sub_q_tokens; ++j) {
                    cs.token_ids.push_back(token_dist(rng));
                }
            }

            cs.current_round++;
        }

        // Round barrier: wait for all threads to finish this round before proceeding
        if (barrier) {
            barrier->Wait();
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
            "  -c <num_clients>        Number of simulated clients (default: 256)\n"
            "  -t <threads>            Number of worker threads (default: 4)\n"
            "  -k <tokens_per_request> Token IDs for initial request (default: 8)\n"
            "  -o <output_tokens>      Simulated output token IDs per round (default: 2)\n"
            "  -K <sub_question_tokens> Sub-question token IDs per round (default: same as -k)\n"
            "  -n <num_rounds>         Number of rounds per client (default: 5)\n"
            "  -R <request_rate>       Target requests per second (default: 1.0)\n"
            "  -D <distribution>       Interval distribution: poisson|uniform (default: poisson)\n"
            "  -b <block_size>         Tokens per block (default: auto-detect via GetInstanceInfo)\n"
            "  -B                      Enable round barrier (sync all clients between rounds)\n"
            "  -s <seed>               Random seed for input tokens (default: current timestamp)\n"
            "  -S <output_seed>        Random seed for output tokens (default: random each run)\n"
            "  -h                      Show this help message\n",
            prog);
}

// ─── main ───

int main(int argc, char *argv[]) {
    HitRateConfig cfg;
    cfg.num_clients = 256;
    cfg.threads = 4;
    cfg.tokens_per_request = 8;
    cfg.output_tokens = 2;
    cfg.sub_question_tokens = 0; // 0 means use tokens_per_request
    cfg.num_rounds = 5;
    cfg.request_rate = 1.0;
    cfg.distribution = "poisson";
    cfg.enable_round_barrier = false;
    cfg.block_size = 0; // 0 means auto-detect
    cfg.seed = std::chrono::steady_clock::now().time_since_epoch().count();
    cfg.output_seed = std::random_device{}();

    int opt;
    while ((opt = getopt(argc, argv, "u:i:c:t:k:o:K:n:R:D:b:Bs:S:h")) != -1) {
        switch (opt) {
        case 'u':
            cfg.uri = optarg;
            break;
        case 'i':
            cfg.instance_id = optarg;
            break;
        case 'c':
            cfg.num_clients = atoi(optarg);
            break;
        case 't':
            cfg.threads = atoi(optarg);
            break;
        case 'k':
            cfg.tokens_per_request = atoi(optarg);
            break;
        case 'o':
            cfg.output_tokens = atoi(optarg);
            break;
        case 'K':
            cfg.sub_question_tokens = atoi(optarg);
            break;
        case 'n':
            cfg.num_rounds = atoi(optarg);
            break;
        case 'R':
            cfg.request_rate = atof(optarg);
            break;
        case 'D':
            cfg.distribution = optarg;
            break;
        case 'b':
            cfg.block_size = atoi(optarg);
            break;
        case 'B':
            cfg.enable_round_barrier = true;
            break;
        case 's':
            cfg.seed = atoll(optarg);
            break;
        case 'S':
            cfg.output_seed = atoll(optarg);
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
    if (cfg.num_clients <= 0 || cfg.threads <= 0 || cfg.tokens_per_request <= 0 || cfg.num_rounds <= 0) {
        fprintf(stderr,
                "Error: num_clients, threads, tokens_per_request, "
                "num_rounds must be > 0\n");
        return 1;
    }
    if (cfg.distribution != "poisson" && cfg.distribution != "uniform") {
        fprintf(stderr, "Error: distribution must be 'poisson' or 'uniform'\n");
        return 1;
    }

    // Cap threads to num_clients
    if (cfg.threads > cfg.num_clients) {
        cfg.threads = cfg.num_clients;
    }

    // Auto-detect block_size if not manually specified
    if (cfg.block_size <= 0) {
        fprintf(stdout, "  Auto-detecting block_size via GetInstanceInfo...\n");
        cfg.block_size = FetchBlockSize(cfg.uri, cfg.instance_id);
        if (cfg.block_size <= 0) {
            fprintf(stderr,
                    "Error: failed to auto-detect block_size. "
                    "Please specify it manually with -b <block_size>\n");
            return 1;
        }
    }

    int actual_sub_q_tokens = (cfg.sub_question_tokens > 0) ? cfg.sub_question_tokens : cfg.tokens_per_request;

    // Register signal handler for graceful shutdown
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    fprintf(stdout,
            "=== MetaService Hit-Rate Benchmark ===\n"
            "  uri:                  %s\n"
            "  instance_id:          %s\n"
            "  num_clients:          %d\n"
            "  threads:              %d\n"
            "  tokens_per_request:   %d\n"
            "  output_tokens:        %d\n"
            "  sub_question_tokens:  %d\n"
            "  num_rounds:           %d\n"
            "  request_rate:         %.1f\n"
            "  distribution:         %s\n"
            "  block_size:           %d\n"
            "  round_barrier:        %s\n"
            "  seed:                 %ld\n"
            "  output_seed:          %ld\n\n",
            cfg.uri.c_str(),
            cfg.instance_id.c_str(),
            cfg.num_clients,
            cfg.threads,
            cfg.tokens_per_request,
            cfg.output_tokens,
            actual_sub_q_tokens,
            cfg.num_rounds,
            cfg.request_rate,
            cfg.distribution.c_str(),
            cfg.block_size,
            cfg.enable_round_barrier ? "enabled" : "disabled",
            cfg.seed,
            cfg.output_seed);
    fflush(stdout);

    // Distribute clients evenly across threads (round-robin)
    std::vector<std::vector<int>> thread_client_ids(cfg.threads);
    for (int c = 0; c < cfg.num_clients; ++c) {
        thread_client_ids[c % cfg.threads].push_back(c);
    }

    // Optional barrier for round synchronization
    std::unique_ptr<Barrier> barrier;
    if (cfg.enable_round_barrier) {
        barrier = std::make_unique<Barrier>(cfg.threads);
    }

    // Launch worker threads
    std::vector<ThreadMetrics> all_metrics(cfg.threads);
    std::vector<std::thread> workers;
    workers.reserve(cfg.threads);

    auto overall_start = SteadyClock::now();

    for (int i = 0; i < cfg.threads; ++i) {
        workers.emplace_back(
            HitRateWorker, i, std::cref(cfg), std::ref(all_metrics[i]), std::cref(thread_client_ids[i]), barrier.get());
    }

    // Wait for all workers to complete
    for (auto &w : workers) {
        w.join();
    }

    double total_duration_s = std::chrono::duration<double>(SteadyClock::now() - overall_start).count();

    // ── Aggregate per-round metrics across all threads ──

    struct AggRound {
        int64_t total_tokens_queried = 0;
        int64_t total_tokens_hit = 0;
        std::vector<double> gl_lats;
        std::vector<double> sw_lats;
        std::vector<double> fw_lats;
        std::vector<double> rt_lats;
        int64_t success_count = 0;
        int64_t fail_count = 0;
    };

    std::vector<AggRound> agg_rounds(cfg.num_rounds);

    for (auto &tm : all_metrics) {
        for (int r = 0; r < cfg.num_rounds && r < static_cast<int>(tm.per_round.size()); ++r) {
            auto &rm = tm.per_round[r];
            auto &ar = agg_rounds[r];
            ar.total_tokens_queried += rm.total_tokens_queried;
            ar.total_tokens_hit += rm.total_tokens_hit;
            ar.gl_lats.insert(
                ar.gl_lats.end(), rm.get_location_latencies_ms.begin(), rm.get_location_latencies_ms.end());
            ar.sw_lats.insert(ar.sw_lats.end(), rm.start_write_latencies_ms.begin(), rm.start_write_latencies_ms.end());
            ar.fw_lats.insert(
                ar.fw_lats.end(), rm.finish_write_latencies_ms.begin(), rm.finish_write_latencies_ms.end());
            ar.rt_lats.insert(ar.rt_lats.end(), rm.round_trip_latencies_ms.begin(), rm.round_trip_latencies_ms.end());
            ar.success_count += rm.success_count;
            ar.fail_count += rm.fail_count;
        }
    }

    // ── Print results ──

    fprintf(stdout, "=== Results (total %.2fs) ===\n\n", total_duration_s);

    // Overall aggregates
    int64_t grand_tokens_queried = 0, grand_tokens_hit = 0;
    int64_t grand_success = 0, grand_fail = 0;
    std::vector<double> all_gl, all_sw, all_fw, all_rt;

    for (int r = 0; r < cfg.num_rounds; ++r) {
        auto &ar = agg_rounds[r];
        grand_tokens_queried += ar.total_tokens_queried;
        grand_tokens_hit += ar.total_tokens_hit;
        grand_success += ar.success_count;
        grand_fail += ar.fail_count;
        all_gl.insert(all_gl.end(), ar.gl_lats.begin(), ar.gl_lats.end());
        all_sw.insert(all_sw.end(), ar.sw_lats.begin(), ar.sw_lats.end());
        all_fw.insert(all_fw.end(), ar.fw_lats.begin(), ar.fw_lats.end());
        all_rt.insert(all_rt.end(), ar.rt_lats.begin(), ar.rt_lats.end());
    }

    double overall_hit_rate =
        (grand_tokens_queried > 0) ? static_cast<double>(grand_tokens_hit) / grand_tokens_queried : 0.0;

    fprintf(stdout,
            "  Overall:\n"
            "    Total requests:      %ld (success: %ld, fail: %ld)\n"
            "    Total tokens queried: %ld\n"
            "    Total tokens hit:    %ld\n"
            "    Cache hit rate:      %.6f\n"
            "    Throughput:          %.1f requests/s\n",
            grand_success + grand_fail,
            grand_success,
            grand_fail,
            grand_tokens_queried,
            grand_tokens_hit,
            overall_hit_rate,
            static_cast<double>(grand_success + grand_fail) / total_duration_s);

    // Overall latency stats
    auto gl_stats = ComputeStats(all_gl);
    auto sw_stats = ComputeStats(all_sw);
    auto fw_stats = ComputeStats(all_fw);
    auto rt_stats = ComputeStats(all_rt);

    if (!all_rt.empty()) {
        fprintf(stdout,
                "\n  Round-trip Latency (ms):  [GetCacheLocation + StartWrite + FinishWrite]\n"
                "    avg      p50      p99      p999     max\n"
                "    %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n",
                rt_stats.avg_ms,
                rt_stats.p50_ms,
                rt_stats.p99_ms,
                rt_stats.p999_ms,
                rt_stats.max_ms);
    }
    if (!all_gl.empty()) {
        fprintf(stdout,
                "\n  GetCacheLocation Latency (ms):\n"
                "    avg      p50      p99      p999     max\n"
                "    %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n",
                gl_stats.avg_ms,
                gl_stats.p50_ms,
                gl_stats.p99_ms,
                gl_stats.p999_ms,
                gl_stats.max_ms);
    }
    if (!all_sw.empty()) {
        fprintf(stdout,
                "\n  StartWriteCache Latency (ms):\n"
                "    avg      p50      p99      p999     max\n"
                "    %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n",
                sw_stats.avg_ms,
                sw_stats.p50_ms,
                sw_stats.p99_ms,
                sw_stats.p999_ms,
                sw_stats.max_ms);
    }
    if (!all_fw.empty()) {
        fprintf(stdout,
                "\n  FinishWriteCache Latency (ms):\n"
                "    avg      p50      p99      p999     max\n"
                "    %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n",
                fw_stats.avg_ms,
                fw_stats.p50_ms,
                fw_stats.p99_ms,
                fw_stats.p999_ms,
                fw_stats.max_ms);
    }

    // Per-round summary
    fprintf(stdout, "\n  Per-round metrics:\n");
    for (int r = 0; r < cfg.num_rounds; ++r) {
        auto &ar = agg_rounds[r];
        double hit_rate =
            (ar.total_tokens_queried > 0) ? static_cast<double>(ar.total_tokens_hit) / ar.total_tokens_queried : 0.0;
        auto round_gl = ComputeStats(ar.gl_lats);
        auto round_rt = ComputeStats(ar.rt_lats);

        fprintf(stdout,
                "    Round %d: hit_rate=%.6f  tokens_queried=%ld  tokens_hit=%ld  "
                "success=%ld  fail=%ld  avg_gl=%.3fms  avg_rt=%.3fms\n",
                r,
                hit_rate,
                ar.total_tokens_queried,
                ar.total_tokens_hit,
                ar.success_count,
                ar.fail_count,
                round_gl.avg_ms,
                round_rt.avg_ms);
    }

    fprintf(stdout, "\n");
    return 0;
}
