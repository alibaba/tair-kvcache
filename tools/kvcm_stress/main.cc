#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>

#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/common/logger.h"
#include "tools/kvcm_stress/workload.h"

namespace {
using namespace kv_cache_manager;
using namespace kv_cache_manager::stress;
static_assert(std::atomic<bool>::is_always_lock_free, "signal handler requires lock-free atomic");
std::atomic<bool> interrupted{false};
void OnSignal(int) { interrupted.store(true, std::memory_order_relaxed); }

struct RunOptions {
    Options work;
    std::string config;
    size_t threads = 32, prefill = 1;
    int seconds = 60, report = 5;
    uint64_t key_base = (uint64_t{std::random_device{}()} << 20) + 1;
};
void Usage() {
    std::cout << "kvcm_stress --config client.json [options]\n"
                 "  --threads 32          concurrent workers (one shared TransferClient)\n"
                 "  --batch 16            blocks per operation\n"
                 "  --seconds 60          measured duration; 0 runs until SIGINT/SIGTERM\n"
                 "  --query-percent 30    fraction of operations doing GetCacheLocation\n"
                 "  --read-percent 100    fraction of queries loading hit blocks\n"
                 "  --pool 4096           recent successful keys retained per worker\n"
                 "  --prefill 1           write batches per worker before measurement\n"
                 "  --write-timeout 60    server write-session TTL in seconds (1..1800)\n"
                 "  --report 5            report interval in seconds\n"
                 "  --key-base N          positive initial key; default random per run\n"
                 "  --verify              fill/check every byte (costs client CPU)\n"
                 "Block bytes/spec, gRPC address/timeouts and SDK threads come from client.json.\n";
}
uint64_t Number(const std::string &s, uint64_t max) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid number: " + s);
    auto n = std::stoull(s);
    if (n > max)
        throw std::runtime_error("number out of range: " + s);
    return n;
}
RunOptions Parse(int argc, char **argv) {
    RunOptions o;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--verify") {
            o.work.verify = true;
            continue;
        }
        if (i + 1 == argc)
            throw std::runtime_error("missing value: " + flag);
        std::string value = argv[++i];
        if (flag == "--config")
            o.config = value;
        else if (flag == "--threads")
            o.threads = Number(value, 4096);
        else if (flag == "--batch")
            o.work.batch = Number(value, 1000000);
        else if (flag == "--seconds")
            o.seconds = Number(value, INT32_MAX);
        else if (flag == "--query-percent")
            o.work.query_percent = Number(value, 100);
        else if (flag == "--read-percent")
            o.work.read_percent = Number(value, 100);
        else if (flag == "--pool")
            o.work.pool = Number(value, 100000000);
        else if (flag == "--prefill")
            o.prefill = Number(value, 1000000);
        else if (flag == "--write-timeout")
            o.work.write_timeout = Number(value, 1800);
        else if (flag == "--report")
            o.report = Number(value, 86400);
        else if (flag == "--key-base")
            o.key_base = Number(value, INT64_MAX);
        else
            throw std::runtime_error("unknown option: " + flag);
    }
    if (o.config.empty() || !o.threads || !o.work.batch || !o.work.pool || !o.work.write_timeout || !o.report ||
        !o.key_base)
        throw std::runtime_error(
            "config is required; threads/batch/pool/write-timeout/report/key-base must be positive");
    if (o.work.query_percent == 100 && o.prefill == 0)
        throw std::runtime_error("100% queries require --prefill > 0");
    return o;
}

void Report(const Stats &s,
            size_t bytes,
            double elapsed,
            double interval,
            std::array<uint64_t, PHASE_COUNT> &previous,
            uint64_t &prev_written,
            uint64_t &prev_read) {
    auto written = s.written.load(), read = s.read.load(), queried = s.queried.load(), hits = s.hits.load();
    std::cout << std::fixed << std::setprecision(2) << "elapsed=" << elapsed
              << "s write_MiB/s=" << (written - prev_written) * double(bytes) / interval / 1048576
              << " read_MiB/s=" << (read - prev_read) * double(bytes) / interval / 1048576
              << " written_blocks=" << written << " read_blocks=" << read << " hit=" << hits << '/' << queried
              << " skipped=" << s.skipped.load() << " corrupt=" << s.corrupt.load() << '\n';
    prev_written = written;
    prev_read = read;
    for (int p = 0; p < PHASE_COUNT; ++p) {
        const auto &m = s.phases[p];
        auto calls = m.calls.load();
        uint64_t total = 0, p99 = 0;
        for (unsigned b = 0; b < 32; ++b) {
            total += m.histogram[b].load();
            if (calls && total >= calls - calls / 100) {
                p99 = uint64_t{1} << b;
                break;
            }
        }
        std::cout << "  " << kPhaseNames[p] << " qps=" << (calls - previous[p]) / interval << " calls=" << calls
                  << " errors=" << m.errors.load() << " last_error=" << m.last_error.load()
                  << " avg_ms=" << (calls ? m.micros.load() / double(calls) / 1000 : 0) << " p99_le_ms=" << p99 / 1000.0
                  << '\n';
        previous[p] = calls;
    }
    std::cout << std::flush;
}
int Run(RunOptions o) {
    std::ifstream file(o.config);
    if (!file)
        throw std::runtime_error("cannot open config: " + o.config);
    const std::string config((std::istreambuf_iterator<char>(file)), {});
    ClientConfig parsed;
    if (!parsed.FromJsonString(config))
        throw std::runtime_error("invalid client config");
    if (parsed.location_spec_infos().size() != 1 || !parsed.location_spec_groups().empty())
        throw std::runtime_error("use exactly one location_spec_info and omit location_spec_groups");
    if (parsed.meta_channel_config().retry_time() != 1)
        throw std::runtime_error("set meta_channel_config.retry_time=1 (no retries, avoids duplicate write sessions)");
    o.work.spec = parsed.location_spec_infos().begin()->first;
    o.work.bytes = parsed.location_spec_infos().begin()->second;
    if (o.work.bytes > std::numeric_limits<size_t>::max() / o.work.batch / o.threads)
        throw std::runtime_error("payload buffer size overflow");
    std::cout << "instance=" << parsed.instance_id() << " workers=" << o.threads << " batch=" << o.work.batch
              << " block_bytes=" << o.work.bytes
              << " payload_MiB=" << o.work.bytes * double(o.work.batch) * o.threads / 1048576
              << " key_base=" << o.key_base << " query_percent=" << o.work.query_percent
              << " read_percent=" << o.work.read_percent << std::endl;
    InitParams params;
    params.role_type = RoleType::SCHEDULER;
    // One channel per worker; create sequentially so instance registration and
    // process-global PACE initialization never race.
    std::vector<std::unique_ptr<MetaClient>> metas;
    for (size_t i = 0; i < o.threads; ++i) {
        auto meta = MetaClient::Create(config, params);
        if (!meta)
            throw std::runtime_error("MetaClient initialization failed");
        metas.push_back(std::move(meta));
    }
    params.role_type = RoleType::WORKER;
    params.self_location_spec_name = o.work.spec;
    params.storage_configs = metas.front()->GetStorageConfig();
    auto transfer = TransferClient::Create(config, params);
    if (!transfer)
        throw std::runtime_error("TransferClient initialization failed; check internal PACE build/config");
    LoggerBroker::SetLogLevel(Logger::LEVEL_WARN);

    Stats stats;
    std::atomic<uint64_t> next_key{o.key_base};
    // Retain prefilled pools and reset counters before measurement threads start.
    std::vector<std::unique_ptr<Worker>> workers;
    for (size_t i = 0; i < o.threads; ++i)
        workers.push_back(std::make_unique<Worker>(*metas[i], *transfer, o.work, stats, next_key, o.key_base + i));
    for (auto &worker : workers) {
        for (size_t i = 0; i < o.prefill && !interrupted; ++i)
            if (!worker->Write())
                throw std::runtime_error("prefill failed; inspect client logs");
        if (o.work.query_percent && o.prefill && !worker->HasKeys())
            throw std::runtime_error("prefill produced no new keys; use a fresh --key-base/instance");
    }
    std::cout << "prefill_blocks=" << stats.written.load() << "; starting measurement" << std::endl;
    // No worker threads exist yet.
    for (auto &m : stats.phases) {
        m.calls = 0;
        m.errors = 0;
        m.micros = 0;
        m.last_error = 0;
        for (auto &h : m.histogram)
            h = 0;
    }
    stats.written = 0;
    stats.skipped = 0;
    std::atomic<bool> stop{false}, crashed{false};
    const auto start = Clock::now();
    const auto deadline = o.seconds ? start + std::chrono::seconds(o.seconds) : Clock::time_point::max();
    std::vector<std::thread> threads;
    try {
        for (auto &worker : workers)
            threads.emplace_back([&, w = worker.get()] {
                try {
                    while (!stop.load(std::memory_order_relaxed) && !interrupted && Clock::now() < deadline) {
                        if (!w->Step())
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                } catch (const std::exception &e) {
                    std::cerr << "worker failed: " << e.what() << std::endl;
                    crashed = true;
                    stop = true;
                }
            });
    } catch (...) {
        stop = true;
        for (auto &t : threads)
            t.join();
        throw;
    }
    auto last = start;
    std::array<uint64_t, PHASE_COUNT> previous{};
    uint64_t prev_written = 0, prev_read = 0;
    while (!interrupted && !stop && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto now = Clock::now();
        if (now - last >= std::chrono::seconds(o.report)) {
            Report(stats,
                   o.work.bytes,
                   std::chrono::duration<double>(now - start).count(),
                   std::chrono::duration<double>(now - last).count(),
                   previous,
                   prev_written,
                   prev_read);
            last = now;
        }
    }
    stop = true;
    for (auto &t : threads)
        t.join(); // Drain in-flight writes, including FinishWrite.
    auto end = Clock::now();
    std::cout << "final (includes in-flight drain):\n";
    previous.fill(0);
    prev_written = prev_read = 0;
    auto elapsed = std::max(0.000001, std::chrono::duration<double>(end - start).count());
    Report(stats, o.work.bytes, elapsed, elapsed, previous, prev_written, prev_read);
    bool failed = crashed;
    for (const auto &m : stats.phases)
        failed = failed || m.errors.load() != 0;
    return failed ? 1 : 0;
}
} // namespace
int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        Usage();
        return 0;
    }
    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);
    try {
        return Run(Parse(argc, argv));
    } catch (const std::exception &e) {
        std::cerr << "kvcm_stress: " << e.what() << '\n';
        return 2;
    }
}
