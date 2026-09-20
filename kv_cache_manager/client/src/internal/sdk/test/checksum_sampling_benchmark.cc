#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"

namespace kv_cache_manager {
namespace {

// This is a manual benchmark, not a correctness test. It exercises the same
// preallocated/pinned-memory GetBlocksHash path used by TransferClient. Keep
// the result schema stable so two runs can be diffed by a simple line-oriented
// tool. See PrintUsage for the canonical invocation.
struct Options {
    size_t block_count = 64;
    size_t iovs_per_block = 8;
    size_t iov_bytes = 1024 * 1024;
    size_t iterations = 100;
    size_t warmup_iterations = 10;
    bool include_full = false;
    size_t full_iterations = 3;
    size_t full_warmup_iterations = 1;
    double max_sample_p95_us = 0.0;
    double max_w128_over_w4 = 0.0;
};

struct Result {
    std::string label;
    size_t sample_window = 0;
    size_t sampled_bytes = 0;
    std::vector<double> latency_us;
    double average_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
};

volatile uint64_t benchmark_sink = 0;

bool CheckedMultiply(size_t lhs, size_t rhs, size_t &out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool ParseSize(std::string_view text, size_t &out) {
    if (text.empty()) {
        return false;
    }
    size_t parsed = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc() || end != text.data() + text.size()) {
        return false;
    }
    out = parsed;
    return true;
}

bool ParsePositiveDouble(std::string_view text, double &out) {
    if (text.empty()) {
        return false;
    }
    std::string copy(text);
    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(copy.c_str(), &end);
    if (errno != 0 || end != copy.c_str() + copy.size() || !std::isfinite(parsed) || parsed <= 0.0) {
        return false;
    }
    out = parsed;
    return true;
}

void PrintUsage(std::string_view binary) {
    std::cout << "Usage: " << binary << " [options]\n\n"
              << "Measures the production GPU head+tail CRC path at W=4/64/128.\n"
              << "Defaults model 64 x 8 MiB KV blocks split into eight 1 MiB IOVs\n"
              << "(512 MiB logical/device data per batch).\n\n"
              << "  --blocks=N                    block count (default 64)\n"
              << "  --iovs_per_block=N            IOVs in each block (default 8)\n"
              << "  --iov_bytes=N                 bytes in each IOV (default 1048576)\n"
              << "  --iterations=N                samples per head+tail window (default 100)\n"
              << "  --warmup_iterations=N         warmups per head+tail window (default 10)\n"
              << "  --include_full                add a full-IOV CRC reference\n"
              << "  --full_iterations=N           full-reference samples (default 3)\n"
              << "  --full_warmup_iterations=N    full-reference warmups (default 1)\n"
              << "  --max_sample_p95_us=F         fail if any sampled case exceeds F\n"
              << "  --max_w128_over_w4=F          fail if avg(W=128)/avg(W=4) exceeds F\n"
              << "  --help                        show this help\n\n"
              << "Run from github-opensource/:\n"
              << "  bazelisk run -c opt --config=cuda \\\n"
              << "    //kv_cache_manager/client/src/internal/sdk/test:checksum_sampling_benchmark\n"
              << "For MUSA, replace --config=cuda with --config=musa. The optional thresholds\n"
              << "turn the same binary into a hardware-specific performance regression gate.\n";
}

enum class ParseResult {
    OK,
    HELP,
    ERROR,
};

ParseResult ParseOptions(int argc, char **argv, Options &options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help") {
            PrintUsage(argv[0]);
            return ParseResult::HELP;
        }
        if (argument == "--include_full") {
            options.include_full = true;
            continue;
        }

        const auto parse_size_option = [&](std::string_view name, size_t &destination) {
            if (argument.size() < name.size() || argument.compare(0, name.size(), name) != 0) {
                return false;
            }
            if (!ParseSize(argument.substr(name.size()), destination)) {
                throw std::invalid_argument("invalid unsigned integer for " + std::string(name));
            }
            return true;
        };
        const auto parse_double_option = [&](std::string_view name, double &destination) {
            if (argument.size() < name.size() || argument.compare(0, name.size(), name) != 0) {
                return false;
            }
            if (!ParsePositiveDouble(argument.substr(name.size()), destination)) {
                throw std::invalid_argument("invalid positive number for " + std::string(name));
            }
            return true;
        };

        try {
            if (parse_size_option("--blocks=", options.block_count) ||
                parse_size_option("--iovs_per_block=", options.iovs_per_block) ||
                parse_size_option("--iov_bytes=", options.iov_bytes) ||
                parse_size_option("--iterations=", options.iterations) ||
                parse_size_option("--warmup_iterations=", options.warmup_iterations) ||
                parse_size_option("--full_iterations=", options.full_iterations) ||
                parse_size_option("--full_warmup_iterations=", options.full_warmup_iterations) ||
                parse_double_option("--max_sample_p95_us=", options.max_sample_p95_us) ||
                parse_double_option("--max_w128_over_w4=", options.max_w128_over_w4)) {
                continue;
            }
        } catch (const std::invalid_argument &error) {
            std::cerr << error.what() << ": " << argument << '\n';
            return ParseResult::ERROR;
        }
        std::cerr << "unknown option: " << argument << '\n';
        return ParseResult::ERROR;
    }
    return ParseResult::OK;
}

#if defined(USING_CUDA)
using DeviceBufferGuard = CudaBufferGuard;

bool FillDeviceBuffer(void *data, size_t bytes) {
    const cudaError_t memset_error = cudaMemset(data, 0xA5, bytes);
    if (memset_error != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(memset_error) << '\n';
        return false;
    }
    const cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(sync_error) << '\n';
        return false;
    }
    return true;
}

constexpr std::string_view kGpuBackend = "cuda";

std::string GetGpuName() {
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return "unknown";
    }
    return properties.name;
}
#elif defined(USING_MUSA)
using DeviceBufferGuard = MusaBufferGuard;

bool FillDeviceBuffer(void *data, size_t bytes) {
    const musaError_t memset_error = musaMemset(data, 0xA5, bytes);
    if (memset_error != musaSuccess) {
        std::cerr << "musaMemset failed: " << musaGetErrorString(memset_error) << '\n';
        return false;
    }
    const musaError_t sync_error = musaDeviceSynchronize();
    if (sync_error != musaSuccess) {
        std::cerr << "musaDeviceSynchronize failed: " << musaGetErrorString(sync_error) << '\n';
        return false;
    }
    return true;
}

constexpr std::string_view kGpuBackend = "musa";

std::string GetGpuName() {
    int device = 0;
    musaDeviceProp properties{};
    if (musaGetDevice(&device) != musaSuccess || musaGetDeviceProperties(&properties, device) != musaSuccess) {
        return "unknown";
    }
    return properties.name;
}
#else
#error "checksum_sampling_benchmark requires --config=cuda or --config=musa"
#endif

class SampleWindowGuard {
public:
    SampleWindowGuard() : saved_(SdkBufferCheckUtil::min_cal_byte_size_) {}
    ~SampleWindowGuard() { SdkBufferCheckUtil::min_cal_byte_size_ = saved_; }

private:
    size_t saved_;
};

std::vector<int64_t> CalculateChecksums(const BlockBuffers &blocks, SdkBufferCheckPool &pool, size_t total_iov_count) {
    auto cell = pool.GetCell();
    return SdkBufferCheckUtil::GetBlocksHash(
        blocks, cell->d_iovs, cell->d_crcs, cell->h_iovs, total_iov_count, cell->gpu_stream);
}

bool RunOnce(const BlockBuffers &blocks, SdkBufferCheckPool &pool, size_t total_iov_count, double *latency_us) {
    const auto begin = std::chrono::steady_clock::now();
    const auto checksums = CalculateChecksums(blocks, pool, total_iov_count);
    const auto end = std::chrono::steady_clock::now();
    if (checksums.size() != blocks.size()) {
        std::cerr << "checksum computation returned " << checksums.size() << " blocks, expected " << blocks.size()
                  << '\n';
        return false;
    }
    uint64_t folded = 0;
    for (const int64_t checksum : checksums) {
        folded ^= static_cast<uint64_t>(checksum) + 0x9E3779B97F4A7C15ULL + (folded << 6U) + (folded >> 2U);
    }
    benchmark_sink ^= folded;
    if (latency_us != nullptr) {
        *latency_us = std::chrono::duration<double, std::micro>(end - begin).count();
    }
    return true;
}

double Percentile(const std::vector<double> &sorted, double fraction) {
    const size_t rank = static_cast<size_t>(std::ceil(fraction * static_cast<double>(sorted.size())));
    return sorted[std::max<size_t>(1, rank) - 1];
}

void FinalizeResult(Result &result) {
    std::sort(result.latency_us.begin(), result.latency_us.end());
    result.average_us = std::accumulate(result.latency_us.begin(), result.latency_us.end(), 0.0) /
                        static_cast<double>(result.latency_us.size());
    result.p50_us = Percentile(result.latency_us, 0.50);
    result.p95_us = Percentile(result.latency_us, 0.95);
}

bool WarmUp(const BlockBuffers &blocks,
            SdkBufferCheckPool &pool,
            size_t total_iov_count,
            size_t sample_window,
            size_t iterations) {
    SdkBufferCheckUtil::min_cal_byte_size_ = sample_window;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        if (!RunOnce(blocks, pool, total_iov_count, nullptr)) {
            return false;
        }
    }
    return true;
}

bool MeasureHeadTailCases(const Options &options,
                          const BlockBuffers &blocks,
                          SdkBufferCheckPool &pool,
                          size_t total_iov_count,
                          std::vector<Result> &results) {
    constexpr std::array<size_t, 3> kSampleWindows = {4, 64, 128};
    results.clear();
    results.reserve(kSampleWindows.size());
    for (const size_t sample_window : kSampleWindows) {
        if (!WarmUp(blocks, pool, total_iov_count, sample_window, options.warmup_iterations)) {
            return false;
        }
        results.push_back({"head_tail", sample_window, 2 * sample_window * total_iov_count, {}});
        results.back().latency_us.reserve(options.iterations);
    }

    // Rotate the first case so temperature/clock drift does not systematically
    // favor W=4. Changing the host-side window happens outside the timer.
    for (size_t iteration = 0; iteration < options.iterations; ++iteration) {
        for (size_t offset = 0; offset < results.size(); ++offset) {
            Result &result = results[(iteration + offset) % results.size()];
            SdkBufferCheckUtil::min_cal_byte_size_ = result.sample_window;
            double latency_us = 0.0;
            if (!RunOnce(blocks, pool, total_iov_count, &latency_us)) {
                return false;
            }
            result.latency_us.push_back(latency_us);
        }
    }
    for (auto &result : results) {
        FinalizeResult(result);
    }
    return true;
}

bool MeasureFullReference(const Options &options,
                          const BlockBuffers &blocks,
                          SdkBufferCheckPool &pool,
                          size_t total_iov_count,
                          Result &result) {
    // This deliberately bypasses the production KVCM_CHECKSUM_MAX_SAMPLE_BYTES
    // configuration cap. It is a cost reference for the same legacy kernel,
    // not a deployable sample-window recommendation.
    const size_t sample_window = options.iov_bytes / 2;
    if (!WarmUp(blocks, pool, total_iov_count, sample_window, options.full_warmup_iterations)) {
        return false;
    }
    result = {"full_iov_reference", sample_window, (options.iov_bytes / 2) * 2 * total_iov_count, {}};
    result.latency_us.reserve(options.full_iterations);
    for (size_t iteration = 0; iteration < options.full_iterations; ++iteration) {
        double latency_us = 0.0;
        if (!RunOnce(blocks, pool, total_iov_count, &latency_us)) {
            return false;
        }
        result.latency_us.push_back(latency_us);
    }
    FinalizeResult(result);
    return true;
}

void PrintResult(const Result &result, size_t logical_bytes, size_t block_count, double w4_average_us) {
    const double seconds = result.average_us / 1'000'000.0;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const double effective_logical_gib_per_second = static_cast<double>(logical_bytes) / gib / seconds;
    const double sampled_gib_per_second = static_cast<double>(result.sampled_bytes) / gib / seconds;
    const double relative_to_w4 = w4_average_us == 0.0 ? 0.0 : result.average_us / w4_average_us;
    std::cout << "CHECKSUM_BENCH_RESULT"
              << ",label=" << result.label << ",window_bytes=" << result.sample_window
              << ",sampled_bytes=" << result.sampled_bytes << ",samples=" << result.latency_us.size()
              << ",avg_us=" << result.average_us << ",p50_us=" << result.p50_us << ",p95_us=" << result.p95_us
              << ",ns_per_block=" << result.average_us * 1000.0 / static_cast<double>(block_count)
              << ",effective_logical_gib_per_s=" << effective_logical_gib_per_second
              << ",sampled_gib_per_s=" << sampled_gib_per_second << ",relative_to_w4=" << relative_to_w4 << '\n';
}

int RunBenchmark(const Options &options) {
    if (options.block_count == 0 || options.iovs_per_block == 0 || options.iov_bytes < 2 || options.iterations == 0 ||
        (options.include_full && options.full_iterations == 0)) {
        std::cerr
            << "blocks, iovs_per_block, iterations and full_iterations must be positive; iov_bytes must be >= 2\n";
        return 1;
    }
    if (options.iov_bytes / 2 < 128) {
        std::cerr << "iov_bytes must be at least 256 so W=128 has an untruncated head+tail window\n";
        return 1;
    }

    size_t total_iov_count = 0;
    size_t logical_bytes = 0;
    if (!CheckedMultiply(options.block_count, options.iovs_per_block, total_iov_count) ||
        total_iov_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        !CheckedMultiply(total_iov_count, options.iov_bytes, logical_bytes)) {
        std::cerr << "benchmark dimensions overflow the checksum kernel limits\n";
        return 1;
    }

    SampleWindowGuard restore_sample_window;
    SdkBufferCheckUtil::min_cal_byte_size_ = 4;
    DeviceBufferGuard device_data;
    if (!device_data.Alloc(logical_bytes) || !FillDeviceBuffer(device_data.Get(), logical_bytes)) {
        return 1;
    }

    BlockBuffers blocks(options.block_count);
    auto *base = static_cast<std::byte *>(device_data.Get());
    for (size_t block_index = 0; block_index < options.block_count; ++block_index) {
        auto &iovs = blocks[block_index].iovs;
        iovs.reserve(options.iovs_per_block);
        for (size_t iov_index = 0; iov_index < options.iovs_per_block; ++iov_index) {
            const size_t flat_index = block_index * options.iovs_per_block + iov_index;
            iovs.push_back({MemoryType::GPU, base + flat_index * options.iov_bytes, options.iov_bytes, false});
        }
    }

    SdkBufferCheckPool pool(1);
    if (!pool.Init(total_iov_count)) {
        std::cerr << "checksum pool initialization failed\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::string gpu_name = GetGpuName();
    std::replace(gpu_name.begin(), gpu_name.end(), ',', '_');
    std::cout << "CHECKSUM_BENCH_CONFIG"
              << ",backend=" << kGpuBackend << ",device=" << gpu_name << ",blocks=" << options.block_count
              << ",iovs_per_block=" << options.iovs_per_block << ",total_iovs=" << total_iov_count
              << ",iov_bytes=" << options.iov_bytes << ",logical_bytes=" << logical_bytes
              << ",iterations=" << options.iterations << ",warmup_iterations=" << options.warmup_iterations
              << ",algorithm=CRC32_XOR_INT64_LEGACY_V0\n";

    std::vector<Result> results;
    if (!MeasureHeadTailCases(options, blocks, pool, total_iov_count, results)) {
        return 1;
    }
    const double w4_average_us = results.front().average_us;
    for (const auto &result : results) {
        PrintResult(result, logical_bytes, options.block_count, w4_average_us);
    }

    if (options.include_full) {
        Result full_result;
        if (!MeasureFullReference(options, blocks, pool, total_iov_count, full_result)) {
            return 1;
        }
        PrintResult(full_result, logical_bytes, options.block_count, w4_average_us);
        if (options.iov_bytes % 2 != 0) {
            std::cout << "CHECKSUM_BENCH_NOTE,full_reference_skips_center_byte=1\n";
        }
    }

    bool regression = false;
    if (options.max_sample_p95_us > 0.0) {
        for (const auto &result : results) {
            if (result.p95_us > options.max_sample_p95_us) {
                std::cerr << "CHECKSUM_BENCH_REGRESSION,reason=p95_limit,window_bytes=" << result.sample_window
                          << ",actual_us=" << result.p95_us << ",limit_us=" << options.max_sample_p95_us << '\n';
                regression = true;
            }
        }
    }
    const double w128_over_w4 = results.back().average_us / w4_average_us;
    if (options.max_w128_over_w4 > 0.0 && w128_over_w4 > options.max_w128_over_w4) {
        std::cerr << "CHECKSUM_BENCH_REGRESSION,reason=w128_over_w4,actual=" << w128_over_w4
                  << ",limit=" << options.max_w128_over_w4 << '\n';
        regression = true;
    }
    std::cout << "CHECKSUM_BENCH_DONE,sink=" << benchmark_sink << ",regression=" << (regression ? 1 : 0) << '\n';
    return regression ? 2 : 0;
}

} // namespace
} // namespace kv_cache_manager

int main(int argc, char **argv) {
    kv_cache_manager::Options options;
    const kv_cache_manager::ParseResult parse_result = kv_cache_manager::ParseOptions(argc, argv, options);
    if (parse_result == kv_cache_manager::ParseResult::HELP) {
        return 0;
    }
    if (parse_result == kv_cache_manager::ParseResult::ERROR) {
        return 1;
    }
    return kv_cache_manager::RunBenchmark(options);
}
