#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/tools/v6d_benchmark/config.h"

namespace kmonitor {
class KMonitor;
class MutableMetric;
} // namespace kmonitor

namespace kv_cache_manager {
namespace v6d_benchmark {

struct BenchmarkMetrics {
    // 实时采集指标
    std::atomic<int64_t> total_requests{0};
    std::atomic<int64_t> success_requests{0};
    std::atomic<int64_t> failed_requests{0};
    std::atomic<int64_t> total_latency_us{0};
    std::atomic<int64_t> min_latency_us{INT64_MAX};
    std::atomic<int64_t> max_latency_us{0};
    std::atomic<int64_t> total_bytes{0};

    // 延迟样本(用于计算百分位)
    std::vector<int64_t> latency_samples;
    std::shared_mutex latency_mutex;

    // 分操作统计
    struct OpMetrics {
        std::atomic<int64_t> count{0};
        std::atomic<int64_t> success{0};
        std::atomic<int64_t> failed{0};
        std::atomic<int64_t> total_latency_us{0};
    };
    OpMetrics add_block_metrics;
    OpMetrics delete_block_metrics;
    OpMetrics query_metrics;

    // 验证统计
    std::atomic<int64_t> verification_passed{0};
    std::atomic<int64_t> verification_failed{0};
};

class BenchmarkMetricsReporter {
public:
    BenchmarkMetricsReporter(const BenchmarkConfig &config);
    ~BenchmarkMetricsReporter();

    bool Init();
    void Start();
    void Stop();

    // 线程安全的指标采集接口
    void RecordRequest(const std::string &op_type, int64_t latency_us, int64_t bytes, bool success);

    // QPS限流器接口
    void AcquireQPSPermit();

    // 获取当前QPS
    double GetCurrentQPS() const;

    // 获取指标快照（返回const引用，避免拷贝atomic）
    const BenchmarkMetrics &GetSnapshot() const;

    // 更新验证统计
    void RecordVerification(bool success);

private:
    void ReportLoop();
    void ReportToKmonitor(const BenchmarkMetrics &snapshot, double current_qps);
    void CalculatePercentiles(const std::vector<int64_t> &samples, double &p50, double &p99, double &p999);

    BenchmarkConfig config_;
    BenchmarkMetrics metrics_;
    std::thread report_thread_;
    std::atomic<bool> running_{false};

    // QPS限流器
    mutable std::mutex qps_mutex_;
    std::chrono::steady_clock::time_point window_start_;
    int64_t current_window_count_ = 0;

    // Kmonitor相关
    std::unique_ptr<kmonitor::KMonitor> kmonitor_;

    // Metrics指针
    kmonitor::MutableMetric *qps_metrics = nullptr;
    kmonitor::MutableMetric *avg_latency_metrics = nullptr;
    kmonitor::MutableMetric *p50_latency_metrics = nullptr;
    kmonitor::MutableMetric *p99_latency_metrics = nullptr;
    kmonitor::MutableMetric *p999_latency_metrics = nullptr;
    kmonitor::MutableMetric *success_rate_metrics = nullptr;
    kmonitor::MutableMetric *bandwidth_metrics = nullptr;
    kmonitor::MutableMetric *add_block_qps_metrics = nullptr;
    kmonitor::MutableMetric *query_qps_metrics = nullptr;
    kmonitor::MutableMetric *delete_block_qps_metrics = nullptr;
    kmonitor::MutableMetric *verification_passed_metrics = nullptr;
    kmonitor::MutableMetric *verification_failed_metrics = nullptr;
    kmonitor::MutableMetric *target_qps_metrics = nullptr;
};

} // namespace v6d_benchmark
} // namespace kv_cache_manager
