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
        std::vector<int64_t> latency_samples;
        mutable std::shared_mutex latency_mutex;
    };
    OpMetrics add_block_metrics;
    OpMetrics delete_block_metrics;
    OpMetrics query_metrics;

    // Query 子类型统计（按 API 区分）
    OpMetrics batch_query_metrics;  // GetBatchCacheLocations
    OpMetrics single_query_metrics; // GetCacheLocation

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
    void ReportToKmonitor(const BenchmarkMetrics &snapshot,
                          double elapsed_seconds,
                          double current_qps,
                          double avg_latency,
                          double success_rate,
                          double bandwidth_mbps,
                          double add_qps,
                          double query_qps,
                          double delete_qps,
                          double add_avg_latency,
                          double query_avg_latency,
                          double delete_avg_latency,
                          double delta_verify_pass,
                          double delta_verify_fail,
                          double batch_query_avg_latency,
                          double single_query_avg_latency);
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
    std::string host_ip_; // 本机IP，用于构造kmonitor上报tag

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
    // 上报当前 batch 配置，方便看板/对比时把 RPC qps 换算成 key qps：
    //   add/delete keys per second  = add_block_qps  * batch_size
    //   query keys per second       = query_qps      * query_batch_size
    kmonitor::MutableMetric *batch_size_metrics = nullptr;
    kmonitor::MutableMetric *query_batch_size_metrics = nullptr;
};

} // namespace v6d_benchmark
} // namespace kv_cache_manager
