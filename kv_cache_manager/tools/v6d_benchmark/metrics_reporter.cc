#include "kv_cache_manager/tools/v6d_benchmark/metrics_reporter.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/MetricsReporter.h"

namespace kv_cache_manager {
namespace v6d_benchmark {

BenchmarkMetricsReporter::BenchmarkMetricsReporter(const BenchmarkConfig &config) : config_(config) {
    window_start_ = std::chrono::steady_clock::now();
}

BenchmarkMetricsReporter::~BenchmarkMetricsReporter() { Stop(); }

bool BenchmarkMetricsReporter::Init() {
    if (!config_.enable_kmonitor) {
        KVCM_LOG_INFO("Kmonitor is disabled, skipping initialization");
        return true;
    }

    try {
        // 初始化KMonitor
        kmonitor::KMonitorFactory::Init(config_.kmonitor_config.empty() ? R"({"domain":"v6d_benchmark"})"
                                                                        : config_.kmonitor_config);
        kmonitor::KMonitorFactory::Start();

        kmonitor_.reset(kmonitor::KMonitorFactory::GetKMonitor("v6d_benchmark"));
        if (!kmonitor_) {
            KVCM_LOG_ERROR("Failed to get KMonitor instance");
            return false;
        }

        // 注册所有metrics (使用RegisterMetric，指定GAUGE类型)
        qps_metrics = kmonitor_->RegisterMetric("v6d_benchmark.qps", kmonitor::GAUGE, kmonitor::FATAL);
        avg_latency_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.avg_latency_us", kmonitor::GAUGE, kmonitor::FATAL);
        p50_latency_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.p50_latency_us", kmonitor::GAUGE, kmonitor::FATAL);
        p99_latency_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.p99_latency_us", kmonitor::GAUGE, kmonitor::FATAL);
        p999_latency_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.p999_latency_us", kmonitor::GAUGE, kmonitor::FATAL);
        success_rate_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.success_rate", kmonitor::GAUGE, kmonitor::FATAL);
        bandwidth_metrics = kmonitor_->RegisterMetric("v6d_benchmark.bandwidth_mbps", kmonitor::GAUGE, kmonitor::FATAL);
        add_block_qps_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.add_block_qps", kmonitor::GAUGE, kmonitor::FATAL);
        query_qps_metrics = kmonitor_->RegisterMetric("v6d_benchmark.query_qps", kmonitor::GAUGE, kmonitor::FATAL);
        delete_block_qps_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.delete_block_qps", kmonitor::GAUGE, kmonitor::FATAL);
        verification_passed_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.verification_passed", kmonitor::GAUGE, kmonitor::FATAL);
        verification_failed_metrics =
            kmonitor_->RegisterMetric("v6d_benchmark.verification_failed", kmonitor::GAUGE, kmonitor::FATAL);
        target_qps_metrics = kmonitor_->RegisterMetric("v6d_benchmark.target_qps", kmonitor::GAUGE, kmonitor::FATAL);

        KVCM_LOG_INFO("Kmonitor initialized successfully");
        return true;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("Failed to initialize kmonitor: %s", e.what());
        return false;
    }
}

void BenchmarkMetricsReporter::Start() {
    if (!running_.load()) {
        running_.store(true);
        report_thread_ = std::thread(&BenchmarkMetricsReporter::ReportLoop, this);
        KVCM_LOG_INFO("Metrics reporter started with interval %d ms", config_.report_interval_ms);
    }
}

void BenchmarkMetricsReporter::Stop() {
    if (running_.load()) {
        running_.store(false);
        if (report_thread_.joinable()) {
            report_thread_.join();
        }
        KVCM_LOG_INFO("Metrics reporter stopped");
    }
}

void BenchmarkMetricsReporter::RecordRequest(const std::string &op_type,
                                             int64_t latency_us,
                                             int64_t bytes,
                                             bool success) {
    // 更新总统计
    metrics_.total_requests.fetch_add(1);
    if (success) {
        metrics_.success_requests.fetch_add(1);
    } else {
        metrics_.failed_requests.fetch_add(1);
    }
    metrics_.total_latency_us.fetch_add(latency_us);
    metrics_.total_bytes.fetch_add(bytes);

    // 更新最小/最大延迟
    int64_t current_min = metrics_.min_latency_us.load();
    while (latency_us < current_min) {
        if (metrics_.min_latency_us.compare_exchange_weak(current_min, latency_us)) {
            break;
        }
    }
    int64_t current_max = metrics_.max_latency_us.load();
    while (latency_us > current_max) {
        if (metrics_.max_latency_us.compare_exchange_weak(current_max, latency_us)) {
            break;
        }
    }

    // 添加延迟样本
    {
        std::unique_lock lock(metrics_.latency_mutex);
        metrics_.latency_samples.push_back(latency_us);
    }

    // 更新分操作统计
    if (op_type == "add") {
        metrics_.add_block_metrics.count.fetch_add(1);
        if (success)
            metrics_.add_block_metrics.success.fetch_add(1);
        else
            metrics_.add_block_metrics.failed.fetch_add(1);
        metrics_.add_block_metrics.total_latency_us.fetch_add(latency_us);
    } else if (op_type == "query") {
        metrics_.query_metrics.count.fetch_add(1);
        if (success)
            metrics_.query_metrics.success.fetch_add(1);
        else
            metrics_.query_metrics.failed.fetch_add(1);
        metrics_.query_metrics.total_latency_us.fetch_add(latency_us);
    } else if (op_type == "delete") {
        metrics_.delete_block_metrics.count.fetch_add(1);
        if (success)
            metrics_.delete_block_metrics.success.fetch_add(1);
        else
            metrics_.delete_block_metrics.failed.fetch_add(1);
        metrics_.delete_block_metrics.total_latency_us.fetch_add(latency_us);
    }
}

void BenchmarkMetricsReporter::AcquireQPSPermit() {
    if (!config_.enable_qps_limit) {
        return;
    }

    std::unique_lock<std::mutex> lock(qps_mutex_);
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start_).count();

    // 如果超过1秒窗口,重置计数
    if (elapsed >= 1000) {
        window_start_ = now;
        current_window_count_ = 0;
    }

    // 如果超过QPS限制,等待到下一个窗口
    while (current_window_count_ >= static_cast<int64_t>(config_.target_qps)) {
        auto sleep_time = 1000 - std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - window_start_)
                                     .count();
        if (sleep_time > 0) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            lock.lock();
        }
        now = std::chrono::steady_clock::now();
        window_start_ = now;
        current_window_count_ = 0;
    }

    current_window_count_++;
}

double BenchmarkMetricsReporter::GetCurrentQPS() const {
    std::lock_guard<std::mutex> lock(qps_mutex_);
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start_).count();
    if (elapsed_ms == 0)
        return 0.0;
    return (current_window_count_ * 1000.0) / elapsed_ms;
}

const BenchmarkMetrics &BenchmarkMetricsReporter::GetSnapshot() const { return metrics_; }

void BenchmarkMetricsReporter::RecordVerification(bool success) {
    if (success) {
        metrics_.verification_passed.fetch_add(1);
    } else {
        metrics_.verification_failed.fetch_add(1);
    }
}

void BenchmarkMetricsReporter::ReportLoop() {
    auto last_report_time = std::chrono::steady_clock::now();
    int64_t last_total_requests = 0;

    while (running_.load()) {
        // 等待报告间隔
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.report_interval_ms));

        if (!running_.load())
            break;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count();

        if (elapsed_seconds == 0)
            continue;

        // 获取当前统计
        const auto &snapshot = GetSnapshot();
        int64_t current_total = snapshot.total_requests.load();
        int64_t delta_requests = current_total - last_total_requests;
        double current_qps = (delta_requests * 1000.0) / config_.report_interval_ms;

        // 上报到Kmonitor
        ReportToKmonitor(snapshot, current_qps);

        // 打印日志
        double avg_latency =
            snapshot.total_requests.load() > 0
                ? static_cast<double>(snapshot.total_latency_us.load()) / snapshot.total_requests.load()
                : 0;
        double success_rate =
            snapshot.total_requests.load() > 0
                ? (static_cast<double>(snapshot.success_requests.load()) / snapshot.total_requests.load()) * 100.0
                : 0;
        double bandwidth_mbps = (snapshot.total_bytes.load() * 8.0) / (1024.0 * 1024.0 * elapsed_seconds);

        KVCM_LOG_INFO("[Metrics] QPS=%.2f, AvgLatency=%.0fus, SuccessRate=%.2f%%, Bandwidth=%.2fMbps, "
                      "AddQPS=%.2f, QueryQPS=%.2f, DeleteQPS=%.2f, "
                      "Verification: Passed=%ld, Failed=%ld",
                      current_qps,
                      avg_latency,
                      success_rate,
                      bandwidth_mbps,
                      static_cast<double>(snapshot.add_block_metrics.count.load()) / elapsed_seconds,
                      static_cast<double>(snapshot.query_metrics.count.load()) / elapsed_seconds,
                      static_cast<double>(snapshot.delete_block_metrics.count.load()) / elapsed_seconds,
                      snapshot.verification_passed.load(),
                      snapshot.verification_failed.load());

        last_report_time = now;
        last_total_requests = current_total;
    }
}

void BenchmarkMetricsReporter::ReportToKmonitor(const BenchmarkMetrics &snapshot, double current_qps) {
    if (!kmonitor_ || !config_.enable_kmonitor) {
        return;
    }

    try {
        // 计算延迟百分位
        std::vector<int64_t> samples;
        {
            std::unique_lock lock(const_cast<std::shared_mutex &>(snapshot.latency_mutex));
            samples = snapshot.latency_samples;
        }

        double p50 = 0, p99 = 0, p999 = 0;
        if (!samples.empty()) {
            CalculatePercentiles(samples, p50, p99, p999);
        }

        double avg_latency =
            snapshot.total_requests.load() > 0
                ? static_cast<double>(snapshot.total_latency_us.load()) / snapshot.total_requests.load()
                : 0;
        double success_rate =
            snapshot.total_requests.load() > 0
                ? (static_cast<double>(snapshot.success_requests.load()) / snapshot.total_requests.load()) * 100.0
                : 100.0;

        // 上报所有指标 (使用Report方法)
        kmonitor::MetricsTags empty_tags;
        if (qps_metrics)
            qps_metrics->Report(&empty_tags, current_qps);
        if (avg_latency_metrics)
            avg_latency_metrics->Report(&empty_tags, avg_latency);
        if (p50_latency_metrics)
            p50_latency_metrics->Report(&empty_tags, p50);
        if (p99_latency_metrics)
            p99_latency_metrics->Report(&empty_tags, p99);
        if (p999_latency_metrics)
            p999_latency_metrics->Report(&empty_tags, p999);
        if (success_rate_metrics)
            success_rate_metrics->Report(&empty_tags, success_rate);
        if (verification_passed_metrics)
            verification_passed_metrics->Report(&empty_tags, static_cast<double>(snapshot.verification_passed.load()));
        if (verification_failed_metrics)
            verification_failed_metrics->Report(&empty_tags, static_cast<double>(snapshot.verification_failed.load()));
        if (target_qps_metrics)
            target_qps_metrics->Report(&empty_tags, config_.target_qps);

        // 清空样本
        {
            std::unique_lock lock(const_cast<std::shared_mutex &>(snapshot.latency_mutex));
            const_cast<BenchmarkMetrics &>(snapshot).latency_samples.clear();
        }
    } catch (const std::exception &e) { KVCM_LOG_WARN("Failed to report metrics to kmonitor: %s", e.what()); }
}

void BenchmarkMetricsReporter::CalculatePercentiles(const std::vector<int64_t> &samples,
                                                    double &p50,
                                                    double &p99,
                                                    double &p999) {
    if (samples.empty()) {
        p50 = p99 = p999 = 0;
        return;
    }

    std::vector<int64_t> sorted_samples = samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    size_t size = sorted_samples.size();
    p50 = static_cast<double>(sorted_samples[size * 50 / 100]);
    p99 = static_cast<double>(sorted_samples[size * 99 / 100]);
    p999 = static_cast<double>(sorted_samples[size * 999 / 1000]);
}

} // namespace v6d_benchmark
} // namespace kv_cache_manager
