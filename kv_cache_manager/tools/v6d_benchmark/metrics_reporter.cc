#include "kv_cache_manager/tools/v6d_benchmark/metrics_reporter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>

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
        // 使用 MetricsConfig 对象初始化 KMonitor（对齐 kmonitor_metrics_reporter 流程）
        auto getEnv = [](const char *key, const char *def) -> std::string {
            const char *v = std::getenv(key);
            return v ? std::string(v) : std::string(def);
        };

        // 1. 配置 MetricsConfig（tenant / service_name 与现有 kvcm_service 保持一致）
        kmonitor::MetricsConfig metrics_config;
        metrics_config.set_tenant_name(getEnv("kmonitorTenant", "default").c_str());
        metrics_config.set_service_name(getEnv("kmonitorServiceName", "kvcm_service").c_str());

        std::string sink_addr = getEnv("kmonitorSinkAddress", getEnv("HIPPO_SLAVE_IP", "127.0.0.1").c_str());
        std::string kmon_port = getEnv("kmonitorPort", "4141");
        if (!kmon_port.empty()) {
            sink_addr += ":" + kmon_port;
        }
        metrics_config.set_sink_address(sink_addr.c_str());

        metrics_config.set_enable_log_file_sink(getEnv("kmonitorEnableLogFileSink", "false") == "true");
        // 关键：manually_mode 必须为 false，否则需要手动 flush 才会真正上报到服务端
        metrics_config.set_manually_mode(getEnv("kmonitorManuallyMode", "false") == "true");
        metrics_config.set_inited(true);

        // 全局 tags（对齐 setHippoTags 惯例）
        std::string slave_ip = getEnv("HIPPO_SLAVE_IP", "");
        metrics_config.AddGlobalTag("hippo_slave_ip", slave_ip.c_str());
        if (std::getenv("HIPPO_ROLE")) {
            metrics_config.AddGlobalTag("host_ip", slave_ip.c_str());
            metrics_config.AddGlobalTag("container_ip", getEnv("RequestedIP", slave_ip.c_str()).c_str());
            metrics_config.AddGlobalTag("hippo_role", getEnv("HIPPO_ROLE", "").c_str());
            metrics_config.AddGlobalTag("hippo_app", getEnv("HIPPO_APP", "").c_str());
            metrics_config.AddGlobalTag("hippo_group", getEnv("HIPPO_SERVICE_NAME", "").c_str());
        }

        // 2. 配置指标采集周期
        kmonitor::MetricLevelConfig level_config;
        level_config.period[kmonitor::FATAL] =
            static_cast<unsigned int>(std::max(1, config_.report_interval_ms / 1000));
        kmonitor::MetricLevelManager::SetGlobalLevelConfig(level_config);

        // 3. 初始化 KMonitor Factory
        if (!kmonitor::KMonitorFactory::Init(metrics_config)) {
            KVCM_LOG_ERROR("KMonitorFactory::Init() failed, kmonitor will be disabled");
            return false;
        }

        // 4. 注册内建指标（必须，否则上报管道不完整）
        std::string metrics_prefix = getEnv("kmonitorMetricsPrefix", "kvcm");
        kmonitor::KMonitorFactory::registerBuildInMetrics(nullptr, metrics_prefix.c_str());
        KVCM_LOG_INFO("KMonitorFactory::registerBuildInMetrics(%s) finished", metrics_prefix.c_str());

        // 5. 启动 KMonitor（与现有代码一致：Start 后再 GetKMonitor / RegisterMetric）
        kmonitor::KMonitorFactory::Start();
        KVCM_LOG_INFO("KMonitorFactory::Start() finished");

        // 6. 获取 KMonitor 实例（名称与现有服务保持一致的 kvcm_default）
        kmonitor_.reset(kmonitor::KMonitorFactory::GetKMonitor("kvcm_default"));
        if (!kmonitor_) {
            KVCM_LOG_ERROR("Failed to get KMonitor instance");
            return false;
        }

        // 7. 缓存本机IP，用于后续构造 kmonitor 上报 tag
        host_ip_ = getEnv("HIPPO_SLAVE_IP", "");
        if (host_ip_.empty()) {
            host_ip_ = "127.0.0.1";
        }

        // 8. 注册自定义指标（均使用 GAUGE 类型）
#define REGISTER_METRIC(ptr, name)                                                                                     \
    do {                                                                                                               \
        ptr = kmonitor_->RegisterMetric(name, kmonitor::GAUGE, kmonitor::FATAL);                                       \
        if (nullptr == ptr) {                                                                                          \
            KVCM_LOG_ERROR("Failed to register metric: %s", name);                                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

        REGISTER_METRIC(qps_metrics, "v6d_benchmark.qps");
        REGISTER_METRIC(avg_latency_metrics, "v6d_benchmark.avg_latency_us");
        REGISTER_METRIC(p50_latency_metrics, "v6d_benchmark.p50_latency_us");
        REGISTER_METRIC(p99_latency_metrics, "v6d_benchmark.p99_latency_us");
        REGISTER_METRIC(p999_latency_metrics, "v6d_benchmark.p999_latency_us");
        REGISTER_METRIC(success_rate_metrics, "v6d_benchmark.success_rate");
        REGISTER_METRIC(bandwidth_metrics, "v6d_benchmark.bandwidth_mbps");
        REGISTER_METRIC(add_block_qps_metrics, "v6d_benchmark.add_block_qps");
        REGISTER_METRIC(query_qps_metrics, "v6d_benchmark.query_qps");
        REGISTER_METRIC(delete_block_qps_metrics, "v6d_benchmark.delete_block_qps");
        REGISTER_METRIC(verification_passed_metrics, "v6d_benchmark.verification_passed");
        REGISTER_METRIC(verification_failed_metrics, "v6d_benchmark.verification_failed");
        REGISTER_METRIC(target_qps_metrics, "v6d_benchmark.target_qps");
        REGISTER_METRIC(batch_size_metrics, "v6d_benchmark.batch_size");
        REGISTER_METRIC(query_batch_size_metrics, "v6d_benchmark.query_batch_size");

#undef REGISTER_METRIC

        // 注册完成
        KVCM_LOG_INFO("Kmonitor initialized successfully with %d metrics", 15);
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

        // 优雅退出：打印最终累计汇总
        const auto &snapshot = GetSnapshot();
        KVCM_LOG_INFO("[Final] TotalRequests=%ld, SuccessRequests=%ld, FailedRequests=%ld",
                      snapshot.total_requests.load(),
                      snapshot.success_requests.load(),
                      snapshot.failed_requests.load());
        KVCM_LOG_INFO("[Final] AddBlocks=%ld, QueryBlocks=%ld, DeleteBlocks=%ld",
                      snapshot.add_block_metrics.count.load(),
                      snapshot.query_metrics.count.load(),
                      snapshot.delete_block_metrics.count.load());
        KVCM_LOG_INFO("[Final] Verification: Passed=%ld, Failed=%ld",
                      snapshot.verification_passed.load(),
                      snapshot.verification_failed.load());
        KVCM_LOG_INFO("[Final] TotalBytes=%ld, AvgLatency=%.0fus",
                      snapshot.total_bytes.load(),
                      snapshot.total_requests.load() > 0
                          ? static_cast<double>(snapshot.total_latency_us.load()) / snapshot.total_requests.load()
                          : 0.0);

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
        {
            std::unique_lock lock(metrics_.add_block_metrics.latency_mutex);
            metrics_.add_block_metrics.latency_samples.push_back(latency_us);
        }
    } else if (op_type == "query_batch" || op_type == "query_single") {
        // 总 query 统计（两种子类型合并）
        metrics_.query_metrics.count.fetch_add(1);
        if (success)
            metrics_.query_metrics.success.fetch_add(1);
        else
            metrics_.query_metrics.failed.fetch_add(1);
        metrics_.query_metrics.total_latency_us.fetch_add(latency_us);
        {
            std::unique_lock lock(metrics_.query_metrics.latency_mutex);
            metrics_.query_metrics.latency_samples.push_back(latency_us);
        }
        // 分 query 子类型统计
        auto &sub_metrics = (op_type == "query_batch") ? metrics_.batch_query_metrics : metrics_.single_query_metrics;
        sub_metrics.count.fetch_add(1);
        if (success)
            sub_metrics.success.fetch_add(1);
        else
            sub_metrics.failed.fetch_add(1);
        sub_metrics.total_latency_us.fetch_add(latency_us);
        {
            std::unique_lock lock(sub_metrics.latency_mutex);
            sub_metrics.latency_samples.push_back(latency_us);
        }
    } else if (op_type == "delete") {
        metrics_.delete_block_metrics.count.fetch_add(1);
        if (success)
            metrics_.delete_block_metrics.success.fetch_add(1);
        else
            metrics_.delete_block_metrics.failed.fetch_add(1);
        metrics_.delete_block_metrics.total_latency_us.fetch_add(latency_us);
        {
            std::unique_lock lock(metrics_.delete_block_metrics.latency_mutex);
            metrics_.delete_block_metrics.latency_samples.push_back(latency_us);
        }
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

    // 追踪所有累计值的上一次快照，用于计算区间增量
    int64_t last_total_requests = 0;
    int64_t last_total_latency_us = 0;
    int64_t last_success_requests = 0;
    int64_t last_total_bytes = 0;
    int64_t last_add_count = 0;
    int64_t last_query_count = 0;
    int64_t last_delete_count = 0;
    int64_t last_verification_passed = 0;
    int64_t last_verification_failed = 0;
    int64_t last_add_latency_us = 0;
    int64_t last_query_latency_us = 0;
    int64_t last_delete_latency_us = 0;
    int64_t last_batch_query_count = 0;
    int64_t last_single_query_count = 0;
    int64_t last_batch_query_latency_us = 0;
    int64_t last_single_query_latency_us = 0;

    while (running_.load()) {
        // 等待报告间隔
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.report_interval_ms));

        if (!running_.load())
            break;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count();

        if (elapsed_seconds == 0)
            continue;

        // 获取当前累计快照
        const auto &snapshot = GetSnapshot();
        int64_t cur_requests = snapshot.total_requests.load();
        int64_t cur_latency_us = snapshot.total_latency_us.load();
        int64_t cur_success = snapshot.success_requests.load();
        int64_t cur_bytes = snapshot.total_bytes.load();
        int64_t cur_add = snapshot.add_block_metrics.count.load();
        int64_t cur_query = snapshot.query_metrics.count.load();
        int64_t cur_delete = snapshot.delete_block_metrics.count.load();
        int64_t cur_vpass = snapshot.verification_passed.load();
        int64_t cur_vfail = snapshot.verification_failed.load();

        // 计算区间增量（delta）
        int64_t delta_requests = cur_requests - last_total_requests;
        int64_t delta_latency_us = cur_latency_us - last_total_latency_us;
        int64_t delta_success = cur_success - last_success_requests;
        int64_t delta_bytes = cur_bytes - last_total_bytes;
        int64_t delta_add = cur_add - last_add_count;
        int64_t delta_query = cur_query - last_query_count;
        int64_t delta_delete = cur_delete - last_delete_count;
        int64_t delta_vpass = cur_vpass - last_verification_passed;
        int64_t delta_vfail = cur_vfail - last_verification_failed;

        // 分算子累计时延
        int64_t cur_add_latency = snapshot.add_block_metrics.total_latency_us.load();
        int64_t cur_query_latency = snapshot.query_metrics.total_latency_us.load();
        int64_t cur_delete_latency = snapshot.delete_block_metrics.total_latency_us.load();
        int64_t delta_add_latency = cur_add_latency - last_add_latency_us;
        int64_t delta_query_latency = cur_query_latency - last_query_latency_us;
        int64_t delta_delete_latency = cur_delete_latency - last_delete_latency_us;

        double elapsed = static_cast<double>(elapsed_seconds);
        double current_qps = (delta_requests * 1000.0) / config_.report_interval_ms;
        double avg_latency = delta_requests > 0 ? static_cast<double>(delta_latency_us) / delta_requests : 0;
        double success_rate =
            delta_requests > 0 ? (static_cast<double>(delta_success) / delta_requests) * 100.0 : 100.0;
        double bandwidth_mbps = (delta_bytes * 8.0) / (1024.0 * 1024.0 * elapsed);
        double add_qps = elapsed > 0 ? static_cast<double>(delta_add) / elapsed : 0;
        double query_qps = elapsed > 0 ? static_cast<double>(delta_query) / elapsed : 0;
        double delete_qps = elapsed > 0 ? static_cast<double>(delta_delete) / elapsed : 0;
        double add_avg_latency = delta_add > 0 ? static_cast<double>(delta_add_latency) / delta_add : 0;
        double query_avg_latency = delta_query > 0 ? static_cast<double>(delta_query_latency) / delta_query : 0;
        double delete_avg_latency = delta_delete > 0 ? static_cast<double>(delta_delete_latency) / delta_delete : 0;

        // Query 子类型增量
        int64_t cur_batch_query = snapshot.batch_query_metrics.count.load();
        int64_t cur_single_query = snapshot.single_query_metrics.count.load();
        int64_t delta_batch_query = cur_batch_query - last_batch_query_count;
        int64_t delta_single_query = cur_single_query - last_single_query_count;
        int64_t cur_batch_query_latency = snapshot.batch_query_metrics.total_latency_us.load();
        int64_t cur_single_query_latency = snapshot.single_query_metrics.total_latency_us.load();
        int64_t delta_batch_query_latency = cur_batch_query_latency - last_batch_query_latency_us;
        int64_t delta_single_query_latency = cur_single_query_latency - last_single_query_latency_us;
        double batch_query_qps = elapsed > 0 ? static_cast<double>(delta_batch_query) / elapsed : 0;
        double single_query_qps = elapsed > 0 ? static_cast<double>(delta_single_query) / elapsed : 0;
        double batch_query_avg_latency = delta_batch_query > 0 ? static_cast<double>(delta_batch_query_latency) / delta_batch_query : 0;
        double single_query_avg_latency = delta_single_query > 0 ? static_cast<double>(delta_single_query_latency) / delta_single_query : 0;

        // 上报到 Kmonitor
        ReportToKmonitor(snapshot,
                         elapsed,
                         current_qps,
                         avg_latency,
                         success_rate,
                         bandwidth_mbps,
                         add_qps,
                         query_qps,
                         delete_qps,
                         add_avg_latency,
                         query_avg_latency,
                         delete_avg_latency,
                         static_cast<double>(delta_vpass),
                         static_cast<double>(delta_vfail),
                         batch_query_avg_latency,
                         single_query_avg_latency);

        // 打印日志
        KVCM_LOG_INFO("[Metrics] QPS=%.2f, AvgLatency=%.0fus, SuccessRate=%.2f%%, Bandwidth=%.2fMbps, "
                      "AddQPS=%.2f, QueryQPS=%.2f(batch=%.2f,single=%.2f), DeleteQPS=%.2f, "
                      "Verification: Passed=%ld, Failed=%ld",
                      current_qps,
                      avg_latency,
                      success_rate,
                      bandwidth_mbps,
                      add_qps,
                      query_qps,
                      batch_query_qps,
                      single_query_qps,
                      delete_qps,
                      delta_vpass,
                      delta_vfail);

        // 更新上一次快照
        last_report_time = now;
        last_total_requests = cur_requests;
        last_total_latency_us = cur_latency_us;
        last_success_requests = cur_success;
        last_total_bytes = cur_bytes;
        last_add_count = cur_add;
        last_query_count = cur_query;
        last_delete_count = cur_delete;
        last_verification_passed = cur_vpass;
        last_verification_failed = cur_vfail;
        last_add_latency_us = cur_add_latency;
        last_query_latency_us = cur_query_latency;
        last_delete_latency_us = cur_delete_latency;
        last_batch_query_count = cur_batch_query;
        last_single_query_count = cur_single_query;
        last_batch_query_latency_us = cur_batch_query_latency;
        last_single_query_latency_us = cur_single_query_latency;
    }
}

void BenchmarkMetricsReporter::ReportToKmonitor(const BenchmarkMetrics &snapshot,
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
                                                double single_query_avg_latency) {
    if (!kmonitor_ || !config_.enable_kmonitor) {
        return;
    }

    try {
        // 获取分算子延迟样本并计算百分位
        auto get_op_samples = [&snapshot](const BenchmarkMetrics::OpMetrics &op) {
            std::vector<int64_t> samples;
            {
                std::unique_lock lock(const_cast<std::shared_mutex &>(op.latency_mutex));
                samples = op.latency_samples;
            }
            return samples;
        };
        auto op_add_samples = get_op_samples(snapshot.add_block_metrics);
        auto op_query_samples = get_op_samples(snapshot.query_metrics);
        auto op_delete_samples = get_op_samples(snapshot.delete_block_metrics);

        double add_p50 = 0, add_p99 = 0, add_p999 = 0;
        double query_p50 = 0, query_p99 = 0, query_p999 = 0;
        double delete_p50 = 0, delete_p99 = 0, delete_p999 = 0;
        if (!op_add_samples.empty())
            CalculatePercentiles(op_add_samples, add_p50, add_p99, add_p999);
        if (!op_query_samples.empty())
            CalculatePercentiles(op_query_samples, query_p50, query_p99, query_p999);
        if (!op_delete_samples.empty())
            CalculatePercentiles(op_delete_samples, delete_p50, delete_p99, delete_p999);

        // 构造上报标签（static, 仅初始化一次）
        auto make_tags = [this](const char *op_type = nullptr) {
            std::map<std::string, std::string> tag_map;
            tag_map["host_ip"] = host_ip_;
            tag_map["instance_id"] = config_.instance_id;
            tag_map["instance_group"] = config_.instance_group;
            tag_map["test_mode"] = config_.test_mode;
            if (op_type)
                tag_map["op_type"] = op_type;
            return kmonitor::MetricsTags(tag_map);
        };
        static const kmonitor::MetricsTags tags = make_tags(nullptr);
        static const kmonitor::MetricsTags tags_add = make_tags("add");
        static const kmonitor::MetricsTags tags_getLocation = make_tags("getLocation");
        static const kmonitor::MetricsTags tags_getBatchLocations = make_tags("getBatchLocations");
        static const kmonitor::MetricsTags tags_delete = make_tags("delete");

        // Query 子类型延迟百分位
        auto op_batch_query_samples = get_op_samples(snapshot.batch_query_metrics);
        auto op_single_query_samples = get_op_samples(snapshot.single_query_metrics);
        double batch_query_p50 = 0, batch_query_p99 = 0, batch_query_p999 = 0;
        double single_query_p50 = 0, single_query_p99 = 0, single_query_p999 = 0;
        if (!op_batch_query_samples.empty())
            CalculatePercentiles(op_batch_query_samples, batch_query_p50, batch_query_p99, batch_query_p999);
        if (!op_single_query_samples.empty())
            CalculatePercentiles(op_single_query_samples, single_query_p50, single_query_p99, single_query_p999);

        // 总QPS
        if (qps_metrics)
            qps_metrics->Report(&tags, current_qps);
        // 分算子平均时延（RT按 op_type 区分，query 拆分为 getLocation + getBatchLocations）
        if (avg_latency_metrics) {
            avg_latency_metrics->Report(&tags_add, add_avg_latency);
            avg_latency_metrics->Report(&tags_getLocation, single_query_avg_latency);
            avg_latency_metrics->Report(&tags_getBatchLocations, batch_query_avg_latency);
            avg_latency_metrics->Report(&tags_delete, delete_avg_latency);
        }
        // 分算子 p50/p99/p999（RT按 op_type 区分）
        if (p50_latency_metrics) {
            p50_latency_metrics->Report(&tags_add, add_p50);
            p50_latency_metrics->Report(&tags_getLocation, single_query_p50);
            p50_latency_metrics->Report(&tags_getBatchLocations, batch_query_p50);
            p50_latency_metrics->Report(&tags_delete, delete_p50);
        }
        if (p99_latency_metrics) {
            p99_latency_metrics->Report(&tags_add, add_p99);
            p99_latency_metrics->Report(&tags_getLocation, single_query_p99);
            p99_latency_metrics->Report(&tags_getBatchLocations, batch_query_p99);
            p99_latency_metrics->Report(&tags_delete, delete_p99);
        }
        if (p999_latency_metrics) {
            p999_latency_metrics->Report(&tags_add, add_p999);
            p999_latency_metrics->Report(&tags_getLocation, single_query_p999);
            p999_latency_metrics->Report(&tags_getBatchLocations, batch_query_p999);
            p999_latency_metrics->Report(&tags_delete, delete_p999);
        }
        // 通用指标
        if (success_rate_metrics)
            success_rate_metrics->Report(&tags, success_rate);
        if (verification_passed_metrics)
            verification_passed_metrics->Report(&tags, delta_verify_pass);
        if (verification_failed_metrics)
            verification_failed_metrics->Report(&tags, delta_verify_fail);
        if (target_qps_metrics)
            target_qps_metrics->Report(&tags, config_.target_qps);
        if (batch_size_metrics)
            batch_size_metrics->Report(&tags, static_cast<double>(config_.batch_size));
        if (query_batch_size_metrics)
            query_batch_size_metrics->Report(&tags, static_cast<double>(config_.query_batch_size));
        if (bandwidth_metrics && elapsed_seconds > 0)
            bandwidth_metrics->Report(&tags, bandwidth_mbps);
        // 分算子QPS（指标名已含算子，无需额外 op_type）
        if (add_block_qps_metrics && elapsed_seconds > 0)
            add_block_qps_metrics->Report(&tags, add_qps);
        if (query_qps_metrics && elapsed_seconds > 0)
            query_qps_metrics->Report(&tags, query_qps);
        if (delete_block_qps_metrics && elapsed_seconds > 0)
            delete_block_qps_metrics->Report(&tags, delete_qps);

        // 清空本区间延迟样本
        {
            std::unique_lock lock(const_cast<std::shared_mutex &>(snapshot.latency_mutex));
            const_cast<BenchmarkMetrics &>(snapshot).latency_samples.clear();
        }
        {
            std::unique_lock lock(const_cast<std::shared_mutex &>(snapshot.add_block_metrics.latency_mutex));
            const_cast<BenchmarkMetrics::OpMetrics &>(snapshot.add_block_metrics).latency_samples.clear();
        }
        {
            std::unique_lock lock(const_cast<std::shared_mutex &>(snapshot.query_metrics.latency_mutex));
            const_cast<BenchmarkMetrics::OpMetrics &>(snapshot.query_metrics).latency_samples.clear();
        }
        {
            std::unique_lock lock(const_cast<std::shared_mutex &>(snapshot.delete_block_metrics.latency_mutex));
            const_cast<BenchmarkMetrics::OpMetrics &>(snapshot.delete_block_metrics).latency_samples.clear();
        }
        {
            std::unique_lock lock(const_cast<std::shared_mutex &>(snapshot.batch_query_metrics.latency_mutex));
            const_cast<BenchmarkMetrics::OpMetrics &>(snapshot.batch_query_metrics).latency_samples.clear();
        }
        {
            std::unique_lock lock(const_cast<std::shared_mutex &>(snapshot.single_query_metrics.latency_mutex));
            const_cast<BenchmarkMetrics::OpMetrics &>(snapshot.single_query_metrics).latency_samples.clear();
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
