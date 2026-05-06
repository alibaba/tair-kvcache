#include <csignal>
#include <iostream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/tools/v6d_benchmark/config.h"
#include "kv_cache_manager/tools/v6d_benchmark/metrics_reporter.h"
#include "kv_cache_manager/tools/v6d_benchmark/v6d_benchmark.h"

using namespace kv_cache_manager;
using namespace kv_cache_manager::v6d_benchmark;

int main(int argc, char *argv[]) {
    // 1. 初始化日志（使用benchmark专用配置文件，输出到 logs/kv_cache_manager_benchmark_v6d.log）
    LoggerBroker::InitLogger("kv_cache_manager/tools/v6d_benchmark/benchmark_alog.conf", false);
    LoggerBroker::SetLogLevel(Logger::LEVEL_INFO); // 确保INFO级别日志可见
    LoggerBroker::InitLogLevelFromEnv();

    // 2. 解析环境变量配置
    BenchmarkConfig config = ParseConfigFromEnv();
    KVCM_LOG_INFO("Benchmark config loaded: TARGET_QPS=%.2f, THREADS=%d, MODE=%s",
                  config.target_qps,
                  config.num_threads,
                  config.test_mode.c_str());

    // 3. 注册信号处理
    signal(SIGTERM, V6DBenchmark::SignalHandler);
    signal(SIGINT, V6DBenchmark::SignalHandler);

    // 4. 创建指标汇报器
    auto metrics = std::make_shared<BenchmarkMetricsReporter>(config);
    if (config.enable_kmonitor) {
        if (!metrics->Init()) {
            KVCM_LOG_WARN("Failed to initialize kmonitor, continuing without metrics reporting");
        } else {
            metrics->Start(); // 启动10s周期的汇报线程
            KVCM_LOG_INFO("Metrics reporter started");
        }
    }

    // 5. 创建并运行压测 (阻塞直到收到信号)
    V6DBenchmark benchmark(config, metrics);
    int exit_code = benchmark.Run();

    // 6. 停止指标汇报并打印最终统计
    metrics->Stop();
    KVCM_LOG_INFO("Benchmark finished. Final metrics logged above.");

    return exit_code;
}
