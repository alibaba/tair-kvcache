#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/tools/v6d_benchmark/config.h"
#include "kv_cache_manager/tools/v6d_benchmark/http_client.h"
#include "kv_cache_manager/tools/v6d_benchmark/metrics_reporter.h"
#include "kv_cache_manager/tools/v6d_benchmark/result_verifier.h"

namespace kv_cache_manager {
namespace v6d_benchmark {

class V6DBenchmark {
public:
    V6DBenchmark(const BenchmarkConfig &config, std::shared_ptr<BenchmarkMetricsReporter> metrics);

    // 持续运行直到进程被杀死
    int Run();

    // 信号处理
    static void SignalHandler(int signal);
    static std::atomic<bool> shutdown_requested_;

private:
    // 初始化阶段
    bool SetupV6DStorage();
    bool RegisterInstance();
    bool RegisterNode();
    std::string GetLocalIP();

    // 数据集生成与管理
    struct DatasetEntry {
        int64_t block_key;
        std::string uri;
        std::string medium;
        std::string data; // 模拟的block数据
        bool exists;      // 当前是否存在
    };

    void GenerateDataset();
    DatasetEntry *GetRandomEntry();

    // 压测工作线程
    void WorkerThread(int thread_id);

    // 压测操作 (带结果验证)
    bool AddBlock(DatasetEntry *entry);
    bool DeleteBlock(DatasetEntry *entry);
    bool QueryLocation();

    // 辅助函数
    void BuildNodeRegisterEvent(rapidjson::Document &event);
    void BuildHeartbeatEvent(rapidjson::Document &event);
    void BuildBlockAddEvent(rapidjson::Document &event,
                            int64_t block_key,
                            const std::string &uri,
                            const std::string &medium);
    void BuildBlockDeleteEvent(rapidjson::Document &event, int64_t block_key, const std::string &medium);

    BenchmarkConfig config_;
    std::shared_ptr<BenchmarkMetricsReporter> metrics_;
    std::unique_ptr<KVCMHttpClient> http_client_;
    ResultVerifier verifier_;

    // 数据集
    std::vector<DatasetEntry> dataset_;
    std::shared_mutex dataset_mutex_;
    std::string local_ip_port_;

    // 线程管理
    std::vector<std::thread> workers_;
    std::atomic<int64_t> entry_index_{0};
};

} // namespace v6d_benchmark
} // namespace kv_cache_manager
