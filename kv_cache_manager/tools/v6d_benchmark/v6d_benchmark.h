#pragma once

#include <atomic>
#include <memory>
#include <mutex>
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
    // 初始化阶段（每个 worker 独立 httplib::Client，避免单 Client 内 request_mutex 串行化）
    bool InitHttpClients();
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
    // 一次取 N 个不同的 dataset entry（按位置去重，避免同一 RPC 内 block_key 折叠
    // 让服务端 RMW 看上去 batch 比真实 batch 更小，从而高估单 RPC 吞吐）。
    void PickRandomEntries(int n, std::vector<DatasetEntry *> &out);
    // 同上，但只在 dataset 的 stride 子集 { start, start+stride, start+2*stride, ... } 中抽样，
    // 用于"每 worker 独占子集"的并发隔离模式（worker t 用 start=t, stride=num_threads）。
    void PickRandomEntriesInRange(int n, size_t start, size_t stride, std::vector<DatasetEntry *> &out);
    // 同 PickRandomEntriesInRange，但只挑选 exists==want_exists 的 entry。
    // 由于 worker 独占自己的 dataset 子集（无跨线程读写），exists 标志在单 worker 视角
    // 下严格可信，因此可以用来强制 add/delete 的语义：
    //   - add:    want_exists=false（只新增当前不存在的 key）
    //   - delete: want_exists=true （只删除当前存在的 key）
    // 当符合条件的 entry 不足 n 个时，按"凑多少算多少"返回；上层会根据 entries 实际大小
    // 决定是否跳过本轮请求，从而避免重复 add 同一个 key 或 delete 不存在的 key。
    void PickRandomEntriesInRangeByExist(
        int n, size_t start, size_t stride, bool want_exists, std::vector<DatasetEntry *> &out);

    // block_key 编码 / 解码（高位 process_id + thread_id，低位随机）
    int64_t EncodeBlockKey(uint32_t thread_id, uint64_t random_payload) const;

    // 压测工作线程
    void WorkerThread(int thread_id);

    // 周期上报 EVENT_HEARTBEAT（独立 HTTP 客户端，避免与 worker 争抢同一 httplib::Client）
    void HeartbeatLoop();

    // 压测操作 (带结果验证)；http_client 须为该 worker 独占的实例以实现并发 HTTP
    // batch 版本：一次 RPC 内打包 entries.size() 个事件 / keys，与服务端 1 RPC -> 1 RMW 的
    // cost 模型对齐，便于跟 QueryLocation 的 batch 流量做严格对等比较。
    bool AddBlocks(KVCMHttpClient &http_client, const std::vector<DatasetEntry *> &entries);
    bool DeleteBlocks(KVCMHttpClient &http_client, const std::vector<DatasetEntry *> &entries);
    // 旧版：在全 dataset 中随机抽 query keys；保留作为向后兼容入口，但 worker 不应再调用它。
    bool QueryLocation(KVCMHttpClient &http_client);
    // 新版：worker 把自己子集中抽好的 entries 传进来，避免跨 worker 撞 key。
    bool QueryLocationWithEntries(KVCMHttpClient &http_client, const std::vector<DatasetEntry *> &entries, bool use_batch = true);

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
    std::vector<std::unique_ptr<KVCMHttpClient>> http_clients_;
    std::unique_ptr<KVCMHttpClient> heartbeat_http_client_;
    ResultVerifier verifier_;

    // block_key 高位 = process_id（机器隔离），由 IP 末段 / env / pid 推导一次后固定
    uint32_t process_id_ = 0;

    // 数据集
    // 注意：worker 通过 PickRandomEntriesInRange(start=thread_id, stride=num_threads) 各取自己的子集，
    // 跨 worker 不会访问同一 entry，因此 dataset_ 在压测阶段不需要互斥锁。
    std::vector<DatasetEntry> dataset_;
    std::string local_ip_port_;

    // 线程管理
    std::vector<std::thread> workers_;
    std::thread heartbeat_thread_;
    std::atomic<uint64_t> heartbeat_seq_{0};
    std::atomic<int64_t> entry_index_{0};
};

} // namespace v6d_benchmark
} // namespace kv_cache_manager
