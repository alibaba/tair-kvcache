#include "kv_cache_manager/tools/v6d_benchmark/v6d_benchmark.h"

#include <arpa/inet.h>
#include <csignal>
#include <cstdlib>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <random>
#include <sys/socket.h>

namespace kv_cache_manager {
namespace v6d_benchmark {

std::atomic<bool> V6DBenchmark::shutdown_requested_{false};

V6DBenchmark::V6DBenchmark(const BenchmarkConfig &config, std::shared_ptr<BenchmarkMetricsReporter> metrics)
    : config_(config), metrics_(metrics) {
    http_client_ = std::make_unique<KVCMHttpClient>(config_.kvcm_base_url, config_.kvcm_admin_url);
}

void V6DBenchmark::SignalHandler(int signal) {
    if (signal == SIGTERM || signal == SIGINT) {
        KVCM_LOG_INFO("Received signal %d, initiating shutdown...", signal);
        shutdown_requested_.store(true);
    }
}

int V6DBenchmark::Run() {
    KVCM_LOG_INFO("V6D Benchmark starting...");

    // 1. 获取本地IP
    local_ip_port_ = GetLocalIP();
    if (local_ip_port_.empty()) {
        KVCM_LOG_ERROR("Failed to get local IP");
        return 1;
    }
    KVCM_LOG_INFO("Local IP:Port: %s", local_ip_port_.c_str());

    // 2. 注册V6D存储
    if (!SetupV6DStorage()) {
        KVCM_LOG_ERROR("Failed to setup V6D storage");
        return 1;
    }

    // 3. 注册实例
    if (!RegisterInstance()) {
        KVCM_LOG_ERROR("Failed to register instance");
        return 1;
    }

    // 4. 注册V6D节点
    if (!RegisterNode()) {
        KVCM_LOG_ERROR("Failed to register V6D node");
        return 1;
    }

    // 5. 生成数据集
    GenerateDataset();
    KVCM_LOG_INFO("Generated dataset: %zu blocks", dataset_.size());

    // 6. 启动工作线程
    KVCM_LOG_INFO("Starting %d worker threads with QPS limit: %.2f", config_.num_threads, config_.target_qps);

    for (int i = 0; i < config_.num_threads; ++i) {
        workers_.emplace_back(&V6DBenchmark::WorkerThread, this, i);
    }

    KVCM_LOG_INFO("Running benchmark (press Ctrl+C to stop)...");

    // 7. 等待所有工作线程(直到收到shutdown信号)
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    // 8. 打印最终统计
    const auto &stats = metrics_->GetSnapshot();
    KVCM_LOG_INFO("Benchmark finished.");
    KVCM_LOG_INFO("Total requests: %ld, Success: %ld, Failed: %ld",
                  stats.total_requests.load(),
                  stats.success_requests.load(),
                  stats.failed_requests.load());

    auto verify_stats = verifier_.GetStats();
    KVCM_LOG_INFO("Verification: Passed=%ld, Failed=%ld, MissingKeys=%ld, UnexpectedKeys=%ld",
                  verify_stats.passed_verifications,
                  verify_stats.failed_verifications,
                  verify_stats.total_missing_keys,
                  verify_stats.total_unexpected_keys);

    // 关键统计信息同时输出到 stderr
    std::cerr << "\n========== V6D Benchmark Results ==========" << std::endl;
    std::cerr << "Total Requests: " << stats.total_requests.load() << std::endl;
    std::cerr << "Success: " << stats.success_requests.load() << std::endl;
    std::cerr << "Failed: " << stats.failed_requests.load() << std::endl;
    std::cerr << "Verification Passed: " << verify_stats.passed_verifications << std::endl;
    std::cerr << "Verification Failed: " << verify_stats.failed_verifications << std::endl;
    std::cerr << "==========================================" << std::endl;

    return 0;
}

std::string V6DBenchmark::GetLocalIP() {
    // 先尝试环境变量
    const char *env_ip = std::getenv("HOST_IP");
    if (env_ip) {
        return std::string(env_ip) + ":" + config_.v6d_port;
    }
    env_ip = std::getenv("MY_POD_IP");
    if (env_ip) {
        return std::string(env_ip) + ":" + config_.v6d_port;
    }

    // 如果配置了固定IP,使用配置的IP
    if (!config_.auto_detect_host && !config_.v6d_host_ip_port.empty()) {
        return config_.v6d_host_ip_port;
    }

    // 自动获取本机IP
    struct ifaddrs *ifaddrs = nullptr;
    if (getifaddrs(&ifaddrs) == -1) {
        KVCM_LOG_ERROR("getifaddrs failed");
        return "";
    }

    std::string ip;
    for (struct ifaddrs *ifa = ifaddrs; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr)
            continue;
        if (ifa->ifa_addr->sa_family != AF_INET)
            continue;

        // 跳过loopback
        if (ifa->ifa_flags & IFF_LOOPBACK)
            continue;

        char host[NI_MAXHOST];
        if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
            ip = std::string(host);
            break;
        }
    }
    freeifaddrs(ifaddrs);

    if (ip.empty()) {
        KVCM_LOG_ERROR("Failed to get local IP address");
        return "";
    }

    return ip + ":" + config_.v6d_port;
}

bool V6DBenchmark::SetupV6DStorage() {
    std::string storage_name = "v6d_" + config_.instance_id;
    KVCM_LOG_INFO("Setting up V6D storage: %s", storage_name.c_str());

    rapidjson::Document response;

    bool success = http_client_->AddStorage("benchmark_setup", storage_name, config_.instance_id, response);

    if (success) {
        KVCM_LOG_INFO("V6D storage registered: %s", storage_name.c_str());
    } else {
        KVCM_LOG_WARN("AddStorage may have failed or already exists");
    }

    return true; // 即使失败也继续(可能已经存在)
}

bool V6DBenchmark::RegisterInstance() {
    KVCM_LOG_INFO("Registering instance: %s, group: %s, block_size: %d",
                  config_.instance_id.c_str(),
                  config_.instance_group.c_str(),
                  config_.block_size);

    rapidjson::Document response;

    bool success = http_client_->RegisterInstance(
        "benchmark_setup", config_.instance_group, config_.instance_id, config_.block_size, response);

    if (success) {
        KVCM_LOG_INFO("Instance registered: %s", config_.instance_id.c_str());
        return true;
    } else {
        KVCM_LOG_WARN("RegisterInstance FAILED or timed out for instance: %s, continuing anyway",
                      config_.instance_id.c_str());
        return true; // 即使失败也继续(可能已经存在或服务端处理慢)
    }
}

bool V6DBenchmark::RegisterNode() {
    KVCM_LOG_INFO("Registering V6D node: %s", local_ip_port_.c_str());

    // 构造NODE_REGISTER事件
    rapidjson::Document register_event;
    BuildNodeRegisterEvent(register_event);

    // 构造HEARTBEAT事件
    rapidjson::Document heartbeat_event;
    BuildHeartbeatEvent(heartbeat_event);

    std::vector<rapidjson::Document> events;
    events.push_back(std::move(register_event));
    events.push_back(std::move(heartbeat_event));

    rapidjson::Document response;
    bool success =
        http_client_->ReportEvent("benchmark_node_register", config_.instance_id, local_ip_port_, events, response);

    if (success) {
        KVCM_LOG_INFO("V6D node registered: %s with mediums [%s]",
                      local_ip_port_.c_str(),
                      StringUtil::Join(config_.mediums, ", ").c_str());
        return true;
    } else {
        KVCM_LOG_WARN("V6D node registration FAILED: %s, continuing anyway", local_ip_port_.c_str());
        return true; // 容错处理，继续运行
    }
}

void V6DBenchmark::GenerateDataset() {
    dataset_.reserve(config_.num_blocks);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<int64_t> key_dist(1000000, 999999999);

    // 为每个block分配一个medium(轮询)
    for (int i = 0; i < config_.num_blocks; ++i) {
        DatasetEntry entry;
        entry.block_key = key_dist(gen);
        entry.medium = config_.mediums[i % config_.mediums.size()];
        entry.uri =
            "vineyard://" + local_ip_port_ + "/" + entry.medium + "?block_key=" + std::to_string(entry.block_key);
        entry.data = std::string(config_.block_size, 'A'); // 模拟数据
        entry.exists = false;

        dataset_.push_back(std::move(entry));
    }
}

V6DBenchmark::DatasetEntry *V6DBenchmark::GetRandomEntry() {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, dataset_.size() - 1);
    size_t idx = dist(gen);
    return &dataset_[idx];
}

void V6DBenchmark::WorkerThread(int thread_id) {
    KVCM_LOG_INFO("Worker thread %d started", thread_id);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);

    while (!shutdown_requested_.load()) {
        // QPS限流
        metrics_->AcquireQPSPermit();

        // 根据test_mode选择操作
        std::string operation;
        if (config_.test_mode == "full") {
            double rand_val = op_dist(gen);
            if (rand_val < config_.add_ratio) {
                operation = "add";
            } else if (rand_val < config_.add_ratio + config_.query_ratio) {
                operation = "query";
            } else {
                operation = "delete";
            }
        } else {
            operation = config_.test_mode;
        }

        // 执行操作
        auto start_time = std::chrono::steady_clock::now();
        bool success = false;
        size_t bytes = 0;

        if (operation == "add") {
            auto *entry = GetRandomEntry();
            success = AddBlock(entry);
            bytes = config_.block_size;
        } else if (operation == "query") {
            success = QueryLocation();
            bytes = 0;
        } else if (operation == "delete") {
            auto *entry = GetRandomEntry();
            success = DeleteBlock(entry);
            bytes = 0;
        }

        auto end_time = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        // 记录指标
        metrics_->RecordRequest(operation, latency, bytes, success);
    }

    KVCM_LOG_INFO("Worker thread %d stopped", thread_id);
}

bool V6DBenchmark::AddBlock(DatasetEntry *entry) {
    // 构造BlockAdd事件
    rapidjson::Document add_event;
    BuildBlockAddEvent(add_event, entry->block_key, entry->uri, entry->medium);

    std::vector<rapidjson::Document> events;
    events.push_back(std::move(add_event));

    rapidjson::Document response;
    std::string trace_id = "add_" + std::to_string(entry->block_key);
    bool success = http_client_->ReportEvent(trace_id, config_.instance_id, local_ip_port_, events, response);

    if (success) {
        entry->exists = true;
        // 记录到验证器
        if (config_.enable_verification) {
            verifier_.RecordAdd(entry->block_key, entry->uri, entry->medium);
        }
    } else {
        KVCM_LOG_WARN("AddBlock failed for key=%ld", entry->block_key);
    }

    return success;
}

bool V6DBenchmark::DeleteBlock(DatasetEntry *entry) {
    if (!entry->exists) {
        return true; // 已经不存在,视为成功
    }

    // 构造BlockDelete事件
    rapidjson::Document delete_event;
    BuildBlockDeleteEvent(delete_event, entry->block_key, entry->medium);

    std::vector<rapidjson::Document> events;
    events.push_back(std::move(delete_event));

    rapidjson::Document response;
    std::string trace_id = "delete_" + std::to_string(entry->block_key);
    bool success = http_client_->ReportEvent(trace_id, config_.instance_id, local_ip_port_, events, response);

    if (success) {
        entry->exists = false;
        // 记录到验证器
        if (config_.enable_verification) {
            verifier_.RecordDelete(entry->block_key, entry->medium);
        }
    }

    return success;
}

bool V6DBenchmark::QueryLocation() {
    // 随机选择batch_size个key
    std::vector<int64_t> query_keys;
    query_keys.reserve(config_.query_batch_size);

    {
        std::shared_lock lock(dataset_mutex_);
        for (int i = 0; i < config_.query_batch_size && !dataset_.empty(); ++i) {
            auto *entry = GetRandomEntry();
            query_keys.push_back(entry->block_key);
        }
    }

    if (query_keys.empty()) {
        return true;
    }

    rapidjson::Document response;
    std::string trace_id = "query_batch_" + std::to_string(query_keys[0]);
    bool success =
        http_client_->GetCacheLocation(trace_id, config_.instance_id, QueryType::QT_BATCH_GET, query_keys, response);

    if (!success) {
        KVCM_LOG_WARN("QueryLocation failed for batch starting with key=%ld", query_keys[0]);
    }

    // 验证结果
    if (success && config_.enable_verification) {
        auto verify_result = verifier_.VerifyQuery(query_keys, response);
        metrics_->RecordVerification(verify_result.success);
    }

    return success;
}

void V6DBenchmark::BuildNodeRegisterEvent(rapidjson::Document &event) {
    event.SetObject();
    auto &allocator = event.GetAllocator();

    event.AddMember("event_type", "EVENT_NODE_REGISTER", allocator);

    rapidjson::Value params(rapidjson::kObjectType);
    rapidjson::Value mediums_array(rapidjson::kArrayType);
    for (const auto &medium : config_.mediums) {
        mediums_array.PushBack(rapidjson::Value(medium.c_str(), allocator), allocator);
    }
    params.AddMember("mediums", mediums_array, allocator);
    event.AddMember("node_register", params, allocator);
}

void V6DBenchmark::BuildHeartbeatEvent(rapidjson::Document &event) {
    event.SetObject();
    auto &allocator = event.GetAllocator();

    event.AddMember("event_type", "EVENT_HEARTBEAT", allocator);

    rapidjson::Value params(rapidjson::kObjectType);
    rapidjson::Value status(rapidjson::kObjectType);
    status.AddMember("version", "benchmark_v1", allocator);
    params.AddMember("system_status", status, allocator);
    event.AddMember("heartbeat", params, allocator);
}

void V6DBenchmark::BuildBlockAddEvent(rapidjson::Document &event,
                                      int64_t block_key,
                                      const std::string &uri,
                                      const std::string &medium) {
    event.SetObject();
    auto &allocator = event.GetAllocator();

    event.AddMember("event_type", "EVENT_BLOCK_ADD", allocator);

    rapidjson::Value params(rapidjson::kObjectType);
    params.AddMember("block_key", rapidjson::Value(std::to_string(block_key).c_str(), allocator), allocator);
    params.AddMember("uri", rapidjson::Value(uri.c_str(), allocator), allocator);
    params.AddMember("medium", rapidjson::Value(medium.c_str(), allocator), allocator);
    event.AddMember("block_add", params, allocator);
}

void V6DBenchmark::BuildBlockDeleteEvent(rapidjson::Document &event, int64_t block_key, const std::string &medium) {
    event.SetObject();
    auto &allocator = event.GetAllocator();

    event.AddMember("event_type", "EVENT_BLOCK_DELETE", allocator);

    rapidjson::Value params(rapidjson::kObjectType);
    params.AddMember("block_key", rapidjson::Value(std::to_string(block_key).c_str(), allocator), allocator);
    params.AddMember("medium", rapidjson::Value(medium.c_str(), allocator), allocator);
    event.AddMember("block_delete", params, allocator);
}

} // namespace v6d_benchmark
} // namespace kv_cache_manager
