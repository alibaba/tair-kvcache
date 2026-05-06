#include "kv_cache_manager/tools/v6d_benchmark/v6d_benchmark.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <random>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_set>

namespace kv_cache_manager {
namespace v6d_benchmark {

std::atomic<bool> V6DBenchmark::shutdown_requested_{false};

namespace {

// ============================================================
// block_key 64 位布局 (int64_t，最高位强制为 0 保证正数)：
//   bit 63       : 0 (sign, 强制正数)
//   bit 62..52   : process_id    (11 bits, 0..2047)   机器隔离
//   bit 51..42   : thread_id     (10 bits, 0..1023)   单机 worker 隔离
//   bit 41..0    : random_payload(42 bits, 0..2^42-1) 单线程随机
//
// 设计目标：
//   - 不同进程（按 IP 末段 / env / pid 推导出唯一 process_id）的 block_key 完全不重叠
//     -> 多机压测互不踩对方的数据。
//   - 同进程不同 worker thread 的 block_key 也不重叠
//     -> 单机内 add/delete/query 不再有跨 worker 的同 key 竞态，
//        verifier 视角与服务端 RMW 视角天然一致。
//   - 高位变化对服务端任何按 key hash 的分片仍均匀（仅高位偏移，分布不被破坏）。
// ============================================================
constexpr int kProcessIdBits = 11;
constexpr int kThreadIdBits = 10;
constexpr int kRandomBits = 64 - 1 - kProcessIdBits - kThreadIdBits; // = 42
constexpr uint32_t kProcessIdMax = (1u << kProcessIdBits) - 1;       // 2047
constexpr uint32_t kThreadIdMax = (1u << kThreadIdBits) - 1;         // 1023
constexpr uint64_t kRandomMask = (1ULL << kRandomBits) - 1;
constexpr int kProcessIdShift = kThreadIdBits + kRandomBits; // 52
constexpr int kThreadIdShift = kRandomBits;                  // 42

// 推导 process_id：依次尝试
//   1. BENCHMARK_PROCESS_ID 环境变量（可手动指定，0..2047）
//   2. HOST_IP / MY_POD_IP 的最后一段 octet（同网段唯一）
//   3. pid % 2048 兜底
uint32_t DeriveProcessId(const std::string &ip_or_iphost) {
    if (const char *env = std::getenv("BENCHMARK_PROCESS_ID"); env && *env) {
        try {
            long v = std::stol(env);
            if (v >= 0 && v <= kProcessIdMax) {
                return static_cast<uint32_t>(v);
            }
            KVCM_LOG_WARN("BENCHMARK_PROCESS_ID=%ld out of range [0..%u], fallback", v, kProcessIdMax);
        } catch (...) { KVCM_LOG_WARN("BENCHMARK_PROCESS_ID=%s not an int, fallback", env); }
    }
    // ip_or_iphost 形如 "33.67.16.145:8080" 或纯 IP；只取第一个 ":" 之前的部分
    std::string ip = ip_or_iphost;
    auto colon = ip.find(':');
    if (colon != std::string::npos) {
        ip = ip.substr(0, colon);
    }
    auto last_dot = ip.rfind('.');
    if (last_dot != std::string::npos && last_dot + 1 < ip.size()) {
        try {
            long octet = std::stol(ip.substr(last_dot + 1));
            if (octet >= 0 && octet <= 255) {
                return static_cast<uint32_t>(octet);
            }
        } catch (...) {}
    }
    return static_cast<uint32_t>(getpid() & kProcessIdMax);
}

// 简单的全局每秒限速器：跨线程共享一个秒级窗口，超过 max_per_sec 后本秒内剩余调用直接 drop。
// 仅用于失败诊断日志，目的是错误风暴时不要把磁盘塞满；可观测性损失换可用性。
bool ShouldEmitFailLog(int max_per_sec) {
    if (max_per_sec <= 0) {
        return true;
    }
    static std::atomic<int64_t> window_sec{0};
    static std::atomic<int> count_in_window{0};
    auto now_sec =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    int64_t expected = window_sec.load(std::memory_order_relaxed);
    if (expected != now_sec) {
        if (window_sec.compare_exchange_strong(expected, now_sec, std::memory_order_relaxed)) {
            count_in_window.store(0, std::memory_order_relaxed);
        }
    }
    int cur = count_in_window.fetch_add(1, std::memory_order_relaxed) + 1;
    return cur <= max_per_sec;
}

// 把 entries 中的首尾 key + 数量拼成一段简短摘要，方便在日志里检索。
// DatasetEntry 是 V6DBenchmark 的私有嵌套类型，这里用模板让匿名命名空间里也能用。
template <typename Entry>
std::string BatchKeyRange(const std::vector<Entry *> &entries) {
    if (entries.empty()) {
        return "[]";
    }
    std::ostringstream oss;
    oss << "size=" << entries.size() << " first_key=" << entries.front()->block_key
        << " last_key=" << entries.back()->block_key;
    return oss.str();
}

std::string FormatFailureInfo(const KVCMHttpClient::LastFailureInfo &fi) {
    std::ostringstream oss;
    oss << "url=" << fi.url;
    if (fi.http_status == 0) {
        oss << " httplib_error=" << fi.httplib_error << " (no response, likely connect/timeout)";
    } else {
        oss << " status=" << fi.http_status;
        if (fi.is_parse_error) {
            oss << " (json_parse_error)";
        }
    }
    if (!fi.request_body.empty()) {
        oss << " req_body=" << fi.request_body;
    }
    if (!fi.response_body.empty()) {
        oss << " resp_body=" << fi.response_body;
    }
    return oss.str();
}

template <typename T>
std::string JoinKeys(const std::vector<T> &keys, size_t max_n) {
    std::ostringstream oss;
    oss << "[";
    size_t n = std::min(keys.size(), max_n);
    for (size_t i = 0; i < n; ++i) {
        if (i)
            oss << ",";
        oss << keys[i];
    }
    if (keys.size() > max_n) {
        oss << ",...(+" << (keys.size() - max_n) << ")";
    }
    oss << "]";
    return oss.str();
}

} // namespace

V6DBenchmark::V6DBenchmark(const BenchmarkConfig &config, std::shared_ptr<BenchmarkMetricsReporter> metrics)
    : config_(config), metrics_(metrics) {}

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

    // 1.1 推导 process_id（用于 block_key 高位段，保证多机数据隔离）
    process_id_ = DeriveProcessId(local_ip_port_);
    KVCM_LOG_INFO("Derived process_id=%u (block_key high bits, machine-level isolation)", process_id_);
    if (config_.num_threads > static_cast<int>(kThreadIdMax) + 1) {
        KVCM_LOG_WARN("NUM_THREADS=%d exceeds thread_id field capacity %u, "
                      "block_key thread isolation may be lossy",
                      config_.num_threads,
                      kThreadIdMax + 1);
    }

    // 2. 为每个 worker 创建独立 HTTP 客户端（httplib::Client 单实例内请求互斥，多客户端才有真并发）
    if (!InitHttpClients()) {
        KVCM_LOG_ERROR("Failed to init HTTP clients");
        return 1;
    }

    // 3. 注册V6D存储
    if (!SetupV6DStorage()) {
        KVCM_LOG_ERROR("Failed to setup V6D storage");
        return 1;
    }

    // 4. 注册实例
    if (!RegisterInstance()) {
        KVCM_LOG_ERROR("Failed to register instance");
        return 1;
    }

    // 5. 注册V6D节点
    if (!RegisterNode()) {
        KVCM_LOG_ERROR("Failed to register V6D node");
        return 1;
    }

    // 6. 生成数据集
    GenerateDataset();
    KVCM_LOG_INFO("Generated dataset: %zu blocks", dataset_.size());

    // 7. 周期 EVENT_HEARTBEAT（与生产节点一致，独立线程 + 独立 HTTP 连接）
    if (config_.enable_periodic_heartbeat && heartbeat_http_client_) {
        heartbeat_thread_ = std::thread(&V6DBenchmark::HeartbeatLoop, this);
        KVCM_LOG_INFO("Periodic EVENT_HEARTBEAT thread started, interval %d ms", config_.heartbeat_interval_ms);
    }

    // 8. 启动工作线程
    KVCM_LOG_INFO("Starting %d worker threads with QPS limit: %.2f", config_.num_threads, config_.target_qps);

    for (int i = 0; i < config_.num_threads; ++i) {
        workers_.emplace_back(&V6DBenchmark::WorkerThread, this, i);
    }

    KVCM_LOG_INFO("Running benchmark (press Ctrl+C to stop)...");

    // 9. 等待所有工作线程(直到收到shutdown信号)
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }

    // 10. 打印最终统计
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

bool V6DBenchmark::InitHttpClients() {
    int n = config_.num_threads;
    if (n < 1) {
        KVCM_LOG_ERROR("NUM_THREADS must be >= 1, got %d", n);
        return false;
    }
    http_clients_.clear();
    http_clients_.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        auto c = std::make_unique<KVCMHttpClient>(config_.kvcm_base_url, config_.kvcm_admin_url);
        c->SetFailLogBodyMaxBytes(config_.fail_log_body_max_bytes);
        http_clients_.emplace_back(std::move(c));
    }
    if (config_.enable_periodic_heartbeat) {
        heartbeat_http_client_ = std::make_unique<KVCMHttpClient>(config_.kvcm_base_url, config_.kvcm_admin_url);
        heartbeat_http_client_->SetFailLogBodyMaxBytes(config_.fail_log_body_max_bytes);
        KVCM_LOG_INFO("Dedicated HTTP client created for periodic EVENT_HEARTBEAT");
    }
    KVCM_LOG_INFO("Initialized %d dedicated HTTP client(s) (httplib::Client is single-flight per instance; "
                  "one client per worker enables concurrent requests to KVCM)",
                  n);
    return true;
}

bool V6DBenchmark::SetupV6DStorage() {
    std::string storage_name = "v6d_" + config_.instance_id;
    KVCM_LOG_INFO("Setting up V6D storage: %s", storage_name.c_str());

    rapidjson::Document response;

    bool success = http_clients_[0]->AddStorage("benchmark_setup", storage_name, config_.instance_id, response);

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

    bool success = http_clients_[0]->RegisterInstance(
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
        http_clients_[0]->ReportEvent("benchmark_node_register", config_.instance_id, local_ip_port_, events, response);

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

int64_t V6DBenchmark::EncodeBlockKey(uint32_t thread_id, uint64_t random_payload) const {
    uint64_t pid_part = (static_cast<uint64_t>(process_id_ & kProcessIdMax)) << kProcessIdShift;
    uint64_t tid_part = (static_cast<uint64_t>(thread_id & kThreadIdMax)) << kThreadIdShift;
    uint64_t rnd_part = random_payload & kRandomMask;
    uint64_t k = pid_part | tid_part | rnd_part;
    // 强制清掉符号位，保证转换到 int64_t 后仍为正
    k &= 0x7FFFFFFFFFFFFFFFULL;
    return static_cast<int64_t>(k);
}

void V6DBenchmark::GenerateDataset() {
    dataset_.reserve(config_.num_blocks);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> rnd_dist(0, kRandomMask);

    // dataset 按 (i % num_threads) 划分给各 worker：
    //   worker t 拥有的 entry 索引集合 = {i : i % num_threads == t}
    //   该 entry 的 block_key 也被编码进 thread_id=t，确保跨 worker 不重叠。
    const int nt = std::max(1, config_.num_threads);
    std::unordered_set<int64_t> seen; // 防御性去重：random 段几乎不会撞，但加一道保险
    seen.reserve(static_cast<size_t>(config_.num_blocks) * 2);

    for (int i = 0; i < config_.num_blocks; ++i) {
        DatasetEntry entry;
        const uint32_t tid = static_cast<uint32_t>(i % nt);

        // 极小概率撞 key，最多重试若干次再放弃（直接用最新值，影响可忽略）
        int64_t k = 0;
        for (int retry = 0; retry < 8; ++retry) {
            k = EncodeBlockKey(tid, rnd_dist(gen));
            if (seen.insert(k).second) {
                break;
            }
        }
        entry.block_key = k;
        entry.medium = config_.mediums[i % config_.mediums.size()];
        entry.uri =
            "vineyard://" + local_ip_port_ + "/" + entry.medium + "?block_key=" + std::to_string(entry.block_key);
        entry.data = std::string(config_.block_size, 'A');
        entry.exists = false;

        dataset_.push_back(std::move(entry));
    }
    KVCM_LOG_INFO("Dataset generated: num_blocks=%d, owners distributed across %d worker(s); "
                  "block_key layout: process_id=%u (bits %d..%d), thread_id (bits %d..%d), random (bits 0..%d)",
                  config_.num_blocks,
                  nt,
                  process_id_,
                  kProcessIdShift + kProcessIdBits - 1,
                  kProcessIdShift,
                  kThreadIdShift + kThreadIdBits - 1,
                  kThreadIdShift,
                  kRandomBits - 1);
}

V6DBenchmark::DatasetEntry *V6DBenchmark::GetRandomEntry() {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, dataset_.size() - 1);
    size_t idx = dist(gen);
    return &dataset_[idx];
}

void V6DBenchmark::PickRandomEntries(int n, std::vector<DatasetEntry *> &out) {
    out.clear();
    if (n <= 0 || dataset_.empty()) {
        return;
    }
    static thread_local std::mt19937 gen(std::random_device{}());
    const size_t ds_size = dataset_.size();
    const size_t want = std::min<size_t>(static_cast<size_t>(n), ds_size);
    out.reserve(want);
    std::unordered_set<size_t> seen;
    seen.reserve(want * 2);
    std::uniform_int_distribution<size_t> dist(0, ds_size - 1);
    const size_t max_attempts = want * 8 + 16;
    size_t attempts = 0;
    while (out.size() < want && attempts < max_attempts) {
        size_t idx = dist(gen);
        if (seen.insert(idx).second) {
            out.push_back(&dataset_[idx]);
        }
        ++attempts;
    }
    if (out.size() < want) {
        for (size_t i = 0; i < ds_size && out.size() < want; ++i) {
            if (seen.insert(i).second) {
                out.push_back(&dataset_[i]);
            }
        }
    }
}

// 抽样 worker 自己拥有的 entry 子集：indices = { thread_id, thread_id+stride, thread_id+2*stride, ... }
// 与全 dataset 抽样比，pool 大小约为 dataset_size / num_threads，但 worker 之间互不重叠，
// 因此 add/delete/query 不会跨 worker 撞同一个 block_key（彻底消除并发竞态）。
void V6DBenchmark::PickRandomEntriesInRange(int n, size_t start, size_t stride, std::vector<DatasetEntry *> &out) {
    out.clear();
    if (n <= 0 || dataset_.empty() || stride == 0) {
        return;
    }
    // worker 拥有的 entry 数量
    size_t pool = (dataset_.size() > start) ? ((dataset_.size() - start + stride - 1) / stride) : 0;
    if (pool == 0) {
        return;
    }
    static thread_local std::mt19937 gen(std::random_device{}());
    const size_t want = std::min<size_t>(static_cast<size_t>(n), pool);
    out.reserve(want);
    std::unordered_set<size_t> seen;
    seen.reserve(want * 2);
    std::uniform_int_distribution<size_t> dist(0, pool - 1);
    const size_t max_attempts = want * 8 + 16;
    size_t attempts = 0;
    while (out.size() < want && attempts < max_attempts) {
        size_t local_idx = dist(gen);
        if (seen.insert(local_idx).second) {
            size_t real_idx = start + local_idx * stride;
            out.push_back(&dataset_[real_idx]);
        }
        ++attempts;
    }
    if (out.size() < want) {
        for (size_t li = 0; li < pool && out.size() < want; ++li) {
            if (seen.insert(li).second) {
                out.push_back(&dataset_[start + li * stride]);
            }
        }
    }
}

// 按 exists 状态筛选的抽样：先随机扫一段子集尝试凑齐，再退化成线性遍历兜底。
// 抽样行为是"位置去重 + 按 exists 过滤"——位置去重保证一次 RPC 内不会重复同一 key，
// 状态过滤保证 add 不踩已存在 key、delete 不踩已删除 key。worker 独占子集，
// 所以读取/修改 entry->exists 不需要锁。
void V6DBenchmark::PickRandomEntriesInRangeByExist(
    int n, size_t start, size_t stride, bool want_exists, std::vector<DatasetEntry *> &out) {
    out.clear();
    if (n <= 0 || dataset_.empty() || stride == 0) {
        return;
    }
    size_t pool = (dataset_.size() > start) ? ((dataset_.size() - start + stride - 1) / stride) : 0;
    if (pool == 0) {
        return;
    }
    static thread_local std::mt19937 gen(std::random_device{}());
    const size_t want = std::min<size_t>(static_cast<size_t>(n), pool);
    out.reserve(want);
    std::unordered_set<size_t> seen;
    seen.reserve(want * 2);
    std::uniform_int_distribution<size_t> dist(0, pool - 1);
    // 随机阶段：尝试若干次抽到符合状态的 entry；上限给得宽一点，覆盖混合状态下的命中率波动。
    const size_t max_attempts = want * 16 + 32;
    size_t attempts = 0;
    while (out.size() < want && attempts < max_attempts) {
        size_t local_idx = dist(gen);
        if (seen.insert(local_idx).second) {
            DatasetEntry *e = &dataset_[start + local_idx * stride];
            if (e->exists == want_exists) {
                out.push_back(e);
            }
        }
        ++attempts;
    }
    // 兜底阶段：随机阶段没抽够时，线性扫剩下的 entry 把符合条件的全部塞进来，
    // 直到凑够 want 或穷尽 pool。这样在子集中"符合状态"的 entry 数量 < want 时，
    // 也能确定性地拿到所有可用的，保证 add/delete 的最大吞吐而不是过早返回 0/1 个。
    if (out.size() < want) {
        for (size_t li = 0; li < pool && out.size() < want; ++li) {
            if (seen.insert(li).second) {
                DatasetEntry *e = &dataset_[start + li * stride];
                if (e->exists == want_exists) {
                    out.push_back(e);
                }
            }
        }
    }
}

void V6DBenchmark::HeartbeatLoop() {
    constexpr int kSleepSliceMs = 100;
    while (!shutdown_requested_.load()) {
        int remaining = config_.heartbeat_interval_ms;
        while (remaining > 0 && !shutdown_requested_.load()) {
            const int sleep_ms = std::min(kSleepSliceMs, remaining);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            remaining -= sleep_ms;
        }
        if (shutdown_requested_.load()) {
            break;
        }
        rapidjson::Document hb_event;
        BuildHeartbeatEvent(hb_event);
        std::vector<rapidjson::Document> events;
        events.push_back(std::move(hb_event));
        rapidjson::Document response;
        const uint64_t seq = heartbeat_seq_.fetch_add(1, std::memory_order_relaxed) + 1;
        const std::string trace_id = "benchmark_heartbeat_" + std::to_string(seq);
        if (!heartbeat_http_client_->ReportEvent(trace_id, config_.instance_id, local_ip_port_, events, response)) {
            if (ShouldEmitFailLog(config_.max_fail_log_per_sec)) {
                if (config_.verbose_fail_log) {
                    KVCM_LOG_ERROR("Periodic EVENT_HEARTBEAT FAILED trace_id=%s %s",
                                   trace_id.c_str(),
                                   FormatFailureInfo(heartbeat_http_client_->GetLastFailureInfo()).c_str());
                } else {
                    KVCM_LOG_WARN("Periodic EVENT_HEARTBEAT failed trace_id=%s", trace_id.c_str());
                }
            }
        }
    }
    KVCM_LOG_INFO("Heartbeat thread exiting");
}

void V6DBenchmark::WorkerThread(int thread_id) {
    KVCM_LOG_INFO("Worker thread %d started", thread_id);

    KVCMHttpClient &client = *http_clients_[static_cast<size_t>(thread_id)];

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

        // 三流量对等：add/delete 与 query 一样，每次 RPC 携带 batch_size 个 key。
        // 服务端把同一 ReportEvent 内的事件按 block_key 折叠成 1 次 RMW，
        // 与 getCacheLocation(batch_get) 的 1 RPC -> 1 RMW 行为对齐。
        //
        // 并发隔离：worker 只在自己的 dataset 子集（thread_id 段）里抽样，
        // 跨 worker 不会撞到同一个 block_key，verifier 也不会因竞态产生假阳性。
        //
        // 语义严格化：
        //   - add 只抽"当前不存在"(exists==false) 的 entry，避免重复新增已存在 key；
        //   - delete 只抽"当前存在"(exists==true)  的 entry，避免删除不存在 key；
        //   - query 不区分状态，可以查任意 key（验证器视角自洽）。
        // 由于 worker 独占自己的 dataset 子集，exists 标志在单 worker 视角严格可信，
        // 不存在原"按存在性过滤会塌陷 batch"的并发风险。
        const size_t start = static_cast<size_t>(thread_id);
        const size_t stride = static_cast<size_t>(std::max(1, config_.num_threads));
        // skip_metrics：本轮 entries 为空（add 时 worker 子集全满，delete 时 worker 子集
        // 还没人 add 过），没有真实 RPC 发出。这种情况下不能调 RecordRequest，否则会以
        // "0 latency 的成功请求"刷高 QPS，让 delete 看上去比真实快。
        bool skip_metrics = false;
        if (operation == "add") {
            std::vector<DatasetEntry *> entries;
            PickRandomEntriesInRangeByExist(config_.batch_size, start, stride, /*want_exists=*/false, entries);
            if (entries.empty()) {
                skip_metrics = true;
            } else {
                success = AddBlocks(client, entries);
                bytes = static_cast<size_t>(config_.block_size) * entries.size();
            }
        } else if (operation == "query") {
            std::vector<DatasetEntry *> entries;
            PickRandomEntriesInRange(config_.query_batch_size, start, stride, entries);
            success = QueryLocationWithEntries(client, entries);
            bytes = 0;
        } else if (operation == "delete") {
            std::vector<DatasetEntry *> entries;
            PickRandomEntriesInRangeByExist(config_.batch_size, start, stride, /*want_exists=*/true, entries);
            if (entries.empty()) {
                // 典型出现在纯 delete 模式启动初期，dataset 还没人 add。
                // 这里只 warn 一次/秒，并跳过 metrics 上报，避免污染 QPS 数据。
                if (ShouldEmitFailLog(1)) {
                    KVCM_LOG_WARN("Worker %d: no existing entry to delete in own subset (pool likely empty); "
                                  "consider running 'full' or 'add' mode first as warm-up",
                                  thread_id);
                }
                skip_metrics = true;
            } else {
                success = DeleteBlocks(client, entries);
                bytes = 0;
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        // 记录指标
        if (!skip_metrics) {
            metrics_->RecordRequest(operation, latency, bytes, success);
        }
    }

    KVCM_LOG_INFO("Worker thread %d stopped", thread_id);
}

bool V6DBenchmark::AddBlocks(KVCMHttpClient &client, const std::vector<DatasetEntry *> &entries) {
    if (entries.empty()) {
        return true;
    }

    // 调用方（WorkerThread）已经按 exists==false 预过滤过 entries，这里收到的都应当是
    // "当前不存在"的 key——保证 add 不会重复新增已存在 key（语义严格 "add only missing"）。
    std::vector<rapidjson::Document> events;
    events.reserve(entries.size());
    for (auto *entry : entries) {
        rapidjson::Document add_event;
        BuildBlockAddEvent(add_event, entry->block_key, entry->uri, entry->medium);
        events.push_back(std::move(add_event));
    }

    rapidjson::Document response;
    // trace_id 用首尾 key + batch size，便于在服务端日志里反查这一批的范围。
    std::string trace_id =
        "add_batch_" + std::to_string(entries.front()->block_key) + "_n" + std::to_string(entries.size());
    bool success = client.ReportEvent(trace_id, config_.instance_id, local_ip_port_, events, response);

    if (success) {
        for (auto *entry : entries) {
            entry->exists = true;
            if (config_.enable_verification) {
                verifier_.RecordAdd(entry->block_key, entry->uri, entry->medium);
            }
        }
    } else if (ShouldEmitFailLog(config_.max_fail_log_per_sec)) {
        if (config_.verbose_fail_log) {
            KVCM_LOG_ERROR("AddBlocks FAILED trace_id=%s batch{%s} %s",
                           trace_id.c_str(),
                           BatchKeyRange(entries).c_str(),
                           FormatFailureInfo(client.GetLastFailureInfo()).c_str());
        } else {
            KVCM_LOG_WARN("AddBlocks failed trace_id=%s batch{%s}", trace_id.c_str(), BatchKeyRange(entries).c_str());
        }
    }

    return success;
}

bool V6DBenchmark::DeleteBlocks(KVCMHttpClient &client, const std::vector<DatasetEntry *> &entries) {
    if (entries.empty()) {
        return true;
    }

    // 调用方（WorkerThread）已经按 exists==true 预过滤过 entries，这里收到的都应当是
    // "当前存在"的 key。worker 独占自己的 dataset 子集，过滤后的 entries 在发送 RPC
    // 前不会被其他线程再改 exists，所以语义上严格 "delete only existing"；
    // 服务端对不存在的 key 仍有 EC_NOENT 兜底（兜底主要面向跨进程并发场景）。
    std::vector<rapidjson::Document> events;
    events.reserve(entries.size());
    for (auto *entry : entries) {
        rapidjson::Document delete_event;
        BuildBlockDeleteEvent(delete_event, entry->block_key, entry->medium);
        events.push_back(std::move(delete_event));
    }

    rapidjson::Document response;
    std::string trace_id =
        "delete_batch_" + std::to_string(entries.front()->block_key) + "_n" + std::to_string(entries.size());
    bool success = client.ReportEvent(trace_id, config_.instance_id, local_ip_port_, events, response);

    if (success) {
        for (auto *entry : entries) {
            entry->exists = false;
            if (config_.enable_verification) {
                verifier_.RecordDelete(entry->block_key, entry->medium);
            }
        }
    } else if (ShouldEmitFailLog(config_.max_fail_log_per_sec)) {
        if (config_.verbose_fail_log) {
            KVCM_LOG_ERROR("DeleteBlocks FAILED trace_id=%s batch{%s} %s",
                           trace_id.c_str(),
                           BatchKeyRange(entries).c_str(),
                           FormatFailureInfo(client.GetLastFailureInfo()).c_str());
        } else {
            KVCM_LOG_WARN(
                "DeleteBlocks failed trace_id=%s batch{%s}", trace_id.c_str(), BatchKeyRange(entries).c_str());
        }
    }

    return success;
}

bool V6DBenchmark::QueryLocation(KVCMHttpClient &client) {
    // 兼容旧调用路径：从全 dataset 抽样
    std::vector<DatasetEntry *> entries;
    PickRandomEntries(config_.query_batch_size, entries);
    return QueryLocationWithEntries(client, entries);
}

bool V6DBenchmark::QueryLocationWithEntries(KVCMHttpClient &client, const std::vector<DatasetEntry *> &entries) {
    if (entries.empty()) {
        return true;
    }
    std::vector<int64_t> query_keys;
    query_keys.reserve(entries.size());
    for (auto *entry : entries) {
        query_keys.push_back(entry->block_key);
    }

    rapidjson::Document response;
    std::string trace_id = "query_batch_" + std::to_string(query_keys[0]);
    bool success =
        client.GetCacheLocation(trace_id, config_.instance_id, QueryType::QT_BATCH_GET, query_keys, response);

    if (!success && ShouldEmitFailLog(config_.max_fail_log_per_sec)) {
        if (config_.verbose_fail_log) {
            KVCM_LOG_ERROR("QueryLocation FAILED trace_id=%s keys=%s %s",
                           trace_id.c_str(),
                           JoinKeys(query_keys, 5).c_str(),
                           FormatFailureInfo(client.GetLastFailureInfo()).c_str());
        } else {
            KVCM_LOG_WARN("QueryLocation failed trace_id=%s first_key=%ld size=%zu",
                          trace_id.c_str(),
                          query_keys[0],
                          query_keys.size());
        }
    }

    // 验证结果
    // 注意：本 verifier 只是对客户端记录的"调用序"做对比，多线程压测下 verifier 视角与
    // 服务端 RMW 序无法严格对齐，会产生大量假阳性 missing/unexpected。
    // - 默认（非 strict）：只在 WARN 级别打一行汇总，不打 response body，避免日志被淹没。
    // - STRICT_VERIFICATION=true：按 ERROR 打印并附带 response body，适合单线程正确性回归。
    if (success && config_.enable_verification) {
        auto verify_result = verifier_.VerifyQuery(query_keys, response);
        metrics_->RecordVerification(verify_result.success);
        if (!verify_result.success && ShouldEmitFailLog(config_.max_fail_log_per_sec)) {
            if (config_.strict_verification) {
                KVCM_LOG_ERROR("Verification FAILED trace_id=%s expected=%d actual=%d missing=%zu unexpected=%zu "
                               "missing_keys=%s unexpected_keys=%s",
                               trace_id.c_str(),
                               verify_result.expected_count,
                               verify_result.actual_count,
                               verify_result.missing_keys.size(),
                               verify_result.unexpected_keys.size(),
                               JoinKeys(verify_result.missing_keys, 5).c_str(),
                               JoinKeys(verify_result.unexpected_keys, 5).c_str());
                if (config_.verbose_fail_log) {
                    rapidjson::StringBuffer buf;
                    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
                    response.Accept(writer);
                    std::string body(buf.GetString(), buf.GetSize());
                    int max_n = config_.fail_log_body_max_bytes;
                    if (max_n > 0 && static_cast<int>(body.size()) > max_n) {
                        body.resize(static_cast<size_t>(max_n));
                        body.append("...<truncated>");
                    }
                    KVCM_LOG_ERROR("Verification FAILED trace_id=%s response_body=%s", trace_id.c_str(), body.c_str());
                }
            } else {
                // 非 strict：只打一条 WARN，附 first 5 key 做抽样调查，不打 response body。
                KVCM_LOG_WARN("Verification mismatch (likely concurrency race, not a bug) trace_id=%s "
                              "expected=%d actual=%d missing=%zu unexpected=%zu missing_keys=%s unexpected_keys=%s",
                              trace_id.c_str(),
                              verify_result.expected_count,
                              verify_result.actual_count,
                              verify_result.missing_keys.size(),
                              verify_result.unexpected_keys.size(),
                              JoinKeys(verify_result.missing_keys, 5).c_str(),
                              JoinKeys(verify_result.unexpected_keys, 5).c_str());
            }
        }
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
