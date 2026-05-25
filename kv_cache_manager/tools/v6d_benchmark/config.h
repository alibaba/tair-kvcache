#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {
namespace v6d_benchmark {

struct BenchmarkConfig {
    // KVCM连接配置
    std::string kvcm_base_url;  // KVCM_BASE_URL
    std::string kvcm_admin_url; // KVCM_ADMIN_URL
    std::string instance_id;    // INSTANCE_ID
    std::string instance_group; // INSTANCE_GROUP

    // V6D节点配置 - 支持自动获取IP
    bool auto_detect_host = true;     // AUTO_DETECT_HOST
    std::string v6d_host_ip_port;     // V6D_HOST (当auto_detect_host=false时使用)
    std::string v6d_port = "8080";    // V6D_PORT
    std::vector<std::string> mediums; // V6D_MEDIUMS

    // 压测参数
    int num_blocks = 10000; // NUM_BLOCKS
    int block_size = 128;   // BLOCK_SIZE
    int num_threads = 1;    // NUM_THREADS

    // QPS限流配置
    double target_qps = 1000.0;   // TARGET_QPS
    bool enable_qps_limit = true; // ENABLE_QPS_LIMIT

    // 压测模式
    std::string test_mode = "full"; // TEST_MODE: "full", "add", "query", "delete"

    // Batch 配置（三流量对等的核心旋钮）
    //
    // 服务端 ReportEvent 与 getCacheLocation 都是 1 RPC -> 1 RMW（按 key 折叠），
    // 因此 add/query/delete 三种 RPC 的 cost 模型只在「每 RPC 携带 key 数」一致时才对等。
    // 默认让三种操作共用 BATCH_SIZE，确保压测出的 RPC qps 反映的是同一种"批写/批读"workload。
    //
    // 兼容性：旧脚本设置 QUERY_BATCH_SIZE 时仍然生效（仅作用于 query），未设置时回退到 BATCH_SIZE。
    int batch_size = 50;       // BATCH_SIZE
    int query_batch_size = 50; // 由 ParseConfigFromEnv 同步为 batch_size，QUERY_BATCH_SIZE 可单独覆写

    // Query API 比例：[0.0, 1.0]，表示使用 GetBatchCacheLocations 的比例，剩余使用 GetCacheLocation
    double batch_query_ratio = 0.8; // BATCH_QUERY_RATIO

    // 操作比例配置 (仅full模式)
    double add_ratio = 0.7;    // ADD_RATIO
    double query_ratio = 0.2;  // QUERY_RATIO
    double delete_ratio = 0.1; // DELETE_RATIO

    // Kmonitor配置
    bool enable_kmonitor = true;    // ENABLE_KMONITOR
    std::string kmonitor_config;    // KMONITOR_CONFIG
    int report_interval_ms = 10000; // REPORT_INTERVAL_MS

    // 结果验证
    // - enable_verification: 是否对 query 结果做客户端校验
    // - strict_verification: 校验失败按 ERROR 打印（含 response body），适合单线程正确性回归；
    //   关闭后只按 WARN 打印汇总（missing/unexpected 计数），适合并发压测，
    //   原因是多线程乱序下 verifier "客户端记录序" vs "服务端 RMW 序" 无法严格对齐，
    //   会大量产生假阳性 unexpected/missing。
    bool enable_verification = true;  // ENABLE_VERIFICATION
    bool strict_verification = false; // STRICT_VERIFICATION

    // 与真实链路一致：注册节点后需持续上报 EVENT_HEARTBEAT 维持存活
    bool enable_periodic_heartbeat = true; // ENABLE_PERIODIC_HEARTBEAT
    int heartbeat_interval_ms = 3000;      // HEARTBEAT_INTERVAL_MS

    // 失败诊断日志
    // - verbose_fail_log: 失败时同时打印请求 body / 响应 body 摘要，便于排查根因
    // - max_fail_log_per_sec: 每秒最多打印多少条失败诊断日志，避免错误风暴时打爆磁盘
    //   （0 表示不限制；负数等同 0；只对失败诊断日志生效，HTTP 层 ERROR 仍按原逻辑）
    // - fail_log_body_max_bytes: 单条诊断日志中请求/响应 body 最多打多少字节
    bool verbose_fail_log = true;       // VERBOSE_FAIL_LOG
    int max_fail_log_per_sec = 50;      // MAX_FAIL_LOG_PER_SEC
    int fail_log_body_max_bytes = 1024; // FAIL_LOG_BODY_MAX_BYTES
};

inline std::string GetEnvOrDefault(const char *env_var, const std::string &default_val) {
    const char *val = std::getenv(env_var);
    return val ? std::string(val) : default_val;
}

inline bool GetEnvBool(const char *env_var, bool default_val) {
    const char *val = std::getenv(env_var);
    if (!val)
        return default_val;
    std::string str(val);
    return str == "true" || str == "1" || str == "yes";
}

inline int GetEnvInt(const char *env_var, int default_val) {
    const char *val = std::getenv(env_var);
    if (!val)
        return default_val;
    try {
        return std::stoi(val);
    } catch (...) {
        KVCM_LOG_WARN("Invalid int value for %s: %s, using default: %d", env_var, val, default_val);
        return default_val;
    }
}

inline double GetEnvDouble(const char *env_var, double default_val) {
    const char *val = std::getenv(env_var);
    if (!val)
        return default_val;
    try {
        return std::stod(val);
    } catch (...) {
        KVCM_LOG_WARN("Invalid double value for %s: %s, using default: %.2f", env_var, val, default_val);
        return default_val;
    }
}

inline std::vector<std::string> ParseStringList(const std::string &str, const std::string &delimiter = ",") {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        std::string token = str.substr(start, end - start);
        if (!token.empty()) {
            result.push_back(token);
        }
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    std::string token = str.substr(start);
    if (!token.empty()) {
        result.push_back(token);
    }
    return result;
}

inline BenchmarkConfig ParseConfigFromEnv() {
    BenchmarkConfig config;

    // KVCM连接配置
    config.kvcm_base_url = GetEnvOrDefault("KVCM_BASE_URL", "http://127.0.0.1:8080");
    config.kvcm_admin_url = GetEnvOrDefault("KVCM_ADMIN_URL", config.kvcm_base_url);
    config.instance_id = GetEnvOrDefault("INSTANCE_ID", "v6d_benchmark_0");
    config.instance_group = GetEnvOrDefault("INSTANCE_GROUP", "default");

    // V6D节点配置
    config.auto_detect_host = GetEnvBool("AUTO_DETECT_HOST", true);
    config.v6d_host_ip_port = GetEnvOrDefault("V6D_HOST", "");
    config.v6d_port = GetEnvOrDefault("V6D_PORT", "8080");

    std::string mediums_str = GetEnvOrDefault("V6D_MEDIUMS", "mem,disk");
    config.mediums = ParseStringList(mediums_str);

    // 压测参数
    config.num_blocks = GetEnvInt("NUM_BLOCKS", 10000);
    config.block_size = GetEnvInt("BLOCK_SIZE", 128);
    config.num_threads = GetEnvInt("NUM_THREADS", 1);

    // QPS限流配置
    config.target_qps = GetEnvDouble("TARGET_QPS", 1000.0);
    config.enable_qps_limit = GetEnvBool("ENABLE_QPS_LIMIT", true);

    // 压测模式
    config.test_mode = GetEnvOrDefault("TEST_MODE", "full");

    // Batch 配置：先取 BATCH_SIZE 作为三流量对等的默认值，再让 QUERY_BATCH_SIZE 单独覆写 query 路径
    config.batch_size = GetEnvInt("BATCH_SIZE", 50);
    if (config.batch_size < 1) {
        KVCM_LOG_WARN("BATCH_SIZE=%d invalid, clamping to 1", config.batch_size);
        config.batch_size = 1;
    }
    config.query_batch_size = GetEnvInt("QUERY_BATCH_SIZE", config.batch_size);
    if (config.query_batch_size < 1) {
        KVCM_LOG_WARN("QUERY_BATCH_SIZE=%d invalid, clamping to 1", config.query_batch_size);
        config.query_batch_size = 1;
    }

    // Query API 比例
    config.batch_query_ratio = GetEnvDouble("BATCH_QUERY_RATIO", 0.8);
    if (config.batch_query_ratio < 0.0) config.batch_query_ratio = 0.0;
    if (config.batch_query_ratio > 1.0) config.batch_query_ratio = 1.0;

    // 操作比例配置
    config.add_ratio = GetEnvDouble("ADD_RATIO", 0.7);
    config.query_ratio = GetEnvDouble("QUERY_RATIO", 0.2);
    config.delete_ratio = GetEnvDouble("DELETE_RATIO", 0.1);

    // Kmonitor配置
    config.enable_kmonitor = GetEnvBool("ENABLE_KMONITOR", true);
    config.kmonitor_config = GetEnvOrDefault("KMONITOR_CONFIG", "");
    config.report_interval_ms = GetEnvInt("REPORT_INTERVAL_MS", 10000);

    // 结果验证
    config.enable_verification = GetEnvBool("ENABLE_VERIFICATION", true);
    config.strict_verification = GetEnvBool("STRICT_VERIFICATION", false);

    // 周期心跳（与生产节点行为对齐）
    config.enable_periodic_heartbeat = GetEnvBool("ENABLE_PERIODIC_HEARTBEAT", true);
    config.heartbeat_interval_ms = GetEnvInt("HEARTBEAT_INTERVAL_MS", 3000);
    if (config.heartbeat_interval_ms < 100) {
        KVCM_LOG_WARN("HEARTBEAT_INTERVAL_MS=%d too small, clamping to 100ms", config.heartbeat_interval_ms);
        config.heartbeat_interval_ms = 100;
    }

    config.verbose_fail_log = GetEnvBool("VERBOSE_FAIL_LOG", true);
    config.max_fail_log_per_sec = GetEnvInt("MAX_FAIL_LOG_PER_SEC", 50);
    if (config.max_fail_log_per_sec < 0) {
        config.max_fail_log_per_sec = 0;
    }
    config.fail_log_body_max_bytes = GetEnvInt("FAIL_LOG_BODY_MAX_BYTES", 1024);
    if (config.fail_log_body_max_bytes < 0) {
        config.fail_log_body_max_bytes = 0;
    }

    // 打印配置信息
    KVCM_LOG_INFO("=== Benchmark Configuration ===");
    KVCM_LOG_INFO("KVCM Base URL: %s", config.kvcm_base_url.c_str());
    KVCM_LOG_INFO("KVCM Admin URL: %s", config.kvcm_admin_url.c_str());
    KVCM_LOG_INFO("Instance ID: %s", config.instance_id.c_str());
    KVCM_LOG_INFO("Auto Detect Host: %s", config.auto_detect_host ? "true" : "false");
    if (!config.v6d_host_ip_port.empty()) {
        KVCM_LOG_INFO("V6D Host: %s", config.v6d_host_ip_port.c_str());
    }
    KVCM_LOG_INFO("V6D Port: %s", config.v6d_port.c_str());
    KVCM_LOG_INFO("V6D Mediums: [%s]", StringUtil::Join(config.mediums, ", ").c_str());
    KVCM_LOG_INFO("Num Blocks: %d", config.num_blocks);
    KVCM_LOG_INFO("Block Size: %d bytes", config.block_size);
    KVCM_LOG_INFO("Num Threads: %d", config.num_threads);
    KVCM_LOG_INFO("Target QPS: %.2f", config.target_qps);
    KVCM_LOG_INFO("Test Mode: %s", config.test_mode.c_str());
    KVCM_LOG_INFO("Batch Size: %d (add/delete RPC keys), Query Batch Size: %d (query RPC keys)",
                  config.batch_size,
                  config.query_batch_size);
    KVCM_LOG_INFO("Batch Query Ratio (GetBatchCacheLocations): %.2f", config.batch_query_ratio);
    KVCM_LOG_INFO("Enable Verification: %s (strict=%s)",
                  config.enable_verification ? "true" : "false",
                  config.strict_verification ? "true" : "false");
    KVCM_LOG_INFO("Enable Kmonitor: %s", config.enable_kmonitor ? "true" : "false");
    KVCM_LOG_INFO("Periodic Heartbeat: %s, interval %d ms",
                  config.enable_periodic_heartbeat ? "true" : "false",
                  config.heartbeat_interval_ms);
    KVCM_LOG_INFO("Verbose Fail Log: %s, MaxFailLogPerSec: %d, BodyMaxBytes: %d",
                  config.verbose_fail_log ? "true" : "false",
                  config.max_fail_log_per_sec,
                  config.fail_log_body_max_bytes);
    KVCM_LOG_INFO("================================");

    return config;
}

} // namespace v6d_benchmark
} // namespace kv_cache_manager
