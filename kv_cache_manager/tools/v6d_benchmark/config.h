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
    int query_batch_size = 50;      // QUERY_BATCH_SIZE

    // 操作比例配置 (仅full模式)
    double add_ratio = 0.7;    // ADD_RATIO
    double query_ratio = 0.2;  // QUERY_RATIO
    double delete_ratio = 0.1; // DELETE_RATIO

    // Kmonitor配置
    bool enable_kmonitor = true;    // ENABLE_KMONITOR
    std::string kmonitor_config;    // KMONITOR_CONFIG
    int report_interval_ms = 10000; // REPORT_INTERVAL_MS

    // 结果验证
    bool enable_verification = true; // ENABLE_VERIFICATION
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
    config.query_batch_size = GetEnvInt("QUERY_BATCH_SIZE", 50);

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
    KVCM_LOG_INFO("Enable Verification: %s", config.enable_verification ? "true" : "false");
    KVCM_LOG_INFO("Enable Kmonitor: %s", config.enable_kmonitor ? "true" : "false");
    KVCM_LOG_INFO("================================");

    return config;
}

} // namespace v6d_benchmark
} // namespace kv_cache_manager
