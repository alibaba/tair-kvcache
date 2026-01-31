#include <iostream>
#include <string>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/analysis/result_analysis.h"
#include "kv_cache_manager/optimizer/config/optimizer_config_loader.h"
#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"

int main(int argc, char *argv[]) {
    // 初始化KVCM日志系统
    kv_cache_manager::LoggerBroker::InitLogger(""); // 使用默认配置
    kv_cache_manager::LoggerBroker::SetLogLevel(kv_cache_manager::Logger::LEVEL_INFO);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <optimizer_config.json>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Optimizer supports three input formats:" << std::endl;
        std::cerr << "  - qwen_bailian: Qwen Bailian dataset format" << std::endl;
        std::cerr << "  - publisher_log: KVCacheManager Event Publisher log format" << std::endl;
        std::cerr << "  - optimizer_schema: Standard format (from trace_anonymous_tool)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Specify trace_type in the config file." << std::endl;
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }

    std::string config_file_path = argv[1];

    KVCM_LOG_INFO("Loading optimizer configuration from file: %s", config_file_path.c_str());

    // 从配置文件中读取参数
    kv_cache_manager::OptimizerConfigLoader config_loader;
    if (!config_loader.Load(config_file_path)) {
        KVCM_LOG_ERROR("Failed to load optimizer configuration from file: %s", config_file_path.c_str());
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }

    const auto &config = config_loader.get_config();
    std::string trace_file_path = config.trace_file_path();
    std::string output_result_path = config.output_result_path();

    // 使用配置文件中指定的 trace_type
    // 支持三种输入格式：qwen_bailian, publisher_log, optimizer_schema

    // 创建优化器管理器
    auto optimizer = std::make_unique<kv_cache_manager::OptimizerManager>(config);
    // 初始化优化器管理器
    if (!optimizer->Init()) {
        KVCM_LOG_ERROR("Failed to initialize optimizer manager.");
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }

    // 运行优化器分析
    KVCM_LOG_INFO("Starting optimization on trace file: %s", trace_file_path.c_str());
    optimizer->DirectRun();

    // 获取结果并分析
    optimizer->AnalyzeResults();

    KVCM_LOG_INFO("Optimization analysis completed. Results saved to: %s", output_result_path.c_str());

    // 销毁日志系统
    kv_cache_manager::LoggerBroker::DestroyLogger();
    return 0;
}
