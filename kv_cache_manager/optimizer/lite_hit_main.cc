#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_lite_hit_config.h"
#include "kv_cache_manager/optimizer/manager/lite_hit_offline_runner.h"

int main(int argc, char *argv[]) {
    kv_cache_manager::LoggerBroker::InitLogger("", false);
    kv_cache_manager::LoggerBroker::SetLogLevel(kv_cache_manager::Logger::LEVEL_INFO);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <lite_hit_config.json>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Replays a standard-format trace through per-instance LiteHit lanes and" << std::endl;
        std::cerr << "atomically publishes one capacity-independent facts CSV:" << std::endl;
        std::cerr << "  <output_result_path>/litehit_facts.csv" << std::endl;
        std::cerr << "Apply capacities afterwards with lite_hit_facts_query_main." << std::endl;
        std::cerr << std::endl;
        std::cerr << "Config fields: trace_file_path, output_result_path (directory)," << std::endl;
        std::cerr << "instance_groups (online OptimizerInstanceGroup list; enable_prefix_hash" << std::endl;
        std::cerr << "lives here), instances (OptimizerInstanceInfo list), override_instance_id," << std::endl;
        std::cerr << "block_size (trace granularity in tokens, default 256; every instance's" << std::endl;
        std::cerr << "block_size must be a multiple of it and requests are re-blocked per lane)," << std::endl;
        std::cerr << "fanout_all_instances (replay every request into every instance, e.g. to" << std::endl;
        std::cerr << "sweep several block sizes in one run), pipeline_worker_count." << std::endl;
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }

    const std::string config_file_path = argv[1];
    KVCM_LOG_INFO("Loading LiteHit configuration from file: %s", config_file_path.c_str());

    std::ifstream ifs(config_file_path, std::ios::in | std::ios::binary);
    if (!ifs) {
        KVCM_LOG_ERROR("Failed to read LiteHit config file: %s", config_file_path.c_str());
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();

    kv_cache_manager::OptimizerLiteHitConfig config;
    if (!config.FromJsonString(oss.str())) {
        KVCM_LOG_ERROR("Failed to parse LiteHit config file: %s", config_file_path.c_str());
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }

    kv_cache_manager::LiteHitOfflineRunner runner(config);
    const bool ok = runner.Run();

    kv_cache_manager::LoggerBroker::DestroyLogger();
    return ok ? 0 : 1;
}
