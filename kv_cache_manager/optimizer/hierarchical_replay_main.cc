#include <iostream>
#include <memory>
#include <string>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/hierarchical_replay_config_loader.h"
#include "kv_cache_manager/optimizer/manager/hierarchical_replay_manager.h"

int main(int argc, char *argv[]) {
    kv_cache_manager::LoggerBroker::InitLogger("", false);
    kv_cache_manager::LoggerBroker::SetLogLevel(kv_cache_manager::Logger::LEVEL_INFO);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <hierarchical_replay_config.json>" << std::endl;
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }

    const std::string config_file_path = argv[1];
    kv_cache_manager::HierarchicalReplayConfigLoader config_loader;
    if (!config_loader.Load(config_file_path)) {
        KVCM_LOG_ERROR("Failed to load hierarchical replay config: %s", config_file_path.c_str());
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }

    auto replay_manager = std::make_unique<kv_cache_manager::HierarchicalReplayManager>(config_loader.get_config());
    if (!replay_manager->Init()) {
        KVCM_LOG_ERROR("Failed to initialize hierarchical replay manager.");
        kv_cache_manager::LoggerBroker::DestroyLogger();
        return 1;
    }

    replay_manager->DirectRun();
    replay_manager->AnalyzeResults();

    kv_cache_manager::LoggerBroker::DestroyLogger();
    return 0;
}
