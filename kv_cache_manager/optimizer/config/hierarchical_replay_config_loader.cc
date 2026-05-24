#include "kv_cache_manager/optimizer/config/hierarchical_replay_config_loader.h"

#include <fstream>
#include <sstream>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

bool HierarchicalReplayConfigLoader::Load(const std::string &config_file) {
    if (config_file.empty()) {
        KVCM_LOG_ERROR("Hierarchical replay config file path is empty.");
        return false;
    }

    std::ifstream ifs(config_file, std::ios::in | std::ios::binary);
    if (!ifs) {
        KVCM_LOG_ERROR("Read hierarchical replay config file [%s] failed.", config_file.c_str());
        return false;
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    const std::string config_str = oss.str();
    if (!config_.FromJsonString(config_str)) {
        KVCM_LOG_ERROR("Parse hierarchical replay config failed, content=[%s]", config_str.c_str());
        return false;
    }
    return true;
}

} // namespace kv_cache_manager
