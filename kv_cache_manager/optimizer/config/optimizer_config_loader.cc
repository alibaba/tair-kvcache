#include "kv_cache_manager/optimizer/config/optimizer_config_loader.h"

#include <fstream>
#include <sstream>

#include "kv_cache_manager/common/logger.h"
namespace kv_cache_manager {

bool OptimizerConfigLoader::Load(const std::string &config_file) {
    std::string config_str;
    if (!config_file.empty()) {
        std::ifstream ifs(config_file, std::ios::in | std::ios::binary);
        if (!ifs) {
            KVCM_LOG_ERROR("Read optimizer config file [%s] failed.", config_file.c_str());
            return false;
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        config_str = oss.str();
    } else {
        KVCM_LOG_ERROR("Optimizer config file path is empty.");
        return false;
    }
    bool ret = config_.FromJsonString(config_str);
    if (!ret) {
        KVCM_LOG_ERROR("Parse optimizer config failed, content=[%s]", config_str.c_str());
        return false;
    }
    return true;
}

} // namespace kv_cache_manager