#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/config/optimizer_config.h"
namespace kv_cache_manager {
class OptimizerConfigLoader {
public:
    bool Load(const std::string &config_file);
    const OptimizerConfig &get_config() const { return config_; }

private:
    OptimizerConfig config_;
};
} // namespace kv_cache_manager