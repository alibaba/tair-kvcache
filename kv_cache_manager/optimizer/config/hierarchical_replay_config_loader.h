#pragma once

#include <string>

#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"

namespace kv_cache_manager {

class HierarchicalReplayConfigLoader {
public:
    bool Load(const std::string &config_file);
    const HierarchicalReplayConfig &get_config() const { return config_; }

private:
    HierarchicalReplayConfig config_;
};

} // namespace kv_cache_manager
