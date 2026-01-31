#include "kv_cache_manager/manager/startup_config_loader.h"

#include <fstream>
#include <sstream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"

namespace kv_cache_manager {

static const std::string DEFAULT_STARTUP_CONFIG = R"(
{
    "storage_config": {
        "type": "file",
        "global_unique_name": "nfs_01",
        "storage_spec": {
            "root_path": "/tmp/nfs/",
            "key_count_per_file": 8
        }
    },
    "instance_group": {
        "name": "default",
        "storage_candidates": [
            "nfs_01"
        ],
        "global_quota_group_name": "default_quota_group",
        "max_instance_count": 100,
        "quota": {
            "capacity": 30000000000,
            "quota_config": [
                {
                    "storage_type": "file",
                    "capacity": 10000000000
                },
                {
                    "storage_type": "hf3fs",
                    "capacity": 10000000000
                },
                {
                    "storage_type": "pace",
                    "capacity": 10000000000
                }
            ]
        },
        "cache_config": {
            "reclaim_strategy": {
                "reclaim_policy": 1,
                "trigger_strategy": {
                    "used_percentage": 0.8
                },
                "delay_before_delete_ms": 1000
            },
            "cache_prefer_strategy": 2,
            "meta_indexer_config": {
                "max_key_count": 1000000,
                "mutex_shard_num": 16,
                "batch_key_size": 16,
                "meta_storage_backend_config": {
                    "storage_type": "local",
                    "storage_uri": ""
                },
                "meta_cache_policy_config": {
                    "type": "LRU",
                    "capacity": 10000,
                    "cache_shard_bits": 0,
                    "high_pri_pool_ratio": 0.0
                }
            }
        },
        "user_data": "{\"description\": \"Default instance group for KV Cache Manager\"}",
        "version": 1
    }
}
)";

bool StartupConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "storage_config", storage_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_group", instance_group_);
    return true;
}

void StartupConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "storage_config", storage_config_);
    Put(writer, "instance_group", instance_group_);
}

bool StartupConfigLoader::Init(std::shared_ptr<RegistryManager> registry_manager) {
    registry_manager_ = registry_manager;
    return true;
}

bool StartupConfigLoader::Load(const std::string &startup_config_file) {
    std::string config_str;
    if (!startup_config_file.empty()) {
        std::ifstream ifs(startup_config_file, std::ios::in | std::ios::binary);
        if (!ifs) {
            KVCM_LOG_ERROR("Read startup config file [%s] failed.", startup_config_file.c_str());
            return false;
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        config_str = oss.str();
    } else {
        config_str = DEFAULT_STARTUP_CONFIG;
    }
    StartupConfig startup_config;
    bool ret = startup_config.FromJsonString(config_str);
    if (!ret) {
        KVCM_LOG_ERROR("Parse startup config failed, content=[%s]", config_str.c_str());
        return false;
    }
    RequestContext empty_context("system_startup");
    auto storage_config = startup_config.storage_config();
    auto ec = registry_manager_->AddStorage(&empty_context, storage_config);
    if (ec != ErrorCode::EC_OK && ec != ErrorCode::EC_EXIST) {
        KVCM_LOG_ERROR("Register storage failed, storage_uniq_name=[%s], error_code=[%d]",
                       storage_config.global_unique_name().c_str(),
                       ec);
        return false;
    }
    auto instance_group = startup_config.instance_group();
    ec = registry_manager_->CreateInstanceGroup(&empty_context, instance_group);
    if (ec != ErrorCode::EC_OK && ec != ErrorCode::EC_EXIST) {
        KVCM_LOG_ERROR("Create instance group failed, instance_group_name=[%s], error_code=[%d]",
                       instance_group.name().c_str(),
                       ec);
        return false;
    }
    return true;
}

} // namespace kv_cache_manager
