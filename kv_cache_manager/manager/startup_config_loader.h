#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class StartupConfig : public Jsonizable {

public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    // Getter methods (Google style)
    const StorageConfig &storage_config() const { return storage_config_; }
    const InstanceGroup &instance_group() const { return instance_group_; }

    // Setter methods (Google style)
    void set_storage_config(const StorageConfig &storage_config) { storage_config_ = storage_config; }
    void set_instance_group(const InstanceGroup &instance_group) { instance_group_ = instance_group; }

private:
    StorageConfig storage_config_;
    InstanceGroup instance_group_;
};

class RegistryManager;

class StartupConfigLoader {
public:
    bool Init(std::shared_ptr<RegistryManager> registry_manager);
    bool Load(const std::string &startup_config_file);

private:
    std::shared_ptr<RegistryManager> registry_manager_;
};

} // namespace kv_cache_manager