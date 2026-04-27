#pragma once

#include <string>

#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// VineyardStorageSpec represents the configuration for one V6D cluster.
// Each cluster is registered as a single VineyardBackend instance; individual
// V6D nodes within the cluster register themselves dynamically via ReportEvent.
class VineyardStorageSpec : public StorageSpec {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const override;
    std::string ToString() const override;

    const std::string &cluster_name() const { return cluster_name_; }
    void set_cluster_name(const std::string &cluster_name) { cluster_name_ = cluster_name; }

private:
    // Human-readable cluster identifier, used for logging only.
    std::string cluster_name_;
};

} // namespace kv_cache_manager
