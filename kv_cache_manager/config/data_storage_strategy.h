#pragma once

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class DataStorageStrategy : public Jsonizable {
public:
    bool PerferMemoryStorage() { return true; }

    ~DataStorageStrategy() override;

public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override { return false; }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {}
};

} // namespace kv_cache_manager