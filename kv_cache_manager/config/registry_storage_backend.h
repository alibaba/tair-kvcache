#pragma once

#include <map>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

class RegistryStorageBackend {
public:
    virtual ~RegistryStorageBackend() {}

    virtual ErrorCode Init(const StandardUri &standard_uri) noexcept = 0;
    virtual ErrorCode Load(const std::string &key, std::map<std::string, std::string> &out_value) noexcept = 0;
    virtual ErrorCode Save(const std::string &key, const std::map<std::string, std::string> &value) noexcept = 0;
    virtual ErrorCode Delete(const std::string &key) noexcept = 0;
};

} // namespace kv_cache_manager
