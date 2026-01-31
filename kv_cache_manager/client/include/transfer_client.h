#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

class TransferClient {
public:
    virtual ~TransferClient() = default;
    static std::unique_ptr<TransferClient> Create(const std::string &client_config, const InitParams &init_params);

    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers) = 0;

protected:
    TransferClient() = default;
    virtual ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) = 0;
};
} // namespace kv_cache_manager