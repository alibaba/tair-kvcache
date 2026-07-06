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

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) {
        return LoadKvCaches(uri_str_vec, block_buffers, LoadKvCachesOptions{});
    }

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                 const BlockBuffers &block_buffers,
                                 std::shared_ptr<TransferTraceInfo> trace_info) {
        auto options = LoadKvCachesOptions::WithTraceInfo(trace_info);
        return LoadKvCaches(uri_str_vec, block_buffers, options);
    }

    // LoadKvCachesOptions carries optional trace_info and expected checksums for
    // read-side verification. expected_checksums must be 1:1 with block_buffers; a
    // zero entry is treated as "no checksum for this block".
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         const LoadKvCachesOptions &options) = 0;

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers) {
        auto [ec, result] = SaveKvCaches(uri_str_vec, block_buffers, SaveKvCachesOptions{});
        return {ec, std::move(result.uri_str_vec)};
    }

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers,
                                                       std::shared_ptr<TransferTraceInfo> trace_info) {
        auto options = SaveKvCachesOptions::WithTraceInfo(trace_info);
        auto [ec, result] = SaveKvCaches(uri_str_vec, block_buffers, options);
        return {ec, std::move(result.uri_str_vec)};
    }

    // SaveKvCachesOptions carries optional trace_info and checksum collection
    // controls. On success, SaveKvCachesResult::checksums matches block_buffers.size().
    virtual std::pair<ClientErrorCode, SaveKvCachesResult> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                        const BlockBuffers &block_buffers,
                                                                        const SaveKvCachesOptions &options) = 0;

protected:
    TransferClient() = default;
    virtual ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) = 0;
};
} // namespace kv_cache_manager
