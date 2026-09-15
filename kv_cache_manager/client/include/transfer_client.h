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
    static std::unique_ptr<TransferClient>
    Create(const std::string &client_config,
           const InitParams &init_params,
           const SharedMemoryRegistration &shared_memory_registration);

    // Keep these legacy virtual methods, including their declaration order, ABI
    // compatible with released clients. New virtual methods are appended below.
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         std::shared_ptr<TransferTraceInfo> trace_info = nullptr) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec>
    SaveKvCaches(const UriStrVec &uri_str_vec,
                 const BlockBuffers &block_buffers,
                 std::shared_ptr<TransferTraceInfo> trace_info = nullptr) = 0;

    // An empty brace was a valid spelling for the legacy optional trace_info.
    // The options overload below would otherwise make that source-compatible
    // call ambiguous, so give nullptr (and therefore `{}`) an exact forwarding
    // target without adding another virtual slot.
    ClientErrorCode
    LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers, std::nullptr_t) {
        return LoadKvCaches(uri_str_vec, block_buffers, std::shared_ptr<TransferTraceInfo>{});
    }
    std::pair<ClientErrorCode, UriStrVec>
    SaveKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers, std::nullptr_t) {
        return SaveKvCaches(uri_str_vec, block_buffers, std::shared_ptr<TransferTraceInfo>{});
    }

protected:
    TransferClient() = default;
    virtual ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) = 0;

public:
    // Appended after every legacy virtual slot. The fallback keeps downstream
    // subclasses that only implement the old interface source-compatible; they
    // can serve default operations but explicitly reject checksum work.
    virtual ClientErrorCode
    LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers, const LoadKvCachesOptions &options) {
        const bool verify_requested = options.verify_checksums || !options.expected_checksums.empty() ||
                                      !options.expected_checksum_present.empty();
        if ((!options.expected_checksum_present.empty() &&
             options.expected_checksum_present.size() != options.expected_checksums.size()) ||
            (verify_requested && options.expected_checksums.size() != block_buffers.size())) {
            return ER_CHECKSUM_MISMATCH;
        }
        bool has_expected_checksum = false;
        for (std::size_t i = 0; i < options.expected_checksums.size(); ++i) {
            if (options.expected_checksum_present.empty() || options.expected_checksum_present[i]) {
                has_expected_checksum = true;
                break;
            }
        }
        if (has_expected_checksum) {
            return ER_CHECKSUM_UNAVAILABLE;
        }
        return LoadKvCaches(uri_str_vec, block_buffers, options.trace_info);
    }

    virtual std::pair<ClientErrorCode, SaveKvCachesResult>
    SaveKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers, const SaveKvCachesOptions &options) {
        const bool verify_requested = options.verify_caller_checksums || !options.expected_checksums.empty();
        if (verify_requested && options.expected_checksums.size() != block_buffers.size()) {
            return {ER_CHECKSUM_MISMATCH, {}};
        }
        if (options.include_checksums || verify_requested) {
            return {ER_CHECKSUM_UNAVAILABLE, {}};
        }
        auto [ec, uris] = SaveKvCaches(uri_str_vec, block_buffers, options.trace_info);
        SaveKvCachesResult result;
        result.uri_str_vec = std::move(uris);
        return {ec, std::move(result)};
    }
};
} // namespace kv_cache_manager
