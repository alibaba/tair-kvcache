#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

class ManagerClient {
public:
    virtual ~ManagerClient() = default;
    static std::unique_ptr<ManagerClient> Create(const std::string &config, InitParams &init_params);

    // Keep the complete legacy virtual sequence unchanged for binary clients.
    // for meta client
    virtual std::pair<ClientErrorCode, Locations>
    MatchLocation(const std::string &trace_id,
                  QueryType query_type,
                  const std::vector<int64_t> &keys,
                  const std::vector<int64_t> &tokens,
                  const BlockMask &block_mask,
                  int32_t sw_size,
                  const std::vector<std::string> &location_spec_names) = 0;

    virtual std::pair<ClientErrorCode, WriteLocation>
    StartWrite(const std::string &trace_id,
               const std::vector<int64_t> &keys,
               const std::vector<int64_t> &tokens,
               const std::vector<std::string> &location_spec_group_names,
               int64_t write_timeout_seconds) = 0;
    virtual ClientErrorCode FinishWrite(const std::string &trace_id,
                                        const std::string &write_session_id,
                                        const BlockMask &success_block,
                                        const Locations &locations) = 0;

    virtual std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &trace_id,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        int32_t detail_level) = 0;

    virtual ClientErrorCode RemoveCache(const std::string &trace_id,
                                        const std::vector<int64_t> &keys,
                                        const std::vector<int64_t> &tokens,
                                        const BlockMask &block_mask) = 0;

    // for transfer client
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers) = 0;

protected:
    ManagerClient() = default;
    virtual ClientErrorCode Init(const std::string &config, InitParams &init_params) = 0;
    virtual void Shutdown() = 0;

public:
    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                        QueryType query_type,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        const std::vector<std::string> &location_spec_names) {
        return MatchLocation(trace_id, query_type, keys, tokens, block_mask, -1, location_spec_names);
    }

    // New virtuals are deliberately appended after every released virtual slot.
    // Default implementations keep old-only downstream subclasses concrete.
    virtual std::pair<ClientErrorCode, MatchLocationResult>
    MatchLocation(const std::string &trace_id,
                  QueryType query_type,
                  const std::vector<int64_t> &keys,
                  const std::vector<int64_t> &tokens,
                  const BlockMask &block_mask,
                  const std::vector<std::string> &location_spec_names,
                  const MatchLocationOptions &options) {
        if (options.include_checksums) {
            return {ER_CHECKSUM_UNAVAILABLE, {}};
        }
        auto [ec, locations] =
            MatchLocation(trace_id, query_type, keys, tokens, block_mask, options.sw_size, location_spec_names);
        MatchLocationResult result;
        result.locations = std::move(locations);
        return {ec, std::move(result)};
    }

    virtual ClientErrorCode FinishWrite(const std::string &trace_id,
                                        const std::string &write_session_id,
                                        const BlockMask &success_block,
                                        const Locations &locations,
                                        const FinishWriteOptions &options) {
        if (!options.checksum_batches.empty()) {
            return ER_CHECKSUM_UNAVAILABLE;
        }
        return FinishWrite(trace_id, write_session_id, success_block, locations);
    }

    virtual std::pair<ClientErrorCode, MatchMetaResult> MatchMeta(const std::string &trace_id,
                                                                  const std::vector<int64_t> &keys,
                                                                  const std::vector<int64_t> &tokens,
                                                                  const BlockMask &block_mask,
                                                                  const MatchMetaOptions &options) {
        if (options.include_checksums) {
            return {ER_CHECKSUM_UNAVAILABLE, {}};
        }
        auto [ec, metas] = MatchMeta(trace_id, keys, tokens, block_mask, options.detail_level);
        MatchMetaResult result;
        result.metas = std::move(metas);
        return {ec, std::move(result)};
    }

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
        return LoadKvCaches(uri_str_vec, block_buffers);
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
        auto [ec, uris] = SaveKvCaches(uri_str_vec, block_buffers);
        SaveKvCachesResult result;
        result.uri_str_vec = std::move(uris);
        return {ec, std::move(result)};
    }
};

} // namespace kv_cache_manager
