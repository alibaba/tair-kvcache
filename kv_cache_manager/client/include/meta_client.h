#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

class MetaClient {
public:
    virtual ~MetaClient() = default;
    static std::unique_ptr<MetaClient> Create(const std::string &config, const InitParams &init_params);

    // Keep the complete legacy virtual sequence unchanged for binary clients.
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

    virtual std::pair<ClientErrorCode, int64_t> MatchLocationLen(const std::string &trace_id,
                                                                 QueryType query_type,
                                                                 const std::vector<int64_t> &keys,
                                                                 const std::vector<int64_t> &tokens,
                                                                 int32_t sw_size) = 0;

    virtual ClientErrorCode RemoveCache(const std::string &trace_id,
                                        const std::vector<int64_t> &keys,
                                        const std::vector<int64_t> &tokens,
                                        const BlockMask &block_mask) = 0;

    virtual const std::string &GetStorageConfig() const = 0;

protected:
    MetaClient() = default;
    virtual ClientErrorCode Init(const std::string &config, const InitParams &init_params) = 0;
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

    virtual std::pair<ClientErrorCode, int64_t> MatchLocationLen(const std::string &trace_id,
                                                                 QueryType query_type,
                                                                 const std::vector<int64_t> &keys,
                                                                 const std::vector<int64_t> &tokens,
                                                                 const MatchLocationLenOptions &options) {
        return MatchLocationLen(trace_id, query_type, keys, tokens, options.sw_size);
    }
};
} // namespace kv_cache_manager
