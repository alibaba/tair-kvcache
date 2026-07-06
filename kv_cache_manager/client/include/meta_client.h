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

    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                        QueryType query_type,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        const std::vector<std::string> &location_spec_names) {
        auto [ec, result] = MatchLocation(
            trace_id, query_type, keys, tokens, block_mask, location_spec_names, MatchLocationOptions{});
        return {ec, std::move(result.locations)};
    }

    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                        QueryType query_type,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        int32_t sw_size,
                                                        const std::vector<std::string> &location_spec_names) {
        auto [ec, result] = MatchLocation(trace_id,
                                          query_type,
                                          keys,
                                          tokens,
                                          block_mask,
                                          location_spec_names,
                                          MatchLocationOptions::WithSlideWindowSize(sw_size));
        return {ec, std::move(result.locations)};
    }

    // MatchLocationOptions carries optional query controls such as slide-window
    // size and checksum collection for later LoadKvCaches verification.
    virtual std::pair<ClientErrorCode, MatchLocationResult>
    MatchLocation(const std::string &trace_id,
                  QueryType query_type,
                  const std::vector<int64_t> &keys,
                  const std::vector<int64_t> &tokens,
                  const BlockMask &block_mask,
                  const std::vector<std::string> &location_spec_names,
                  const MatchLocationOptions &options) = 0;

    virtual std::pair<ClientErrorCode, WriteLocation>
    StartWrite(const std::string &trace_id,
               const std::vector<int64_t> &keys,
               const std::vector<int64_t> &tokens,
               const std::vector<std::string> &location_spec_group_names,
               int64_t write_timeout_seconds) = 0;
    ClientErrorCode FinishWrite(const std::string &trace_id,
                                const std::string &write_session_id,
                                const BlockMask &success_block,
                                const Locations &locations) {
        return FinishWrite(trace_id, write_session_id, success_block, locations, FinishWriteOptions{});
    }

    // FinishWriteOptions::checksums parallels the keys captured at StartWrite
    // (full batch, 0 means "not reported").
    virtual ClientErrorCode FinishWrite(const std::string &trace_id,
                                        const std::string &write_session_id,
                                        const BlockMask &success_block,
                                        const Locations &locations,
                                        const FinishWriteOptions &options) = 0;

    std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &trace_id,
                                                const std::vector<int64_t> &keys,
                                                const std::vector<int64_t> &tokens,
                                                const BlockMask &block_mask,
                                                int32_t detail_level) {
        return MatchMeta(trace_id, keys, tokens, block_mask, detail_level, MatchMetaOptions{});
    }

    virtual std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &trace_id,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        int32_t detail_level,
                                                        const MatchMetaOptions &options) = 0;

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
};
} // namespace kv_cache_manager
