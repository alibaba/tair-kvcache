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

    // for meta client
    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                        QueryType query_type,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        const std::vector<std::string> &location_spec_names) {
        return MatchLocation(trace_id,
                             query_type,
                             keys,
                             tokens,
                             block_mask,
                             location_spec_names,
                             MatchLocationOptions{});
    }

    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                        QueryType query_type,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        int32_t sw_size,
                                                        const std::vector<std::string> &location_spec_names) {
        return MatchLocation(trace_id,
                             query_type,
                             keys,
                             tokens,
                             block_mask,
                             location_spec_names,
                             MatchLocationOptions::WithSlideWindowSize(sw_size));
    }

    // MatchLocationOptions carries optional query controls such as slide-window
    // size and checksum collection for later LoadKvCaches verification.
    virtual std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
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
    // (length == keys.size, full batch including failed positions which carry 0).
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

    virtual ClientErrorCode RemoveCache(const std::string &trace_id,
                                        const std::vector<int64_t> &keys,
                                        const std::vector<int64_t> &tokens,
                                        const BlockMask &block_mask) = 0;

    // for transfer client
    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) {
        return LoadKvCaches(uri_str_vec, block_buffers, LoadKvCachesOptions{});
    }

    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         const LoadKvCachesOptions &options) = 0;

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers) {
        return SaveKvCaches(uri_str_vec, block_buffers, SaveKvCachesOptions{});
    }

    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers,
                                                               const SaveKvCachesOptions &options) = 0;

protected:
    ManagerClient() = default;
    virtual ClientErrorCode Init(const std::string &config, InitParams &init_params) = 0;
    virtual void Shutdown() = 0;
};

} // namespace kv_cache_manager
