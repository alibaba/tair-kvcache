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

    // out_checksums (optional): filled with the per-block checksum the server stored
    // at write time, parallel to the returned Locations. Pass nullptr to keep the
    // legacy behavior (no verification possible on read).
    virtual std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                                QueryType query_type,
                                                                const std::vector<int64_t> &keys,
                                                                const std::vector<int64_t> &tokens,
                                                                const BlockMask &block_mask,
                                                                int32_t sw_size,
                                                                const std::vector<std::string> &location_spec_names,
                                                                std::vector<int64_t> *out_checksums = nullptr) = 0;

    virtual std::pair<ClientErrorCode, WriteLocation>
    StartWrite(const std::string &trace_id,
               const std::vector<int64_t> &keys,
               const std::vector<int64_t> &tokens,
               const std::vector<std::string> &location_spec_group_names,
               int64_t write_timeout_seconds) = 0;
    // checksums parallels the keys captured at StartWrite (full batch, 0 means "not
    // reported"). MetaClientImpl fills each one into the corresponding
    // FinishWriteCacheRequest.locations[i].checksum field on the wire.
    virtual ClientErrorCode FinishWrite(const std::string &trace_id,
                                        const std::string &write_session_id,
                                        const BlockMask &success_block,
                                        const Locations &locations,
                                        const std::vector<int64_t> &checksums = {}) = 0;

    virtual std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &trace_id,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        int32_t detail_level,
                                                        std::vector<int64_t> *out_checksums = nullptr) = 0;

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