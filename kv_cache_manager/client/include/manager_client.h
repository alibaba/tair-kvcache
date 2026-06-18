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
    // out_checksums (optional): when non-null, filled with the per-block checksum the
    // server stored at write time, parallel to the returned Locations. The caller is
    // expected to pass this same vector to LoadKvCaches's expected_checksums to enable
    // read-side verification.
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
    // checksums parallels the keys captured at StartWrite (length == keys.size, full
    // batch including failed positions which carry 0). Pass an empty vector when the
    // client did not enable meta_checksum; server then keeps existing CacheLocation
    // checksums untouched.
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

    virtual ClientErrorCode RemoveCache(const std::string &trace_id,
                                        const std::vector<int64_t> &keys,
                                        const std::vector<int64_t> &tokens,
                                        const BlockMask &block_mask) = 0;

    // for transfer client
    // out_checksums is forwarded to TransferClient::SaveKvCaches; when non-null the SDK
    // writes the computed checksums into it and the caller passes the same vector to
    // FinishWrite later.
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         const std::vector<int64_t> *expected_checksums = nullptr) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers,
                                                               std::vector<int64_t> *out_checksums = nullptr) = 0;

protected:
    ManagerClient() = default;
    virtual ClientErrorCode Init(const std::string &config, InitParams &init_params) = 0;
    virtual void Shutdown() = 0;
};

} // namespace kv_cache_manager