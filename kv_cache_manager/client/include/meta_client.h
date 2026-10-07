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

    virtual std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                                QueryType query_type,
                                                                const std::vector<int64_t> &keys,
                                                                const std::vector<int64_t> &tokens,
                                                                const BlockMask &block_mask,
                                                                int32_t sw_size,
                                                                const std::vector<std::string> &location_spec_names,
                                                                std::vector<ClientReplicationHint> &out_hints) = 0;

    virtual std::pair<ClientErrorCode, WriteLocation>
    StartWrite(const std::string &trace_id,
               const std::vector<int64_t> &keys,
               const std::vector<int64_t> &tokens,
               const std::vector<std::string> &location_spec_group_names,
               int64_t write_timeout_seconds,
               bool is_replication = false) = 0;
    // Explicit-target form used by replication writes. The default preserves
    // source compatibility for test/custom clients while MetaClientImpl sends
    // the target through the wire protocol.
    virtual std::pair<ClientErrorCode, WriteLocation>
    StartReplicationWrite(const std::string &trace_id,
                          const std::vector<int64_t> &keys,
                          const std::vector<std::string> &location_spec_group_names,
                          int64_t write_timeout_seconds,
                          const std::string &target_node_id) {
        return StartWrite(trace_id, keys, {}, location_spec_group_names, write_timeout_seconds, true);
    }
    virtual ClientErrorCode ReplicateCache(const std::string &trace_id,
                                           const ClientReplicationHint &hint,
                                           int32_t write_timeout_seconds) {
        return ER_SERVICE_UNSUPPORTED;
    }
    virtual std::vector<ClientReplicationRpcResult>
    ReplicateCaches(const std::string &trace_id,
                    const std::vector<ClientReplicationHint> &hints,
                    int32_t write_timeout_seconds) {
        std::vector<ClientReplicationRpcResult> results;
        results.reserve(hints.size());
        for (const auto &hint : hints) {
            results.push_back({ReplicateCache(trace_id, hint, write_timeout_seconds), false});
        }
        return results;
    }
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

    virtual std::string GetCallerNode() const = 0;

protected:
    MetaClient() = default;
    virtual ClientErrorCode Init(const std::string &config, const InitParams &init_params) = 0;
    virtual void Shutdown() = 0;
};
} // namespace kv_cache_manager
