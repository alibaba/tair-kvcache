#pragma once

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/config/instance_info.h"

namespace kv_cache_manager {

class Stub {
public:
    using KeyType = int64_t;
    using KeyVector = std::vector<KeyType>;
    using TokenIds = int64_t;
    using TokenIdsVector = std::vector<KeyType>;
    using LocationSpecInfoMap = std::map<std::string, int64_t>;
    using LocationSpecGroups = std::map<std::string, std::vector<std::string>>;

    virtual ~Stub() = default;

    virtual ClientErrorCode AddConnection(const std::string &address, uint32_t connection_timeout) = 0;

    virtual std::pair<ClientErrorCode, std::string>
    RegisterInstance(const std::string &trace_id,
                     const std::string instance_group,
                     const std::string &instance_id,
                     int32_t block_size,
                     const LocationSpecInfoMap &location_spec_infos,
                     const ModelDeployment &model_deployment,
                     const LocationSpecGroups &location_spec_groups) = 0;
    virtual std::pair<ClientErrorCode, InstanceInfo> GetInstanceInfo(const std::string &trace_id,
                                                                     const std::string &instance_id) = 0;

    // TODO : remove this
    virtual std::pair<ClientErrorCode, Metas> GetCacheMeta(const std::string &trace_id,
                                                           const std::string &instance_id,
                                                           const KeyVector &keys,
                                                           const TokenIdsVector &tokens,
                                                           const BlockMask &block_mask,
                                                           int32_t detail_level) = 0;

    virtual std::pair<ClientErrorCode, Locations>
    GetCacheLocation(const std::string &trace_id,
                     const std::string &instance_id,
                     QueryType query_type,
                     const KeyVector &keys,
                     const TokenIdsVector &tokens,
                     const BlockMask &block_mask,
                     int32_t sw_size,
                     const std::vector<std::string> &location_spec_names) = 0;

    virtual std::pair<ClientErrorCode, WriteLocation>
    StartWriteCache(const std::string &trace_id,
                    const std::string &instance_id,
                    const KeyVector &keys,
                    const TokenIdsVector &tokens,
                    const std::vector<std::string> &location_spec_group_names,
                    int64_t write_timeout_seconds) = 0;
    virtual ClientErrorCode FinishWriteCache(const std::string &trace_id,
                                             const std::string &instance_id,
                                             const std::string write_session_id,
                                             const BlockMask &success_block,
                                             const Locations &locations) = 0;

    virtual ClientErrorCode RemoveCache(const std::string &trace_id,
                                        const std::string &instance_id,
                                        const KeyVector &keys,
                                        const TokenIdsVector &tokens,
                                        const BlockMask &block_mask) = 0;
    virtual bool TrimCache() = 0;
};

} // namespace kv_cache_manager