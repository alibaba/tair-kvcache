#pragma once

#include <deque>
#include <shared_mutex>

#include "kv_cache_manager/client/src/internal/stub/stub.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"

namespace kv_cache_manager {

class GrpcStub : public Stub {
public:
    GrpcStub() = default;
    GrpcStub(uint32_t retry_time, uint32_t call_timeout);
    ClientErrorCode AddConnection(const std::string &address, uint32_t connection_timeout) override;

    std::pair<ClientErrorCode, std::string> RegisterInstance(const std::string &trace_id,
                                                             const std::string instance_group,
                                                             const std::string &instance_id,
                                                             int32_t block_size,
                                                             const LocationSpecInfoMap &location_spec_infos,
                                                             const ModelDeployment &model_deployment,
                                                             const LocationSpecGroups &location_spec_groups) override;

    std::pair<ClientErrorCode, InstanceInfo> GetInstanceInfo(const std::string &trace_id,
                                                             const std::string &instance_id) override;

    std::pair<ClientErrorCode, Metas> GetCacheMeta(const std::string &trace_id,
                                                   const std::string &instance_id,
                                                   const KeyVector &keys,
                                                   const TokenIdsVector &tokens,
                                                   const BlockMask &block_mask,
                                                   int32_t detail_level) override;

    std::pair<ClientErrorCode, Locations>
    GetCacheLocation(const std::string &trace_id,
                     const std::string &instance_id,
                     QueryType query_type,
                     const KeyVector &keys,
                     const TokenIdsVector &tokens,
                     const BlockMask &block_mask,
                     int32_t sw_size,
                     const std::vector<std::string> &location_spec_names) override;

    std::pair<ClientErrorCode, WriteLocation> StartWriteCache(const std::string &trace_id,
                                                              const std::string &instance_id,
                                                              const KeyVector &keys,
                                                              const TokenIdsVector &tokens,
                                                              const std::vector<std::string> &location_spec_group_names,
                                                              int64_t write_timeout_seconds) override;

    ClientErrorCode FinishWriteCache(const std::string &trace_id,
                                     const std::string &instance_id,
                                     const std::string write_session_id,
                                     const BlockMask &success_block,
                                     const Locations &locations) override;

    ClientErrorCode RemoveCache(const std::string &trace_id,
                                const std::string &instance_id,
                                const KeyVector &keys,
                                const TokenIdsVector &tokens,
                                const BlockMask &block_mask) override;

    bool TrimCache() override;

private:
    std::shared_ptr<proto::meta::MetaService::Stub> GetStub() const;
    uint32_t retry_time_ = 3;
    uint32_t call_timeout_ = 100;
    mutable std::shared_mutex stubs_mutex_;
    // TODO : add stub proxy to manage stub
    std::deque<std::shared_ptr<proto::meta::MetaService::Stub>> stubs_;
    mutable size_t next_stub_ = 0;
};

} // namespace kv_cache_manager