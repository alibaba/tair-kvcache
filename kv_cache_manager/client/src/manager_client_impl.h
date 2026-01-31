#pragma once

#include "kv_cache_manager/client/include/manager_client.h"

namespace kv_cache_manager {
class MetaClient;
class TransferClient;
class ManagerClientImpl : public ManagerClient {
public:
    ManagerClientImpl();
    ~ManagerClientImpl() override;

    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                        QueryType query_type,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        int32_t sw_size,
                                                        const std::vector<std::string> &location_spec_names) override;

    std::pair<ClientErrorCode, WriteLocation> StartWrite(const std::string &trace_id,
                                                         const std::vector<int64_t> &keys,
                                                         const std::vector<int64_t> &tokens,
                                                         const std::vector<std::string> &location_spec_group_names,
                                                         int64_t write_timeout_seconds) override;
    ClientErrorCode FinishWrite(const std::string &trace_id,
                                const std::string &write_session_id,
                                const BlockMask &success_block,
                                const Locations &locations) override;

    std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &trace_id,
                                                const std::vector<int64_t> &keys,
                                                const std::vector<int64_t> &tokens,
                                                const BlockMask &block_mask,
                                                int32_t detail_level) override;

    ClientErrorCode RemoveCache(const std::string &trace_id,
                                const std::vector<int64_t> &keys,
                                const std::vector<int64_t> &tokens,
                                const BlockMask &block_mask) override;

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) override;

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers) override;

protected:
    ClientErrorCode Init(const std::string &client_config, InitParams &init_params) override;
    void Shutdown() override;

private:
    ClientErrorCode Connect(const std::string &address);

private:
    friend class ManagerClient;
    std::unique_ptr<MetaClient> meta_client_;
    std::unique_ptr<TransferClient> transfer_client_;
};
} // namespace kv_cache_manager