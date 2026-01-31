#pragma once

#include <memory>
#include <shared_mutex>

#include "kv_cache_manager/client/include/meta_client.h"

namespace kv_cache_manager {
class Stub;
class ClientConfig;

class MetaClientImpl : public MetaClient {
public:
    MetaClientImpl();
    ~MetaClientImpl() override;

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

    const std::string &GetStorageConfig() const override;

protected:
    ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) override;
    void Shutdown() override;

private:
    ClientErrorCode IsValid(const std::unique_ptr<ClientConfig> &client_config) const;
    ClientErrorCode Connect(const std::string &address);
    const ClientConfig *GetClientConfig() const;
    const ClientConfig *GetClientConfigUnsafe() const;
    const std::string &GetInstanceId() const;

private:
    friend class MetaClient;
    std::unique_ptr<ClientConfig> client_config_;
    std::unique_ptr<Stub> stub_;
    std::string storage_config_;
    mutable std::shared_mutex config_mutex_;
};
} // namespace kv_cache_manager