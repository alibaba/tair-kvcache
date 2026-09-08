#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "kv_cache_manager/client/include/kv_meta_client.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.grpc.pb.h"

namespace kv_cache_manager {

class KvMetaClientImpl final : public KvMetaClient {
public:
    KvMetaClientImpl() = default;
    ~KvMetaClientImpl() override = default;

    std::pair<ClientErrorCode, std::string>
    RegisterInstance(const std::string &trace_id,
                     const std::string &instance_group,
                     const std::string &user_data) override;
    std::pair<ClientErrorCode, KvMetaInstanceInfo> GetInstanceInfo(const std::string &trace_id) override;
    std::pair<ClientErrorCode, KvMetaGetResult>
    Get(const std::string &trace_id, const std::vector<std::string> &keys) override;
    std::pair<ClientErrorCode, KvMetaStartWriteResult>
    StartWrite(const std::string &trace_id,
               const std::vector<std::string> &keys,
               const std::vector<std::uint64_t> &value_sizes,
               std::int32_t write_timeout_seconds) override;
    ClientErrorCode FinishWrite(const std::string &trace_id,
                                const std::string &write_session_id,
                                const std::vector<bool> &success_keys) override;
    ClientErrorCode Remove(const std::string &trace_id, const std::vector<std::string> &keys) override;
    ClientErrorCode TrimAll(const std::string &trace_id, bool metadata_only) override;

private:
    enum class TransportRetryPolicy {
        kSafe,
        kUnsafe,
    };

    friend class KvMetaClient;
    ClientErrorCode Init(const KvMetaClientConfig &config);

    template <typename Response, typename Rpc>
    ClientErrorCode Call(Response *response, TransportRetryPolicy transport_retry_policy, Rpc &&rpc);

    KvMetaClientConfig config_;
    std::vector<std::unique_ptr<proto::kv_meta::MetaService::Stub>> stubs_;
    std::atomic<std::size_t> preferred_stub_{0};
};

} // namespace kv_cache_manager
