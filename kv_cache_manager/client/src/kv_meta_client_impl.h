#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "kv_cache_manager/client/include/kv_meta_client.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.grpc.pb.h"

namespace kv_cache_manager {

class KvMetaClientImpl final : public KvMetaClient {
public:
    KvMetaClientImpl() = default;
    ~KvMetaClientImpl() override = default;

    std::pair<ClientErrorCode, std::string> RegisterInstance(const std::string &trace_id,
                                                             const std::string &instance_group,
                                                             const std::string &user_data) override;
    std::pair<ClientErrorCode, KvMetaInstanceInfo> GetInstanceInfo(const std::string &trace_id) override;
    std::pair<ClientErrorCode, KvMetaGetResult> Get(const std::string &trace_id,
                                                    const std::vector<std::string> &keys) override;
    std::pair<ClientErrorCode, KvMetaStartWriteResult> StartWrite(const std::string &trace_id,
                                                                  const std::vector<std::string> &keys,
                                                                  const std::vector<std::uint64_t> &value_sizes,
                                                                  std::int32_t write_timeout_seconds) override;
    std::pair<ClientErrorCode, KvMetaStartWriteResult>
    StartWriteWithUserData(const std::string &trace_id,
                           const std::vector<std::string> &keys,
                           const std::vector<std::uint64_t> &value_sizes,
                           std::int32_t write_timeout_seconds,
                           const std::vector<KvMetaObjectUserData> &user_data) override;
    ClientErrorCode FinishWrite(const std::string &trace_id,
                                const std::string &write_session_id,
                                const std::vector<bool> &success_keys) override;
    ClientErrorCode Remove(const std::string &trace_id, const std::vector<std::string> &keys) override;
    ClientErrorCode UpdateAddresses(const std::vector<std::string> &addresses) override;
    std::pair<ClientErrorCode, KvMetaGetResult>
    GetWithDeadline(const std::string &trace_id,
                    const std::vector<std::string> &keys,
                    std::chrono::steady_clock::time_point deadline) override;

private:
    enum class TransportRetryPolicy {
        kSafe,
        kUnsafe,
    };

    friend class KvMetaClient;
    ClientErrorCode Init(const KvMetaClientConfig &config);

    template <typename Response, typename Rpc>
    ClientErrorCode Call(const std::string &trace_id,
                         Response *response,
                         TransportRetryPolicy transport_retry_policy,
                         Rpc &&rpc,
                         std::optional<std::chrono::steady_clock::time_point> caller_deadline = std::nullopt);

    std::pair<ClientErrorCode, KvMetaGetResult>
    GetImpl(const std::string &trace_id,
            const std::vector<std::string> &keys,
            std::optional<std::chrono::steady_clock::time_point> caller_deadline);

    KvMetaClientConfig config_;
    struct EndpointSnapshot {
        std::vector<std::string> addresses;
        std::vector<std::string> safe_addresses;
        std::vector<std::unique_ptr<proto::kv_meta::MetaService::Stub>> stubs;
        std::atomic<std::size_t> preferred{0};
    };
    // Use atomic shared_ptr operations: old RPCs own their stubs until return.
    std::shared_ptr<EndpointSnapshot> endpoints_;
};

} // namespace kv_cache_manager
