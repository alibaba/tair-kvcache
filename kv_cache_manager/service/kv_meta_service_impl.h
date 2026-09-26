#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.pb.h"
#include "kv_cache_manager/service/service_impl_base.h"

namespace kv_cache_manager {

class CacheManager;
class KvMetaManager;
class MetricsReporter;
class RequestContext;

// Protocol adapter for the generic object metadata path. This service has its
// own leader gate and is registered only when explicitly enabled. Its fully
// qualified gRPC methods are disjoint from the fixed-block MetaService even
// though both services share the primary listener.
class KvMetaServiceImpl final : public ServiceImplBase {
public:
    KvMetaServiceImpl(std::shared_ptr<CacheManager> cache_manager,
                      std::shared_ptr<KvMetaManager> kv_meta_manager,
                      std::shared_ptr<MetricsReporter> metrics_reporter);
    ~KvMetaServiceImpl() override = default;

    void RegisterInstance(RequestContext *request_context,
                          const proto::kv_meta::RegisterInstanceRequest *request,
                          proto::kv_meta::RegisterInstanceResponse *response);
    void GetInstanceInfo(RequestContext *request_context,
                         const proto::kv_meta::GetInstanceInfoRequest *request,
                         proto::kv_meta::GetInstanceInfoResponse *response);
    void Get(RequestContext *request_context,
             const proto::kv_meta::GetRequest *request,
             proto::kv_meta::GetResponse *response);
    void PutStart(RequestContext *request_context,
                  const proto::kv_meta::PutStartRequest *request,
                  proto::kv_meta::PutStartResponse *response);
    void PutFinish(RequestContext *request_context,
                   const proto::kv_meta::PutFinishRequest *request,
                   proto::kv_meta::CommonResponse *response);
    void Remove(RequestContext *request_context,
                const proto::kv_meta::RemoveRequest *request,
                proto::kv_meta::CommonResponse *response);
    void Trim(RequestContext *request_context,
              const proto::kv_meta::TrimRequest *request,
              proto::kv_meta::CommonResponse *response);

private:
    ErrorCode AbortMalformedPutStart(RequestContext *request_context,
                                     const std::string &instance_id,
                                     const std::string &write_session_id,
                                     std::size_t session_item_count) noexcept;

    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<KvMetaManager> kv_meta_manager_;
    std::shared_ptr<MetricsReporter> metrics_reporter_;
};

} // namespace kv_cache_manager
