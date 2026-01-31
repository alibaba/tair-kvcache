#pragma once

#include <memory>

#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/service_impl_base.h"

namespace kv_cache_manager {

class CacheManager;
class MetricsReporter;
class MetricsRegistry;
class RequestContext;

class MetaServiceImpl : public ServiceImplBase {
public:
    MetaServiceImpl(std::shared_ptr<CacheManager> cache_manager, std::shared_ptr<MetricsReporter> metrics_reporter);
    ~MetaServiceImpl() override = default;

    // 实现所有MetaService的接口方法
    void RegisterInstance(RequestContext *request_context,
                          const proto::meta::RegisterInstanceRequest *request,
                          proto::meta::RegisterInstanceResponse *response);

    void GetInstanceInfo(RequestContext *request_context,
                         const proto::meta::GetInstanceInfoRequest *request,
                         proto::meta::GetInstanceInfoResponse *response);

    void GetCacheLocation(RequestContext *request_context,
                          const proto::meta::GetCacheLocationRequest *request,
                          proto::meta::GetCacheLocationResponse *response);

    void GetCacheMeta(RequestContext *request_context,
                      const proto::meta::GetCacheMetaRequest *request,
                      proto::meta::GetCacheMetaResponse *response);

    void StartWriteCache(RequestContext *request_context,
                         const proto::meta::StartWriteCacheRequest *request,
                         proto::meta::StartWriteCacheResponse *response);

    void FinishWriteCache(RequestContext *request_context,
                          const proto::meta::FinishWriteCacheRequest *request,
                          proto::meta::CommonResponse *response);

    void RemoveCache(RequestContext *request_context,
                     const proto::meta::RemoveCacheRequest *request,
                     proto::meta::CommonResponse *response);

    void TrimCache(RequestContext *request_context,
                   const proto::meta::TrimCacheRequest *request,
                   proto::meta::CommonResponse *response);

private:
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetricsReporter> metrics_reporter_;
};

} // namespace kv_cache_manager
