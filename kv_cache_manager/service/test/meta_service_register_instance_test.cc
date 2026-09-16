#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/location_spec_group.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "stub.h"

namespace kv_cache_manager {
namespace {

std::string selected_meta_service_url;

std::pair<ErrorCode, std::string>
RegisterInstanceStub(void *,
                     RequestContext *,
                     const std::string &,
                     const std::string &,
                     int32_t,
                     const std::vector<LocationSpecInfo> &,
                     const ModelDeployment &,
                     const std::vector<LocationSpecGroup> &,
                     CacheManager::QueryType,
                     std::string *tair_mempool_metaservice_url) {
    if (tair_mempool_metaservice_url != nullptr) {
        *tair_mempool_metaservice_url = selected_meta_service_url;
    }
    return {EC_OK, "[]"};
}

std::string GetExtraInfoStub(void *, RequestContext *, const std::string &) { return "{}"; }

proto::meta::RegisterInstanceRequest MakeRequest() {
    proto::meta::RegisterInstanceRequest request;
    request.set_trace_id("register-selected-meta");
    request.set_instance_group("group-a");
    request.set_instance_id("instance-a");
    request.set_block_size(64);
    auto *deployment = request.mutable_model_deployment();
    deployment->set_model_name("model-a");
    deployment->set_dtype("fp8");
    auto *spec = request.add_location_spec_infos();
    spec->set_name("tp0");
    spec->set_size(512);
    return request;
}

} // namespace

class MetaServiceRegisterInstanceTest : public TESTBASE {
protected:
    void SetUp() override {
        stub_.set(ADDR(CacheManager, RegisterInstance), RegisterInstanceStub);
        stub_.set(ADDR(CacheManager, GetExtraInfo), GetExtraInfoStub);
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("local://", metrics_registry_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        service_ = std::make_shared<MetaServiceImpl>(cache_manager_, nullptr, nullptr);
        request_context_ = std::make_shared<RequestContext>("register-selected-meta");
    }

    Stub stub_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetaServiceImpl> service_;
    std::shared_ptr<RequestContext> request_context_;
};

TEST_F(MetaServiceRegisterInstanceTest, ReturnsSelectedTairMempoolMetaServiceUrl) {
    selected_meta_service_url = "spectrum://v-selected?port=12348";
    auto request = MakeRequest();
    proto::meta::RegisterInstanceResponse response;

    service_->RegisterInstance(request_context_.get(), &request, &response);

    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_EQ("[]", response.storage_configs());
    EXPECT_EQ("spectrum://v-selected?port=12348", response.tair_mempool_metaservice_url());
}

TEST_F(MetaServiceRegisterInstanceTest, KeepsRegistrationSuccessfulWhenSelectionIsEmpty) {
    selected_meta_service_url.clear();
    auto request = MakeRequest();
    proto::meta::RegisterInstanceResponse response;

    service_->RegisterInstance(request_context_.get(), &request, &response);

    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_TRUE(response.tair_mempool_metaservice_url().empty());
}

} // namespace kv_cache_manager
