#include <memory>
#include <set>
#include <string>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "rapidjson/document.h"

namespace kv_cache_manager {

class MetaServiceStorageConfigTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("local://", metrics_registry_);
        ASSERT_TRUE(registry_manager_->Init());
        request_context_ = std::make_shared<RequestContext>("meta-storage-config-query");
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        service_ = std::make_shared<MetaServiceImpl>(cache_manager_, nullptr, nullptr);

        auto spec = std::make_shared<NfsStorageSpec>();
        spec->set_root_path(GetPrivateTestRuntimeDataPath() + "pace_primary/");
        spec->set_key_count_per_file(1);
        StorageConfig storage(DataStorageType::DATA_STORAGE_TYPE_NFS, "pace_primary", spec);
        ASSERT_EQ(EC_OK,
                  registry_manager_->data_storage_manager()->RegisterStorage(
                      request_context_.get(), "pace_primary", storage));

        InstanceGroup group;
        group.set_name("group_a");
        group.set_storage_candidates({"pace_primary"});
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(request_context_.get(), group));
    }

    proto::meta::GetStorageConfigsByInstanceGroupResponse Query(const std::string &instance_group) {
        proto::meta::GetStorageConfigsByInstanceGroupRequest request;
        request.set_trace_id("storage-bootstrap");
        request.set_instance_group(instance_group);
        proto::meta::GetStorageConfigsByInstanceGroupResponse response;
        service_->GetStorageConfigsByInstanceGroup(request_context_.get(), &request, &response);
        return response;
    }

    static std::set<std::string> ParseStorageNames(const std::string &json) {
        rapidjson::Document document;
        document.Parse(json.c_str());
        EXPECT_FALSE(document.HasParseError());
        EXPECT_TRUE(document.IsArray());
        std::set<std::string> names;
        if (!document.IsArray()) {
            return names;
        }
        for (const auto &config : document.GetArray()) {
            if (config.HasMember("global_unique_name") && config["global_unique_name"].IsString()) {
                names.insert(config["global_unique_name"].GetString());
            }
        }
        return names;
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<RequestContext> request_context_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetaServiceImpl> service_;
};

TEST_F(MetaServiceStorageConfigTest, RejectsEmptyInstanceGroup) {
    auto response = Query("");

    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
}

TEST_F(MetaServiceStorageConfigTest, ReportsMissingInstanceGroup) {
    auto response = Query("missing");

    EXPECT_EQ(proto::meta::INSTANCE_NOT_EXIST, response.header().status().code());
    EXPECT_NE(std::string::npos, response.header().status().message().find("instance group"));
}

TEST_F(MetaServiceStorageConfigTest, ReturnsVisibleStorageConfigs) {
    auto response = Query("group_a");

    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_EQ((std::set<std::string>{"pace_primary"}), ParseStorageNames(response.storage_configs()));
}

} // namespace kv_cache_manager
