#include <gtest/gtest.h>

#include <set>

#include "kv_cache_manager/client/src/internal/async_rpc/api.h"
#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"
#include "kv_cache_manager/client/src/internal/async_rpc/json_codec.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"

namespace kv_cache_manager::async_rpc {
namespace {

namespace admin = kv_cache_manager::proto::admin;
namespace meta = kv_cache_manager::proto::meta;

TEST(ApiTableTest, ContainsOnlyClientRpcOperations) {
    static const std::set<std::string> kForbidden = {
        "AddStorage",
        "UpdateStorage",
        "RemoveStorage",
        "CreateInstanceGroup",
        "UpdateInstanceGroup",
        "RemoveInstanceGroup",
        "MigrateCache",
        "TrimCache",
        "RemoveInstance",
    };
    for (const auto &info : AllApis()) {
        EXPECT_EQ(kForbidden.count(std::string(info.name)), 0u) << info.name;
    }
    EXPECT_EQ(AllApis().size(), 8u);
}

TEST(ApiTableTest, RoutesMetaAndAdminServicesSeparately) {
    EXPECT_EQ(GetApiInfo(Api::kRegisterInstance).http_path, "/api/registerInstance");
    EXPECT_EQ(GetApiInfo(Api::kRegisterInstance).endpoint, ServiceEndpoint::kMeta);
    EXPECT_EQ(GetApiInfo(Api::kCheckHealth).http_path, "/api/checkHealth");
    EXPECT_EQ(GetApiInfo(Api::kCheckHealth).endpoint, ServiceEndpoint::kAdmin);
    for (const auto &info : AllApis()) {
        if (info.api == Api::kCheckHealth) {
            EXPECT_EQ(info.grpc_method, "/kv_cache_manager.proto.admin.AdminService/CheckHealth");
        } else {
            EXPECT_EQ(info.grpc_method.rfind("/kv_cache_manager.proto.meta.MetaService/", 0), 0u) << info.name;
        }
    }
}

TEST(JsonCodecTest, PreservesWireFieldNamesEnumsAndInt64Values) {
    meta::GetCacheLocationsByBackendRequest request;
    request.set_instance_id("inst");
    request.set_query_type(meta::QT_BATCH_GET);
    request.add_block_keys(-6631516056321758883LL);
    request.add_block_keys(4052814987019828902LL);
    request.add_location_spec_names("v6d_4096");
    request.add_location_spec_names("v6d_1024");
    auto *selector = request.add_backend_selectors();
    selector->set_backend_type(meta::ST_EVENT_REPORT_L2);
    selector->set_strategy(meta::LSS_V6D_PREFIX);

    std::string json;
    ASSERT_TRUE(MessageToJson(request, &json));
    EXPECT_NE(json.find("\"instance_id\""), std::string::npos);
    EXPECT_NE(json.find("\"ST_EVENT_REPORT_L2\""), std::string::npos);

    meta::GetCacheLocationsByBackendRequest parsed;
    std::string error;
    ASSERT_TRUE(JsonToMessage(json, &parsed, &error)) << error;
    EXPECT_EQ(parsed.SerializeAsString(), request.SerializeAsString());
}

TEST(ServiceStatusTest, ExtractsMetaAndAdminStatusThroughReflection) {
    meta::ReportEventResponse report;
    report.mutable_header()->mutable_status()->set_code(meta::SERVER_NOT_LEADER);
    report.mutable_header()->mutable_status()->set_message("not leader");
    EXPECT_EQ(ExtractServiceStatus(report), kStatusServerNotLeader);
    EXPECT_EQ(ExtractServiceMessage(report), "not leader");

    admin::CheckHealthResponse health;
    health.mutable_header()->mutable_status()->set_code(admin::OK);
    EXPECT_EQ(ExtractServiceStatus(health), kStatusOk);
}

TEST(EndpointValidationTest, RejectsTlsAndMalformedEndpoints) {
    std::string error;
    EXPECT_TRUE(ValidateInsecureEndpoint("http://127.0.0.1:8080", true, &error));
    EXPECT_TRUE(ValidateInsecureEndpoint("127.0.0.1:8081", false, &error));
    EXPECT_FALSE(ValidateInsecureEndpoint("https://127.0.0.1:8080", true, &error));
    EXPECT_FALSE(ValidateInsecureEndpoint("http://127.0.0.1:8081", false, &error));
    EXPECT_FALSE(ValidateInsecureEndpoint("127.0.0.1:not-a-port", false, &error));
}

} // namespace
} // namespace kv_cache_manager::async_rpc
