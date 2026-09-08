#include "kv_cache_manager/service/grpc_service/kv_meta_service_grpc.h"

#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {
namespace {

TEST(KvMetaServiceGRpcTest, MetricsAreNamespacedAwayFromMainService) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto main_metric =
        registry->GetCounter("service.query_counter", MetricsTags{{"api_name", "RegisterInstance"}});

    KvMetaServiceGRpc service(registry, nullptr);
    service.Init();

    auto kv_meta_metric =
        registry->GetCounter("service.query_counter", MetricsTags{{"api_name", "KvMeta.RegisterInstance"}});
    EXPECT_NE(main_metric.GetRaw(), kv_meta_metric.GetRaw());

    auto metrics_data = registry->GetMetricsData("service.query_counter");
    ASSERT_NE(nullptr, metrics_data);
    EXPECT_EQ(8U, metrics_data->GetSize());
}

} // namespace
} // namespace kv_cache_manager
