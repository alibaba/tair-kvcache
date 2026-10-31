#include "kv_cache_manager/service/grpc_service/grpc_server_options.h"

#include <algorithm>
#include <array>
#include <google/protobuf/descriptor.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/resource_quota.h>
#include <grpcpp/support/server_interceptor.h>
#include <iterator>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"

namespace kv_cache_manager {
namespace {

constexpr std::pair<grpc::StatusCode, const char *> kStatusNames[] = {
    {grpc::StatusCode::OK, "OK"},
    {grpc::StatusCode::CANCELLED, "CANCELLED"},
    {grpc::StatusCode::UNKNOWN, "UNKNOWN"},
    {grpc::StatusCode::INVALID_ARGUMENT, "INVALID_ARGUMENT"},
    {grpc::StatusCode::DEADLINE_EXCEEDED, "DEADLINE_EXCEEDED"},
    {grpc::StatusCode::NOT_FOUND, "NOT_FOUND"},
    {grpc::StatusCode::ALREADY_EXISTS, "ALREADY_EXISTS"},
    {grpc::StatusCode::PERMISSION_DENIED, "PERMISSION_DENIED"},
    {grpc::StatusCode::RESOURCE_EXHAUSTED, "RESOURCE_EXHAUSTED"},
    {grpc::StatusCode::FAILED_PRECONDITION, "FAILED_PRECONDITION"},
    {grpc::StatusCode::ABORTED, "ABORTED"},
    {grpc::StatusCode::OUT_OF_RANGE, "OUT_OF_RANGE"},
    {grpc::StatusCode::UNIMPLEMENTED, "UNIMPLEMENTED"},
    {grpc::StatusCode::INTERNAL, "INTERNAL"},
    {grpc::StatusCode::UNAVAILABLE, "UNAVAILABLE"},
    {grpc::StatusCode::DATA_LOSS, "DATA_LOSS"},
    {grpc::StatusCode::UNAUTHENTICATED, "UNAUTHENTICATED"},
};

struct GrpcResponseMetrics {
    GrpcResponseMetrics(const std::string &server_name,
                        const std::string &method,
                        const std::shared_ptr<MetricsRegistry> &registry,
                        std::shared_ptr<MetricsReporter> metrics_reporter)
        : reporter(std::move(metrics_reporter)) {
        // Cache the bounded set of collectors once per method, so the
        // response path does not register series or take registry locks.
        for (const auto &[code, name] : kStatusNames) {
            collectors[code] = std::make_shared<GrpcMetricsCollector>(
                registry, MetricsTags{{"server", server_name}, {"method", method}, {"grpc_code", name}});
            collectors[code]->Init();
        }
    }

    void Record(grpc::StatusCode status) const {
        const int code = static_cast<int>(status);
        const size_t index = code >= 0 && code < collectors.size() ? code : grpc::StatusCode::UNKNOWN;
        reporter->ReportPerQuery(collectors[index].get());
    }

    std::array<std::shared_ptr<GrpcMetricsCollector>, std::size(kStatusNames)> collectors;
    const std::shared_ptr<MetricsReporter> reporter;
};

class GrpcResponseInterceptor final : public grpc::experimental::Interceptor {
public:
    explicit GrpcResponseInterceptor(std::shared_ptr<const GrpcResponseMetrics> metrics)
        : metrics_(std::move(metrics)) {}

    void Intercept(grpc::experimental::InterceptorBatchMethods *methods) override {
        if (methods->QueryInterceptionHookPoint(grpc::experimental::InterceptionHookPoints::PRE_SEND_STATUS) &&
            !reported_) {
            reported_ = true;
            metrics_->Record(methods->GetSendStatus().error_code());
        }
        methods->Proceed();
    }

private:
    const std::shared_ptr<const GrpcResponseMetrics> metrics_;
    bool reported_ = false;
};

class GrpcResponseInterceptorFactory final : public grpc::experimental::ServerInterceptorFactoryInterface {
public:
    GrpcResponseInterceptorFactory(const std::string &server_name,
                                   const std::shared_ptr<MetricsRegistry> &registry,
                                   std::shared_ptr<MetricsReporter> reporter)
        : server_name_(server_name)
        , registry_(registry)
        , reporter_(std::move(reporter))
        , unknown_metrics_(std::make_shared<GrpcResponseMetrics>(server_name_, "unknown", registry_, reporter_)) {}

    grpc::experimental::Interceptor *CreateServerInterceptor(grpc::experimental::ServerRpcInfo *info) override {
        const std::string method = info->method() ? info->method() : "";
        {
            std::shared_lock lock(mutex_);
            auto it = method_metrics_.find(method);
            if (it != method_metrics_.end()) {
                return new GrpcResponseInterceptor(it->second);
            }
        }

        // Unknown RPC paths are client-controlled. Only protobuf-defined methods
        // get their own labels and cache entries, keeping cardinality bounded.
        const auto separator = method.rfind('/');
        if (separator == std::string::npos || separator == 0 || method.front() != '/') {
            return new GrpcResponseInterceptor(unknown_metrics_);
        }
        const auto descriptor_name = method.substr(1, separator - 1) + "." + method.substr(separator + 1);
        const auto *descriptor = google::protobuf::DescriptorPool::generated_pool()->FindMethodByName(descriptor_name);
        if (!descriptor) {
            return new GrpcResponseInterceptor(unknown_metrics_);
        }

        std::unique_lock lock(mutex_);
        auto &metrics = method_metrics_[method];
        if (!metrics) {
            const auto method_label = descriptor->service()->name() + "/" + descriptor->name();
            metrics = std::make_shared<GrpcResponseMetrics>(server_name_, method_label, registry_, reporter_);
        }
        return new GrpcResponseInterceptor(metrics);
    }

private:
    const std::string server_name_;
    const std::shared_ptr<MetricsRegistry> registry_;
    const std::shared_ptr<MetricsReporter> reporter_;
    const std::shared_ptr<const GrpcResponseMetrics> unknown_metrics_;
    std::shared_mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<const GrpcResponseMetrics>> method_metrics_;
};

} // namespace

int ResolveGrpcMaxThreads(int configured_max_threads, unsigned int hardware_threads) {
    if (configured_max_threads != 0) {
        return configured_max_threads;
    }
    return static_cast<int>(
        std::clamp(hardware_threads, 2u, static_cast<unsigned int>(std::numeric_limits<int>::max())));
}

int ConfigureGrpcServer(grpc::ServerBuilder &builder,
                        int configured_max_threads,
                        std::size_t memory_quota_bytes,
                        const std::string &server_name,
                        const std::shared_ptr<MetricsRegistry> &metrics_registry,
                        const std::shared_ptr<MetricsReporter> &metrics_reporter) {
    const int max_threads = ResolveGrpcMaxThreads(configured_max_threads, std::thread::hardware_concurrency());
    if (max_threads > 0 || memory_quota_bytes > 0) {
        grpc::ResourceQuota quota;
        if (max_threads > 0) {
            quota.SetMaxThreads(max_threads);
        }
        if (memory_quota_bytes > 0) {
            quota.Resize(memory_quota_bytes);
        }
        builder.SetResourceQuota(quota);
    }

    std::vector<std::unique_ptr<grpc::experimental::ServerInterceptorFactoryInterface>> factories;
    factories.push_back(
        std::make_unique<GrpcResponseInterceptorFactory>(server_name, metrics_registry, metrics_reporter));
    builder.experimental().SetInterceptorCreators(std::move(factories));
    return max_threads;
}

} // namespace kv_cache_manager
