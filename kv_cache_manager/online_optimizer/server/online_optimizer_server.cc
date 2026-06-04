#include "kv_cache_manager/online_optimizer/server/online_optimizer_server.h"

#include <chrono>
#include <fstream>

#include <grpcpp/grpcpp.h>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/online_optimizer/manager/online_optimizer_manager.h"
#include "kv_cache_manager/online_optimizer/metrics/optimizer_metrics_reporter.h"
#include "kv_cache_manager/online_optimizer/server/grpc/optimizer_service_grpc.h"
#include "kv_cache_manager/online_optimizer/server/http/optimizer_service_http.h"
#include "kv_cache_manager/online_optimizer/server/optimizer_service_impl.h"

namespace kv_cache_manager {

OnlineOptimizerServer::~OnlineOptimizerServer() { Stop(); }

bool OnlineOptimizerServer::Init(const std::string &config_file, const EnvironMap &environ) {
    std::ifstream ifs(config_file);
    if (!ifs.is_open()) {
        KVCM_LOG_ERROR("Failed to open config file: %s", config_file.c_str());
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (!config_.FromJsonString(content)) {
        KVCM_LOG_ERROR("Failed to parse config file: %s", config_file.c_str());
        return false;
    }
    if (!config_.OverrideFromEnviron(environ)) {
        KVCM_LOG_ERROR("Failed to override config from environ");
        return false;
    }

    // Create registry first, then manager holds it
    registry_manager_ = std::make_shared<OptimizerRegistryManager>(config_.registry_storage_uri());
    if (!registry_manager_->Init()) {
        KVCM_LOG_ERROR("Failed to init registry manager");
        return false;
    }

    manager_ = std::make_shared<OnlineOptimizerManager>(registry_manager_);

    ErrorCode recover_ec = manager_->Recover();
    if (recover_ec != EC_OK) {
        KVCM_LOG_WARN("Recovery failed (ec=%d), starting fresh", static_cast<int>(recover_ec));
    }

    metrics_registry_ = std::make_shared<MetricsRegistry>();
    metrics_reporter_ = std::make_shared<OptimizerMetricsReporter>(
        manager_, metrics_registry_, config_.prometheus_prefix());
    metrics_reporter_->InitKmonitor();

    service_impl_ = std::make_shared<OptimizerServiceImpl>(manager_, metrics_reporter_);

    KVCM_LOG_INFO("OnlineOptimizerServer initialized");
    return true;
}

bool OnlineOptimizerServer::InitRpcServer() {
    grpc_service_ = std::make_shared<OptimizerServiceGRpc>(service_impl_, metrics_registry_);

    std::string server_address = "0.0.0.0:" + std::to_string(config_.rpc_port());
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(grpc_service_.get());
    grpc_server_ = builder.BuildAndStart();
    if (!grpc_server_) {
        KVCM_LOG_ERROR("Failed to start gRPC server on %s", server_address.c_str());
        return false;
    }
    KVCM_LOG_INFO("gRPC server started on %s", server_address.c_str());
    return true;
}

bool OnlineOptimizerServer::InitHttpServer() {
    http_service_ = std::make_shared<OptimizerServiceHttp>(service_impl_, metrics_registry_);
    http_service_->Init();
    http_service_->RegisterHandler();

    if (config_.enable_prometheus() && metrics_registry_) {
        http_service_->RegisterPrometheusEndpoint(metrics_registry_, config_.prometheus_prefix());
    }

    int32_t port = config_.http_port();
    size_t threads = config_.io_thread_num();
    http_thread_ = std::thread([this, port, threads]() {
        KVCM_LOG_INFO("HTTP server starting on port %d", port);
        if (!http_service_->Start(port, threads)) {
            KVCM_LOG_ERROR("Failed to start HTTP server on port %d", port);
        } else {
            KVCM_LOG_INFO("HTTP server exited on port %d", port);
        }
    });

    return true;
}

bool OnlineOptimizerServer::Start() {
    if (!InitRpcServer()) return false;
    if (!InitHttpServer()) return false;

    running_ = true;
    if (config_.metrics_report_interval_ms() > 0) {
        metrics_thread_ = std::thread(&OnlineOptimizerServer::MetricsReportLoop, this);
    }

    KVCM_LOG_INFO("OnlineOptimizerServer started: rpc_port=%d http_port=%d",
                  config_.rpc_port(), config_.http_port());
    return true;
}

void OnlineOptimizerServer::Stop() {
    running_ = false;
    if (metrics_thread_.joinable()) {
        metrics_thread_.join();
    }
    if (grpc_server_) {
        grpc_server_->Shutdown();
        grpc_server_.reset();
    }
    if (http_service_) {
        http_service_->Stop();
    }
    if (http_thread_.joinable()) {
        http_thread_.join();
    }
    KVCM_LOG_INFO("OnlineOptimizerServer stopped");
}

void OnlineOptimizerServer::WaitForShutdown() {
    if (grpc_server_) {
        grpc_server_->Wait();
    }
    if (http_thread_.joinable()) {
        http_thread_.join();
    }
}

void OnlineOptimizerServer::MetricsReportLoop() {
    while (running_) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.metrics_report_interval_ms()));
        if (!running_) break;
        metrics_reporter_->ReportInterval();
    }
}

} // namespace kv_cache_manager
