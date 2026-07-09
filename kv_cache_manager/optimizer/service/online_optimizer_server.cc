#include "kv_cache_manager/optimizer/service/online_optimizer_server.h"

#include <chrono>
#include <fstream>
#include <grpcpp/grpcpp.h>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/service/grpc/optimizer_service_grpc.h"
#include "kv_cache_manager/optimizer/service/http/optimizer_service_http.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_reporter.h"
#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"

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
        KVCM_LOG_WARN("Recovery failed (ec=%d), will retry asynchronously after Start", static_cast<int>(recover_ec));
        recovery_needed_ = true;
    }

    metrics_registry_ = std::make_shared<MetricsRegistry>();
    metrics_reporter_ =
        std::make_shared<OptimizerMetricsReporter>(manager_, metrics_registry_, config_.prometheus_prefix());
    if (!metrics_reporter_->InitKmonitor()) {
        KVCM_LOG_WARN("KMonitor init failed, kmonitor metrics disabled");
    }

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
    if (port < 1 || port > 65535) {
        KVCM_LOG_ERROR("Invalid http_port %d: must be in range [1, 65535]", port);
        return false;
    }
    size_t threads = config_.io_thread_num() > 0 ? static_cast<size_t>(config_.io_thread_num())
                                                 : std::thread::hardware_concurrency();
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
    if (!InitRpcServer())
        return false;
    if (!InitHttpServer())
        return false;

    running_ = true;

    if (recovery_needed_) {
        recovery_thread_ = std::thread(&OnlineOptimizerServer::RecoveryRetryLoop, this);
    }

    if (config_.metrics_report_interval_ms() > 0) {
        metrics_thread_ = std::thread(&OnlineOptimizerServer::MetricsReportLoop, this);
    }

    KVCM_LOG_INFO("OnlineOptimizerServer started: rpc_port=%d http_port=%d", config_.rpc_port(), config_.http_port());
    return true;
}

void OnlineOptimizerServer::Stop() {
    std::call_once(stop_flag_, [this]() { DoStop(); });
}

void OnlineOptimizerServer::RequestShutdown() {
    // Async-signal-safe: only atomic store + gRPC Shutdown (unblocks Wait()).
    // Full cleanup is deferred to Stop() called from the main thread.
    running_ = false;
    if (grpc_server_) {
        grpc_server_->Shutdown();
    }
}

void OnlineOptimizerServer::DoStop() {
    running_ = false;

    // Stop listeners first so no new requests are accepted and in-flight
    // requests can drain before we tear down metrics infrastructure.
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

    // Now that all request threads have finished, safe to join background
    // threads and shut down kmonitor without racing with ReportPerQuery().
    if (recovery_thread_.joinable()) {
        recovery_thread_.join();
    }
    if (metrics_thread_.joinable()) {
        metrics_thread_.join();
    }
    if (metrics_reporter_) {
        metrics_reporter_->ShutdownKmonitor();
    }
    KVCM_LOG_INFO("OnlineOptimizerServer stopped");
}

void OnlineOptimizerServer::WaitForShutdown() {
    if (grpc_server_) {
        grpc_server_->Wait();
    }
    Stop();
}

void OnlineOptimizerServer::MetricsReportLoop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.metrics_report_interval_ms()));
        if (!running_)
            break;
        metrics_reporter_->ReportInterval();
    }
}

void OnlineOptimizerServer::RecoveryRetryLoop() {
    constexpr int kMaxRetries = 10;
    constexpr int kBaseIntervalMs = 3000;

    for (int attempt = 1; attempt <= kMaxRetries; attempt++) {
        int wait_ms = kBaseIntervalMs * attempt; // linear backoff
        // Sleep in small increments to allow early exit on shutdown
        for (int elapsed = 0; elapsed < wait_ms && running_; elapsed += 100) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!running_)
            break;

        KVCM_LOG_INFO("Recovery retry attempt %d/%d", attempt, kMaxRetries);
        ErrorCode ec = manager_->Recover();
        if (ec == EC_OK) {
            KVCM_LOG_INFO("Recovery retry succeeded on attempt %d", attempt);
            return;
        }
        KVCM_LOG_WARN("Recovery retry attempt %d failed (ec=%d)", attempt, static_cast<int>(ec));
    }
    KVCM_LOG_ERROR("Recovery failed after %d retries, running without persisted state", kMaxRetries);
}

} // namespace kv_cache_manager
