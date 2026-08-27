#include "kv_cache_manager/optimizer/service/online_optimizer_server.h"

#include <chrono>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <map>
#include <set>
#include <sstream>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_kmonitor_metrics_reporter.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_reporter.h"
#include "kv_cache_manager/optimizer/quota_runtime/quota_plan.h"
#include "kv_cache_manager/optimizer/service/event_subscriber/kvcm_event_subscriber.h"
#include "kv_cache_manager/optimizer/service/grpc/optimizer_service_grpc.h"
#include "kv_cache_manager/optimizer/service/http/optimizer_service_http.h"
#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"

namespace kv_cache_manager {

OnlineOptimizerServer::OnlineOptimizerServer() = default;

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
    if (config_.metrics_reporter_type() == "kmonitor") {
        kmonitor_metrics_reporter_ = std::make_shared<OptimizerKmonitorMetricsReporter>(config_.prometheus_prefix());
        if (!kmonitor_metrics_reporter_->Init()) {
            KVCM_LOG_WARN("KMonitor init failed, kmonitor metrics disabled");
            kmonitor_metrics_reporter_.reset();
        }
    }
    metrics_reporter_ =
        std::make_shared<OptimizerMetricsReporter>(manager_, metrics_registry_, kmonitor_metrics_reporter_);

    if (config_.quota_planner_config().enable) {
        quota_plan_store_ = std::make_shared<InMemoryQuotaPlanStore>();
        quota_planner_ = std::make_unique<ShadowQuotaPlanner>(config_.quota_planner_config());
    }
    service_impl_ =
        std::make_shared<OptimizerServiceImpl>(manager_, metrics_reporter_, quota_plan_store_, metrics_registry_);

    kvcm_event_subscribers_.clear();
    kvcm_event_subscribers_.reserve(config_.kvcm_event_subscriptions().size());
    for (const auto &subscription_config : config_.kvcm_event_subscriptions()) {
        auto subscriber = std::make_unique<KvcmEventSubscriber>(
            subscription_config, service_impl_, metrics_registry_, metrics_reporter_);
        if (!subscriber->Init()) {
            KVCM_LOG_ERROR("Failed to init KVCM event subscriber for discovery URL[%s]",
                           subscription_config.service_discovery_url().c_str());
            kvcm_event_subscribers_.clear();
            return false;
        }
        kvcm_event_subscribers_.push_back(std::move(subscriber));
    }

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

    for (auto &subscriber : kvcm_event_subscribers_) {
        if (!subscriber->Start()) {
            KVCM_LOG_ERROR("Failed to start KVCM event subscriber");
            Stop();
            return false;
        }
    }

    if (recovery_needed_) {
        recovery_thread_ = std::thread(&OnlineOptimizerServer::RecoveryRetryLoop, this);
    }

    if (config_.metrics_report_interval_ms() > 0) {
        metrics_report_thread_ =
            LoopThread::CreateLoopThread([reporter = metrics_reporter_]() { reporter->ReportInterval(); },
                                         config_.metrics_report_interval_ms() * 1000,
                                         "OptimizerMetricsReporter");
        if (!metrics_report_thread_) {
            KVCM_LOG_ERROR("Failed to start optimizer metrics reporter");
            Stop();
            return false;
        }
    }

    if (quota_planner_) {
        PlanQuotaOnce();
        quota_planner_thread_ =
            LoopThread::CreateLoopThread([this]() { PlanQuotaOnce(); },
                                         config_.quota_planner_config().period_seconds * 1000 * 1000,
                                         "KVBrainQuotaPlanner");
        if (!quota_planner_thread_) {
            KVCM_LOG_ERROR("Failed to start KVBrain quota planner");
            Stop();
            return false;
        }
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

    for (auto &subscriber : kvcm_event_subscribers_) {
        subscriber->Stop();
    }

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
    // threads and shut down kmonitor without racing with request/query reporting.
    if (recovery_thread_.joinable()) {
        recovery_thread_.join();
    }
    if (metrics_report_thread_) {
        metrics_report_thread_->Stop();
        metrics_report_thread_.reset();
    }
    if (quota_planner_thread_) {
        quota_planner_thread_->Stop();
        quota_planner_thread_.reset();
    }
    if (kmonitor_metrics_reporter_) {
        kmonitor_metrics_reporter_->Shutdown();
        kmonitor_metrics_reporter_.reset();
    }
    KVCM_LOG_INFO("OnlineOptimizerServer stopped");
}

void OnlineOptimizerServer::WaitForShutdown() {
    if (grpc_server_) {
        grpc_server_->Wait();
    }
    Stop();
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

void OnlineOptimizerServer::PlanQuotaOnce() {
    if (!manager_ || !quota_planner_ || !quota_plan_store_) {
        return;
    }
    std::map<std::string, std::set<uint64_t>> capacity_sets_by_source;
    for (const auto &pool : config_.quota_planner_config().pools) {
        for (const auto &member : pool.members) {
            const int64_t maximum = std::min(member.configured_max_quota_bytes, member.hardware_max_quota_bytes);
            auto &capacities = capacity_sets_by_source[member.source_id];
            capacities.insert(static_cast<uint64_t>(member.min_quota_bytes));
            capacities.insert(static_cast<uint64_t>(member.current_quota_bytes));
            capacities.insert(static_cast<uint64_t>(maximum));
            for (int64_t candidate = member.min_quota_bytes;
                 candidate < maximum && candidate <= maximum - pool.candidate_step_bytes;
                 candidate += pool.candidate_step_bytes) {
                capacities.insert(static_cast<uint64_t>(candidate + pool.candidate_step_bytes));
            }
        }
    }

    std::map<std::string, std::vector<uint64_t>> capacities_by_source;
    for (const auto &[source_id, capacities] : capacity_sets_by_source) {
        capacities_by_source.emplace(source_id, std::vector<uint64_t>(capacities.begin(), capacities.end()));
    }

    const auto snapshot = manager_->TakeQuotaDecisionSnapshot(capacities_by_source);
    for (const auto &plan : quota_planner_->BuildPlans(snapshot, quota_plan_store_->GetObservedQuotas())) {
        if (!quota_plan_store_->Publish(plan)) {
            KVCM_LOG_WARN("quota_decision_audit event=plan_publish_skipped pool_id=%s reason=active_plan",
                          plan->pool_id.c_str());
            continue;
        }
        std::ostringstream allocations;
        for (const auto &allocation : plan->allocations) {
            allocations << allocation.quota_target_id << ':' << allocation.current_quota_bytes << "->"
                        << allocation.target_quota_bytes << ',';
        }
        KVCM_LOG_INFO("quota_decision_audit event=plan_published system=KVBrain plan_id=%s plan_hash=%s pool_id=%s "
                      "snapshot_id=%llu status=%s phase=%s writes_quota=%d reason=%s allocations=%s",
                      plan->plan_id.c_str(),
                      plan->plan_hash.c_str(),
                      plan->pool_id.c_str(),
                      static_cast<unsigned long long>(plan->mrc_snapshot_id),
                      plan->status.c_str(),
                      plan->execution_phase.c_str(),
                      static_cast<int>(plan->writes_quota),
                      plan->reason.c_str(),
                      allocations.str().c_str());
        if (metrics_registry_) {
            MetricsTags tags{{"pool_id", plan->pool_id}, {"status", plan->status}, {"phase", plan->execution_phase}};
            metrics_registry_->GetCounter("quota_plan.published_total", tags) += 1;
            REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                                  "quota_plan.execution_revision",
                                  tags,
                                  static_cast<double>(plan->execution_revision));
        }
    }
}

} // namespace kv_cache_manager
