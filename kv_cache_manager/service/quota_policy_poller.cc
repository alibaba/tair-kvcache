#include "kv_cache_manager/service/quota_policy_poller.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <limits>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/service_discovery_factory.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

QuotaPolicyPoller::QuotaPolicyPoller(QuotaPolicyPollerConfig config,
                                     std::shared_ptr<RegistryManager> registry_manager,
                                     IsLeaderFunction is_leader,
                                     std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                     std::shared_ptr<MetricsRegistry> metrics_registry)
    : config_(std::move(config))
    , registry_manager_(std::move(registry_manager))
    , meta_indexer_manager_(std::move(meta_indexer_manager))
    , metrics_registry_(std::move(metrics_registry))
    , is_leader_(std::move(is_leader)) {}

QuotaPolicyPoller::~QuotaPolicyPoller() { Stop(); }

bool QuotaPolicyPoller::Init() {
    if (!registry_manager_ || !is_leader_ || config_.optimizer_service_discovery_url.empty() ||
        config_.pool_id.empty() || config_.quota_target_id.empty() || config_.instance_group.empty() ||
        config_.state_file.empty() || config_.poll_interval_seconds <= 0 || config_.rpc_timeout_ms <= 0) {
        return false;
    }
    discovery_ = ServiceDiscoveryFactory::CreateServiceDiscovery(config_.optimizer_service_discovery_url);
    return discovery_ && LoadState();
}

bool QuotaPolicyPoller::Start() {
    if (!discovery_) {
        return false;
    }
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return true;
    }
    thread_ = std::thread(&QuotaPolicyPoller::Loop, this);
    return true;
}

void QuotaPolicyPoller::Stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

void QuotaPolicyPoller::Loop() {
    while (running_) {
        if (is_leader_()) {
            PollOnce();
        }
        for (int64_t elapsed = 0; elapsed < config_.poll_interval_seconds && running_; ++elapsed) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

bool QuotaPolicyPoller::PollOnce() {
    if (!is_leader_() || !EnsureStub()) {
        return false;
    }
    RequestContext local_context("quota_policy_poller");
    auto [ec, instance_group] = registry_manager_->GetInstanceGroup(&local_context, config_.instance_group);
    if (ec != EC_OK || !instance_group) {
        KVCM_LOG_WARN(
            "quota_resize_audit event=plan_skipped pool_id=%s quota_target_id=%s reason=instance_group_missing",
            config_.pool_id.c_str(),
            config_.quota_target_id.c_str());
        return false;
    }
    const int64_t quota_before = instance_group->quota().capacity();
    const int64_t version_before = instance_group->version();

    proto::optimizer::PullQuotaAllocationRequest request;
    request.set_trace_id("quota-policy-poller");
    request.set_pool_id(config_.pool_id);
    request.set_quota_target_id(config_.quota_target_id);
    request.set_last_leader_epoch(last_leader_epoch_);
    request.set_last_allocation_epoch(last_allocation_epoch_);
    request.set_last_execution_revision(last_execution_revision_);
    request.set_current_quota_bytes(instance_group->quota().capacity());
    const auto used_bytes = GetGroupUsedBytes(&local_context);
    request.set_current_used_bytes(used_bytes.value_or(-1));

    proto::optimizer::PullQuotaAllocationResponse response;
    grpc::ClientContext rpc_context;
    rpc_context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(config_.rpc_timeout_ms));
    const grpc::Status rpc_status = stub_->PullQuotaAllocation(&rpc_context, request, &response);
    if (!rpc_status.ok() || response.header().status().code() != proto::optimizer::OK) {
        if (discovery_) {
            discovery_->Refresh();
            stub_.reset();
            connected_endpoint_.clear();
        }
        KVCM_LOG_WARN(
            "quota_resize_audit event=plan_pull_failed pool_id=%s quota_target_id=%s grpc_code=%d status_code=%d",
            config_.pool_id.c_str(),
            config_.quota_target_id.c_str(),
            static_cast<int>(rpc_status.error_code()),
            static_cast<int>(response.header().status().code()));
        return false;
    }
    if (response.pull_status() == proto::optimizer::QUOTA_PULL_NO_PLAN ||
        response.pull_status() == proto::optimizer::QUOTA_PULL_NOT_MODIFIED) {
        return true;
    }

    std::string result_status = "PLAN_REJECTED";
    std::string reason = response.reason();
    const int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    if (response.plan_id().empty() || response.plan_hash().empty() || response.pool_id() != config_.pool_id) {
        reason = "plan_identity_mismatch";
    } else if (response.pull_status() == proto::optimizer::QUOTA_PULL_FROZEN) {
        result_status = "FROZEN_ACKNOWLEDGED";
        reason = reason.empty() ? "optimizer_frozen" : reason;
    } else if (!response.has_allocation() || response.allocation().quota_target_id() != config_.quota_target_id ||
               response.allocation().instance_group() != config_.instance_group) {
        reason = "allocation_identity_mismatch";
    } else if (response.valid_until_ns() <= now_ns) {
        reason = "plan_expired";
    } else if (response.leader_epoch() < last_leader_epoch_ ||
               (response.leader_epoch() == last_leader_epoch_ &&
                (response.allocation_epoch() < last_allocation_epoch_ ||
                 (response.allocation_epoch() == last_allocation_epoch_ &&
                  response.execution_revision() < last_execution_revision_)))) {
        reason = "stale_epoch";
    } else if (response.leader_epoch() == last_leader_epoch_ && response.allocation_epoch() == last_allocation_epoch_ &&
               response.execution_revision() == last_execution_revision_ && !last_plan_hash_.empty() &&
               response.plan_hash() != last_plan_hash_) {
        reason = "epoch_plan_hash_conflict";
    } else if (response.allocation().target_quota_bytes() < response.allocation().min_quota_bytes() ||
               response.allocation().target_quota_bytes() > response.allocation().max_quota_bytes()) {
        reason = "target_out_of_bounds";
    } else if (response.execution_phase() == "RECEIVER_GROW" &&
               response.allocation().target_quota_bytes() <= response.allocation().current_quota_bytes()) {
        reason = "execution_phase_direction_mismatch";
    } else if (response.writes_quota() && !config_.enable_hard_resize) {
        result_status = "HARD_RESIZE_REJECTED";
        reason = "kvcm_hard_resize_disabled";
    } else if (response.writes_quota() && response.execution_phase() == "HOLD") {
        result_status = "HOLD_ACKNOWLEDGED";
        reason = "waiting_for_optimizer_phase_transition";
    } else if (response.writes_quota() && response.execution_phase() == "COMPLETE") {
        result_status = "COMPLETE_ACKNOWLEDGED";
        reason = "two_phase_hard_resize_complete";
    } else if (!response.writes_quota() || !response.executable() || response.execution_phase() == "SHADOW") {
        result_status = "DRY_RUN_ACCEPTED";
        reason = "writes_quota=false";
    } else if (!is_leader_()) {
        KVCM_LOG_WARN("quota_resize_audit event=plan_aborted pool_id=%s quota_target_id=%s reason=leadership_lost",
                      config_.pool_id.c_str(),
                      config_.quota_target_id.c_str());
        return false;
    } else if (!used_bytes.has_value()) {
        result_status = "USAGE_OBSERVATION_UNAVAILABLE";
        reason = "authoritative_group_usage_unavailable";
    } else if (response.execution_phase() == "DONOR_SHRINK") {
        int64_t observed_quota = instance_group->quota().capacity();
        int64_t observed_version = instance_group->version();
        if (response.release_deadline_ns() <= 0 || response.release_consecutive_samples() <= 0) {
            result_status = "DONOR_SHRINK_FAILED";
            reason = "invalid_release_confirmation_contract";
        } else if (now_ns > response.release_deadline_ns()) {
            result_status = "DONOR_RELEASE_TIMEOUT";
            reason = "release_deadline_exceeded";
        } else {
            const bool requires_hard_shrink =
                response.allocation().target_quota_bytes() < response.allocation().current_quota_bytes();
            if (requires_hard_shrink && !ApplyHardQuota(response, &observed_quota, &observed_version, &reason)) {
                result_status = "DONOR_SHRINK_FAILED";
            } else if (!requires_hard_shrink &&
                       instance_group->quota().capacity() != response.allocation().current_quota_bytes()) {
                result_status = "DONOR_SHRINK_FAILED";
                reason = "release_wait_quota_precondition_mismatch";
            }
            if (result_status != "DONOR_SHRINK_FAILED") {
                if (release_plan_id_ != response.plan_id()) {
                    release_plan_id_ = response.plan_id();
                    consecutive_release_samples_ = 0;
                }
                if (*used_bytes <= response.allocation().target_quota_bytes()) {
                    ++consecutive_release_samples_;
                } else {
                    consecutive_release_samples_ = 0;
                }
                if (consecutive_release_samples_ >= response.release_consecutive_samples()) {
                    result_status = "DONOR_RELEASE_CONFIRMED";
                    reason = "usage_at_or_below_target_for_consecutive_samples";
                } else {
                    result_status = "DONOR_SHRINK_APPLIED";
                    reason = "waiting_for_physical_release";
                }
                const auto refreshed = registry_manager_->GetInstanceGroup(&local_context, config_.instance_group);
                if (refreshed.first == EC_OK && refreshed.second) {
                    instance_group = refreshed.second;
                }
            }
        }
    } else if (response.execution_phase() == "RECEIVER_GROW") {
        int64_t observed_quota = instance_group->quota().capacity();
        int64_t observed_version = instance_group->version();
        if (ApplyHardQuota(response, &observed_quota, &observed_version, &reason)) {
            result_status = "RECEIVER_GROW_APPLIED";
            reason = "hard_quota_cas_confirmed";
            const auto refreshed = registry_manager_->GetInstanceGroup(&local_context, config_.instance_group);
            if (refreshed.first == EC_OK && refreshed.second) {
                instance_group = refreshed.second;
            }
        } else {
            result_status = "RECEIVER_GROW_FAILED";
        }
    } else {
        result_status = "PLAN_REJECTED";
        reason = "unknown_execution_phase";
    }

    KVCM_LOG_INFO("quota_resize_audit event=plan_consumed plan_id=%s plan_hash=%s pool_id=%s quota_target_id=%s "
                  "leader_epoch=%llu allocation_epoch=%llu instance_group=%s quota_before=%ld quota_target=%ld "
                  "observed_used_bytes=%ld instance_group_version_before=%ld instance_group_version_after=%ld "
                  "executable=%s status=%s reason=%s writes_quota=%s execution_phase=%s execution_revision=%llu",
                  response.plan_id().c_str(),
                  response.plan_hash().c_str(),
                  response.pool_id().c_str(),
                  config_.quota_target_id.c_str(),
                  static_cast<unsigned long long>(response.leader_epoch()),
                  static_cast<unsigned long long>(response.allocation_epoch()),
                  config_.instance_group.c_str(),
                  static_cast<long>(quota_before),
                  static_cast<long>(response.has_allocation() ? response.allocation().target_quota_bytes() : -1),
                  static_cast<long>(used_bytes.value_or(-1)),
                  static_cast<long>(version_before),
                  static_cast<long>(instance_group->version()),
                  response.executable() ? "true" : "false",
                  result_status.c_str(),
                  reason.c_str(),
                  response.writes_quota() ? "true" : "false",
                  response.execution_phase().c_str(),
                  static_cast<unsigned long long>(response.execution_revision()));

    if (!ReportResult(response,
                      response.execution_phase(),
                      result_status,
                      reason,
                      instance_group->quota().capacity(),
                      used_bytes.value_or(-1),
                      instance_group->version())) {
        return false;
    }
    ReportMetrics(response, used_bytes.value_or(-1), result_status);
    if (result_status == "DONOR_SHRINK_APPLIED" || result_status == "USAGE_OBSERVATION_UNAVAILABLE") {
        // Keep pulling the same revision while physical deletion catches up,
        // or while the authoritative usage observation is temporarily absent.
        return true;
    }
    return SaveState(
        response.leader_epoch(), response.allocation_epoch(), response.execution_revision(), response.plan_hash());
}

std::optional<int64_t> QuotaPolicyPoller::GetGroupUsedBytes(RequestContext *context) const {
    const auto [ec, instances] = registry_manager_->ListInstanceInfo(context, config_.instance_group);
    if (ec != EC_OK) {
        return std::nullopt;
    }
    uint64_t total = 0;
    for (const auto &instance : instances) {
        if (!meta_indexer_manager_) {
            return std::nullopt;
        }
        const auto indexer = meta_indexer_manager_->GetMetaIndexer(instance->instance_id());
        if (!indexer) {
            return std::nullopt;
        }
        const uint64_t usage = indexer->GetStorageUsage();
        if (total > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - usage) {
            return std::nullopt;
        }
        total += usage;
    }
    return static_cast<int64_t>(total);
}

bool QuotaPolicyPoller::ApplyHardQuota(const proto::optimizer::PullQuotaAllocationResponse &plan,
                                       int64_t *observed_quota_bytes,
                                       int64_t *instance_group_version,
                                       std::string *reason) {
    if (!is_leader_()) {
        *reason = "leadership_lost_before_quota_cas";
        return false;
    }
    RequestContext context("quota-hard-resize-cas");
    const auto [ec, current] = registry_manager_->GetInstanceGroup(&context, config_.instance_group);
    if (ec != EC_OK || !current || !plan.has_allocation()) {
        *reason = "instance_group_or_allocation_missing";
        return false;
    }
    const int64_t target = plan.allocation().target_quota_bytes();
    *observed_quota_bytes = current->quota().capacity();
    *instance_group_version = current->version();
    if (target < plan.allocation().min_quota_bytes() || target > plan.allocation().max_quota_bytes()) {
        *reason = "target_out_of_bounds";
        return false;
    }
    if (current->quota().capacity() == target) {
        *reason = "hard_quota_already_at_target";
        return true;
    }
    if (current->quota().capacity() != plan.allocation().current_quota_bytes()) {
        *reason = "hard_quota_cas_precondition_mismatch";
        if (metrics_registry_) {
            metrics_registry_->GetCounter(
                "quota_resize.cas_conflicts_total",
                {{"pool_id", config_.pool_id}, {"quota_target_id", config_.quota_target_id}}) += 1;
        }
        return false;
    }
    InstanceGroup updated = *current;
    InstanceGroupQuota quota = current->quota();
    quota.set_capacity(target);
    updated.set_quota(quota);
    updated.set_version(current->version() + 1);
    if (registry_manager_->UpdateInstanceGroup(&context, updated, current->version()) != EC_OK) {
        *reason = "hard_quota_registry_cas_failed";
        if (metrics_registry_) {
            metrics_registry_->GetCounter(
                "quota_resize.cas_conflicts_total",
                {{"pool_id", config_.pool_id}, {"quota_target_id", config_.quota_target_id}}) += 1;
        }
        return false;
    }
    const auto [confirm_ec, confirmed] = registry_manager_->GetInstanceGroup(&context, config_.instance_group);
    if (confirm_ec != EC_OK || !confirmed || confirmed->quota().capacity() != target ||
        confirmed->version() != updated.version()) {
        *reason = "hard_quota_post_cas_confirmation_failed";
        return false;
    }
    *observed_quota_bytes = confirmed->quota().capacity();
    *instance_group_version = confirmed->version();
    KVCM_LOG_INFO("quota_resize_audit event=hard_quota_cas_applied plan_id=%s plan_hash=%s pool_id=%s "
                  "quota_target_id=%s instance_group=%s execution_phase=%s quota_before=%ld quota_after=%ld "
                  "version_before=%ld version_after=%ld",
                  plan.plan_id().c_str(),
                  plan.plan_hash().c_str(),
                  config_.pool_id.c_str(),
                  config_.quota_target_id.c_str(),
                  config_.instance_group.c_str(),
                  plan.execution_phase().c_str(),
                  static_cast<long>(current->quota().capacity()),
                  static_cast<long>(target),
                  static_cast<long>(current->version()),
                  static_cast<long>(confirmed->version()));
    *reason = "hard_quota_cas_confirmed";
    return true;
}

void QuotaPolicyPoller::ReportMetrics(const proto::optimizer::PullQuotaAllocationResponse &plan,
                                      int64_t used_bytes,
                                      const std::string &status) {
    if (!metrics_registry_ || !plan.has_allocation()) {
        return;
    }
    MetricsTags tags{{"pool_id", config_.pool_id}, {"quota_target_id", config_.quota_target_id}};
    RequestContext context("quota-resize-metrics");
    const auto [ec, current] = registry_manager_->GetInstanceGroup(&context, config_.instance_group);
    if (ec == EC_OK && current) {
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              "quota_resize.current_quota_bytes",
                              tags,
                              static_cast<double>(current->quota().capacity()));
        REPORT_DYNAMIC_GAUGE_(
            metrics_registry_, "quota_resize.instance_group_version", tags, static_cast<double>(current->version()));
    }
    REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                          "quota_resize.target_quota_bytes",
                          tags,
                          static_cast<double>(plan.allocation().target_quota_bytes()));
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, "quota_resize.observed_used_bytes", tags, static_cast<double>(used_bytes));
    REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                          "quota_resize.release_consecutive_samples",
                          tags,
                          static_cast<double>(consecutive_release_samples_));
    MetricsTags result_tags = tags;
    result_tags["phase"] = plan.execution_phase();
    result_tags["status"] = status;
    metrics_registry_->GetCounter("quota_resize.operations_total", result_tags) += 1;
    if (status.find("FAILED") != std::string::npos || status.find("TIMEOUT") != std::string::npos ||
        status.find("REJECTED") != std::string::npos) {
        metrics_registry_->GetCounter("quota_resize.failures_total", result_tags) += 1;
    }
}

bool QuotaPolicyPoller::EnsureStub() {
    if (stub_ && !connected_endpoint_.empty()) {
        return true;
    }
    std::vector<ServiceEndpoint> endpoints;
    if (!discovery_->GetAllEndpoints(endpoints)) {
        return false;
    }
    std::vector<std::string> healthy;
    for (const auto &endpoint : endpoints) {
        if (endpoint.healthy && endpoint.port > 0 && (!endpoint.host.empty() || !endpoint.ip.empty())) {
            healthy.push_back(endpoint.host.empty() ? endpoint.ip + ":" + std::to_string(endpoint.port)
                                                    : endpoint.host);
        }
    }
    if (healthy.empty()) {
        return false;
    }
    const std::string endpoint = *std::min_element(healthy.begin(), healthy.end());
    stub_ =
        proto::optimizer::OptimizerService::NewStub(grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials()));
    if (!stub_) {
        return false;
    }
    connected_endpoint_ = endpoint;
    return true;
}

bool QuotaPolicyPoller::ReportResult(const proto::optimizer::PullQuotaAllocationResponse &plan,
                                     const std::string &phase,
                                     const std::string &status,
                                     const std::string &reason,
                                     int64_t current_quota_bytes,
                                     int64_t used_bytes,
                                     int64_t instance_group_version) {
    proto::optimizer::ReportQuotaResizeResultRequest request;
    request.set_trace_id("quota-policy-poller-result");
    request.set_plan_id(plan.plan_id());
    request.set_plan_hash(plan.plan_hash());
    request.set_pool_id(plan.pool_id());
    request.set_quota_target_id(config_.quota_target_id);
    request.set_leader_epoch(plan.leader_epoch());
    request.set_allocation_epoch(plan.allocation_epoch());
    request.set_phase(phase);
    request.set_status(status);
    request.set_reason(reason);
    request.set_observed_quota_bytes(current_quota_bytes);
    request.set_observed_used_bytes(used_bytes);
    request.set_instance_group_version(instance_group_version);
    request.set_execution_revision(plan.execution_revision());
    proto::optimizer::ReportQuotaResizeResultResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(config_.rpc_timeout_ms));
    const auto status_result = stub_->ReportQuotaResizeResult(&context, request, &response);
    const bool success = status_result.ok() && response.header().status().code() == proto::optimizer::OK;
    if (!success && discovery_) {
        discovery_->Refresh();
        stub_.reset();
        connected_endpoint_.clear();
    }
    return success;
}

bool QuotaPolicyPoller::LoadState() {
    std::ifstream input(config_.state_file);
    if (!input.is_open()) {
        return true;
    }
    uint64_t leader_epoch = 0, allocation_epoch = 0, execution_revision = 0;
    std::string plan_hash;
    if (!(input >> leader_epoch >> allocation_epoch >> execution_revision >> plan_hash)) {
        KVCM_LOG_ERROR("quota policy poller state file is invalid: %s", config_.state_file.c_str());
        return false;
    }
    last_leader_epoch_ = leader_epoch;
    last_allocation_epoch_ = allocation_epoch;
    last_execution_revision_ = execution_revision;
    last_plan_hash_ = std::move(plan_hash);
    return true;
}

bool QuotaPolicyPoller::SaveState(uint64_t leader_epoch,
                                  uint64_t allocation_epoch,
                                  uint64_t execution_revision,
                                  const std::string &plan_hash) {
    const std::string temporary = config_.state_file + ".tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output.is_open()) {
            return false;
        }
        output << leader_epoch << ' ' << allocation_epoch << ' ' << execution_revision << ' ' << plan_hash << '\n';
        output.flush();
        if (!output.good()) {
            return false;
        }
    }
    if (std::rename(temporary.c_str(), config_.state_file.c_str()) != 0) {
        return false;
    }
    last_leader_epoch_ = leader_epoch;
    last_allocation_epoch_ = allocation_epoch;
    last_execution_revision_ = execution_revision;
    last_plan_hash_ = plan_hash;
    return true;
}

} // namespace kv_cache_manager
