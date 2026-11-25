#include <algorithm>
#include <array>
#include <bitset>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/service_discovery_url.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/pace_reclaim_free.h"
#include "kv_cache_manager/data_storage/pace_uri.h"
#include "kv_cache_manager/data_storage/reclaim_storage.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "stub_source/kv_cache_manager/data_storage/tair_mempool_backend.h"

namespace kv_cache_manager {
namespace {

bool ParsePaceReclaimGA(const DataStorageUri &uri, PaceReclaimGA &target) {
    if (uri.GetProtocol() != kTairMempoolUriScheme) {
        return false;
    }
    target = {};
    target.ga = ParsePaceGAPath(uri.GetPath());
    if (target.ga == 0) {
        return false;
    }
    if (!uri.HasParam("alloc_generation")) {
        return true;
    }
    uri.GetParamAs("alloc_generation", target.generation);
    uri.GetParamAs("node_id", target.node_id);
    return target.HasIdentity() && IsValidPaceNodeId(target.node_id);
}

// Each request uses either bare GAs or allocation identities. Only the adapter needs to know
// that one Location may depend on two requests, or which of those has already finished.
class PaceReclaimBatch final : public ReclaimStorageBatch {
public:
    explicit PaceReclaimBatch(const std::string &name) : name_(name) {}

    bool AddLocation(const std::vector<ReclaimDeleteUri> &specs) override;
    bool HasRequests() const override { return !attempt_.gas[0].empty() || !attempt_.gas[1].empty(); }
    ReclaimStorageResult Send(const DataStorageManager &storage,
                              const std::string &trace_id,
                              const std::function<const char *()> &stop_reason) override;

private:
    PaceReclaimFreeResult Free(const DataStorageManager &storage,
                               const std::string &trace_id,
                               const std::vector<PaceReclaimGA> &gas,
                               const std::function<bool()> &can_send);
    using Modes = std::bitset<2>;
    static constexpr uint8_t kSkipped = 2;
    struct LocationParts {
        Modes required;
        size_t spec_end;
    };
    struct Attempt {
        std::array<std::vector<PaceReclaimGA>, 2> gas;
        std::vector<LocationParts> locations;
        std::vector<uint8_t> spec_modes;
    };
    Attempt attempt_;
    const std::string name_;
    std::weak_ptr<DataStorageBackend> backend_;
    bool attempted_{false};
    Modes outstanding_{0b11};
    std::array<const char *, 2> retain_reason_{};
};

bool PaceReclaimBatch::AddLocation(const std::vector<ReclaimDeleteUri> &specs) {
    if (specs.empty()) {
        return false;
    }
    const std::array<size_t, 2> starts{attempt_.gas[0].size(), attempt_.gas[1].size()};
    const size_t first_spec = attempt_.spec_modes.size();
    Modes required;
    for (const auto &spec : specs) {
        const DataStorageUri uri(spec.uri);
        PaceReclaimGA ga;
        if (name_.empty() || uri.GetHostName() != name_ || !ParsePaceReclaimGA(uri, ga)) {
            KVCM_LOG_WARN("Invalid PACE reclaim URI storage[%s] uri[%s]", name_.c_str(), spec.uri.c_str());
            for (size_t mode = 0; mode < starts.size(); ++mode) {
                attempt_.gas[mode].resize(starts[mode]);
            }
            attempt_.spec_modes.resize(first_spec);
            return false;
        }
        const uint8_t mode = ga.HasIdentity();
        attempt_.spec_modes.push_back(spec.skip ? kSkipped : mode);
        if (!spec.skip) {
            required.set(mode);
            if (outstanding_[mode]) {
                attempt_.gas[mode].push_back(ga);
            }
        }
    }
    attempt_.locations.push_back({required, attempt_.spec_modes.size()});
    return true;
}

PaceReclaimFreeResult PaceReclaimBatch::Free(const DataStorageManager &storage,
                                             const std::string &trace_id,
                                             const std::vector<PaceReclaimGA> &gas,
                                             const std::function<bool()> &can_send) {
    const bool check_original = std::exchange(attempted_, true);
    return storage.WithRegisteredBackend(name_, [&](const auto &current) -> PaceReclaimFreeResult {
        if (!current || (check_original && backend_.lock() != current)) {
            return {check_original ? PaceReclaimFreeResult::kBackendChanged : PaceReclaimFreeResult::kNotSent};
        }
        auto *backend = dynamic_cast<TairMempoolBackend *>(current.get());
        if (!backend || !can_send()) {
            return {PaceReclaimFreeResult::kNotSent};
        }
        backend_ = current;
        return backend->DeleteForReclaim(gas, trace_id, can_send);
    });
}

ReclaimStorageResult PaceReclaimBatch::Send(const DataStorageManager &storage,
                                            const std::string &trace_id,
                                            const std::function<const char *()> &stop_reason) {
    // Release all per-attempt arrays before waiting for another retry.
    auto attempt = std::exchange(attempt_, Attempt{});
    ReclaimStorageResult result;
    Modes busy, errors;
    auto can_send = [&] { return stop_reason() == nullptr; };
    for (size_t mode = 0; mode < attempt.gas.size(); ++mode) {
        const auto &gas = attempt.gas[mode];
        if (gas.empty()) {
            continue;
        }
        ++result.requests;
        const auto reply = Free(storage, trace_id, gas, can_send);
        switch (reply.status) {
        case PaceReclaimFreeResult::kBusy:
            busy.set(mode);
            ++result.busy_replies;
            result.retry_after_ms = std::max(result.retry_after_ms, reply.retry_after_ms);
            break;
        case PaceReclaimFreeResult::kSuccess:
            result.recovery_confirmed |= reply.recovery_confirmed;
            break;
        case PaceReclaimFreeResult::kTooLarge:
            KVCM_LOG_WARN(
                "PACE Free too large storage[%s] gas[%zu] max_gas[%lu]", name_.c_str(), gas.size(), reply.max_gas);
            retain_reason_[mode] = "batch_too_large";
            break;
        case PaceReclaimFreeResult::kNotSent:
            retain_reason_[mode] = stop_reason();
            if (!retain_reason_[mode]) {
                retain_reason_[mode] = "backend_unavailable";
            }
            break;
        case PaceReclaimFreeResult::kBackendChanged:
            retain_reason_[mode] = "backend_changed";
            break;
        case PaceReclaimFreeResult::kRejected:
            retain_reason_[mode] = "protocol_rejected";
            break;
        case PaceReclaimFreeResult::kReleaseUnconfirmed:
            retain_reason_[mode] = "release_unconfirmed";
            break;
        case PaceReclaimFreeResult::kError:
            errors.set(mode);
            break;
        }
    }
    outstanding_ = busy;
    result.locations.reserve(attempt.locations.size());
    size_t first_spec = 0;
    for (const auto &[required, end] : attempt.locations) {
        const auto index = result.locations.size();
        auto &location = result.locations.emplace_back();
        if ((required & busy).any()) {
            location.action = ReclaimStorageResult::Action::Retry;
        } else {
            for (size_t mode = 0; mode < retain_reason_.size(); ++mode) {
                if (required[mode] && retain_reason_[mode]) {
                    location = {ReclaimStorageResult::Action::Retain, retain_reason_[mode]};
                    break;
                }
            }
        }
        if ((required & errors).any()) {
            for (size_t i = first_spec; i < end; ++i) {
                const auto mode = attempt.spec_modes[i];
                if (mode != kSkipped && errors[mode]) {
                    result.failed_specs.emplace_back(index, i - first_spec);
                }
            }
        }
        first_spec = end;
    }
    return result;
}
} // namespace

bool SupportsAdaptiveDelete(DataStorageType type) { return IsTairMempoolStorageType(type); }

std::unique_ptr<ReclaimStorageBatch> CreateReclaimStorageBatch(DataStorageType type, const std::string &name) {
    return SupportsAdaptiveDelete(type) ? std::make_unique<PaceReclaimBatch>(name) : nullptr;
}

ReclaimProviderCount GetReclaimProviderCount(const std::weak_ptr<DataStorageBackend> &source) {
    const auto backend = std::dynamic_pointer_cast<TairMempoolBackend>(source.lock());
    const auto sample = backend ? backend->GetReclaimProviderCount() : PaceProviderCount{};
    return {sample.count, sample.observed_at};
}

std::string GetReclaimClusterEndpoint(const StorageConfig &config) {
    const auto *spec = dynamic_cast<const TairMemPoolStorageSpec *>(config.storage_spec().get());
    if (!spec) {
        return {};
    }
    const auto &url = spec->service_discovery_url();
    if (url.empty()) {
        return spec->domain();
    }
    const auto query = url.find('?');
    if (query == std::string::npos || (url.rfind("spectrum://", 0) != 0 && url.rfind("vipserver://", 0) != 0)) {
        return url;
    }
    ServiceDiscoveryUrl parsed;
    if (!ServiceDiscoveryUrl::Parse(url, parsed) || parsed.body.empty()) {
        return url;
    }
    // Client timing options don't identify the Meta. Preserve routing options such as Spectrum's port.
    parsed.params.erase("timeout");
    if (parsed.scheme == "spectrum") {
        parsed.params.erase("cache_time");
        parsed.params.erase("retry_time");
    }
    auto endpoint = url.substr(0, query);
    char separator = '?';
    for (const auto &[name, value] : parsed.params) {
        endpoint += separator;
        endpoint.append(name).append("=").append(value);
        separator = '&';
    }
    return endpoint;
}

} // namespace kv_cache_manager
