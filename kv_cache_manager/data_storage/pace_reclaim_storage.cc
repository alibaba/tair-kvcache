#include "kv_cache_manager/data_storage/pace_reclaim_storage.h"

#include <charconv>
#include <string_view>

#include "kv_cache_manager/common/service_discovery_url.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "stub_source/kv_cache_manager/data_storage/tair_mempool_backend.h"

namespace kv_cache_manager {

bool ParsePaceReclaimGA(const DataStorageUri &uri, PaceReclaimGA &target) {
    std::string_view path = uri.GetPath();
    if (uri.GetProtocol() != kTairMempoolUriScheme || path.size() < 2 || path.front() != '/') {
        return false;
    }
    path.remove_prefix(1);
    const auto end = path.data() + path.size();
    target = {};
    const auto result = std::from_chars(path.data(), end, target.ga);
    if (result.ec != std::errc{} || result.ptr != end || target.ga == 0) {
        return false;
    }
    if (!uri.HasParam("alloc_generation")) {
        return true;
    }
    uri.GetParamAs("alloc_generation", target.generation);
    uri.GetParamAs("node_id", target.node_id);
    return target.HasIdentity() && IsValidPaceNodeId(target.node_id);
}

PaceReclaimFreeResult FreePaceGAsForReclaim(const DataStorageManager &storage,
                                            const std::string &trace_id,
                                            const std::string &name,
                                            const std::vector<PaceReclaimGA> &gas,
                                            std::weak_ptr<DataStorageBackend> &original,
                                            bool retry,
                                            const std::function<bool()> &can_send) {
    return storage.WithRegisteredBackend(name, [&](const auto &current) -> PaceReclaimFreeResult {
        if (!current || (retry && original.lock() != current)) {
            return {retry ? PaceReclaimFreeResult::kBackendChanged : PaceReclaimFreeResult::kNotSent};
        }
        auto *backend = dynamic_cast<TairMempoolBackend *>(current.get());
        if (!backend || gas.empty() || !can_send()) {
            return {PaceReclaimFreeResult::kNotSent};
        }
        original = current;
        return backend->DeleteForReclaim(gas, trace_id, can_send);
    });
}

PaceProviderCount GetPaceProviderCount(const std::weak_ptr<DataStorageBackend> &source) {
    const auto backend = std::dynamic_pointer_cast<TairMempoolBackend>(source.lock());
    return backend ? backend->GetReclaimProviderCount() : PaceProviderCount{};
}

std::string GetPaceClusterEndpoint(const StorageConfig &config) {
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
