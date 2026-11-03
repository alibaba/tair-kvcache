#include "kv_cache_manager/metrics/redis_health_metrics.h"

#include <hiredis.h>
#include <shared_mutex>
#include <utility>

#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {
MetricsTags RedisMetricTags(const std::string &group, const std::string &uri) {
    const auto parsed = StandardUri::FromUri(uri);
    return {{"instance_group", group}, {"host", parsed.GetHostName()}, {"port", std::to_string(parsed.GetPort())}};
}
} // namespace

RedisHealthMetrics::RedisHealthMetrics(std::shared_ptr<MetricsRegistry> registry,
                                       std::shared_ptr<MetricsLifecycle> lifecycle,
                                       TargetProvider targets,
                                       Probe probe,
                                       std::function<bool()> active)
    : registry_(std::move(registry))
    , lifecycle_(std::move(lifecycle))
    , targets_(std::move(targets))
    , probe_(std::move(probe))
    , active_(std::move(active)) {}

std::optional<std::string> RedisHealthMetrics::RedisUri(const std::string &storage_type,
                                                      const std::string &storage_uri) {
    auto type = storage_type;
    if (type == "cached") {
        type = StandardUri::FromUri(storage_uri).GetParam("persistent_type");
        if (type.empty()) {
            type = "redis"; // same default as MetaStorageBackendManager
        }
    }
    if (type == "redis" || type == "async_redis") {
        return storage_uri;
    }
    return std::nullopt;
}

bool RedisHealthMetrics::ProbeRedis(const std::string &uri, int timeout_ms) {
    const auto parsed = StandardUri::FromUri(uri);
    if (parsed.GetProtocol() != "redis" || parsed.GetHostName().empty() || parsed.GetPort() <= 0 ||
        parsed.GetPort() > 65535 || timeout_ms <= 0) {
        return false;
    }
    int64_t db = 0;
    if (parsed.HasParam("db")) {
        db = -1;
        parsed.GetParamAs("db", db);
        if (db < 0) {
            return false;
        }
    }
    const timeval timeout{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    std::unique_ptr<redisContext, decltype(&redisFree)> context(
        redisConnectWithTimeout(parsed.GetHostName().c_str(), parsed.GetPort(), timeout), redisFree);
    if (!context || context->err || redisSetTimeout(context.get(), timeout) != REDIS_OK) {
        return false;
    }
    using Reply = std::unique_ptr<redisReply, decltype(&freeReplyObject)>;
    auto status_is = [](const Reply &reply, const char *status) {
        return reply && reply->type == REDIS_REPLY_STATUS && reply->str &&
               std::string(reply->str, reply->len) == status;
    };
    Reply ping(static_cast<redisReply *>(redisCommand(context.get(), "PING")), freeReplyObject);
    if (!status_is(ping, "PONG")) {
        // Match the existing RedisClient authentication convention: a single AUTH token.
        if (!ping || ping->type != REDIS_REPLY_ERROR || !ping->str ||
            std::string(ping->str, ping->len).rfind("NOAUTH", 0) != 0) {
            return false;
        }
        const auto &auth = parsed.GetUserInfo();
        Reply authenticated(
            static_cast<redisReply *>(redisCommand(context.get(), "AUTH %b", auth.data(), auth.size())), freeReplyObject);
        if (!status_is(authenticated, "OK")) {
            return false;
        }
        ping.reset(static_cast<redisReply *>(redisCommand(context.get(), "PING")));
        if (!status_is(ping, "PONG")) {
            return false;
        }
    }
    if (db > 0) {
        Reply selected(static_cast<redisReply *>(redisCommand(context.get(), "SELECT %lld", static_cast<long long>(db))),
                       freeReplyObject);
        if (!status_is(selected, "OK")) {
            return false;
        }
    }
    return true;
}

std::optional<RedisHealthMetrics::Targets> RedisHealthMetrics::CollectTargets(const RegistryManager &registry,
                                                                              const MetaIndexerManager &indexers) {
    const auto config = registry.TryGetMetricsSnapshot();
    const auto runtime = indexers.TryGetStorageBackendConfigs();
    if (!config || !runtime) {
        return std::nullopt;
    }
    Targets targets;
    for (const auto &group : config->groups) {
        const auto cache = group->cache_config();
        if (!cache || !cache->meta_indexer_config()) {
            continue;
        }
        const auto backend = cache->meta_indexer_config()->GetMetaStorageBackendConfig();
        if (backend) {
            if (const auto uri = RedisUri(backend->GetStorageType(), backend->GetStorageUri())) {
                targets.emplace(group->name(), *uri);
            }
        }
    }
    for (const auto &[id, backend] : *runtime) {
        const auto group = config->instance_groups.find(id);
        if (group != config->instance_groups.end()) {
            if (const auto uri = RedisUri(backend.first, backend.second)) {
                targets.emplace(group->second, *uri);
            }
        }
    }
    return targets;
}

RedisHealthMetrics::Series RedisHealthMetrics::GroupByLabels(const Targets &targets) {
    Series series;
    for (const auto &[group, uri] : targets) {
        series[RedisMetricTags(group, uri)].insert(uri);
    }
    return series;
}

void RedisHealthMetrics::RemoveStale(const Series &targets) {
    for (auto it = published_.begin(); it != published_.end();) {
        const auto target = targets.find(it->first);
        if (target == targets.end() || target->second != it->second) {
            registry_->Remove(kMetricName, it->first);
            it = published_.erase(it);
        } else {
            ++it;
        }
    }
}

void RedisHealthMetrics::Report() {
    if (stopped_.load(std::memory_order_relaxed)) {
        return;
    }
    Series snapshot;
    {
        std::shared_lock lock(lifecycle_->mut_, std::try_to_lock);
        if (!lock.owns_lock()) {
            return;
        }
        const auto targets = targets_();
        if (!targets) {
            return;
        }
        snapshot = GroupByLabels(*targets);
        RemoveStale(snapshot);
    }
    // All network work uses private connections outside business/lifecycle locks.
    // Snapshot the indexer/registry maps only twice per cycle, not per endpoint.
    std::map<std::string, bool> results;
    for (const auto &[tags, uris] : snapshot) {
        if (!active_()) {
            break;
        }
        for (const auto &uri : uris) {
            if (stopped_.load(std::memory_order_relaxed)) {
                return;
            }
            if (!active_()) {
                break;
            }
            if (results.find(uri) == results.end()) {
                results.emplace(uri, probe_(uri));
            }
        }
    }
    if (stopped_.load(std::memory_order_relaxed)) {
        return;
    }
    std::shared_lock lock(lifecycle_->mut_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return;
    }
    const auto targets = targets_();
    if (!targets) {
        return;
    }
    const auto current = GroupByLabels(*targets);
    RemoveStale(current);
    for (const auto &[tags, uris] : snapshot) {
        const auto target = current.find(tags);
        if (target == current.end() || target->second != uris) {
            continue;
        }
        bool complete = true;
        bool healthy = true;
        for (const auto &uri : uris) {
            const auto result = results.find(uri);
            if (result == results.end()) {
                complete = false;
                break;
            }
            // A shared host/port is healthy only if every live DB/auth setting is.
            healthy = healthy && result->second;
        }
        if (complete) {
            registry_->GetGauge(kMetricName, tags) = healthy ? 1.0 : 0.0;
            published_[tags] = uris;
        }
    }
}

CoordinationRedisHealthMetrics::CoordinationRedisHealthMetrics(std::shared_ptr<MetricsRegistry> registry,
                                                               std::string uri,
                                                               RedisHealthMetrics::Probe probe)
    : registry_(std::move(registry)), uri_(std::move(uri)), probe_(std::move(probe)) {
    const auto parsed = StandardUri::FromUri(uri_);
    enabled_ = parsed.GetProtocol() == "redis";
    tags_ = {{"host", parsed.GetHostName()}, {"port", std::to_string(parsed.GetPort())}};
}

void CoordinationRedisHealthMetrics::Report() {
    if (!enabled_ || stopped_.load(std::memory_order_relaxed)) {
        return;
    }
    const bool healthy = probe_(uri_);
    if (!stopped_.load(std::memory_order_relaxed)) {
        registry_->GetGauge(kMetricName, tags_) = healthy ? 1.0 : 0.0;
    }
}

} // namespace kv_cache_manager
