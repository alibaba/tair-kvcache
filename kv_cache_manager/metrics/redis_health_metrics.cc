#include "kv_cache_manager/metrics/redis_health_metrics.h"

#include <hiredis.h>
#include <shared_mutex>
#include <utility>

#include "kv_cache_manager/common/standard_uri.h"
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
                                     Probe probe)
    : registry_(std::move(registry))
    , lifecycle_(std::move(lifecycle))
    , targets_(std::move(targets))
    , probe_(std::move(probe)) {}

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

void RedisHealthMetrics::RemoveStale(const Targets &targets) {
    for (auto it = published_.begin(); it != published_.end();) {
        const auto target = targets.find(it->first);
        if (target == targets.end() || target->second != it->second) {
            registry_->Remove(kMetricName, RedisMetricTags(it->first, it->second));
            it = published_.erase(it);
        } else {
            ++it;
        }
    }
}

void RedisHealthMetrics::Report() {
    Targets snapshot;
    {
        std::shared_lock lock(lifecycle_->mut_);
        snapshot = targets_();
        RemoveStale(snapshot);
    }
    // Deduplicate groups sharing the same endpoint/auth/DB configuration in this cycle.
    std::map<std::string, bool> results;
    for (const auto &[group, uri] : snapshot) {
        if (results.find(uri) == results.end()) {
            results.emplace(uri, probe_(uri));
        }
        // Do not hold the lifecycle lock during network I/O. Revalidate before
        // publication so removal, config replacement or demotion cannot revive
        // a series using an obsolete probe result.
        std::shared_lock lock(lifecycle_->mut_);
        const auto current = targets_();
        RemoveStale(current);
        const auto target = current.find(group);
        if (target != current.end() && target->second == uri) {
            registry_->GetGauge(kMetricName, RedisMetricTags(group, uri)) = results.at(uri) ? 1.0 : 0.0;
            published_[group] = uri;
        }
        if (current.empty()) {
            break;
        }
    }
}

} // namespace kv_cache_manager
