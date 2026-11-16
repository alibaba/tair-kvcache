#include "kv_cache_manager/manager/kv_meta_manager.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/kv_meta_instance.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

constexpr std::uint64_t kObjectKeyHashSeed = 0x8bc5'1f2d'671a'94e3ULL;
constexpr std::size_t kScanBatchSize = 1000;

void AddError(RequestContext *request_context, const std::string &message) {
    if (request_context && request_context->error_tracer()) {
        request_context->error_tracer()->AddErrorMsg(message);
    }
}

ErrorCode FirstHardError(ErrorCode current, ErrorCode candidate) {
    if (current != EC_OK) {
        return current;
    }
    return candidate == EC_OK || candidate == EC_NOENT || candidate == EC_EXIST ? EC_OK : candidate;
}

std::string HexEncode(std::string_view input) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output(input.size() * 2, '\0');
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto value = static_cast<unsigned char>(input[i]);
        output[2 * i] = kHex[value >> 4];
        output[2 * i + 1] = kHex[value & 0x0f];
    }
    return output;
}

bool HexDecode(std::string_view input, std::string &output) {
    if (input.empty() || (input.size() & 1U) != 0) {
        return false;
    }
    const auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        return -1;
    };
    output.resize(input.size() / 2);
    for (std::size_t i = 0; i < output.size(); ++i) {
        const int high = nibble(input[2 * i]);
        const int low = nibble(input[2 * i + 1]);
        if (high < 0 || low < 0) {
            output.clear();
            return false;
        }
        output[i] = static_cast<char>((high << 4) | low);
    }
    return true;
}

std::vector<std::vector<std::size_t>> MakeUniqueKeyLayers(const std::vector<std::int64_t> &keys) {
    std::vector<std::vector<std::size_t>> layers;
    std::vector<std::unordered_set<std::int64_t>> used;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        std::size_t layer = 0;
        while (layer < used.size() && !used[layer].insert(keys[i]).second) {
            ++layer;
        }
        if (layer == used.size()) {
            used.emplace_back();
            used.back().insert(keys[i]);
            layers.emplace_back();
        }
        layers[layer].push_back(i);
    }
    return layers;
}

bool ReadLogicalSize(const CacheLocation &location, std::uint64_t &out_size) {
    out_size = 0;
    if (location.spec_size() != 1 || location.location_specs().size() != 1 ||
        location.location_specs().front().name() != kKvMetaValueSpecName) {
        return false;
    }
    const std::string &text = location.location_specs().front().uri();
    return IsValidKvMetaLocation(text, out_size);
}

bool ToValueLocation(const CacheLocation &location, KvMetaManager::ValueLocation &out) {
    std::uint64_t size = 0;
    if (!ReadLogicalSize(location, size)) {
        return false;
    }
    out = {};
    out.type = location.type();
    out.value_size = size;
    out.specs.reserve(location.location_specs().size());
    for (const auto &spec : location.location_specs()) {
        out.specs.emplace_back(spec.name(), spec.uri());
    }
    return true;
}

std::uint64_t SaturatingAdd(std::uint64_t lhs, std::uint64_t rhs) {
    return rhs > std::numeric_limits<std::uint64_t>::max() - lhs ? std::numeric_limits<std::uint64_t>::max()
                                                                 : lhs + rhs;
}

bool GetKvMetaCapacity(const InstanceGroup &group,
                       const std::shared_ptr<DataStorageManager> &storage_manager,
                       std::int64_t &capacity,
                       DataStorageType &storage_type) {
    if (!storage_manager || group.storage_candidates().empty() || group.quota().capacity() <= 0) {
        return false;
    }
    storage_type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    for (const auto &name : group.storage_candidates()) {
        const auto backend = storage_manager->GetDataStorageBackend(name);
        if (!backend || !IsTairMempoolStorageType(backend->GetType()) ||
            (storage_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN && storage_type != backend->GetType())) {
            return false;
        }
        storage_type = backend->GetType();
    }
    capacity = group.quota().capacity();
    for (const auto &quota : group.quota().quota_config()) {
        if (quota.storage_spec() == storage_type) {
            if (quota.capacity() <= 0) {
                return false;
            }
            capacity = std::min(capacity, quota.capacity());
        }
    }
    return storage_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
}

bool UsesMemoryPrimaryMetadata(const InstanceGroup &group) {
    const auto cache = group.cache_config();
    const auto indexer = cache ? cache->meta_indexer_config() : nullptr;
    const auto backend = indexer ? indexer->GetMetaStorageBackendConfig() : nullptr;
    return backend && backend->GetMemoryPrimary();
}

bool MatchesKvMetaStoragePreference(CachePreferStrategy preference, DataStorageType storage_type) {
    switch (preference) {
    case CachePreferStrategy::CPS_ALWAYS_TAIR_MEMPOOL:
        return storage_type == DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
    case CachePreferStrategy::CPS_ALWAYS_TAIR_MEMPOOL_SSD:
        return storage_type == DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD;
    case CachePreferStrategy::CPS_ALWAYS_3FS:
    case CachePreferStrategy::CPS_ALWAYS_MOONCAKE:
    case CachePreferStrategy::CPS_ALWAYS_VCNS_3FS:
        return false;
    default:
        return true; // PREFER strategies explicitly allow fallback.
    }
}

} // namespace

struct KvMetaManager::SessionItem {
    std::int64_t internal_key = 0;
    std::string location_id;
    CacheLocationConstPtr location;
    std::uint64_t value_size = 0;
};

struct KvMetaManager::ExactLocation {
    std::int64_t internal_key = 0;
    std::string location_id;
    ErrorCode ec = EC_UNKNOWN;
    CacheLocationConstPtr location;
};

class KvMetaManager::MetadataMutationGuard {
public:
    MetadataMutationGuard(const KvMetaManager &owner,
                          const std::string &instance_id,
                          const std::vector<std::int64_t> &keys)
        : count_(owner.mutation_lock_count_), wait_(owner.mutation_lock_wait_us_), hold_(owner.mutation_lock_hold_us_) {
        std::vector<std::size_t> shards;
        shards.reserve(keys.size());
        const auto instance_hash = std::hash<std::string>{}(instance_id);
        for (const auto key : keys) {
            shards.push_back((instance_hash ^ std::hash<std::int64_t>{}(key)) %
                             owner.metadata_mutation_mutexes_.size());
        }
        std::sort(shards.begin(), shards.end());
        shards.erase(std::unique(shards.begin(), shards.end()), shards.end());
        locks_.reserve(shards.size());
        const auto begin = TimestampUtil::GetCurrentTimeUs();
        for (const auto shard : shards) {
            locks_.emplace_back(owner.metadata_mutation_mutexes_[shard]);
        }
        acquired_us_ = TimestampUtil::GetCurrentTimeUs();
        wait_ += std::max<int64_t>(0, acquired_us_ - begin);
        count_ += 1;
    }
    ~MetadataMutationGuard() { unlock(); }
    void unlock() {
        if (locks_.empty()) {
            return;
        }
        const auto elapsed = std::max<int64_t>(0, TimestampUtil::GetCurrentTimeUs() - acquired_us_);
        locks_.clear();
        hold_ += elapsed;
    }

private:
    std::vector<std::unique_lock<std::mutex>> locks_;
    Counter count_, wait_, hold_;
    int64_t acquired_us_ = 0;
};

KvMetaManager::MetadataMutationGuard KvMetaManager::LockMetadataMutation(const std::string &instance_id,
                                                                         const std::vector<std::int64_t> &keys) const {
    return MetadataMutationGuard(*this, instance_id, keys);
}

std::string KvMetaManager::UsageReservationKey(const std::string &instance_id, const SessionItem &item) {
    std::string key = instance_id + '\0' + std::to_string(item.internal_key) + '\0' + item.location_id;
    if (item.location) {
        key += '\0' + std::to_string(item.location->create_time());
        if (!item.location->location_specs().empty()) {
            key += '\0' + item.location->location_specs().front().uri();
        }
    }
    return key;
}

void KvMetaManager::ReserveUncertainUsage(const std::string &instance_id, const SessionItem &item) {
    std::lock_guard<std::mutex> lock(uncertain_usage_mutex_);
    uncertain_usage_.insert(UsageReservationKey(instance_id, item));
}

bool KvMetaManager::TakeUncertainUsage(const std::string &instance_id, const SessionItem &item) {
    std::lock_guard<std::mutex> lock(uncertain_usage_mutex_);
    return uncertain_usage_.erase(UsageReservationKey(instance_id, item)) != 0;
}

bool KvMetaManager::HasUncertainUsage(const std::string &instance_id, const SessionItem &item) {
    std::lock_guard<std::mutex> lock(uncertain_usage_mutex_);
    return !uncertain_usage_.empty() && uncertain_usage_.count(UsageReservationKey(instance_id, item)) != 0;
}

class KvMetaWriteSessionManager {
public:
    using Clock = std::chrono::steady_clock;

    enum class PutResult {
        kOk,
        kDuplicate,
        kStopped,
        kFull,
        kExpired
    };

    struct Session {
        std::string internal_instance_id;
        Clock::time_point deadline;
        std::vector<KvMetaManager::SessionItem> items;
        bool cleanup_only = false;
        bool release_allocation = false;
        std::shared_ptr<const InstanceInfo> registration;
    };

    KvMetaWriteSessionManager(KvMetaManager *owner, std::size_t max_sessions)
        : owner_(owner), max_sessions_(max_sessions) {
        if (owner_ && owner_->cache_manager_->metrics_registry()) {
            auto metrics = owner_->cache_manager_->metrics_registry();
            active_sessions_ = metrics->GetGauge("kv_meta.active_write_sessions");
            metrics->GetGauge("kv_meta.write_session_capacity") = max_sessions_;
            rejected_sessions_ = metrics->GetCounter("kv_meta.write_session_rejected_total");
            active_sessions_ = 0;
            rejected_sessions_ += 0;
        }
    }
    ~KvMetaWriteSessionManager() { StopAndDiscard(); }

    bool Start() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (thread_.joinable() && !stopping_) {
                return true;
            }
        }
        // A finished or stopping worker remains joinable until joined.
        // Never join while holding the mutex needed by its exit path.
        if (thread_.joinable()) {
            thread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = false;
        }
        try {
            thread_ = std::thread([this]() { ExpireLoop(); });
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            KVCM_LOG_ERROR("failed to start KVMeta session worker: %s", e.what());
            return false;
        }
        return true;
    }

    void RequestStop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
    }

    void StopAndDiscard() noexcept {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        RequestStop();
        if (thread_.joinable()) {
            thread_.join();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        sessions_.clear();
        expirations_.clear();
        active_sessions_ = 0;
    }

    PutResult Availability() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return PutResult::kStopped;
        }
        if (sessions_.size() >= max_sessions_) {
            rejected_sessions_ += 1;
            return PutResult::kFull;
        }
        return PutResult::kOk;
    }

    PutResult Put(const std::string &session_id, Session session) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return PutResult::kStopped;
        }
        if (session.deadline <= Clock::now()) {
            return PutResult::kExpired;
        }
        if (sessions_.size() >= max_sessions_) {
            rejected_sessions_ += 1;
            return PutResult::kFull;
        }
        const auto deadline = session.deadline;
        if (!sessions_.emplace(session_id, std::move(session)).second) {
            return PutResult::kDuplicate;
        }
        active_sessions_ = sessions_.size();
        expirations_.emplace(deadline, session_id);
        if (expirations_.begin()->second == session_id) {
            condition_.notify_one();
        }
        return PutResult::kOk;
    }

    ErrorCode
    Take(const std::string &session_id, const std::string &internal_instance_id, std::size_t item_count, Session &out) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return EC_NOENT;
        }
        if (it->second.internal_instance_id != internal_instance_id) {
            return EC_BADARGS;
        }
        if (it->second.cleanup_only) {
            return EC_NOENT;
        }
        if (it->second.items.size() != item_count) {
            return EC_MISMATCH;
        }
        expirations_.erase({it->second.deadline, session_id});
        out = std::move(it->second);
        sessions_.erase(it);
        active_sessions_ = sessions_.size();
        return out.deadline <= Clock::now() ? EC_TIMEOUT : EC_OK;
    }

    void ScheduleCleanup(const std::string &session_id, Session session, bool release_allocation = true) noexcept {
        try {
            session.deadline = Clock::now() + std::chrono::seconds(1);
            session.cleanup_only = true;
            session.release_allocation = release_allocation;
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            const auto deadline = session.deadline;
            if (!sessions_.emplace(session_id, std::move(session)).second) {
                KVCM_LOG_ERROR("failed to reschedule KVMeta session cleanup for [%s]", session_id.c_str());
                return;
            }
            active_sessions_ = sessions_.size();
            expirations_.emplace(deadline, session_id);
            condition_.notify_one();
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("failed to schedule KVMeta session cleanup: %s", e.what());
        } catch (...) { KVCM_LOG_ERROR("failed to schedule KVMeta session cleanup with unknown exception"); }
    }

private:
    void ExpireLoop() noexcept {
        while (true) {
            std::vector<std::pair<std::string, Session>> expired;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (stopping_) {
                    return;
                }
                if (expirations_.empty()) {
                    condition_.wait(lock, [this]() { return stopping_ || !expirations_.empty(); });
                    continue;
                }
                const auto next = expirations_.begin()->first;
                if (Clock::now() < next) {
                    condition_.wait_until(lock, next);
                    continue;
                }
                const auto now = Clock::now();
                while (!expirations_.empty() && expirations_.begin()->first <= now) {
                    auto entry = expirations_.extract(expirations_.begin());
                    auto session = sessions_.find(entry.value().second);
                    if (session != sessions_.end()) {
                        expired.emplace_back(session->first, std::move(session->second));
                        sessions_.erase(session);
                    }
                }
                active_sessions_ = sessions_.size();
            }
            for (auto &[session_id, session] : expired) {
                const bool release_allocation = session.release_allocation;
                if (owner_ && !owner_->ExpireSession(session_id,
                                                     session.internal_instance_id,
                                                     session.items,
                                                     release_allocation,
                                                     session.registration)) {
                    // A retry must not turn an unacknowledged write into a
                    // physically reclaimable allocation.
                    ScheduleCleanup(session_id, std::move(session), release_allocation);
                }
            }
        }
    }

    Gauge active_sessions_;
    mutable Counter rejected_sessions_;
    std::set<std::pair<Clock::time_point, std::string>> expirations_;
    KvMetaManager *owner_ = nullptr;
    std::size_t max_sessions_ = 0;
    mutable std::mutex mutex_;
    std::mutex lifecycle_mutex_;
    std::condition_variable condition_;
    bool stopping_ = true;
    std::unordered_map<std::string, Session> sessions_;
    std::thread thread_;
};

namespace {
std::uint64_t SaturatingMultiply(std::uint64_t a, std::uint64_t b) {
    return a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a ? std::numeric_limits<std::uint64_t>::max()
                                                                       : a * b;
}
} // namespace

class KvMetaReclaimer {
public:
    explicit KvMetaReclaimer(KvMetaManager *owner) : owner_(owner) {}
    ~KvMetaReclaimer() { StopAndJoin(); }

    bool Start() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (thread_.joinable() && !stopping_) {
                return true;
            }
        }
        // A finished or stopping worker remains joinable until joined.
        // Never join while holding the mutex needed by its exit path.
        if (thread_.joinable()) {
            thread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = false;
            cancel_deletes_ = std::make_shared<std::atomic<bool>>(false);
        }
        try {
            thread_ = std::thread([this]() { Loop(); });
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            KVCM_LOG_ERROR("failed to start KVMeta reclaimer: %s", e.what());
            return false;
        }
        return true;
    }

    void RequestStop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            if (cancel_deletes_) {
                cancel_deletes_->store(true, std::memory_order_release);
            }
        }
        condition_.notify_all();
    }

    void StopAndJoin() noexcept {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        RequestStop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    struct Candidate {
        std::string instance_id;
        std::int64_t key = 0;
        std::int64_t last_access_time_us = 0;
    };

    using PendingLocationKey = CacheReclaimer::PendingLocationKey;

    struct PendingDelete {
        std::string group;
        std::vector<PendingLocationKey> ids;
        std::uint64_t bytes = 0;
        std::future<PlanExecuteResult> future;
        std::chrono::steady_clock::time_point submitted_at;
        std::chrono::steady_clock::time_point credit_deadline;
        bool credit_enabled = true;
    };

    static PendingLocationKey PendingId(const std::string &instance, std::int64_t key,
                                        const std::string &location) {
        return {instance, key, location};
    }

    bool IsPendingKey(const Candidate &candidate) const {
        const auto it = pending_ids_.lower_bound(PendingId(candidate.instance_id, candidate.key, {}));
        return it != pending_ids_.end() && it->instance_id == candidate.instance_id &&
               it->block_key == candidate.key;
    }

    void PollDeletes() {
        for (auto it = pending_deletes_.begin(); it != pending_deletes_.end();) {
            if (it->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                if (it->credit_enabled && std::chrono::steady_clock::now() >= it->credit_deadline) {
                    it->credit_enabled = false;
                    owner_->ObserveReclaimStage(it->group, "delete_timeout", "error");
                    KVCM_LOG_ERROR("KVMeta reclaim pending beyond credit deadline, group[%s] objects[%zu] bytes[%lu]",
                                   it->group.c_str(),
                                   it->ids.size(),
                                   it->bytes);
                }
                ++it;
                continue;
            }
            const auto result = it->future.get();
            owner_->ObserveReclaimGauge(it->group, "deleted_count", result.reclaimed_locations);
            owner_->ObserveReclaimGauge(it->group, "deleted_bytes", result.reclaimed_bytes);
            owner_->AddReclaimCounter(it->group, "deleted_objects_total", result.reclaimed_locations);
            owner_->AddReclaimCounter(it->group, "deleted_bytes_total", result.reclaimed_bytes);
            owner_->AddReclaimCounter(it->group, "physical_deleted_objects_total", result.physical_deleted_locations);
            owner_->AddReclaimCounter(it->group, "physical_deleted_bytes_total", result.physical_deleted_bytes);
            owner_->ObserveReclaimStage(it->group, "delete_complete", result.status == EC_OK ? "success" : "error");
            if (result.status != EC_OK) {
                KVCM_LOG_ERROR("KVMeta async reclaim failed group[%s] ec[%d] detail[%s]",
                               it->group.c_str(),
                               result.status,
                               result.error_message.c_str());
            }
            for (const auto &id : it->ids) {
                pending_ids_.erase(id);
            }
            it = pending_deletes_.erase(it);
        }
    }

    const CacheReclaimerAsyncDeleteConfig &DeleteConfig() const {
        static const CacheReclaimerAsyncDeleteConfig defaults;
        auto reclaimer = owner_->cache_manager_->cache_reclaimer();
        return reclaimer ? reclaimer->async_delete_config() : defaults;
    }

    const CacheReclaimerGroupLruConfig &LruConfig() const {
        static const CacheReclaimerGroupLruConfig defaults;
        auto reclaimer = owner_->cache_manager_->cache_reclaimer();
        return reclaimer ? reclaimer->group_lru_config() : defaults;
    }

    bool CanSubmit(const std::string &group, std::uint64_t bytes, std::size_t count) const {
        const auto &config = DeleteConfig();
        std::uint64_t total_bytes = 0, group_bytes = 0, group_count = 0;
        for (const auto &pending : pending_deletes_) {
            total_bytes = SaturatingAdd(total_bytes, pending.bytes);
            if (pending.group == group) {
                group_bytes = SaturatingAdd(group_bytes, pending.bytes);
                group_count = SaturatingAdd(group_count, pending.ids.size());
            }
        }
        return pending_deletes_.size() < config.pending_delete_handler_limit &&
               SaturatingAdd(total_bytes, bytes) <= config.pending_bytes_limit &&
               SaturatingAdd(group_bytes, bytes) <= config.pending_bytes_limit_per_group_type &&
               SaturatingAdd(group_count, count) <= config.pending_location_limit_per_group_type;
    }

    bool Stopping() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopping_;
    }

    std::pair<std::size_t, std::size_t> BatchSizes() const {
        if (!owner_ || !owner_->cache_manager_ || !owner_->cache_manager_->cache_reclaimer()) {
            return {256, 64};
        }
        RequestContext context("kv_meta_reclaimer_config");
        const std::size_t sampling = owner_->cache_manager_->cache_reclaimer()->GetSamplingSize(&context);
        const std::size_t batching = owner_->cache_manager_->cache_reclaimer()->GetBatchingSize(&context);
        return {sampling == 0 ? 256 : sampling, batching == 0 ? 64 : batching};
    }

    std::uint32_t SleepIntervalMs() const {
        if (!owner_ || !owner_->cache_manager_ || !owner_->cache_manager_->cache_reclaimer()) {
            return 1000;
        }
        RequestContext context("kv_meta_reclaimer_config");
        return std::max<std::uint32_t>(1, owner_->cache_manager_->cache_reclaimer()->GetSleepIntervalMs(&context));
    }

    void Loop() noexcept {
        while (!Stopping()) {
            try {
                PollDeletes();
                if (!owner_->recovery_complete_.load(std::memory_order_acquire)) {
                    const ErrorCode ec = owner_->DoRecover();
                    if (ec != EC_OK && ec != EC_SERVICE_NOT_LEADER) {
                        KVCM_LOG_WARN("KVMeta recovery failed, ec[%d]", static_cast<int>(ec));
                    }
                } else {
                    RequestContext context("kv_meta_reclaim_groups");
                    const auto [ec, groups] = owner_->registry_manager_->ListInstanceGroup(&context);
                    if (ec != EC_OK) {
                        KVCM_LOG_WARN("KVMeta group listing failed, ec[%d]", static_cast<int>(ec));
                    }
                    for (const auto &group : groups) {
                        if (Stopping()) {
                            break;
                        }
                        if (group) {
                            ReclaimGroup(group);
                        }
                    }
                }
            } catch (const std::exception &e) {
                KVCM_LOG_WARN("KVMeta reclaim round failed: %s", e.what());
            } catch (...) { KVCM_LOG_WARN("KVMeta reclaim round failed with unknown exception"); }

            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait_for(lock, std::chrono::milliseconds(SleepIntervalMs()), [this]() { return stopping_; });
        }
    }

    void ReclaimGroup(const std::shared_ptr<const InstanceGroup> &group) {
        RequestContext context("kv_meta_reclaim");
        if (!group || !group->cache_config() || !group->cache_config()->reclaim_strategy()) {
            return;
        }
        const auto strategy = group->cache_config()->reclaim_strategy();
        const double threshold = strategy->trigger_strategy().used_percentage();
        if (strategy->reclaim_policy() != ReclaimPolicy::POLICY_LRU || !std::isfinite(threshold) || threshold < 0.0 ||
            threshold >= 1.0 || strategy->delay_before_delete_ms() < 0) {
            return;
        }
        const auto [instances_ec, instances] = owner_->registry_manager_->ListInstanceInfo(&context, group->name());
        if (instances_ec != EC_OK || instances.empty()) {
            return;
        }

        std::uint64_t group_usage = 0;
        std::vector<std::pair<InstanceInfoConstPtr, std::shared_ptr<MetaIndexer>>> indexed_instances;
        for (const auto &instance : instances) {
            if (!instance || !IsKvMetaInstance(*instance)) {
                // KVMeta groups are dedicated. Do not let this side path
                // inspect or reclaim ordinary KV-cache metadata.
                return;
            }
            auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                return;
            }
            group_usage = SaturatingAdd(group_usage, indexer->GetStorageUsage());
            indexed_instances.emplace_back(instance, std::move(indexer));
        }

        std::int64_t capacity = 0;
        DataStorageType storage_type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        if (!GetKvMetaCapacity(*group, owner_->registry_manager_->data_storage_manager(), capacity, storage_type)) {
            return;
        }
        const std::uint64_t target = static_cast<std::uint64_t>(static_cast<long double>(capacity) * threshold);
        owner_->ObserveReclaimGauge(group->name(), "usage_bytes", group_usage);
        owner_->ObserveReclaimGauge(group->name(), "target_bytes", target);
        std::uint64_t pending_bytes = 0, pending_credit_bytes = 0;
        std::size_t pending_requests = 0;
        std::size_t pending_locations = 0;
        for (const auto &pending : pending_deletes_) {
            if (pending.group == group->name()) {
                pending_bytes = SaturatingAdd(pending_bytes, pending.bytes);
                if (pending.credit_enabled) {
                    pending_credit_bytes = SaturatingAdd(pending_credit_bytes, pending.bytes);
                }
                ++pending_requests;
                pending_locations += pending.ids.size();
            }
        }
        owner_->ObserveReclaimGauge(group->name(), "pending_bytes", pending_bytes);
        owner_->ObserveReclaimGauge(group->name(), "pending_credit_bytes", pending_credit_bytes);
        owner_->ObserveReclaimGauge(group->name(), "pending_objects", pending_locations);
        owner_->ObserveReclaimGauge(group->name(), "pending_requests", pending_requests);
        std::uint64_t oldest_ms = 0;
        for (const auto &pending : pending_deletes_) {
            if (pending.group == group->name()) {
                oldest_ms =
                    std::max(oldest_ms,
                             static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                            std::chrono::steady_clock::now() - pending.submitted_at)
                                                            .count()));
            }
        }
        owner_->ObserveReclaimGauge(group->name(), "pending_oldest_ms", oldest_ms);
        const std::uint64_t projected_usage =
            group_usage > pending_credit_bytes ? group_usage - pending_credit_bytes : 0;
        if (projected_usage <= target) {
            owner_->ObserveReclaimStage(group->name(), "round", "skipped");
            return;
        }

        if (!CanSubmit(group->name(), 0, 1)) {
            owner_->ObserveReclaimStage(group->name(), "pending_limit", "skipped");
            return;
        }
        const auto [sampling_baseline, batching_size] = BatchSizes();
        const auto &lru_config = LruConfig();
        const auto sampling_size = std::min<std::uint64_t>(lru_config.max_sampling_size,
            SaturatingAdd(pending_locations, std::max<std::uint64_t>(sampling_baseline,
                SaturatingMultiply(batching_size, lru_config.min_sampling_ratio))));
        const auto round_limit = SaturatingMultiply(batching_size, lru_config.max_delete_requests_per_round);
        const std::int64_t round_us = TimestampUtil::GetCurrentTimeUs();
        std::vector<Candidate> candidates;
        bool round_error = false;
        for (const auto &[instance, indexer] : indexed_instances) {
            ReclaimCandidateVector sampled;
            const auto sample_ec =
                indexer->SampleReclaimCandidates(&context, static_cast<std::int64_t>(sampling_size), sampled, true);
            owner_->ObserveReclaimStage(group->name(), "sample", sample_ec == EC_OK ? "success" : "error");
            if (sample_ec != EC_OK) {
                round_error = true;
                KVCM_LOG_WARN("KVMeta reclaim sampling failed group[%s] instance[%s] ec[%d]",
                              group->name().c_str(),
                              instance->instance_id().c_str(),
                              sample_ec);
                continue;
            }
            for (const auto &candidate : sampled) {
                candidates.push_back({instance->instance_id(), candidate.key, candidate.last_access_time_us});
            }
        }
        // Ordinary Cache also excludes pending locations before selecting
        // victims. Pending keys must not consume either metadata-read work or
        // the sampling window while their reader grace is outstanding.
        std::map<std::string, KeyVector> pending_keys;
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
            [this, &pending_keys](const Candidate &candidate) {
                if (!IsPendingKey(candidate)) {
                    return false;
                }
                pending_keys[candidate.instance_id].push_back(candidate.key);
                return true;
            }), candidates.end());
        for (const auto &[instance_id, keys] : pending_keys) {
            if (auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance_id)) {
                // Yield rejected keys just like ordinary Cache. Otherwise an
                // oldest-shard sample can repeatedly return only pending keys.
                owner_->AddReclaimCounter(group->name(), "maintenance_touch_keys_total",
                                         indexer->TouchKeysForMaintenance(keys));
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate &lhs, const Candidate &rhs) {
            if (lhs.last_access_time_us != rhs.last_access_time_us) {
                return lhs.last_access_time_us < rhs.last_access_time_us;
            }
            if (lhs.instance_id != rhs.instance_id) {
                return lhs.instance_id < rhs.instance_id;
            }
            return lhs.key < rhs.key;
        });

        owner_->ObserveReclaimGauge(group->name(), "candidate_count", candidates.size());
        KVCM_LOG_DEBUG("KVMeta LRU round round[%ld] group[%s] usage[%lu] target[%lu] requested[%lu] candidates[%lu] "
                       "batching[%lu]",
                       round_us,
                       group->name().c_str(),
                       group_usage,
                       target,
                       sampling_size,
                       candidates.size(),
                       batching_size);
        std::size_t selected_count = 0;
        std::uint64_t selected_bytes = 0;
        std::map<std::string, std::vector<KvMetaManager::SessionItem>> by_instance;
        // Grow candidate-read batches only while more victims are needed. A
        // nearly balanced group must not deserialize the entire sample each round.
        std::size_t read_batch_size = std::min<std::size_t>(16, batching_size);
        for (std::size_t offset = 0;
             offset < candidates.size() && selected_count < round_limit && selected_bytes < projected_usage - target;) {
            const auto end = std::min(candidates.size(), offset + read_batch_size);
            std::map<std::string, KeyVector> keys_by_instance;
            for (auto i = offset; i < end; ++i) {
                keys_by_instance[candidates[i].instance_id].push_back(candidates[i].key);
            }
            std::map<std::string, std::unordered_map<std::int64_t, CacheLocationMap>> locations_by_instance;
            for (const auto &[instance_id, keys] : keys_by_instance) {
                auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance_id);
                if (!indexer) {
                    continue;
                }
                CacheLocationMapVector maps;
                const auto result = indexer->GetLocationMapsForMaintenance(&context, keys, maps);
                if (result.error_codes.size() != keys.size() || maps.size() != keys.size()) {
                    round_error = true;
                    continue;
                }
                auto &locations = locations_by_instance[instance_id];
                for (size_t i = 0; i < keys.size(); ++i) {
                    if (result.error_codes[i] == EC_OK) {
                        locations.emplace(keys[i], std::move(maps[i]));
                    }
                }
            }
            for (auto i = offset; i < end && selected_count < round_limit && selected_bytes < projected_usage - target;
                 ++i) {
                const auto &candidate = candidates[i];
                auto &maps = locations_by_instance[candidate.instance_id];
                const auto found = maps.find(candidate.key);
                if (found == maps.end()) {
                    continue;
                }
                for (const auto &[location_id, location] : found->second) {
                    if (!location || location->status() != CLS_SERVING ||
                        location_id.compare(0, kKvMetaLocationIdPrefix.size(), kKvMetaLocationIdPrefix) != 0 ||
                        pending_ids_.count(PendingId(candidate.instance_id, candidate.key, location_id)) != 0) {
                        continue;
                    }
                    std::uint64_t size = 0;
                    if (owner_->ValidateLocation(&context, candidate.key, location_id, *location, size) != EC_OK) {
                        continue;
                    }
                    KvMetaManager::SessionItem item{candidate.key, location_id, location, size};
                    // Do not compete with an in-progress/uncertain commit's
                    // accounting and cleanup ownership.
                    if (owner_->HasUncertainUsage(candidate.instance_id, item)) {
                        owner_->ObserveReclaimStage(group->name(), "uncertain_commit", "skipped");
                        continue;
                    }
                    by_instance[candidate.instance_id].push_back(std::move(item));
                    selected_bytes = SaturatingAdd(selected_bytes, size);
                    ++selected_count;
                    if (selected_count >= round_limit || selected_bytes >= projected_usage - target) {
                        break;
                    }
                }
            }
            offset = end;
            read_batch_size = std::min(batching_size, read_batch_size * 2);
        }

        owner_->ObserveReclaimGauge(group->name(), "selected_count", selected_count);
        std::uint64_t submitted_bytes = 0;
        std::size_t submitted_count = 0;
        std::size_t requests = 0;
        for (auto &[instance_id, items] : by_instance) {
            for (std::size_t begin = 0; begin < items.size(); begin += batching_size) {
                if (requests >= lru_config.max_delete_requests_per_round) {
                    break;
                }
                CacheLocationDelRequest request;
                request.instance_id = instance_id;
                request.delay = std::chrono::milliseconds(strategy->delay_before_delete_ms());
                request.authoritative_read = true;
                request.cancel_before_physical = cancel_deletes_;
                std::vector<PendingLocationKey> ids;
                std::uint64_t bytes = 0;
                const auto end = std::min(items.size(), begin + batching_size);
                for (auto i = begin; i < end; ++i) {
                    const auto &item = items[i];
                    const auto id = PendingId(instance_id, item.internal_key, item.location_id);
                    if (!pending_ids_.insert(id).second) {
                        continue;
                    }
                    ids.push_back(id);
                    request.block_keys.push_back(item.internal_key);
                    request.location_ids.push_back({item.location_id});
                    request.expected_location_values.push_back({item.location->ToJsonString()});
                    bytes = SaturatingAdd(bytes, item.value_size);
                }
                if (ids.empty()) {
                    continue;
                }
                if (!CanSubmit(group->name(), bytes, ids.size())) {
                    for (const auto &id : ids) {
                        pending_ids_.erase(id);
                    }
                    owner_->ObserveReclaimStage(group->name(), "pending_limit", "skipped");
                    break;
                }
                auto result = owner_->cache_manager_->schedule_plan_executor()->SubmitAsync(request);
                if (!result.accepted) {
                    for (const auto &id : ids) {
                        pending_ids_.erase(id);
                    }
                    round_error = true;
                    owner_->ObserveReclaimStage(group->name(), "delete_submit", "error");
                    continue;
                }
                submitted_count += ids.size();
                submitted_bytes = SaturatingAdd(submitted_bytes, bytes);
                const auto now = std::chrono::steady_clock::now();
                pending_deletes_.push_back(
                    {group->name(),
                     std::move(ids),
                     bytes,
                     std::move(result.future),
                     now,
                     now + request.delay + std::chrono::milliseconds(DeleteConfig().inflight_delete_timeout_ms)});
                ++requests;
                owner_->ObserveReclaimStage(group->name(), "delete_submit", "success");
            }
        }
        owner_->AddReclaimCounter(group->name(), "submitted_objects_total", submitted_count);
        owner_->AddReclaimCounter(group->name(), "submitted_bytes_total", submitted_bytes);
        owner_->ObserveReclaimGauge(group->name(), "round_time_us", TimestampUtil::GetCurrentTimeUs() - round_us);
        owner_->ObserveReclaimGauge(group->name(), "submitted_count", submitted_count);
        owner_->ObserveReclaimGauge(group->name(), "submitted_bytes", submitted_bytes);
        owner_->ObserveReclaimStage(group->name(),
                                    "round",
                                    round_error             ? "error"
                                    : submitted_count == 0 ? "skipped"
                                                            : "success");
    }

    KvMetaManager *owner_ = nullptr;
    mutable std::mutex mutex_;
    std::mutex lifecycle_mutex_;
    std::condition_variable condition_;
    bool stopping_ = true;
    std::shared_ptr<std::atomic<bool>> cancel_deletes_;
    std::vector<PendingDelete> pending_deletes_;
    std::set<PendingLocationKey> pending_ids_;
    std::thread thread_;
};

KvMetaManager::KvMetaManager(std::shared_ptr<CacheManager> cache_manager,
                             std::shared_ptr<RegistryManager> registry_manager)
    : KvMetaManager(std::move(cache_manager), std::move(registry_manager), Limits{}) {}

KvMetaManager::KvMetaManager(std::shared_ptr<CacheManager> cache_manager,
                             std::shared_ptr<RegistryManager> registry_manager,
                             Limits limits)
    : cache_manager_(std::move(cache_manager)), registry_manager_(std::move(registry_manager)), limits_(limits) {}

KvMetaManager::~KvMetaManager() { Shutdown(); }

bool KvMetaManager::Init() {
    if (initialized_.load(std::memory_order_acquire)) {
        return true;
    }
    if (!cache_manager_ || !registry_manager_ || !cache_manager_->meta_indexer_manager() ||
        !registry_manager_->data_storage_manager() || limits_.max_batch_items == 0 || limits_.max_key_bytes == 0 ||
        limits_.max_instance_id_bytes == 0 || limits_.max_instance_group_bytes == 0 ||
        limits_.max_write_session_id_bytes == 0 || limits_.max_user_data_bytes == 0 ||
        limits_.max_active_write_sessions == 0 || limits_.max_value_bytes == 0 || limits_.max_batch_bytes == 0 ||
        limits_.max_write_timeout_seconds <= 0) {
        KVCM_LOG_ERROR("KVMeta manager init failed: dependency or limits are invalid");
        return false;
    }
    if (const auto metrics = cache_manager_->metrics_registry()) {
        mutation_lock_count_ = metrics->GetCounter("kv_meta.mutation_lock_count");
        mutation_lock_wait_us_ = metrics->GetCounter("kv_meta.mutation_lock_wait_us");
        mutation_lock_hold_us_ = metrics->GetCounter("kv_meta.mutation_lock_hold_us");
        mutation_lock_count_ += 0;
        mutation_lock_wait_us_ += 0;
        mutation_lock_hold_us_ += 0;
    }
    data_storage_selector_ =
        std::make_unique<DataStorageSelector>(cache_manager_->meta_indexer_manager(), registry_manager_);
    write_session_manager_ = std::make_unique<KvMetaWriteSessionManager>(this, limits_.max_active_write_sessions);
    reclaimer_ = std::make_unique<KvMetaReclaimer>(this);
    if (!write_session_manager_->Start()) {
        reclaimer_.reset();
        write_session_manager_.reset();
        data_storage_selector_.reset();
        return false;
    }
    maintenance_cancelled_.store(false, std::memory_order_release);
    recovery_complete_.store(false, std::memory_order_release);
    initialized_.store(true, std::memory_order_release);
    return true;
}

void KvMetaManager::Shutdown() {
    CancelMaintenance();
    initialized_.store(false, std::memory_order_release);
    if (reclaimer_) {
        reclaimer_->StopAndJoin();
        reclaimer_.reset();
    }
    if (write_session_manager_) {
        write_session_manager_->StopAndDiscard();
        write_session_manager_.reset();
    }
    data_storage_selector_.reset();
}

void KvMetaManager::DoCleanup() {
    CancelMaintenance();
    if (reclaimer_) {
        reclaimer_->StopAndJoin();
    }
    if (write_session_manager_) {
        write_session_manager_->StopAndDiscard();
    }
}

void KvMetaManager::CancelMaintenance() noexcept {
    recovery_complete_.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(maintenance_mutex_);
        maintenance_cancelled_.store(true, std::memory_order_release);
    }
    maintenance_condition_.notify_all();
    if (reclaimer_) {
        reclaimer_->RequestStop();
    }
    if (write_session_manager_) {
        write_session_manager_->RequestStop();
    }
}

bool KvMetaManager::ResumeMaintenance() {
    if (!initialized_.load(std::memory_order_acquire) || !write_session_manager_ || !reclaimer_) {
        return false;
    }
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        // Drain the old workers before clearing cancellation, otherwise an old
        // reclaimer could resume its reader grace wait during restart.
        reclaimer_->StopAndJoin();
        write_session_manager_->StopAndDiscard();
    }
    if (!write_session_manager_->Start()) {
        return false;
    }
    recovery_complete_.store(false, std::memory_order_release);
    maintenance_cancelled_.store(false, std::memory_order_release);
    if (!reclaimer_->Start()) {
        maintenance_cancelled_.store(true, std::memory_order_release);
        write_session_manager_->RequestStop();
        return false;
    }
    return true;
}

std::string KvMetaManager::InternalInstanceId(const std::string &instance_id) {
    return std::string(kKvMetaInternalInstancePrefix) + HexEncode(instance_id);
}

std::int64_t KvMetaManager::InternalKey(const std::string &key) {
    const std::uint64_t hash = Hash64(key.data(), key.size(), kObjectKeyHashSeed);
    std::int64_t result = 0;
    static_assert(sizeof(result) == sizeof(hash));
    std::memcpy(&result, &hash, sizeof(result));
    return result;
}

std::string KvMetaManager::StableLocationId(const std::string &key) {
    return std::string(kKvMetaLocationIdPrefix) + HexEncode(key);
}

bool KvMetaManager::SameGeneration(const CacheLocation &lhs, const CacheLocation &rhs) {
    if (lhs.id() != rhs.id() || lhs.type() != rhs.type() || lhs.spec_size() != rhs.spec_size() ||
        lhs.create_time() != rhs.create_time() || lhs.location_specs().size() != rhs.location_specs().size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.location_specs().size(); ++i) {
        if (lhs.location_specs()[i].name() != rhs.location_specs()[i].name() ||
            lhs.location_specs()[i].uri() != rhs.location_specs()[i].uri()) {
            return false;
        }
    }
    return true;
}

ErrorCode KvMetaManager::ValidateInstanceId(RequestContext *request_context, const std::string &instance_id) const {
    if (!request_context || instance_id.empty() || instance_id.size() > limits_.max_instance_id_bytes) {
        AddError(request_context, "KVMeta instance_id is empty or too long");
        return EC_BADARGS;
    }
    return EC_OK;
}

ErrorCode KvMetaManager::CheckReady(RequestContext *request_context) const {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }
    if (!recovery_complete_.load(std::memory_order_acquire)) {
        AddError(request_context, "KVMeta recovery is not complete");
        return EC_CONFIG_ERROR;
    }
    return EC_OK;
}

ErrorCode KvMetaManager::ValidateKeys(RequestContext *request_context, const std::vector<std::string> &keys) const {
    if (!request_context || keys.empty() || keys.size() > limits_.max_batch_items) {
        AddError(request_context, "KVMeta keys are empty or exceed the batch limit");
        return EC_BADARGS;
    }
    std::unordered_set<std::string> unique;
    unique.reserve(keys.size());
    for (const auto &key : keys) {
        if (key.empty() || key.size() > limits_.max_key_bytes) {
            AddError(request_context, "KVMeta key is empty or too long");
            return EC_BADARGS;
        }
        if (!unique.insert(key).second) {
            AddError(request_context, "KVMeta request contains duplicate keys");
            return EC_DUPLICATE_ENTITY;
        }
    }
    return EC_OK;
}

ErrorCode KvMetaManager::ValidateCacheConfiguration(RequestContext *request_context,
                                                    const std::string &instance_group) const {
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(request_context, instance_group);
    if (group_ec != EC_OK || !group) {
        return group_ec == EC_OK ? EC_INSTANCE_NOT_EXIST : group_ec;
    }
    const auto cache_config = group->cache_config();
    if (UsesMemoryPrimaryMetadata(*group)) {
        // GC samples the persistent keyspace. Best-effort asynchronous Redis
        // backup can omit locally committed keys, so this mode is not safe yet.
        AddError(request_context, "KVMeta does not support memory_primary metadata");
        return EC_CONFIG_ERROR;
    }
    const auto strategy = cache_config ? cache_config->reclaim_strategy() : nullptr;
    const double threshold = strategy ? strategy->trigger_strategy().used_percentage() : -1.0;
    if (!strategy || strategy->reclaim_policy() != ReclaimPolicy::POLICY_LRU || !std::isfinite(threshold) ||
        threshold < 0.0 || threshold >= 1.0 || strategy->delay_before_delete_ms() < 0) {
        AddError(request_context, "KVMeta requires an LRU reclaim percentage in [0, 1)");
        return EC_CONFIG_ERROR;
    }
    if (!cache_config->migration_strategies().empty()) {
        AddError(request_context, "KVMeta does not support storage migration");
        return EC_CONFIG_ERROR;
    }
    std::int64_t effective_capacity = 0;
    DataStorageType storage_type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    if (!GetKvMetaCapacity(*group, registry_manager_->data_storage_manager(), effective_capacity, storage_type)) {
        AddError(request_context, "KVMeta requires one positive-capacity PACE storage tier per group");
        return EC_CONFIG_ERROR;
    }
    if (!MatchesKvMetaStoragePreference(cache_config->cache_prefer_strategy(), storage_type)) {
        AddError(request_context, "KVMeta storage preference must match its PACE storage tier");
        return EC_CONFIG_ERROR;
    }
    return EC_OK;
}

std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
KvMetaManager::GetValidatedInstanceInfo(RequestContext *request_context, const std::string &instance_id) const {
    if (const ErrorCode ec = ValidateInstanceId(request_context, instance_id); ec != EC_OK) {
        return {ec, nullptr};
    }
    auto info = registry_manager_->GetInstanceInfo(request_context, InternalInstanceId(instance_id));
    if (!info) {
        return {EC_INSTANCE_NOT_EXIST, nullptr};
    }
    if (!IsKvMetaInstance(*info)) {
        AddError(request_context, "KVMeta instance schema is invalid");
        return {EC_CORRUPTION, nullptr};
    }
    return {EC_OK, std::move(info)};
}

ErrorCode KvMetaManager::ValidateLocation(RequestContext *request_context,
                                          std::int64_t internal_key,
                                          const std::string &location_id,
                                          const CacheLocation &location,
                                          std::uint64_t &value_size) const {
    std::string original_key;
    const bool owned_key =
        location_id.size() > kKvMetaLocationIdPrefix.size() &&
        location_id.compare(0, kKvMetaLocationIdPrefix.size(), kKvMetaLocationIdPrefix) == 0 &&
        HexDecode(std::string_view(location_id).substr(kKvMetaLocationIdPrefix.size()), original_key) &&
        original_key.size() <= limits_.max_key_bytes && InternalKey(original_key) == internal_key;
    if (location.id() != location_id || !owned_key ||
        (location.status() != CLS_WRITING && location.status() != CLS_SERVING && location.status() != CLS_DELETING) ||
        !IsTairMempoolStorageType(location.type()) || !ReadLogicalSize(location, value_size) ||
        value_size > limits_.max_value_bytes) {
        AddError(request_context, "KVMeta location is malformed");
        return EC_CORRUPTION;
    }
    const DataStorageUri uri(location.location_specs().front().uri());
    const auto storage_manager = registry_manager_->data_storage_manager();
    const auto backend = storage_manager ? storage_manager->GetDataStorageBackend(uri.GetHostName()) : nullptr;
    if (!backend || backend->GetType() != location.type()) {
        AddError(request_context, "KVMeta location does not belong to a registered storage backend");
        return EC_CORRUPTION;
    }
    return EC_OK;
}

std::pair<ErrorCode, std::string> KvMetaManager::RegisterInstance(RequestContext *request_context,
                                                                  const std::string &instance_group,
                                                                  const std::string &instance_id,
                                                                  const std::string &user_data) {
    if (const ErrorCode ec = CheckReady(request_context); ec != EC_OK) {
        return {ec, {}};
    }
    if (!request_context || instance_group.empty() || instance_group.size() > limits_.max_instance_group_bytes ||
        instance_id.empty() || instance_id.size() > limits_.max_instance_id_bytes ||
        user_data.size() > limits_.max_user_data_bytes) {
        return {EC_BADARGS, {}};
    }
    ModelDeployment deployment;
    deployment.set_model_name(std::string(kKvMetaModelName));
    deployment.set_dtype(std::string(kKvMetaDtype));
    deployment.set_tp_size(1);
    deployment.set_dp_size(1);
    deployment.set_pp_size(1);
    deployment.set_extra(std::string(kKvMetaDeploymentExtra));
    deployment.set_user_data(user_data);

    const std::string internal_instance_id = InternalInstanceId(instance_id);
    if (!registry_manager_->GetInstanceInfo(request_context, internal_instance_id)) {
        if (const ErrorCode ec = ValidateCacheConfiguration(request_context, instance_group); ec != EC_OK) {
            return {ec, {}};
        }
    }
    auto result = cache_manager_->RegisterInstance(request_context,
                                                   instance_group,
                                                   internal_instance_id,
                                                   1,
                                                   {LocationSpecInfo(std::string(kKvMetaValueSpecName), 1)},
                                                   deployment,
                                                   {},
                                                   CacheManager::QueryType::QT_BATCH_GET);
    return result;
}

std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
KvMetaManager::GetInstanceInfo(RequestContext *request_context, const std::string &instance_id) const {
    if (const ErrorCode ec = CheckReady(request_context); ec != EC_OK) {
        return {ec, nullptr};
    }
    auto [ec, info] = GetValidatedInstanceInfo(request_context, instance_id);
    if (ec != EC_OK) {
        return {ec, nullptr};
    }
    auto public_info = std::make_shared<InstanceInfo>(*info);
    public_info->set_instance_id(instance_id);
    return {EC_OK, std::move(public_info)};
}

ErrorCode KvMetaManager::LoadExactLocations(RequestContext *request_context,
                                            const std::string &internal_instance_id,
                                            const std::vector<std::string> &keys,
                                            std::vector<ExactLocation> &out) const {
    out.assign(keys.size(), ExactLocation{});
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        return EC_INSTANCE_NOT_EXIST;
    }
    std::vector<std::int64_t> internal_keys;
    internal_keys.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        out[i].internal_key = InternalKey(keys[i]);
        out[i].location_id = StableLocationId(keys[i]);
        internal_keys.push_back(out[i].internal_key);
    }
    ErrorCode overall = EC_OK;
    for (const auto &layer : MakeUniqueKeyLayers(internal_keys)) {
        KeyVector layer_keys;
        LocationIdsPerKey ids;
        for (const std::size_t index : layer) {
            layer_keys.push_back(out[index].internal_key);
            ids.push_back({out[index].location_id});
        }
        LocationsPerKey locations;
        const auto result = indexer->GetLocations(request_context, layer_keys, ids, locations);
        if (locations.size() != layer.size() || result.per_location_error_codes.size() != layer.size()) {
            return EC_MISMATCH;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            const std::size_t output_index = layer[i];
            if (locations[i].size() != 1 || result.per_location_error_codes[i].size() != 1) {
                out[output_index].ec = EC_MISMATCH;
            } else {
                out[output_index].ec = result.per_location_error_codes[i][0];
                out[output_index].location = locations[i][0];
            }
            overall = FirstHardError(overall, out[output_index].ec);
        }
    }
    return overall;
}

std::pair<ErrorCode, std::vector<KvMetaManager::GetResult>> KvMetaManager::Get(
    RequestContext *request_context, const std::string &instance_id, const std::vector<std::string> &keys) const {
    if (const ErrorCode ec = CheckReady(request_context); ec != EC_OK) {
        return {ec, {}};
    }
    if (const auto [ec, _] = GetValidatedInstanceInfo(request_context, instance_id); ec != EC_OK) {
        return {ec, {}};
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return {ec, {}};
    }
    std::vector<ExactLocation> exact;
    if (const ErrorCode ec = LoadExactLocations(request_context, InternalInstanceId(instance_id), keys, exact);
        ec != EC_OK) {
        return {ec, {}};
    }
    std::vector<GetResult> result(keys.size());
    for (std::size_t i = 0; i < exact.size(); ++i) {
        if (exact[i].ec == EC_NOENT) {
            continue;
        }
        if (exact[i].ec != EC_OK || !exact[i].location) {
            return {exact[i].ec == EC_OK ? EC_CORRUPTION : exact[i].ec, {}};
        }
        std::uint64_t size = 0;
        if (const ErrorCode ec = ValidateLocation(
                request_context, exact[i].internal_key, exact[i].location_id, *exact[i].location, size);
            ec != EC_OK) {
            return {ec, {}};
        }
        if (exact[i].location->status() != CLS_SERVING) {
            continue;
        }
        if (!ToValueLocation(*exact[i].location, result[i].location)) {
            return {EC_CORRUPTION, {}};
        }
        result[i].found = true;
    }

    // Only a fully validated public hit is an access. In particular, pure
    // Redis backends do not maintain an in-memory LRU clock on GetLocations.
    // Re-read under the same shard fence used by deletion and compare the
    // exact generation: a delayed reader must neither recreate a deleted key
    // nor refresh a replacement object. Maintenance reads themselves stay cold.
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(InternalInstanceId(instance_id));
    if (indexer) {
        const bool access_time_already_updated = indexer->TracksAccessTimeOnRead();
        std::vector<std::int64_t> hit_keys;
        std::vector<std::size_t> hit_indices;
        for (std::size_t i = 0; i < result.size(); ++i) {
            if (result[i].found) {
                hit_keys.push_back(exact[i].internal_key);
                hit_indices.push_back(i);
            }
        }
        for (const auto &layer : MakeUniqueKeyLayers(hit_keys)) {
            KeyVector layer_keys;
            LocationIdsPerKey ids;
            std::vector<CacheLocationConstPtr> expected;
            for (const std::size_t index : layer) {
                const auto &item = exact[hit_indices[index]];
                layer_keys.push_back(item.internal_key);
                ids.push_back({item.location_id});
                expected.push_back(item.location);
            }
            std::vector<bool> generation_valid(expected.size(), false);
            auto modifier = [&expected, &generation_valid, access_time_already_updated](
                                const std::vector<ErrorCode> &get_ecs,
                                const LocationIdVector &,
                                std::size_t key_index,
                                CacheLocationVector &locations,
                                PropertyMap &properties) -> LocationModifierResult {
                if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= expected.size()) {
                    return {MA_FAIL, {EC_MISMATCH}};
                }
                if (get_ecs[0] == EC_NOENT) {
                    return {MA_SKIP, {EC_OK}};
                }
                if (get_ecs[0] != EC_OK || !locations[0] || !expected[key_index]) {
                    return {MA_FAIL, {get_ecs[0] == EC_OK ? EC_CORRUPTION : get_ecs[0]}};
                }
                if (locations[0]->status() != CLS_SERVING || !SameGeneration(*locations[0], *expected[key_index])) {
                    return {MA_SKIP, {EC_OK}};
                }
                generation_valid[key_index] = true;
                if (access_time_already_updated) {
                    return {MA_SKIP, {EC_OK}};
                }
                properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
                return {MA_OK, {EC_OK}};
            };
            const auto touch = indexer->IsRecovering()
                                   ? indexer->ReadModifyWriteLocation(request_context, layer_keys, ids, modifier, false, true)
                                   : indexer->ReadModifyWriteLocationsForMaintenance(request_context, layer_keys, ids, modifier);
            for (std::size_t i = 0; i < layer.size(); ++i) {
                if (!generation_valid[i]) {
                    result[hit_indices[layer[i]]] = GetResult{};
                }
            }
            // LRU bookkeeping is best effort; a failed auxiliary write does
            // not turn an already validated data-plane hit into a read error.
            if (touch.ec != EC_OK) {
                KVCM_LOG_WARN("KVMeta hit LRU update failed, instance[%s], ec[%d]", instance_id.c_str(), touch.ec);
            }
        }
    } else {
        // The instance disappeared after the initial lookup.
        result.assign(keys.size(), GetResult{});
    }
    return {EC_OK, std::move(result)};
}

bool KvMetaManager::DeletePhysicalBestEffort(RequestContext *request_context,
                                             const std::vector<SessionItem> &items) const {
    const auto storage_manager = registry_manager_->data_storage_manager();
    if (!storage_manager) {
        return false;
    }
    std::map<std::string, std::vector<DataStorageUri>> by_storage;
    std::size_t valid_count = 0;
    for (const auto &item : items) {
        if (!item.location || item.location->location_specs().size() != 1) {
            continue;
        }
        DataStorageUri uri(item.location->location_specs().front().uri());
        if (uri.Valid() && !uri.GetHostName().empty()) {
            by_storage[uri.GetHostName()].push_back(std::move(uri));
            ++valid_count;
        }
    }
    bool success = valid_count == items.size();
    for (auto &[storage_name, uris] : by_storage) {
        const auto results = storage_manager->Delete(request_context, storage_name, uris, nullptr);
        if (results.size() != uris.size() ||
            std::any_of(results.begin(), results.end(), [](ErrorCode ec) { return ec != EC_OK && ec != EC_NOENT; })) {
            // Physical deletion is best effort. Metadata is already absent, so
            // a provider failure cannot block cache traffic or corrupt usage.
            const auto failed = [](ErrorCode ec) { return ec != EC_OK && ec != EC_NOENT; };
            const auto first_failure = std::find_if(results.begin(), results.end(), failed);
            KVCM_LOG_ERROR("KVMeta physical cleanup failed trace_id [%s] storage [%s] requested [%zu] returned [%zu] "
                           "failed_items [%zu] first_failed_index [%zu] first_error [%d]",
                           request_context ? request_context->trace_id().c_str() : "",
                           storage_name.c_str(),
                           uris.size(),
                           results.size(),
                           static_cast<std::size_t>(std::count_if(results.begin(), results.end(), failed)),
                           static_cast<std::size_t>(std::distance(results.begin(), first_failure)),
                           static_cast<int>(first_failure == results.end() ? EC_MISMATCH : *first_failure));
            success = false;
        }
    }
    return success;
}

void KvMetaManager::ObserveReclaimStage(const std::string &group,
                                        const std::string &stage,
                                        const std::string &status) const noexcept {
    const auto metrics = cache_manager_->metrics_registry();
    if (!metrics) {
        return;
    }
    try {
        for (const auto *label : {"total", "success", "error", "skipped"}) {
            metrics->GetCounter("kv_meta.reclaim.stage_count",
                                {{"instance_group", group}, {"stage", stage}, {"status", label}}) +=
                std::string_view(label) == "total" || status == label ? 1 : 0;
        }
    } catch (...) { KVCM_LOG_WARN("KVMeta reclaim metric reporting failed"); }
}

void KvMetaManager::AddReclaimCounter(const std::string &group,
                                      const std::string &name,
                                      std::uint64_t value) const noexcept {
    try {
        if (const auto metrics = cache_manager_->metrics_registry()) {
            metrics->GetCounter("kv_meta.reclaim." + name, {{"instance_group", group}}) += value;
        }
    } catch (...) { KVCM_LOG_ERROR("KVMeta reclaim counter reporting failed"); }
}

void KvMetaManager::ObserveReclaimGauge(const std::string &group,
                                        const std::string &name,
                                        std::uint64_t value) const noexcept {
    const auto metrics = cache_manager_->metrics_registry();
    if (!metrics) {
        return;
    }
    try {
        metrics->GetGauge("kv_meta.reclaim." + name, {{"instance_group", group}}) = static_cast<double>(value);
    } catch (...) { KVCM_LOG_WARN("KVMeta reclaim gauge reporting failed"); }
}

ErrorCode KvMetaManager::DeleteItems(RequestContext *request_context,
                                     const std::string &internal_instance_id,
                                     const std::vector<SessionItem> &items,
                                     bool delete_physical,
                                     bool maintenance_read,
                                     std::vector<SessionItem> *deleted_items,
                                     bool refresh_cache_from_persistent) {
    if (deleted_items) {
        deleted_items->clear();
    }
    if (items.empty()) {
        return EC_OK;
    }
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        return EC_INSTANCE_NOT_EXIST;
    }
    std::vector<std::int64_t> keys;
    keys.reserve(items.size());
    for (const auto &item : items) {
        keys.push_back(item.internal_key);
    }
    auto mutation_lock = LockMetadataMutation(internal_instance_id, keys);
    std::vector<SessionItem> physically_deleted;
    ErrorCode overall = EC_OK;
    for (const auto &layer : MakeUniqueKeyLayers(keys)) {
        KeyVector layer_keys;
        LocationIdsPerKey layer_ids;
        std::vector<CacheLocationConstPtr> expected;
        for (const std::size_t index : layer) {
            layer_keys.push_back(items[index].internal_key);
            layer_ids.push_back({items[index].location_id});
            expected.push_back(items[index].location);
        }
        std::vector<bool> accepted(layer.size(), false);
        std::vector<bool> replaced(layer.size(), false);
        std::vector<CacheLocationConstPtr> removed_locations(layer.size());
        auto modifier = [&expected, &accepted, &removed_locations, &replaced](const std::vector<ErrorCode> &get_ecs,
                                                                              const LocationIdVector &,
                                                                              std::size_t key_index,
                                                                              CacheLocationVector &locations,
                                                                              PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= expected.size()) {
                return {MA_FAIL, {EC_MISMATCH}};
            }
            if (get_ecs[0] == EC_NOENT) {
                return {MA_SKIP, {EC_NOENT}};
            }
            if (get_ecs[0] != EC_OK || !locations[0] || !expected[key_index]) {
                return {MA_FAIL, {get_ecs[0] == EC_OK ? EC_CORRUPTION : get_ecs[0]}};
            }
            if (!SameGeneration(*locations[0], *expected[key_index])) {
                replaced[key_index] = true;
                return {MA_SKIP, {EC_EXIST}};
            }
            if (locations[0]->status() == CLS_DELETING) {
                // The executor owns physical deletion after its atomic mark.
                // Removing the metadata here could authorize a second release.
                return {MA_SKIP, {EC_EXIST}};
            }
            removed_locations[key_index] = locations[0];
            accepted[key_index] = true;
            return {MA_DELETE, {EC_OK}};
        };
        const auto result =
            (refresh_cache_from_persistent || (maintenance_read && indexer->IsRecovering()))
                ? indexer->ReadModifyWriteLocation(request_context, layer_keys, layer_ids, modifier, true, true)
            : maintenance_read
                ? indexer->ReadModifyWriteLocationsForMaintenance(request_context, layer_keys, layer_ids, modifier)
                : indexer->ReadModifyWriteLocation(request_context, layer_keys, layer_ids, modifier);
        if (result.per_location_error_codes.size() != layer.size()) {
            overall = FirstHardError(overall, EC_MISMATCH);
            continue;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            const ErrorCode ec =
                result.per_location_error_codes[i].size() == 1 ? result.per_location_error_codes[i][0] : EC_MISMATCH;
            if ((ec == EC_NOENT || (ec == EC_EXIST && replaced[i])) && items[layer[i]].location &&
                TakeUncertainUsage(internal_instance_id, items[layer[i]])) {
                // Absence or replacement proves an old HDEL took effect,
                // even if its acknowledgement was lost.
                // Only the failed FinishWrite's outstanding reservation is
                // refundable; an ordinary miss has no usage to subtract.
                indexer->SubStorageUsageByType(items[layer[i]].location->type(), items[layer[i]].value_size);
                if (delete_physical) {
                    // The session still owns this URI even though its
                    // metadata is gone. Reader grace below protects a
                    // possible SERVING value before physical release.
                    physically_deleted.push_back(items[layer[i]]);
                }
            }
            if (accepted[i] && ec == EC_OK) {
                SessionItem item = items[layer[i]];
                item.location = removed_locations[i];
                physically_deleted.push_back(item);
                // Cleanup can own a conservative reservation even when an
                // uncertain SERVING upsert actually left the value WRITING.
                const auto &expected = items[layer[i]].location;
                const bool reserved = TakeUncertainUsage(internal_instance_id, items[layer[i]]);
                if ((item.location && item.location->status() == CLS_SERVING) || reserved ||
                    (expected && expected->status() == CLS_SERVING)) {
                    indexer->SubStorageUsageByType(item.location->type(), item.value_size);
                }
            } else {
                overall = FirstHardError(overall, ec);
            }
        }
        if (result.ec != EC_OK && result.ec != EC_PARTIAL_OK) {
            overall = FirstHardError(overall, result.ec);
        }
    }
    if ((delete_physical || deleted_items) && !physically_deleted.empty()) {
        KeyVector deleted_keys;
        LocationIdsPerKey deleted_ids;
        for (const auto &item : physically_deleted) {
            deleted_keys.push_back(item.internal_key);
            deleted_ids.push_back({item.location_id});
        }
        if (!indexer->SyncDeletedLocations(request_context, deleted_keys, deleted_ids)) {
            // Retain orphaned storage rather than free an address that recovery
            // could still expose. Usage follows the accepted metadata mutation.
            physically_deleted.clear();
            overall = FirstHardError(overall, EC_ERROR);
        }
    }
    mutation_lock.unlock();
    if (deleted_items) {
        *deleted_items = physically_deleted;
    }
    if (delete_physical) {
        const bool needs_reader_grace =
            std::any_of(physically_deleted.begin(), physically_deleted.end(), [](const SessionItem &item) {
                return item.location && item.location->status() == CLS_SERVING;
            });
        if (needs_reader_grace) {
            // FinishWrite can expose a committed subset before a later metadata
            // update fails. Its rollback (including cleanup retries) must give
            // those readers the same grace period as Remove and the reclaimer.
            const auto instance = registry_manager_->GetInstanceInfo(request_context, internal_instance_id);
            if (!instance) {
                return FirstHardError(overall, EC_INSTANCE_NOT_EXIST);
            }
            const auto [group_ec, group] =
                registry_manager_->GetInstanceGroup(request_context, instance->instance_group_name());
            const auto strategy = group && group->cache_config() ? group->cache_config()->reclaim_strategy() : nullptr;
            if (group_ec != EC_OK || !strategy || strategy->delay_before_delete_ms() < 0) {
                // No safe grace period is known; retain the allocation.
                return overall != EC_OK ? overall : EC_CONFIG_ERROR;
            }
            if (!WaitForDeletionDelay(strategy->delay_before_delete_ms())) {
                return FirstHardError(overall, EC_ERROR);
            }
        }
        DeletePhysicalBestEffort(request_context, physically_deleted);
    }
    return overall;
}

std::pair<ErrorCode, KvMetaManager::StartWriteResult>
KvMetaManager::StartWrite(RequestContext *request_context,
                          const std::string &instance_id,
                          const std::vector<std::string> &keys,
                          const std::vector<std::uint64_t> &value_sizes,
                          std::int64_t write_timeout_seconds) {
    StartWriteResult response;
    if (const ErrorCode ec = CheckReady(request_context); ec != EC_OK) {
        return {ec, {}};
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return {ec, {}};
    }
    if (value_sizes.size() != keys.size() || write_timeout_seconds <= 0 ||
        write_timeout_seconds > limits_.max_write_timeout_seconds) {
        return {EC_BADARGS, {}};
    }
    std::uint64_t batch_bytes = 0;
    for (const std::uint64_t size : value_sizes) {
        if (size == 0 || size > limits_.max_value_bytes || size > limits_.max_batch_bytes ||
            size > std::numeric_limits<std::size_t>::max() || batch_bytes > limits_.max_batch_bytes - size) {
            return {EC_OUT_OF_LIMIT, {}};
        }
        batch_bytes += size;
    }
    const auto [instance_ec, instance] = GetValidatedInstanceInfo(request_context, instance_id);
    if (instance_ec != EC_OK || !instance) {
        return {instance_ec, {}};
    }
    const std::string internal_instance_id = InternalInstanceId(instance_id);
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        return {EC_INSTANCE_NOT_EXIST, {}};
    }

    std::vector<ExactLocation> existing;
    if (const ErrorCode ec = LoadExactLocations(request_context, internal_instance_id, keys, existing); ec != EC_OK) {
        return {ec, {}};
    }
    response.key_mask.assign(keys.size(), false);
    std::vector<std::size_t> missing;
    for (std::size_t i = 0; i < existing.size(); ++i) {
        if (existing[i].ec == EC_NOENT) {
            missing.push_back(i);
            continue;
        }
        if (existing[i].ec != EC_OK || !existing[i].location) {
            return {existing[i].ec == EC_OK ? EC_CORRUPTION : existing[i].ec, {}};
        }
        std::uint64_t existing_size = 0;
        if (const ErrorCode ec = ValidateLocation(request_context,
                                                  existing[i].internal_key,
                                                  existing[i].location_id,
                                                  *existing[i].location,
                                                  existing_size);
            ec != EC_OK) {
            return {ec, {}};
        }
        if (existing[i].location->status() == CLS_WRITING || existing[i].location->status() == CLS_DELETING) {
            return {EC_EXIST, {}};
        }
        if (existing_size != value_sizes[i]) {
            return {EC_MISMATCH, {}};
        }
        response.key_mask[i] = true;
    }
    if (missing.empty()) {
        return {EC_OK, std::move(response)};
    }
    if (!write_session_manager_ ||
        write_session_manager_->Availability() != KvMetaWriteSessionManager::PutResult::kOk) {
        return {EC_NOSPC, {}};
    }
    if (const ErrorCode ec = ValidateCacheConfiguration(request_context, instance->instance_group_name());
        ec != EC_OK) {
        return {ec, {}};
    }

    // This is deliberately a snapshot-only admission check. It does not add
    // requested bytes and it takes no capacity lock: concurrent starts can
    // cross the quota/watermark, after which the background reclaimer
    // converges usage. The storage provider remains the hard capacity guard.
    const auto selected =
        data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context, instance->instance_group_name());
    if (selected.ec != EC_OK || selected.name.empty() || selected.type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
        // Configuration was validated above; no selectable candidate here is
        // normal cache capacity/availability pressure, not missing metadata.
        return {EC_NOSPC, {}};
    }

    const auto storage_manager = registry_manager_->data_storage_manager();
    const auto backend = storage_manager ? storage_manager->GetDataStorageBackend(selected.name) : nullptr;
    if (!backend || backend->GetType() != selected.type || !IsTairMempoolStorageType(selected.type)) {
        return {EC_CONFIG_ERROR, {}};
    }

    const auto deadline = KvMetaWriteSessionManager::Clock::now() + std::chrono::seconds(write_timeout_seconds);
    auto publish_writing = [&](const std::vector<SessionItem> &batch, std::vector<bool> &inserted) {
        inserted.assign(batch.size(), false);
        KeyVector batch_keys;
        LocationIdsPerKey ids;
        for (const auto &candidate : batch) {
            batch_keys.push_back(candidate.internal_key);
            ids.push_back({candidate.location_id});
        }
        auto modifier = [&batch, &inserted](const std::vector<ErrorCode> &get_ecs,
                                            const LocationIdVector &,
                                            std::size_t key_index,
                                            CacheLocationVector &locations,
                                            PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= batch.size()) {
                return {MA_FAIL, {EC_MISMATCH}};
            }
            if (get_ecs[0] != EC_NOENT) {
                return {get_ecs[0] == EC_OK ? MA_SKIP : MA_FAIL, {get_ecs[0] == EC_OK ? EC_EXIST : get_ecs[0]}};
            }
            locations[0] = batch[key_index].location;
            inserted[key_index] = true;
            return {MA_OK, {EC_OK}};
        };
        auto mutation_lock = LockMetadataMutation(internal_instance_id, batch_keys);
        const auto result = indexer->ReadModifyWriteTargetLocations(request_context, batch_keys, ids, modifier);
        std::vector<ErrorCode> errors(batch.size(), EC_MISMATCH);
        if (result.per_location_error_codes.size() != batch.size()) {
            return errors;
        }
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (result.per_location_error_codes[i].size() != 1) {
                continue;
            }
            const auto item_ec = result.per_location_error_codes[i][0];
            errors[i] = item_ec != EC_OK                                   ? item_ec
                        : result.ec != EC_OK && result.ec != EC_PARTIAL_OK ? result.ec
                        : inserted[i]                                      ? EC_OK
                                                                           : EC_EXIST;
        }
        return errors;
    };

    std::vector<SessionItem> candidates;
    candidates.reserve(missing.size());
    std::vector<SessionItem> unpublished(keys.size());
    const auto rollback_candidates = [&]() {
        std::vector<SessionItem> private_allocations;
        for (auto &item : unpublished) {
            if (item.location) {
                private_allocations.push_back(std::move(item));
            }
        }
        DeletePhysicalBestEffort(request_context, private_allocations);
        if (candidates.empty()) {
            return;
        }
        const ErrorCode cleanup_ec = DeleteItems(request_context, internal_instance_id, candidates, true, false);
        if (cleanup_ec != EC_OK && write_session_manager_) {
            KvMetaWriteSessionManager::Session cleanup{
                internal_instance_id, KvMetaWriteSessionManager::Clock::now(), candidates};
            cleanup.registration = instance;
            write_session_manager_->ScheduleCleanup(StringUtil::GenerateRandomString(32), std::move(cleanup));
        }
    };
    // The backend already accepts a batch with one size per key. Group equal
    // sizes instead of turning a multi-object request into N PACE allocations.
    std::map<std::uint64_t, std::vector<std::size_t>> by_size;
    for (const auto index : missing) {
        by_size[value_sizes[index]].push_back(index);
    }
    for (const auto &[size, indices] : by_size) {
        if (maintenance_cancelled_.load(std::memory_order_acquire) ||
            KvMetaWriteSessionManager::Clock::now() >= deadline) {
            rollback_candidates();
            return {maintenance_cancelled_.load(std::memory_order_acquire) ? EC_SERVICE_NOT_LEADER : EC_TIMEOUT, {}};
        }
        std::vector<std::string> object_keys;
        object_keys.reserve(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            object_keys.push_back("kvmeta/" + StringUtil::GenerateRandomString(kKvMetaObjectNonceBytes));
        }
        const auto created = storage_manager->Create(
            request_context, selected.name, object_keys, static_cast<std::size_t>(size), nullptr);
        ErrorCode allocation_ec = created.size() == indices.size() ? EC_OK : EC_IO_ERROR;
        for (std::size_t i = 0; i < created.size(); ++i) {
            if (created[i].first != EC_OK || !created[i].second.Valid()) {
                if (allocation_ec == EC_OK) {
                    allocation_ec = created[i].first != EC_OK ? created[i].first : EC_IO_ERROR;
                }
                continue;
            }
            auto location = std::make_shared<CacheLocation>();
            location->set_status(CLS_WRITING);
            location->set_type(selected.type);
            location->set_spec_size(1);
            location->set_create_time(TimestampUtil::GetCurrentTimeUs());
            location->push_location_spec(
                LocationSpec(std::string(kKvMetaValueSpecName), created[i].second.ToUriString()));
            if (i >= indices.size()) {
                // Even a malformed oversized response can contain allocations
                // owned only by this request; no writer has received them.
                DeletePhysicalBestEffort(request_context, {{0, {}, std::move(location), size}});
                continue;
            }
            const auto request_index = indices[i];
            location->set_id(existing[request_index].location_id);
            unpublished[request_index] = {
                existing[request_index].internal_key, existing[request_index].location_id, location, size};
            std::uint64_t actual_size = 0;
            if (ValidateLocation(request_context,
                                 existing[request_index].internal_key,
                                 existing[request_index].location_id,
                                 *location,
                                 actual_size) != EC_OK ||
                actual_size != size) {
                if (allocation_ec == EC_OK) {
                    allocation_ec = EC_CORRUPTION;
                }
            }
        }
        if (allocation_ec != EC_OK) {
            rollback_candidates();
            return {allocation_ec, {}};
        }
    }
    KeyVector missing_keys;
    for (const auto request_index : missing) {
        missing_keys.push_back(existing[request_index].internal_key);
    }
    std::vector<std::size_t> candidate_positions;
    for (const auto &layer : MakeUniqueKeyLayers(missing_keys)) {
        if (maintenance_cancelled_.load(std::memory_order_acquire) ||
            KvMetaWriteSessionManager::Clock::now() >= deadline) {
            rollback_candidates();
            return {maintenance_cancelled_.load(std::memory_order_acquire) ? EC_SERVICE_NOT_LEADER : EC_TIMEOUT, {}};
        }
        std::vector<SessionItem> batch;
        std::vector<std::string> business_keys;
        for (const auto index : layer) {
            const auto request_index = missing[index];
            batch.push_back(std::move(unpublished[request_index]));
            business_keys.push_back(keys[request_index]);
        }
        std::vector<bool> inserted;
        const auto errors = publish_writing(batch, inserted);
        const auto failure = std::find_if(errors.begin(), errors.end(), [](ErrorCode ec) { return ec != EC_OK; });
        if (failure == errors.end()) {
            for (std::size_t i = 0; i < batch.size(); ++i) {
                candidates.push_back(std::move(batch[i]));
                candidate_positions.push_back(layer[i]);
            }
            continue;
        }
        // A modifier can run before admission rejects a new key, or before a
        // backend reply is lost. Verify uncertain publications once as a batch:
        // release only allocations whose generation is proven absent/replaced.
        std::vector<ExactLocation> observed;
        const bool lookup_needed = std::any_of(inserted.begin(), inserted.end(), [](bool value) { return value; });
        const bool lookup_ok =
            lookup_needed &&
            LoadExactLocations(request_context, internal_instance_id, business_keys, observed) == EC_OK &&
            observed.size() == batch.size();
        std::vector<SessionItem> private_allocations;
        for (std::size_t i = 0; i < batch.size(); ++i) {
            const bool absent = lookup_ok && (observed[i].ec == EC_NOENT ||
                                              (observed[i].ec == EC_OK && observed[i].location &&
                                               !SameGeneration(*observed[i].location, *batch[i].location)));
            if (!inserted[i] || absent) {
                private_allocations.push_back(std::move(batch[i]));
            } else {
                candidates.push_back(std::move(batch[i]));
            }
        }
        DeletePhysicalBestEffort(request_context, private_allocations);
        rollback_candidates();
        return {*failure, {}};
    }

    // Collision layers may be visited out of request order. Keep both the
    // returned locations and write-session flags aligned with missing keys.
    std::vector<SessionItem> ordered_candidates(missing.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        ordered_candidates[candidate_positions[i]] = std::move(candidates[i]);
    }
    candidates = std::move(ordered_candidates);

    std::vector<ValueLocation> output_locations;
    output_locations.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        ValueLocation location;
        if (!candidate.location || !ToValueLocation(*candidate.location, location)) {
            rollback_candidates();
            return {EC_CORRUPTION, {}};
        }
        output_locations.push_back(std::move(location));
    }

    std::string session_id;
    KvMetaWriteSessionManager::PutResult put_result = KvMetaWriteSessionManager::PutResult::kDuplicate;
    for (int attempt = 0; attempt < 8 && put_result == KvMetaWriteSessionManager::PutResult::kDuplicate; ++attempt) {
        session_id = StringUtil::GenerateRandomString(32);
        KvMetaWriteSessionManager::Session session{internal_instance_id, deadline, candidates};
        session.registration = instance;
        put_result = write_session_manager_->Put(session_id, std::move(session));
    }
    if (put_result != KvMetaWriteSessionManager::PutResult::kOk) {
        rollback_candidates();
        if (put_result == KvMetaWriteSessionManager::PutResult::kStopped) {
            return {EC_SERVICE_NOT_LEADER, {}};
        }
        return {put_result == KvMetaWriteSessionManager::PutResult::kExpired ? EC_TIMEOUT : EC_NOSPC, {}};
    }

    response.write_session_id = std::move(session_id);
    response.locations = std::move(output_locations);
    return {EC_OK, std::move(response)};
}

ErrorCode KvMetaManager::FinishWrite(RequestContext *request_context,
                                     const std::string &instance_id,
                                     const std::string &write_session_id,
                                     const std::vector<bool> &success_keys) {
    if (const ErrorCode ec = CheckReady(request_context); ec != EC_OK) {
        return ec;
    }
    if (!request_context || write_session_id.empty() || write_session_id.size() > limits_.max_write_session_id_bytes ||
        success_keys.empty() || success_keys.size() > limits_.max_batch_items) {
        return EC_BADARGS;
    }
    if (const ErrorCode ec = ValidateInstanceId(request_context, instance_id); ec != EC_OK) {
        return ec;
    }
    const std::string internal_instance_id = InternalInstanceId(instance_id);
    KvMetaWriteSessionManager::Session session;
    const ErrorCode take_ec =
        write_session_manager_->Take(write_session_id, internal_instance_id, success_keys.size(), session);
    if (take_ec != EC_OK) {
        if (take_ec == EC_TIMEOUT) {
            if (DeleteItems(request_context, internal_instance_id, session.items, true, false) != EC_OK) {
                write_session_manager_->ScheduleCleanup(write_session_id, std::move(session));
            }
        }
        return take_ec;
    }
    // A removed instance may be registered again under the same public ID.
    // Its fresh indexer namespace must never accept an old write lease.
    const auto current_registration = registry_manager_->GetInstanceInfo(request_context, internal_instance_id);
    if (!current_registration || current_registration != session.registration) {
        return EC_INSTANCE_NOT_EXIST;
    }
    if (std::any_of(success_keys.begin(), success_keys.end(), [](bool success) { return !success; })) {
        const ErrorCode cleanup_ec = DeleteItems(request_context, internal_instance_id, session.items, true, false);
        if (cleanup_ec != EC_OK) {
            write_session_manager_->ScheduleCleanup(write_session_id, std::move(session));
        }
        return cleanup_ec;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        // Without the indexer we cannot confirm that persistent metadata no
        // longer references these allocations.
        return EC_INSTANCE_NOT_EXIST;
    }
    std::vector<std::int64_t> keys;
    for (const auto &item : session.items) {
        keys.push_back(item.internal_key);
    }
    auto mutation_lock = LockMetadataMutation(internal_instance_id, keys);
    ErrorCode overall = EC_OK;
    std::vector<bool> committed(session.items.size(), false);
    std::vector<bool> accounted(session.items.size(), false);
    for (const auto &layer : MakeUniqueKeyLayers(keys)) {
        KeyVector layer_keys;
        LocationIdsPerKey ids;
        std::vector<CacheLocationConstPtr> expected;
        for (const std::size_t index : layer) {
            layer_keys.push_back(session.items[index].internal_key);
            ids.push_back({session.items[index].location_id});
            expected.push_back(session.items[index].location);
        }
        std::vector<bool> accepted(layer.size(), false);
        auto modifier = [this, &expected, &accepted, &accounted, &layer, &session, &internal_instance_id, &indexer](
                            const std::vector<ErrorCode> &get_ecs,
                            const LocationIdVector &,
                            std::size_t key_index,
                            CacheLocationVector &locations,
                            PropertyMap &properties) -> LocationModifierResult {
            if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= expected.size()) {
                return {MA_FAIL, {EC_MISMATCH}};
            }
            if (get_ecs[0] != EC_OK || !locations[0] || !expected[key_index]) {
                return {MA_FAIL, {get_ecs[0] == EC_OK ? EC_CORRUPTION : get_ecs[0]}};
            }
            if (locations[0]->status() != CLS_WRITING || !SameGeneration(*locations[0], *expected[key_index])) {
                return {MA_SKIP, {EC_EXIST}};
            }
            auto serving = std::make_shared<CacheLocation>(*locations[0]);
            serving->set_status(CLS_SERVING);
            const auto item_index = layer[key_index];
            // Charge before publication; GC must never see a SERVING value
            // whose bytes have not been added. Keep cleanup ownership until
            // the metadata acknowledgement is known.
            ReserveUncertainUsage(internal_instance_id, session.items[item_index]);
            indexer->AddStorageUsageByType(session.items[item_index].location->type(),
                                           session.items[item_index].value_size);
            accounted[item_index] = true;
            locations[0] = std::move(serving);
            properties[PROPERTY_LRU_TIME] = std::to_string(TimestampUtil::GetCurrentTimeUs());
            accepted[key_index] = true;
            return {MA_OK, {EC_OK}};
        };
        const auto result = indexer->ReadModifyWriteLocation(request_context, layer_keys, ids, modifier);
        if (result.per_location_error_codes.size() != layer.size()) {
            overall = FirstHardError(overall, EC_MISMATCH);
            continue;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            const ErrorCode ec =
                result.per_location_error_codes[i].size() == 1 ? result.per_location_error_codes[i][0] : EC_MISMATCH;
            if (accepted[i] && ec == EC_OK) {
                const std::size_t item_index = layer[i];
                committed[item_index] = true;
            } else {
                overall = FirstHardError(overall, ec);
            }
        }
        if (result.ec != EC_OK && result.ec != EC_PARTIAL_OK) {
            overall = FirstHardError(overall, result.ec);
        }
    }
    if (overall == EC_OK && std::all_of(committed.begin(), committed.end(), [](bool value) { return value; })) {
        for (const auto &item : session.items) {
            TakeUncertainUsage(internal_instance_id, item);
        }
        return EC_OK;
    }
    // A partial metadata transition is not exposed as a partial cache write:
    // remove every generation owned by this session and let the caller retry.
    // SERVING in this cleanup expectation records reserved usage; DeleteItems
    // still uses the actual removed status to decide reader grace.
    std::vector<SessionItem> cleanup = session.items;
    for (std::size_t i = 0; i < cleanup.size(); ++i) {
        if (accounted[i]) {
            auto serving = std::make_shared<CacheLocation>(*cleanup[i].location);
            serving->set_status(CLS_SERVING);
            cleanup[i].location = std::move(serving);
        }
    }
    mutation_lock.unlock();
    if (DeleteItems(request_context, internal_instance_id, cleanup, true, false) != EC_OK) {
        session.items = std::move(cleanup);
        write_session_manager_->ScheduleCleanup(write_session_id, std::move(session));
    }
    return overall == EC_OK ? EC_ERROR : overall;
}

ErrorCode KvMetaManager::Remove(RequestContext *request_context,
                                const std::string &instance_id,
                                const std::vector<std::string> &keys) {
    if (const ErrorCode ec = CheckReady(request_context); ec != EC_OK) {
        return ec;
    }
    const auto [instance_ec, instance] = GetValidatedInstanceInfo(request_context, instance_id);
    if (instance_ec != EC_OK) {
        return instance_ec;
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return ec;
    }
    const std::string internal_instance_id = InternalInstanceId(instance_id);
    std::vector<ExactLocation> exact;
    if (const ErrorCode ec = LoadExactLocations(request_context, internal_instance_id, keys, exact); ec != EC_OK) {
        return ec;
    }
    std::vector<SessionItem> items;
    for (std::size_t i = 0; i < exact.size(); ++i) {
        if (exact[i].ec == EC_NOENT) {
            continue;
        }
        if (exact[i].ec != EC_OK || !exact[i].location) {
            return exact[i].ec == EC_OK ? EC_CORRUPTION : exact[i].ec;
        }
        std::uint64_t size = 0;
        if (const ErrorCode ec = ValidateLocation(
                request_context, exact[i].internal_key, exact[i].location_id, *exact[i].location, size);
            ec != EC_OK) {
            return ec;
        }
        if (exact[i].location->status() == CLS_WRITING) {
            AddError(request_context, "KVMeta object is still being written");
            return EC_EXIST;
        }
        if (exact[i].location->status() == CLS_DELETING) {
            continue; // Already unreadable and exclusively owned by GC.
        }
        items.push_back({exact[i].internal_key, exact[i].location_id, exact[i].location, size});
    }
    std::vector<SessionItem> deleted;
    const ErrorCode delete_ec = DeleteItems(request_context, internal_instance_id, items, false, false, &deleted);
    if (!deleted.empty()) {
        const auto [group_ec, group] =
            registry_manager_->GetInstanceGroup(request_context, instance->instance_group_name());
        const auto strategy = group && group->cache_config() ? group->cache_config()->reclaim_strategy() : nullptr;
        if (group_ec != EC_OK || !strategy || strategy->delay_before_delete_ms() < 0) {
            // Metadata is already unpublished. Retain the allocation when the
            // reader grace policy is unavailable rather than freeing it early.
            // A not-ready/leader status could replay this deletion against a
            // new generation after configuration or leadership changes.
            return EC_ERROR;
        }
        if (!WaitForDeletionDelay(strategy->delay_before_delete_ms())) {
            // Metadata has already changed; a leader redirect would invite
            // automatic retries against a replacement generation.
            return EC_ERROR;
        }
        DeletePhysicalBestEffort(request_context, deleted);
    }
    return !deleted.empty() && delete_ec != EC_OK ? EC_ERROR : delete_ec;
}

bool KvMetaManager::WaitForDeletionDelay(std::int32_t delay_ms) {
    std::unique_lock<std::mutex> lock(maintenance_mutex_);
    return !maintenance_condition_.wait_for(lock, std::chrono::milliseconds(delay_ms), [this]() {
        return maintenance_cancelled_.load(std::memory_order_acquire);
    });
}

bool KvMetaManager::ExpireSession(const std::string &session_id,
                                  const std::string &internal_instance_id,
                                  const std::vector<SessionItem> &items,
                                  bool delete_physical,
                                  const std::shared_ptr<const InstanceInfo> &registration) noexcept {
    try {
        RequestContext context("kv_meta_session_expire_" + session_id);
        if (registration && registry_manager_->GetInstanceInfo(&context, internal_instance_id) != registration) {
            return true;
        }
        // A lease timeout does not fence an outstanding PACE write. Only
        // acknowledged or never-published allocations may be released.
        return DeleteItems(&context, internal_instance_id, items, delete_physical, true) == EC_OK;
    } catch (const std::exception &e) { KVCM_LOG_WARN("KVMeta session expiry failed: %s", e.what()); } catch (...) {
        KVCM_LOG_WARN("KVMeta session expiry failed with unknown exception");
    }
    return false;
}

ErrorCode KvMetaManager::DoRecover() {
    {
        std::lock_guard<std::mutex> lock(uncertain_usage_mutex_);
        uncertain_usage_.clear();
    }
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    recovery_complete_.store(false, std::memory_order_release);
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }
    if (!registry_manager_->IsRecoverComplete()) {
        return EC_CONFIG_ERROR;
    }
    RequestContext context("kv_meta_recover");
    const auto [groups_ec, groups] = registry_manager_->ListInstanceGroup(&context);
    if (groups_ec != EC_OK) {
        return groups_ec;
    }
    ErrorCode overall = EC_OK;
    for (const auto &group : groups) {
        if (maintenance_cancelled_.load(std::memory_order_acquire)) {
            return EC_SERVICE_NOT_LEADER;
        }
        if (!group) {
            overall = FirstHardError(overall, EC_CORRUPTION);
            continue;
        }
        const auto [instances_ec, instances] = registry_manager_->ListInstanceInfo(&context, group->name());
        if (instances_ec != EC_OK) {
            overall = FirstHardError(overall, instances_ec);
            continue;
        }
        const bool has_kv_meta = std::any_of(instances.begin(), instances.end(), [](const auto &instance) {
            return instance && IsKvMetaInstance(*instance);
        });
        const bool has_other = std::any_of(instances.begin(), instances.end(), [](const auto &instance) {
            return !instance || !IsKvMetaInstance(*instance);
        });
        if (has_kv_meta && has_other) {
            AddError(&context, "KVMeta instance group contains an ordinary KV-cache instance");
            overall = FirstHardError(overall, EC_CONFIG_ERROR);
            continue;
        }
        if (has_kv_meta && UsesMemoryPrimaryMetadata(*group)) {
            AddError(&context, "KVMeta does not support memory_primary metadata");
            overall = FirstHardError(overall, EC_CONFIG_ERROR);
            continue;
        }
        for (const auto &instance : instances) {
            if (!instance || !IsKvMetaInstance(*instance)) {
                continue;
            }
            auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                overall = FirstHardError(overall, EC_INSTANCE_NOT_EXIST);
                continue;
            }
            ErrorCode instance_ec = EC_OK;
            std::unordered_set<std::int64_t> seen_keys;
            std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> usage{};
            std::string cursor = SCAN_BASE_CURSOR;
            do {
                if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                    return EC_SERVICE_NOT_LEADER;
                }
                MaintenanceScanBatch batch;
                const ErrorCode scan_ec =
                    indexer->ScanPersistentLocationsForRecovery(&context, cursor, kScanBatchSize, batch);
                if (scan_ec != EC_OK) {
                    instance_ec = scan_ec;
                    break;
                }
                if (!batch.keys.empty()) {
                    std::vector<SessionItem> stale;
                    for (std::size_t i = 0; i < batch.keys.size() && instance_ec == EC_OK; ++i) {
                        // SCAN is not a snapshot: a listed key can disappear
                        // before the location read, and may recur on later pages.
                        if (batch.location_results[i] == EC_NOENT) {
                            continue;
                        }
                        if (batch.location_results[i] != EC_OK) {
                            instance_ec = batch.location_results[i];
                            break;
                        }
                        if (!seen_keys.insert(batch.keys[i]).second) {
                            continue;
                        }
                        for (const auto &[location_id, location] : batch.locations[i]) {
                            std::uint64_t size = 0;
                            if (!location ||
                                ValidateLocation(&context, batch.keys[i], location_id, *location, size) != EC_OK) {
                                instance_ec = EC_CORRUPTION;
                                break;
                            }
                            if (location->status() == CLS_WRITING) {
                                stale.push_back({batch.keys[i], location_id, location, size});
                            } else {
                                const auto base_type = ToBaseType(location->type());
                                const std::size_t type_index = ToIndex(base_type);
                                if (type_index >= usage.size()) {
                                    instance_ec = EC_CORRUPTION;
                                    break;
                                }
                                usage[type_index] = SaturatingAdd(usage[type_index], size);
                            }
                        }
                    }
                    if (instance_ec == EC_OK && !stale.empty()) {
                        // The previous writer may still own these addresses.
                        // Hide abandoned writes, but do not recycle them.
                        instance_ec = DeleteItems(&context, instance->instance_id(), stale, false, true, nullptr, true);
                    }
                }
                cursor = std::move(batch.next_cursor);
            } while (instance_ec == EC_OK && cursor != SCAN_BASE_CURSOR);

            if (instance_ec == EC_OK) {
                for (std::size_t i = 1; i < usage.size(); ++i) {
                    const auto type = static_cast<DataStorageType>(i);
                    if (ToBaseType(type) == type) {
                        indexer->SetStorageUsageByType(type, usage[i]);
                    }
                }
                indexer->PersistMetaData();
            }
            // Unlike an individual vanished key, an incomplete instance scan
            // must keep recovery unready, including a batch-level EC_NOENT.
            if (overall == EC_OK) {
                overall = instance_ec;
            }
        }
    }
    if (overall == EC_OK && !maintenance_cancelled_.load(std::memory_order_acquire)) {
        recovery_complete_.store(true, std::memory_order_release);
    } else if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }
    return overall;
}

} // namespace kv_cache_manager
