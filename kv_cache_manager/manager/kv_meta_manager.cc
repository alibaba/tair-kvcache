#include "kv_cache_manager/manager/kv_meta_manager.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
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
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/kv_meta_instance.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

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
                       std::int64_t &capacity) {
    if (!storage_manager || group.storage_candidates().empty() || group.quota().capacity() <= 0) {
        return false;
    }
    DataStorageType storage_type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
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
    };

    KvMetaWriteSessionManager(KvMetaManager *owner, std::size_t max_sessions)
        : owner_(owner), max_sessions_(max_sessions) {}
    ~KvMetaWriteSessionManager() { StopAndDiscard(); }

    bool Start() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (thread_.joinable()) {
            return true;
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
    }

    PutResult Availability() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return PutResult::kStopped;
        }
        return sessions_.size() >= max_sessions_ ? PutResult::kFull : PutResult::kOk;
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
            return PutResult::kFull;
        }
        if (!sessions_.emplace(session_id, std::move(session)).second) {
            return PutResult::kDuplicate;
        }
        condition_.notify_all();
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
        out = std::move(it->second);
        sessions_.erase(it);
        return out.deadline <= Clock::now() ? EC_TIMEOUT : EC_OK;
    }

    void ScheduleCleanup(const std::string &session_id, Session session) noexcept {
        try {
            session.deadline = Clock::now() + std::chrono::seconds(1);
            session.cleanup_only = true;
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            if (!sessions_.emplace(session_id, std::move(session)).second) {
                KVCM_LOG_ERROR("failed to reschedule KVMeta session cleanup for [%s]", session_id.c_str());
                return;
            }
            condition_.notify_all();
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
                auto next = Clock::time_point::max();
                for (const auto &[_, session] : sessions_) {
                    next = std::min(next, session.deadline);
                }
                if (next == Clock::time_point::max()) {
                    condition_.wait(lock, [this]() { return stopping_ || !sessions_.empty(); });
                    continue;
                }
                if (Clock::now() < next) {
                    condition_.wait_until(lock, next);
                    continue;
                }
                const auto now = Clock::now();
                for (auto it = sessions_.begin(); it != sessions_.end();) {
                    if (it->second.deadline <= now) {
                        expired.emplace_back(it->first, std::move(it->second));
                        it = sessions_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            for (auto &[session_id, session] : expired) {
                if (owner_ && !owner_->ExpireSession(session_id, session.internal_instance_id, session.items)) {
                    ScheduleCleanup(session_id, std::move(session));
                }
            }
        }
    }

    KvMetaManager *owner_ = nullptr;
    std::size_t max_sessions_ = 0;
    mutable std::mutex mutex_;
    std::mutex lifecycle_mutex_;
    std::condition_variable condition_;
    bool stopping_ = true;
    std::unordered_map<std::string, Session> sessions_;
    std::thread thread_;
};

class KvMetaReclaimer {
public:
    explicit KvMetaReclaimer(KvMetaManager *owner) : owner_(owner) {}
    ~KvMetaReclaimer() { StopAndJoin(); }

    bool Start() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (thread_.joinable()) {
            return true;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = false;
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
        if (!GetKvMetaCapacity(*group, owner_->registry_manager_->data_storage_manager(), capacity)) {
            return;
        }
        const std::uint64_t target = static_cast<std::uint64_t>(static_cast<long double>(capacity) * threshold);
        if (group_usage <= target) {
            return;
        }

        const auto [sampling_size, batching_size] = BatchSizes();
        std::vector<Candidate> candidates;
        for (const auto &[instance, indexer] : indexed_instances) {
            ReclaimCandidateVector sampled;
            if (indexer->SampleReclaimCandidates(&context, static_cast<std::int64_t>(sampling_size), sampled, true) !=
                EC_OK) {
                continue;
            }
            for (const auto &candidate : sampled) {
                candidates.push_back({instance->instance_id(), candidate.key, candidate.last_access_time_us});
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

        std::size_t selected_count = 0;
        std::uint64_t selected_bytes = 0;
        std::map<std::string, std::vector<KvMetaManager::SessionItem>> by_instance;
        for (const auto &candidate : candidates) {
            if (selected_count >= batching_size || selected_bytes >= group_usage - target) {
                break;
            }
            auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(candidate.instance_id);
            if (!indexer) {
                continue;
            }
            CacheLocationMapVector maps;
            const auto result = indexer->GetLocationMapsForMaintenance(&context, {candidate.key}, maps);
            if (result.error_codes.size() != 1 || maps.size() != 1 || result.error_codes[0] != EC_OK) {
                continue;
            }
            for (const auto &[location_id, location] : maps[0]) {
                if (!location || location->status() != CLS_SERVING ||
                    location_id.compare(0, kKvMetaLocationIdPrefix.size(), kKvMetaLocationIdPrefix) != 0) {
                    continue;
                }
                std::uint64_t size = 0;
                if (owner_->ValidateLocation(&context, candidate.key, location_id, *location, size) != EC_OK) {
                    continue;
                }
                by_instance[candidate.instance_id].push_back(
                    KvMetaManager::SessionItem{candidate.key, location_id, location, size});
                selected_bytes = SaturatingAdd(selected_bytes, size);
                ++selected_count;
                if (selected_count >= batching_size || selected_bytes >= group_usage - target) {
                    break;
                }
            }
        }

        for (auto &[instance_id, items] : by_instance) {
            std::vector<KvMetaManager::SessionItem> deleted;
            const ErrorCode ec = owner_->DeleteItems(&context, instance_id, items, false, true, &deleted);
            if (ec != EC_OK) {
                KVCM_LOG_WARN("KVMeta reclaim failed for instance [%s], ec[%d]", instance_id.c_str(), ec);
            }
            if (!deleted.empty()) {
                const auto delay = std::chrono::milliseconds(strategy->delay_before_delete_ms());
                if (delay.count() > 0) {
                    std::this_thread::sleep_for(delay);
                }
                owner_->DeletePhysicalBestEffort(&context, deleted);
            }
        }
    }

    KvMetaManager *owner_ = nullptr;
    mutable std::mutex mutex_;
    std::mutex lifecycle_mutex_;
    std::condition_variable condition_;
    bool stopping_ = true;
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
    maintenance_cancelled_.store(true, std::memory_order_release);
    if (reclaimer_) {
        reclaimer_->RequestStop();
    }
    if (write_session_manager_) {
        write_session_manager_->RequestStop();
    }
}

bool KvMetaManager::ResumeMaintenance() {
    if (!initialized_.load(std::memory_order_acquire) || !write_session_manager_ || !reclaimer_ ||
        !write_session_manager_->Start()) {
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
    if (!GetKvMetaCapacity(*group, registry_manager_->data_storage_manager(), effective_capacity)) {
        AddError(request_context, "KVMeta requires one positive-capacity PACE storage tier per group");
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
        (location.status() != CLS_WRITING && location.status() != CLS_SERVING) ||
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
        const auto [list_ec, existing] = registry_manager_->ListInstanceInfo(request_context, instance_group);
        if (list_ec != EC_OK) {
            return {list_ec, {}};
        }
        if (std::any_of(existing.begin(), existing.end(), [](const auto &instance) {
                return !instance || !IsKvMetaInstance(*instance);
            })) {
            AddError(request_context, "KVMeta requires a dedicated instance group");
            return {EC_CONFIG_ERROR, {}};
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
    return {EC_OK, std::move(result)};
}

void KvMetaManager::DeletePhysicalBestEffort(RequestContext *request_context,
                                             const std::vector<SessionItem> &items) const {
    const auto storage_manager = registry_manager_->data_storage_manager();
    if (!storage_manager) {
        return;
    }
    std::map<std::string, std::vector<DataStorageUri>> by_storage;
    for (const auto &item : items) {
        if (!item.location || item.location->location_specs().size() != 1) {
            continue;
        }
        DataStorageUri uri(item.location->location_specs().front().uri());
        if (uri.Valid() && !uri.GetHostName().empty()) {
            by_storage[uri.GetHostName()].push_back(std::move(uri));
        }
    }
    for (auto &[storage_name, uris] : by_storage) {
        const auto results = storage_manager->Delete(request_context, storage_name, uris, nullptr);
        if (results.size() != uris.size() ||
            std::any_of(results.begin(), results.end(), [](ErrorCode ec) { return ec != EC_OK && ec != EC_NOENT; })) {
            // Physical deletion is best effort. Metadata is already absent, so
            // a provider failure cannot block cache traffic or corrupt usage.
            KVCM_LOG_WARN("KVMeta physical delete failed for storage [%s]", storage_name.c_str());
        }
    }
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
    const std::size_t mutation_shard =
        std::hash<std::string>{}(internal_instance_id) % metadata_mutation_mutexes_.size();
    std::unique_lock<std::mutex> mutation_lock(metadata_mutation_mutexes_[mutation_shard]);
    std::vector<std::int64_t> keys;
    keys.reserve(items.size());
    for (const auto &item : items) {
        keys.push_back(item.internal_key);
    }
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
        std::vector<CacheLocationConstPtr> removed_locations(layer.size());
        auto modifier = [&expected, &accepted, &removed_locations](const std::vector<ErrorCode> &get_ecs,
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
                return {MA_SKIP, {EC_EXIST}};
            }
            removed_locations[key_index] = locations[0];
            accepted[key_index] = true;
            return {MA_DELETE, {EC_OK}};
        };
        const auto result =
            refresh_cache_from_persistent
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
            if (accepted[i] && ec == EC_OK) {
                SessionItem item = items[layer[i]];
                item.location = removed_locations[i];
                physically_deleted.push_back(item);
                if (item.location && item.location->status() == CLS_SERVING) {
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
    mutation_lock.unlock();
    if (deleted_items) {
        *deleted_items = physically_deleted;
    }
    if (delete_physical) {
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
        if (existing[i].location->status() == CLS_WRITING) {
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
    auto publish_writing = [&](const SessionItem &candidate, bool &inserted) {
        inserted = false;
        auto modifier = [&candidate, &inserted](const std::vector<ErrorCode> &get_ecs,
                                                const LocationIdVector &,
                                                std::size_t,
                                                CacheLocationVector &locations,
                                                PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || locations.size() != 1) {
                return {MA_FAIL, {EC_MISMATCH}};
            }
            if (get_ecs[0] != EC_NOENT) {
                return {get_ecs[0] == EC_OK ? MA_SKIP : MA_FAIL, {get_ecs[0] == EC_OK ? EC_EXIST : get_ecs[0]}};
            }
            locations[0] = candidate.location;
            inserted = true;
            return {MA_OK, {EC_OK}};
        };
        const auto result = indexer->ReadModifyWriteTargetLocations(
            request_context, {candidate.internal_key}, {{candidate.location_id}}, modifier);
        if (result.per_location_error_codes.size() != 1 || result.per_location_error_codes[0].size() != 1) {
            return EC_MISMATCH;
        }
        const ErrorCode item_ec = result.per_location_error_codes[0][0];
        if (inserted && item_ec == EC_OK && (result.ec == EC_OK || result.ec == EC_PARTIAL_OK)) {
            return EC_OK;
        }
        if (result.ec != EC_OK && result.ec != EC_PARTIAL_OK) {
            return result.ec;
        }
        return item_ec == EC_OK ? EC_EXIST : item_ec;
    };

    std::vector<SessionItem> candidates;
    candidates.reserve(missing.size());
    const auto rollback_candidates = [&]() {
        if (candidates.empty()) {
            return;
        }
        const ErrorCode cleanup_ec = DeleteItems(request_context, internal_instance_id, candidates, true, false);
        if (cleanup_ec != EC_OK && write_session_manager_) {
            KvMetaWriteSessionManager::Session cleanup{
                internal_instance_id, KvMetaWriteSessionManager::Clock::now(), candidates};
            write_session_manager_->ScheduleCleanup(StringUtil::GenerateRandomString(32), std::move(cleanup));
        }
    };
    for (const std::size_t request_index : missing) {
        if (maintenance_cancelled_.load(std::memory_order_acquire) ||
            KvMetaWriteSessionManager::Clock::now() >= deadline) {
            rollback_candidates();
            return {maintenance_cancelled_.load(std::memory_order_acquire) ? EC_SERVICE_NOT_LEADER : EC_TIMEOUT, {}};
        }
        const std::string object_key = "kvmeta/" + StringUtil::GenerateRandomString(kKvMetaObjectNonceBytes);
        const auto created = storage_manager->Create(request_context,
                                                     selected.name,
                                                     {object_key},
                                                     static_cast<std::size_t>(value_sizes[request_index]),
                                                     nullptr);
        if (created.size() != 1 || created[0].first != EC_OK || !created[0].second.Valid()) {
            rollback_candidates();
            return {created.size() == 1 && created[0].first != EC_OK ? created[0].first : EC_IO_ERROR, {}};
        }
        auto location = std::make_shared<CacheLocation>();
        location->set_id(existing[request_index].location_id);
        location->set_status(CLS_WRITING);
        location->set_type(selected.type);
        location->set_spec_size(1);
        location->set_create_time(TimestampUtil::GetCurrentTimeUs());
        location->push_location_spec(LocationSpec(std::string(kKvMetaValueSpecName), created[0].second.ToUriString()));
        std::uint64_t actual_size = 0;
        if (ValidateLocation(request_context,
                             existing[request_index].internal_key,
                             existing[request_index].location_id,
                             *location,
                             actual_size) != EC_OK ||
            actual_size != value_sizes[request_index]) {
            DeletePhysicalBestEffort(request_context,
                                     {{existing[request_index].internal_key,
                                       existing[request_index].location_id,
                                       location,
                                       value_sizes[request_index]}});
            rollback_candidates();
            return {EC_CORRUPTION, {}};
        }
        SessionItem candidate{existing[request_index].internal_key,
                              existing[request_index].location_id,
                              std::move(location),
                              value_sizes[request_index]};
        bool inserted = false;
        if (const ErrorCode ec = publish_writing(candidate, inserted); ec != EC_OK) {
            if (inserted) {
                candidates.push_back(std::move(candidate));
            } else {
                DeletePhysicalBestEffort(request_context, {candidate});
            }
            rollback_candidates();
            return {ec, {}};
        }
        candidates.push_back(std::move(candidate));
    }

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
    if (std::any_of(success_keys.begin(), success_keys.end(), [](bool success) { return !success; })) {
        const ErrorCode cleanup_ec = DeleteItems(request_context, internal_instance_id, session.items, true, false);
        if (cleanup_ec != EC_OK) {
            write_session_manager_->ScheduleCleanup(write_session_id, std::move(session));
        }
        return cleanup_ec;
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        DeletePhysicalBestEffort(request_context, session.items);
        return EC_INSTANCE_NOT_EXIST;
    }
    const std::size_t mutation_shard =
        std::hash<std::string>{}(internal_instance_id) % metadata_mutation_mutexes_.size();
    std::unique_lock<std::mutex> mutation_lock(metadata_mutation_mutexes_[mutation_shard]);
    std::vector<std::int64_t> keys;
    for (const auto &item : session.items) {
        keys.push_back(item.internal_key);
    }
    ErrorCode overall = EC_OK;
    std::vector<bool> committed(session.items.size(), false);
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
        auto modifier = [&expected, &accepted](const std::vector<ErrorCode> &get_ecs,
                                               const LocationIdVector &,
                                               std::size_t key_index,
                                               CacheLocationVector &locations,
                                               PropertyMap &) -> LocationModifierResult {
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
            locations[0] = std::move(serving);
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
                indexer->AddStorageUsageByType(session.items[item_index].location->type(),
                                               session.items[item_index].value_size);
            } else {
                overall = FirstHardError(overall, ec);
            }
        }
        if (result.ec != EC_OK && result.ec != EC_PARTIAL_OK) {
            overall = FirstHardError(overall, result.ec);
        }
    }
    if (overall == EC_OK && std::all_of(committed.begin(), committed.end(), [](bool value) { return value; })) {
        return EC_OK;
    }
    // A partial metadata transition is not exposed as a partial cache write:
    // remove every generation owned by this session and let the caller retry.
    std::vector<SessionItem> cleanup = session.items;
    for (std::size_t i = 0; i < cleanup.size(); ++i) {
        if (committed[i]) {
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
        items.push_back({exact[i].internal_key, exact[i].location_id, exact[i].location, size});
    }
    std::vector<SessionItem> deleted;
    const ErrorCode delete_ec = DeleteItems(request_context, internal_instance_id, items, false, false, &deleted);
    if (!deleted.empty()) {
        std::int32_t delay_ms = 0;
        const auto [group_ec, group] =
            registry_manager_->GetInstanceGroup(request_context, instance->instance_group_name());
        if (group_ec == EC_OK && group && group->cache_config() && group->cache_config()->reclaim_strategy()) {
            delay_ms = std::max<std::int32_t>(0, group->cache_config()->reclaim_strategy()->delay_before_delete_ms());
        }
        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        DeletePhysicalBestEffort(request_context, deleted);
    }
    return delete_ec;
}

bool KvMetaManager::ExpireSession(const std::string &session_id,
                                  const std::string &internal_instance_id,
                                  const std::vector<SessionItem> &items) noexcept {
    try {
        RequestContext context("kv_meta_session_expire_" + session_id);
        return DeleteItems(&context, internal_instance_id, items, true, true) == EC_OK;
    } catch (const std::exception &e) { KVCM_LOG_WARN("KVMeta session expiry failed: %s", e.what()); } catch (...) {
        KVCM_LOG_WARN("KVMeta session expiry failed with unknown exception");
    }
    return false;
}

ErrorCode KvMetaManager::DoRecover() {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    recovery_complete_.store(false, std::memory_order_release);
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
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
                        if (batch.location_results[i] != EC_OK) {
                            instance_ec = batch.location_results[i];
                            break;
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
                        instance_ec = DeleteItems(&context, instance->instance_id(), stale, true, true, nullptr, true);
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
            overall = FirstHardError(overall, instance_ec);
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
