#include "kv_cache_manager/manager/kv_meta_manager.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <optional>
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
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/kv_meta_instance.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

namespace {

constexpr std::uint64_t kObjectKeyHashSeed = 0x8bc5'1f2d'671a'94e3ULL;
constexpr std::uint64_t kInstancePathHashSeed = 0x6e91'ca34'0bd7'52f8ULL;
constexpr std::size_t kRecoveryScanBatchSize = 1000;
constexpr std::size_t kMaintenanceDeleteBatchSize = 256;

std::string HexEncode(std::string_view input) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output;
    output.resize(input.size() * 2);
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

void AddError(RequestContext *request_context, const std::string &message) {
    if (request_context && request_context->error_tracer()) {
        request_context->error_tracer()->AddErrorMsg(message);
    }
}

ErrorCode FirstHardError(ErrorCode current, ErrorCode candidate) {
    if (current != EC_OK) {
        return current;
    }
    return candidate == EC_OK || candidate == EC_NOENT ? EC_OK : candidate;
}

std::vector<std::vector<std::size_t>> MakeUniqueKeyLayers(const std::vector<std::int64_t> &keys) {
    std::vector<std::vector<std::size_t>> layers;
    std::vector<std::unordered_set<std::int64_t>> layer_keys;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        std::size_t layer = 0;
        for (; layer < layer_keys.size(); ++layer) {
            if (layer_keys[layer].insert(keys[i]).second) {
                layers[layer].push_back(i);
                break;
            }
        }
        if (layer == layer_keys.size()) {
            layer_keys.emplace_back();
            layer_keys.back().insert(keys[i]);
            layers.push_back({i});
        }
    }
    return layers;
}

bool ReadLogicalSize(const CacheLocation &location, std::uint64_t &out_size) {
    out_size = 0;
    if (location.spec_size() != 1 || location.location_specs().size() != 1 ||
        location.location_specs().front().name() != kKvMetaValueSpecName) {
        return false;
    }
    const DataStorageUri uri(location.location_specs().front().uri());
    if (!uri.Valid() || uri.GetHostName().empty()) {
        return false;
    }
    std::uint64_t size = 0;
    uri.GetParamAs<std::uint64_t>("size", size);
    if (size == 0) {
        return false;
    }
    out_size = size;
    return true;
}

bool HasMatchingStorageBackend(const CacheLocation &location,
                               const std::shared_ptr<DataStorageManager> &data_storage_manager) {
    if (!data_storage_manager || location.location_specs().size() != 1) {
        return false;
    }
    const DataStorageUri uri(location.location_specs().front().uri());
    if (!uri.Valid() || uri.GetHostName().empty()) {
        return false;
    }
    const DataStorageType uri_type = ToDataStorageType(uri.GetProtocol());
    const bool scheme_matches =
        IsTairMempoolStorageType(location.type())
            ? uri.GetProtocol() == kTairMempoolUriScheme
            : uri_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN &&
                  ToBaseType(uri_type) == ToBaseType(location.type());
    if (!scheme_matches) {
        return false;
    }
    const auto backend = data_storage_manager->GetDataStorageBackend(uri.GetHostName());
    return backend && backend->GetType() == location.type();
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

bool IsCommittedObject(const CacheLocation &location) {
    // CLS_NEW keeps this path invisible to the KV-cache reclaimer/migration
    // machinery. A negative create_time is the KVMeta-private commit marker;
    // an in-flight allocation always has a positive wall-clock timestamp.
    return location.status() == CLS_NEW && location.create_time() < 0;
}

bool SamePhysicalAllocation(const CacheLocation &lhs, const CacheLocation &rhs) {
    if (lhs.type() != rhs.type() || lhs.location_specs().size() != rhs.location_specs().size()) {
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

std::shared_ptr<CacheLocation> MakeCommittedLocation(const CacheLocation &location) {
    auto committed = std::make_shared<CacheLocation>(location);
    const std::int64_t create_time = location.create_time();
    committed->set_create_time(create_time > 0 ? -create_time : -1);
    return committed;
}

} // namespace

struct KvMetaManager::SessionItem {
    std::size_t request_index = 0;
    std::string original_key;
    std::int64_t internal_key = 0;
    std::string location_id;
    // Exact value expected by metadata CAS/delete.
    CacheLocationConstPtr metadata_location;
    // Physical allocation owned by this generation. It can differ from
    // metadata_location while reconciling a failed start/commit operation.
    CacheLocationConstPtr data_location;
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
    enum class TakeResult { kOk, kNotFound, kInstanceMismatch, kSizeMismatch };
    enum class PutResult { kOk, kDuplicate, kStopped, kFull };

    struct Session {
        std::string internal_instance_id;
        std::vector<KvMetaManager::SessionItem> items;
    };

    KvMetaWriteSessionManager(KvMetaManager *owner, std::size_t max_sessions)
        : owner_(owner), max_sessions_(max_sessions) {}
    ~KvMetaWriteSessionManager() { StopAndDiscard(); }

    bool Start() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        if (thread_.joinable()) {
            return !stopping_;
        }
        stopping_ = false;
        try {
            thread_ = std::thread([this]() { ExpireLoop(); });
        } catch (const std::exception &e) {
            stopping_ = true;
            KVCM_LOG_ERROR("failed to start KVMeta write-session expiry worker: %s", e.what());
            return false;
        }
        return true;
    }

    PutResult Put(const std::string &session_id,
                  const std::string &internal_instance_id,
                  std::vector<KvMetaManager::SessionItem> items,
                  std::int64_t timeout_seconds) {
        auto entry = std::make_shared<Entry>();
        entry->session_id = session_id;
        entry->deadline = Clock::now() + std::chrono::seconds(timeout_seconds);
        entry->session.internal_instance_id = internal_instance_id;
        entry->session.items = std::move(items);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return PutResult::kStopped;
            }
            if (sessions_.size() >= max_sessions_) {
                return PutResult::kFull;
            }
            if (sessions_.find(session_id) != sessions_.end()) {
                return PutResult::kDuplicate;
            }
            entry->sequence = next_sequence_++;
            sessions_.emplace(session_id, entry);
            deadlines_.emplace(DeadlineKey{entry->deadline, entry->sequence}, entry);
        }
        condition_.notify_all();
        return PutResult::kOk;
    }

    PutResult Availability() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return PutResult::kStopped;
        }
        return sessions_.size() < max_sessions_ ? PutResult::kOk : PutResult::kFull;
    }

    // Close admission and wake the expiry loop without joining it. Server
    // demotion uses this first so the existing KV-cache drain/GC/migration
    // sequence never waits behind KVMeta backend I/O. StopAndDiscard performs
    // the eventual join before CacheManager teardown.
    void RequestStop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
    }

    TakeResult Take(const std::string &session_id,
                    const std::string &internal_instance_id,
                    std::optional<std::size_t> expected_size,
                    Session &out) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return TakeResult::kNotFound;
        }
        const auto &entry = it->second;
        if (entry->session.internal_instance_id != internal_instance_id) {
            return TakeResult::kInstanceMismatch;
        }
        if (expected_size && entry->session.items.size() != *expected_size) {
            return TakeResult::kSizeMismatch;
        }
        deadlines_.erase(DeadlineKey{entry->deadline, entry->sequence});
        out = std::move(entry->session);
        sessions_.erase(it);
        condition_.notify_all();
        return TakeResult::kOk;
    }

    void StopAndDiscard() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        std::thread expiry_thread;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            if (thread_.joinable()) {
                expiry_thread = std::move(thread_);
            }
        }
        condition_.notify_all();
        if (expiry_thread.joinable()) {
            expiry_thread.join();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        sessions_.clear();
        deadlines_.clear();
    }

private:
    using Clock = std::chrono::steady_clock;
    using DeadlineKey = std::pair<Clock::time_point, std::uint64_t>;

    struct Entry {
        std::string session_id;
        Clock::time_point deadline;
        std::uint64_t sequence = 0;
        Session session;
    };

    void ExpireLoop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            if (deadlines_.empty()) {
                condition_.wait(lock, [this]() { return stopping_ || !deadlines_.empty(); });
                continue;
            }
            const auto deadline = deadlines_.begin()->first.first;
            condition_.wait_until(lock, deadline);
            if (stopping_) {
                break;
            }
            const auto now = Clock::now();
            std::optional<Session> expired;
            if (!deadlines_.empty() && deadlines_.begin()->first.first <= now) {
                auto entry = deadlines_.begin()->second;
                deadlines_.erase(deadlines_.begin());
                const auto session_it = sessions_.find(entry->session_id);
                if (session_it != sessions_.end() && session_it->second == entry) {
                    expired.emplace(std::move(entry->session));
                    sessions_.erase(session_it);
                }
            }
            if (expired) {
                lock.unlock();
                Expire(std::move(*expired));
                lock.lock();
            }
        }
    }

    void Expire(Session session) {
        if (!owner_ || session.items.empty() ||
            owner_->maintenance_cancelled_.load(std::memory_order_acquire)) {
            return;
        }
        RequestContext request_context("kv_meta_write_session_expired");
        std::vector<bool> failed(session.items.size(), false);
        const ErrorCode ec = owner_->FinishWriteInternal(
            &request_context, session.internal_instance_id, failed, session.items);
        if (ec != EC_OK) {
            KVCM_LOG_WARN("KVMeta write-session cleanup failed, instance[%s], item_count[%zu], ec[%d]",
                          session.internal_instance_id.c_str(),
                          session.items.size(),
                          ec);
        }
    }

    KvMetaManager *owner_ = nullptr;
    const std::size_t max_sessions_;
    std::mutex lifecycle_mutex_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = true;
    std::uint64_t next_sequence_ = 0;
    std::unordered_map<std::string, std::shared_ptr<Entry>> sessions_;
    std::map<DeadlineKey, std::shared_ptr<Entry>> deadlines_;
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
        limits_.max_user_data_bytes == 0 || limits_.max_active_write_sessions == 0 ||
        limits_.max_value_bytes == 0 || limits_.max_batch_bytes == 0 || limits_.max_write_timeout_seconds <= 0) {
        KVCM_LOG_ERROR("KVMeta manager init failed: dependency or limits are invalid");
        return false;
    }
    data_storage_selector_ =
        std::make_unique<DataStorageSelector>(cache_manager_->meta_indexer_manager(), registry_manager_);
    write_session_manager_ =
        std::make_unique<KvMetaWriteSessionManager>(this, limits_.max_active_write_sessions);
    if (!write_session_manager_->Start()) {
        write_session_manager_.reset();
        data_storage_selector_.reset();
        return false;
    }
    maintenance_cancelled_.store(false, std::memory_order_release);
    initialized_.store(true, std::memory_order_release);
    return true;
}

void KvMetaManager::Shutdown() {
    CancelMaintenance();
    initialized_.store(false, std::memory_order_release);
    if (write_session_manager_) {
        write_session_manager_->StopAndDiscard();
        write_session_manager_.reset();
    }
    data_storage_selector_.reset();
}

void KvMetaManager::DoCleanup() {
    CancelMaintenance();
    if (write_session_manager_) {
        write_session_manager_->StopAndDiscard();
    }
}

void KvMetaManager::CancelMaintenance() noexcept {
    maintenance_cancelled_.store(true, std::memory_order_release);
    if (write_session_manager_) {
        write_session_manager_->RequestStop();
    }
}

bool KvMetaManager::ResumeMaintenance() {
    if (!initialized_.load(std::memory_order_acquire) || !write_session_manager_ ||
        !write_session_manager_->Start()) {
        return false;
    }
    maintenance_cancelled_.store(false, std::memory_order_release);
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

bool KvMetaManager::IsOwnedLocation(std::int64_t internal_key, const std::string &location_id) const {
    if (location_id.size() <= kKvMetaLocationIdPrefix.size() ||
        location_id.compare(0, kKvMetaLocationIdPrefix.size(), kKvMetaLocationIdPrefix) != 0) {
        return false;
    }
    const std::string_view encoded_key =
        std::string_view(location_id).substr(kKvMetaLocationIdPrefix.size());
    if ((encoded_key.size() & 1U) != 0 || encoded_key.size() / 2 > limits_.max_key_bytes) {
        return false;
    }
    std::string original_key;
    if (!HexDecode(encoded_key, original_key)) {
        return false;
    }
    return StableLocationId(original_key) == location_id && InternalKey(original_key) == internal_key;
}

ErrorCode KvMetaManager::ValidateOwnedLocation(RequestContext *request_context,
                                               std::int64_t internal_key,
                                               const std::string &location_id,
                                               const CacheLocation &location,
                                               std::uint64_t &value_size) const {
    const auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!IsOwnedLocation(internal_key, location_id) || location.id() != location_id ||
        location.status() != CLS_NEW || location.create_time() == 0 ||
        !HasMatchingStorageBackend(location, data_storage_manager) || !ReadLogicalSize(location, value_size)) {
        AddError(request_context, "KVMeta location does not match its exact key or registered storage backend");
        return EC_CORRUPTION;
    }
    return EC_OK;
}

ErrorCode KvMetaManager::ValidateInstanceId(RequestContext *request_context, const std::string &instance_id) const {
    if (!request_context || instance_id.empty() || instance_id.size() > limits_.max_instance_id_bytes) {
        AddError(request_context, "KVMeta instance_id is empty or exceeds max_instance_id_bytes");
        return EC_BADARGS;
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
        AddError(request_context, "KVMeta internal instance marker does not match the generic-object schema");
        return {EC_CORRUPTION, nullptr};
    }
    return {EC_OK, std::move(info)};
}

ErrorCode KvMetaManager::ValidateKeys(RequestContext *request_context, const std::vector<std::string> &keys) const {
    if (!request_context || keys.empty() || keys.size() > limits_.max_batch_items) {
        AddError(request_context, "KVMeta keys must be non-empty and within max_batch_items");
        return EC_BADARGS;
    }
    std::unordered_set<std::string> unique_keys;
    unique_keys.reserve(keys.size());
    for (const auto &key : keys) {
        if (key.empty() || key.size() > limits_.max_key_bytes) {
            AddError(request_context, "KVMeta key is empty or exceeds max_key_bytes");
            return EC_BADARGS;
        }
        if (!unique_keys.insert(key).second) {
            AddError(request_context, "KVMeta request contains duplicate keys");
            return EC_DUPLICATE_ENTITY;
        }
    }
    return EC_OK;
}

ErrorCode KvMetaManager::CheckDynamicByteAdmission(RequestContext *request_context,
                                                   const std::string &instance_group,
                                                   DataStorageType storage_type,
                                                   std::uint64_t requested_bytes) const {
    if (!request_context || instance_group.empty() || requested_bytes == 0) {
        return EC_BADARGS;
    }
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(request_context, instance_group);
    if (group_ec != EC_OK || !group) {
        return group_ec == EC_OK ? EC_INSTANCE_NOT_EXIST : group_ec;
    }
    const auto [instances_ec, instances] = registry_manager_->ListInstanceInfo(request_context, instance_group);
    if (instances_ec != EC_OK) {
        return instances_ec;
    }

    std::uint64_t total_usage = 0;
    std::uint64_t type_usage = 0;
    const DataStorageType base_type = ToBaseType(storage_type);
    const bool check_type_quota = storage_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    const auto saturating_add = [](std::uint64_t &sum, std::uint64_t value) {
        sum = value > std::numeric_limits<std::uint64_t>::max() - sum
                  ? std::numeric_limits<std::uint64_t>::max()
                  : sum + value;
    };
    for (const auto &instance : instances) {
        if (!instance || !IsKvMetaInstance(*instance)) {
            AddError(request_context,
                     "KVMeta byte admission requires an instance group containing only generic-object instances");
            return instance ? EC_BADARGS : EC_CORRUPTION;
        }
        auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
        if (!indexer) {
            AddError(request_context, "KVMeta byte admission could not read an instance indexer");
            return EC_INSTANCE_NOT_EXIST;
        }
        saturating_add(total_usage, indexer->GetStorageUsage());
        if (check_type_quota) {
            saturating_add(type_usage, indexer->GetStorageUsageByType(base_type));
        }
    }

    const auto fits = [requested_bytes](std::int64_t capacity, std::uint64_t used) {
        return capacity >= 0 && used <= static_cast<std::uint64_t>(capacity) &&
               requested_bytes <= static_cast<std::uint64_t>(capacity) - used;
    };
    if (!fits(group->quota().capacity(), total_usage)) {
        AddError(request_context, "KVMeta requested bytes exceed the remaining instance-group capacity");
        return EC_NOSPC;
    }
    // A pre-selection check uses UNKNOWN to validate only the group quota.
    // This preserves the selector's existing behavior while ensuring that an
    // already-full KVMeta group is reported as EC_NOSPC instead of EC_ERROR.
    if (!check_type_quota) {
        return EC_OK;
    }
    for (const auto &type_quota : group->quota().quota_config()) {
        if (ToBaseType(type_quota.storage_spec()) == base_type && !fits(type_quota.capacity(), type_usage)) {
            AddError(request_context, "KVMeta requested bytes exceed the remaining storage-type capacity");
            return EC_NOSPC;
        }
    }
    return EC_OK;
}

std::pair<ErrorCode, std::string> KvMetaManager::RegisterInstance(RequestContext *request_context,
                                                                  const std::string &instance_group,
                                                                  const std::string &instance_id,
                                                                  const std::string &user_data) {
    if (!initialized_.load(std::memory_order_acquire) || !request_context || instance_group.empty() ||
        instance_group.size() > limits_.max_instance_group_bytes || instance_id.empty() ||
        instance_id.size() > limits_.max_instance_id_bytes || user_data.size() > limits_.max_user_data_bytes) {
        AddError(request_context, "KVMeta instance_group, instance_id, or user_data exceeds a configured limit");
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

    std::lock_guard<std::mutex> lock(registration_mutex_);
    const auto [list_ec, existing_instances] = registry_manager_->ListInstanceInfo(request_context, instance_group);
    if (list_ec != EC_OK) {
        return {list_ec, {}};
    }
    if (std::any_of(existing_instances.begin(), existing_instances.end(), [](const auto &instance) {
            return !instance || !IsKvMetaInstance(*instance);
        })) {
        AddError(request_context,
                 "KVMeta requires a dedicated instance group so generic-object usage cannot affect KV-cache quota");
        return {EC_BADARGS, {}};
    }
    return cache_manager_->RegisterInstance(request_context,
                                            instance_group,
                                            InternalInstanceId(instance_id),
                                            1,
                                            {LocationSpecInfo(std::string(kKvMetaValueSpecName), 1)},
                                            deployment,
                                            {},
                                            CacheManager::QueryType::QT_BATCH_GET);
}

std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>>
KvMetaManager::GetInstanceInfo(RequestContext *request_context, const std::string &instance_id) const {
    if (!initialized_.load(std::memory_order_acquire)) {
        return {EC_ERROR, nullptr};
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
        AddError(request_context, "KVMeta instance does not exist or its indexer is unavailable");
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
        LocationIdsPerKey layer_ids;
        layer_keys.reserve(layer.size());
        layer_ids.reserve(layer.size());
        for (const std::size_t index : layer) {
            layer_keys.push_back(out[index].internal_key);
            layer_ids.push_back({out[index].location_id});
        }
        LocationsPerKey locations;
        const auto result = indexer->GetLocations(request_context, layer_keys, layer_ids, locations);
        if (locations.size() != layer.size() || result.per_location_error_codes.size() != layer.size()) {
            AddError(request_context, "KVMeta exact metadata read returned a malformed batch");
            return EC_MISMATCH;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            const std::size_t output_index = layer[i];
            if (locations[i].size() != 1 || result.per_location_error_codes[i].size() != 1) {
                out[output_index].ec = EC_MISMATCH;
            } else {
                out[output_index].ec = result.per_location_error_codes[i][0];
                out[output_index].location = locations[i][0];
                if (out[output_index].ec == EC_OK &&
                    (!out[output_index].location ||
                     out[output_index].location->id() != out[output_index].location_id)) {
                    out[output_index].ec = EC_CORRUPTION;
                }
            }
            overall = FirstHardError(overall, out[output_index].ec);
        }
    }
    return overall;
}

std::pair<ErrorCode, std::vector<KvMetaManager::GetResult>>
KvMetaManager::Get(RequestContext *request_context,
                   const std::string &instance_id,
                   const std::vector<std::string> &keys) const {
    if (!initialized_.load(std::memory_order_acquire)) {
        return {EC_ERROR, {}};
    }
    const auto validated_instance = GetValidatedInstanceInfo(request_context, instance_id);
    if (validated_instance.first != EC_OK) {
        return {validated_instance.first, {}};
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return {ec, {}};
    }
    std::vector<ExactLocation> exact;
    const ErrorCode load_ec = LoadExactLocations(request_context, InternalInstanceId(instance_id), keys, exact);
    if (load_ec != EC_OK) {
        return {load_ec, {}};
    }
    std::vector<GetResult> result(keys.size());
    for (std::size_t i = 0; i < exact.size(); ++i) {
        if (exact[i].ec == EC_NOENT) {
            continue;
        }
        if (exact[i].ec != EC_OK || !exact[i].location) {
            return {exact[i].ec == EC_OK ? EC_CORRUPTION : exact[i].ec, {}};
        }
        std::uint64_t value_size = 0;
        if (const ErrorCode ec = ValidateOwnedLocation(request_context,
                                                       exact[i].internal_key,
                                                       exact[i].location_id,
                                                       *exact[i].location,
                                                       value_size);
            ec != EC_OK) {
            return {ec, {}};
        }
        if (!IsCommittedObject(*exact[i].location)) {
            continue;
        }
        if (!ToValueLocation(*exact[i].location, result[i].location) ||
            result[i].location.value_size != value_size) {
            AddError(request_context, "KVMeta committed location is malformed");
            return {EC_CORRUPTION, {}};
        }
        result[i].found = true;
    }
    return {EC_OK, std::move(result)};
}

ErrorCode KvMetaManager::DeleteAllocatedLocations(RequestContext *request_context,
                                                   const std::vector<SessionItem> &items) const {
    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!data_storage_manager) {
        return EC_ERROR;
    }
    std::map<std::string, std::vector<DataStorageUri>> uris_by_storage;
    std::unordered_set<std::string> seen_uris;
    ErrorCode overall = EC_OK;
    for (const auto &item : items) {
        if (!item.data_location) {
            continue;
        }
        if (!HasMatchingStorageBackend(*item.data_location, data_storage_manager)) {
            overall = FirstHardError(overall, EC_CORRUPTION);
            continue;
        }
        for (const auto &spec : item.data_location->location_specs()) {
            DataStorageUri uri(spec.uri());
            if (!uri.Valid() || uri.GetHostName().empty()) {
                overall = FirstHardError(overall, EC_CORRUPTION);
                continue;
            }
            const std::string canonical = uri.ToUriString();
            if (seen_uris.insert(canonical).second) {
                uris_by_storage[uri.GetHostName()].push_back(std::move(uri));
            }
        }
    }
    for (auto &[storage_name, uris] : uris_by_storage) {
        const auto delete_results = data_storage_manager->Delete(request_context, storage_name, uris, nullptr);
        if (delete_results.size() != uris.size()) {
            overall = FirstHardError(overall, EC_MISMATCH);
            continue;
        }
        for (const ErrorCode ec : delete_results) {
            if (ec != EC_OK && ec != EC_NOENT) {
                overall = FirstHardError(overall, ec);
            }
        }
    }
    return overall;
}

ErrorCode KvMetaManager::DeleteItems(RequestContext *request_context,
                                     const std::string &internal_instance_id,
                                     const std::vector<SessionItem> &items,
                                     bool metadata_only,
                                     bool adjust_storage_usage) {
    if (items.empty()) {
        return EC_OK;
    }
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        AddError(request_context, "KVMeta indexer is unavailable during exact delete");
        return EC_INSTANCE_NOT_EXIST;
    }
    MetaSearcher searcher(indexer);
    std::vector<std::int64_t> item_keys;
    item_keys.reserve(items.size());
    for (const auto &item : items) {
        item_keys.push_back(item.internal_key);
    }

    std::vector<bool> metadata_deleted(items.size(), false);
    std::vector<bool> metadata_already_absent(items.size(), false);
    ErrorCode overall = EC_OK;
    for (const auto &layer : MakeUniqueKeyLayers(item_keys)) {
        KeyVector keys;
        LocationIdsPerKey ids;
        std::vector<std::vector<std::string>> expected_values;
        keys.reserve(layer.size());
        ids.reserve(layer.size());
        expected_values.reserve(layer.size());
        for (const std::size_t index : layer) {
            if (!items[index].metadata_location) {
                overall = FirstHardError(overall, EC_BADARGS);
                continue;
            }
            keys.push_back(items[index].internal_key);
            ids.push_back({items[index].location_id});
            expected_values.push_back({items[index].metadata_location->ToJsonString()});
        }
        if (keys.size() != layer.size()) {
            continue;
        }
        std::vector<std::vector<ErrorCode>> per_location_ec;
        const ErrorCode delete_ec = searcher.BatchDeleteLocations(request_context,
                                                                   keys,
                                                                   ids,
                                                                   per_location_ec,
                                                                   expected_values,
                                                                   adjust_storage_usage);
        overall = FirstHardError(overall, delete_ec);
        if (per_location_ec.size() != layer.size()) {
            overall = FirstHardError(overall, EC_MISMATCH);
            continue;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            if (per_location_ec[i].size() != 1) {
                overall = FirstHardError(overall, EC_MISMATCH);
                continue;
            }
            const ErrorCode ec = per_location_ec[i][0];
            if (ec == EC_OK) {
                metadata_deleted[layer[i]] = true;
            } else if (ec == EC_NOENT) {
                metadata_already_absent[layer[i]] = true;
            } else {
                overall = FirstHardError(overall, ec);
            }
        }
    }

    KeyVector deleted_keys;
    std::unordered_set<std::int64_t> unique_deleted_keys;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (metadata_deleted[i] && unique_deleted_keys.insert(items[i].internal_key).second) {
            deleted_keys.push_back(items[i].internal_key);
        }
    }
    const bool metadata_delete_is_durable = deleted_keys.empty() || indexer->Sync(deleted_keys);
    if (!metadata_delete_is_durable) {
        overall = FirstHardError(overall, EC_TIMEOUT);
        // BatchDeleteLocations adjusts the in-memory counter when the delete
        // is accepted. If its persistence barrier fails, restore a
        // conservative upper bound; the next KVMeta recovery rebuilds the
        // exact value from durable metadata.
        if (adjust_storage_usage) {
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (metadata_deleted[i] && items[i].metadata_location) {
                    indexer->AddStorageUsageByType(items[i].metadata_location->type(), items[i].value_size);
                }
            }
        }
    }

    if (!metadata_only) {
        std::vector<SessionItem> physical_items;
        physical_items.reserve(items.size());
        for (std::size_t i = 0; i < items.size(); ++i) {
            const bool safe_to_delete = metadata_already_absent[i] ||
                                        (metadata_deleted[i] && metadata_delete_is_durable);
            if (safe_to_delete && items[i].data_location) {
                physical_items.push_back(items[i]);
            }
        }
        overall = FirstHardError(overall, DeleteAllocatedLocations(request_context, physical_items));
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
    if (!initialized_.load(std::memory_order_acquire)) {
        return {EC_ERROR, std::move(response)};
    }
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return {EC_SERVICE_NOT_LEADER, std::move(response)};
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return {ec, std::move(response)};
    }
    if (instance_id.empty() || instance_id.size() > limits_.max_instance_id_bytes ||
        value_sizes.size() != keys.size() || write_timeout_seconds <= 0 ||
        write_timeout_seconds > limits_.max_write_timeout_seconds) {
        AddError(request_context, "KVMeta value_sizes or write_timeout_seconds is invalid");
        return {EC_BADARGS, std::move(response)};
    }
    std::uint64_t batch_bytes = 0;
    for (const std::uint64_t size : value_sizes) {
        if (size == 0 || size > limits_.max_value_bytes || size > limits_.max_batch_bytes ||
            size > std::numeric_limits<std::size_t>::max() || batch_bytes > limits_.max_batch_bytes - size) {
            AddError(request_context, "KVMeta value size exceeds a configured limit");
            return {EC_OUT_OF_LIMIT, std::move(response)};
        }
        batch_bytes += size;
    }

    const std::string internal_instance_id = InternalInstanceId(instance_id);
    auto [instance_ec, instance_info] = GetValidatedInstanceInfo(request_context, instance_id);
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (instance_ec != EC_OK || !indexer) {
        return {instance_ec != EC_OK ? instance_ec : EC_INSTANCE_NOT_EXIST, std::move(response)};
    }

    // The existing selector admits based on current usage because KV-cache
    // blocks have a fixed registered size. KVMeta additionally holds a
    // side-path-only shard lock and checks used + this request's exact bytes,
    // so differently sized values cannot overshoot group/type quota through
    // concurrent PutStart calls.
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % quota_admission_mutexes_.size();
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[quota_shard]);

    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!data_storage_manager) {
        return {EC_ERROR, StartWriteResult{}};
    }

    std::vector<ExactLocation> existing;
    if (const ErrorCode ec = LoadExactLocations(request_context, internal_instance_id, keys, existing); ec != EC_OK) {
        return {ec, std::move(response)};
    }
    response.key_mask.assign(keys.size(), false);
    std::vector<std::size_t> missing_indices;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (existing[i].ec == EC_NOENT) {
            missing_indices.push_back(i);
        } else if (existing[i].ec == EC_OK && existing[i].location) {
            std::uint64_t existing_size = 0;
            if (const ErrorCode ec = ValidateOwnedLocation(request_context,
                                                           existing[i].internal_key,
                                                           existing[i].location_id,
                                                           *existing[i].location,
                                                           existing_size);
                ec != EC_OK) {
                return {ec, StartWriteResult{}};
            }
            if (existing_size != value_sizes[i]) {
                AddError(request_context, "KVMeta existing value size does not match PutStart value_sizes");
                return {EC_MISMATCH, StartWriteResult{}};
            }
            response.key_mask[i] = true;
        } else {
            return {existing[i].ec == EC_OK ? EC_CORRUPTION : existing[i].ec, StartWriteResult{}};
        }
    }
    if (missing_indices.empty()) {
        return {EC_OK, std::move(response)};
    }
    if (!write_session_manager_) {
        return {EC_SERVICE_NOT_LEADER, StartWriteResult{}};
    }
    switch (write_session_manager_->Availability()) {
    case KvMetaWriteSessionManager::PutResult::kOk:
        break;
    case KvMetaWriteSessionManager::PutResult::kFull:
        AddError(request_context, "KVMeta active write-session limit has been reached");
        return {EC_NOSPC, StartWriteResult{}};
    case KvMetaWriteSessionManager::PutResult::kStopped:
        return {EC_SERVICE_NOT_LEADER, StartWriteResult{}};
    case KvMetaWriteSessionManager::PutResult::kDuplicate:
        return {EC_ERROR, StartWriteResult{}};
    }

    std::uint64_t missing_bytes = 0;
    for (const std::size_t index : missing_indices) {
        missing_bytes += value_sizes[index];
    }

    if (!data_storage_selector_) {
        return {EC_ERROR, StartWriteResult{}};
    }
    if (const ErrorCode ec = CheckDynamicByteAdmission(request_context,
                                                       instance_info->instance_group_name(),
                                                       DataStorageType::DATA_STORAGE_TYPE_UNKNOWN,
                                                       missing_bytes);
        ec != EC_OK) {
        return {ec, StartWriteResult{}};
    }
    const auto selected = data_storage_selector_->SelectCacheWriteDataStorageBackend(
        request_context, instance_info->instance_group_name());
    if (selected.ec != EC_OK || selected.name.empty() || selected.type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
        return {selected.ec == EC_OK ? EC_NOENT : selected.ec, StartWriteResult{}};
    }
    if (const ErrorCode ec = CheckDynamicByteAdmission(request_context,
                                                       instance_info->instance_group_name(),
                                                       selected.type,
                                                       missing_bytes);
        ec != EC_OK) {
        return {ec, StartWriteResult{}};
    }
    const auto selected_backend = data_storage_manager->GetDataStorageBackend(selected.name);
    if (!selected_backend || selected_backend->GetType() != selected.type) {
        AddError(request_context, "KVMeta selected storage backend changed before allocation");
        return {EC_CORRUPTION, StartWriteResult{}};
    }

    // A singleton Create call is intentional. Several existing filesystem
    // backends pack a batch into one file; singleton allocation prevents a
    // later per-key Remove from deleting another generic object.
    std::vector<SessionItem> candidates;
    candidates.reserve(missing_indices.size());
    for (const std::size_t request_index : missing_indices) {
        if (maintenance_cancelled_.load(std::memory_order_acquire)) {
            const ErrorCode cleanup_ec = DeleteAllocatedLocations(request_context, candidates);
            if (cleanup_ec != EC_OK) {
                KVCM_LOG_WARN("KVMeta cancellation could not release all uncommitted allocations, ec[%d]",
                              cleanup_ec);
            }
            return {EC_SERVICE_NOT_LEADER, StartWriteResult{}};
        }
        const std::uint64_t instance_path_hash =
            Hash64(internal_instance_id.data(), internal_instance_id.size(), kInstancePathHashSeed);
        const std::string object_key =
            "kvmeta/" + StringUtil::Uint64ToHex(instance_path_hash) + "/" +
            StringUtil::Uint64ToHex(static_cast<std::uint64_t>(existing[request_index].internal_key)) + "/" +
            StringUtil::GenerateRandomString(32);
        const auto create_result = data_storage_manager->Create(request_context,
                                                                 selected.name,
                                                                 {object_key},
                                                                 static_cast<std::size_t>(value_sizes[request_index]),
                                                                 nullptr);
        if (create_result.size() != 1 || create_result[0].first != EC_OK || !create_result[0].second.Valid() ||
            create_result[0].second.GetHostName() != selected.name) {
            const ErrorCode create_ec = create_result.size() == 1 ? create_result[0].first : EC_MISMATCH;
            if (create_result.size() == 1 && create_result[0].first == EC_OK && create_result[0].second.Valid()) {
                const auto delete_result =
                    data_storage_manager->Delete(request_context, selected.name, {create_result[0].second}, nullptr);
                if (delete_result.size() != 1 ||
                    (delete_result[0] != EC_OK && delete_result[0] != EC_NOENT)) {
                    KVCM_LOG_WARN("KVMeta could not release a malformed new allocation");
                }
            }
            DeleteAllocatedLocations(request_context, candidates);
            AddError(request_context, "KVMeta singleton storage allocation failed");
            return {create_ec == EC_OK ? EC_CORRUPTION : create_ec, StartWriteResult{}};
        }

        auto location = std::make_shared<CacheLocation>();
        location->set_id(existing[request_index].location_id);
        location->set_status(CLS_NEW);
        location->set_type(selected.type);
        location->set_spec_size(1);
        location->set_create_time(std::max<std::int64_t>(1, TimestampUtil::GetCurrentTimeUs()));
        location->push_location_spec(
            LocationSpec(std::string(kKvMetaValueSpecName), create_result[0].second.ToUriString()));
        location->set_validated_total_size(value_sizes[request_index]);
        std::uint64_t uri_size = 0;
        const ErrorCode location_ec = ValidateOwnedLocation(request_context,
                                                            existing[request_index].internal_key,
                                                            existing[request_index].location_id,
                                                            *location,
                                                            uri_size);
        if (location_ec != EC_OK || uri_size != value_sizes[request_index]) {
            const auto delete_result =
                data_storage_manager->Delete(request_context, selected.name, {create_result[0].second}, nullptr);
            if (delete_result.size() != 1 ||
                (delete_result[0] != EC_OK && delete_result[0] != EC_NOENT)) {
                KVCM_LOG_WARN("KVMeta could not release an invalid-size new allocation");
            }
            DeleteAllocatedLocations(request_context, candidates);
            AddError(request_context,
                     location_ec == EC_OK ? "KVMeta storage returned a mismatched allocation size"
                                          : "KVMeta storage returned a malformed allocation URI");
            return {location_ec == EC_OK ? EC_MISMATCH : location_ec, StartWriteResult{}};
        }
        candidates.push_back(SessionItem{request_index,
                                         keys[request_index],
                                         existing[request_index].internal_key,
                                         existing[request_index].location_id,
                                         location,
                                         location,
                                         value_sizes[request_index]});
    }

    std::vector<std::int64_t> candidate_keys;
    candidate_keys.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        candidate_keys.push_back(candidate.internal_key);
    }
    std::vector<bool> inserted(candidates.size(), false);
    std::vector<bool> lost_race(candidates.size(), false);
    ErrorCode insert_error = EC_OK;
    for (const auto &layer : MakeUniqueKeyLayers(candidate_keys)) {
        KeyVector layer_keys;
        LocationIdsPerKey layer_ids;
        std::vector<CacheLocationConstPtr> layer_locations;
        layer_keys.reserve(layer.size());
        layer_ids.reserve(layer.size());
        layer_locations.reserve(layer.size());
        for (const std::size_t index : layer) {
            layer_keys.push_back(candidates[index].internal_key);
            layer_ids.push_back({candidates[index].location_id});
            layer_locations.push_back(candidates[index].metadata_location);
        }
        std::vector<bool> modifier_inserted(layer.size(), false);
        auto modifier = [&layer_locations, &modifier_inserted](const std::vector<ErrorCode> &get_ecs,
                                                               const LocationIdVector &,
                                                               std::size_t key_index,
                                                               CacheLocationVector &locations,
                                                               PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= layer_locations.size()) {
                return {MA_FAIL, {EC_MISMATCH}};
            }
            if (get_ecs[0] == EC_NOENT) {
                locations[0] = layer_locations[key_index];
                modifier_inserted[key_index] = true;
                return {MA_OK, {EC_OK}};
            }
            if (get_ecs[0] == EC_OK) {
                return {MA_SKIP, {EC_EXIST}};
            }
            return {MA_FAIL, {get_ecs[0]}};
        };
        const auto rmw = indexer->ReadModifyWriteTargetLocations(request_context, layer_keys, layer_ids, modifier);
        if (rmw.per_location_error_codes.size() != layer.size()) {
            insert_error = FirstHardError(insert_error, EC_MISMATCH);
            break;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            if (rmw.per_location_error_codes[i].size() != 1) {
                insert_error = FirstHardError(insert_error, EC_MISMATCH);
                continue;
            }
            const ErrorCode ec = rmw.per_location_error_codes[i][0];
            if (ec == EC_OK && modifier_inserted[i]) {
                inserted[layer[i]] = true;
            } else if (ec == EC_EXIST) {
                lost_race[layer[i]] = true;
                response.key_mask[candidates[layer[i]].request_index] = true;
            } else {
                insert_error = FirstHardError(insert_error, ec == EC_OK ? EC_MISMATCH : ec);
            }
        }
        if (rmw.ec != EC_OK && rmw.ec != EC_PARTIAL_OK) {
            insert_error = FirstHardError(insert_error, rmw.ec);
        }
        if (insert_error != EC_OK) {
            break;
        }
    }

    auto rollback_start = [&](ErrorCode original_error) {
        std::vector<std::string> candidate_original_keys;
        candidate_original_keys.reserve(candidates.size());
        for (const auto &candidate : candidates) {
            candidate_original_keys.push_back(candidate.original_key);
        }
        indexer->Sync(candidate_keys);
        std::vector<ExactLocation> current;
        const ErrorCode reload_ec =
            LoadExactLocations(request_context, internal_instance_id, candidate_original_keys, current);
        std::vector<SessionItem> exact_deletes;
        std::vector<SessionItem> direct_deletes;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (i < current.size() && current[i].ec == EC_OK && current[i].location) {
                if (current[i].location->ToJsonString() == candidates[i].metadata_location->ToJsonString() ||
                    SamePhysicalAllocation(*current[i].location, *candidates[i].data_location)) {
                    SessionItem item = candidates[i];
                    item.metadata_location = current[i].location;
                    exact_deletes.push_back(std::move(item));
                } else {
                    direct_deletes.push_back(candidates[i]);
                }
            } else if (i < current.size() && current[i].ec == EC_NOENT) {
                direct_deletes.push_back(candidates[i]);
            } else if (inserted[i]) {
                exact_deletes.push_back(candidates[i]);
            }
        }
        const ErrorCode meta_cleanup_ec =
            DeleteItems(request_context, internal_instance_id, exact_deletes, false, false);
        const ErrorCode direct_cleanup_ec = DeleteAllocatedLocations(request_context, direct_deletes);
        if (reload_ec != EC_OK || meta_cleanup_ec != EC_OK || direct_cleanup_ec != EC_OK) {
            AddError(request_context, "KVMeta start rollback was incomplete; uncertain allocations were retained");
        }
        return original_error;
    };

    if (insert_error != EC_OK) {
        return {rollback_start(insert_error), StartWriteResult{}};
    }

    // A different process can win the exact metadata insertion between our
    // initial read and conditional insert. Treat it as an existing value only
    // after validating the complete schema and the caller-declared size.
    std::vector<std::string> race_winner_keys;
    std::vector<std::size_t> race_winner_indices;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (lost_race[i]) {
            race_winner_keys.push_back(candidates[i].original_key);
            race_winner_indices.push_back(i);
        }
    }
    if (!race_winner_keys.empty()) {
        std::vector<ExactLocation> race_winners;
        const ErrorCode reload_ec =
            LoadExactLocations(request_context, internal_instance_id, race_winner_keys, race_winners);
        if (reload_ec != EC_OK || race_winners.size() != race_winner_indices.size()) {
            return {rollback_start(reload_ec == EC_OK ? EC_MISMATCH : reload_ec), StartWriteResult{}};
        }
        for (std::size_t i = 0; i < race_winners.size(); ++i) {
            const std::size_t candidate_index = race_winner_indices[i];
            std::uint64_t winner_size = 0;
            if (race_winners[i].ec != EC_OK || !race_winners[i].location) {
                return {rollback_start(race_winners[i].ec == EC_OK ? EC_CORRUPTION : race_winners[i].ec),
                        StartWriteResult{}};
            }
            const ErrorCode validate_ec = ValidateOwnedLocation(request_context,
                                                                race_winners[i].internal_key,
                                                                race_winners[i].location_id,
                                                                *race_winners[i].location,
                                                                winner_size);
            if (validate_ec != EC_OK || winner_size != candidates[candidate_index].value_size) {
                if (validate_ec == EC_OK) {
                    AddError(request_context, "KVMeta concurrent winner has a different value size");
                }
                return {rollback_start(validate_ec == EC_OK ? EC_MISMATCH : validate_ec), StartWriteResult{}};
            }
        }
    }

    std::vector<SessionItem> race_losers;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (lost_race[i]) {
            race_losers.push_back(candidates[i]);
        }
    }
    if (const ErrorCode ec = DeleteAllocatedLocations(request_context, race_losers); ec != EC_OK) {
        KVCM_LOG_WARN("KVMeta failed to release one or more race-loser allocations, ec[%d]", ec);
    }

    std::vector<SessionItem> session_items;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (inserted[i]) {
            session_items.push_back(candidates[i]);
        }
    }
    if (session_items.empty()) {
        return {EC_OK, std::move(response)};
    }

    std::vector<std::int64_t> inserted_keys;
    inserted_keys.reserve(session_items.size());
    for (const auto &item : session_items) {
        inserted_keys.push_back(item.internal_key);
    }
    if (!indexer->Sync(inserted_keys)) {
        DeleteItems(request_context, internal_instance_id, session_items, false, false);
        AddError(request_context, "KVMeta metadata reservation did not reach its persistence barrier");
        return {EC_TIMEOUT, StartWriteResult{}};
    }

    for (const auto &item : session_items) {
        KvMetaManager::ValueLocation location;
        if (!ToValueLocation(*item.metadata_location, location)) {
            DeleteItems(request_context, internal_instance_id, session_items, false, false);
            return {EC_CORRUPTION, StartWriteResult{}};
        }
        response.locations.push_back(std::move(location));
    }
    for (const auto &item : session_items) {
        indexer->AddStorageUsageByType(item.metadata_location->type(), item.value_size);
    }

    std::string session_id;
    auto session_result = KvMetaWriteSessionManager::PutResult::kDuplicate;
    for (int attempt = 0;
         attempt < 8 && session_result == KvMetaWriteSessionManager::PutResult::kDuplicate;
         ++attempt) {
        session_id = StringUtil::GenerateRandomString(32);
        auto items_for_attempt = session_items;
        session_result = write_session_manager_
                             ? write_session_manager_->Put(session_id,
                                                           internal_instance_id,
                                                           std::move(items_for_attempt),
                                                           write_timeout_seconds)
                             : KvMetaWriteSessionManager::PutResult::kStopped;
    }
    if (session_result != KvMetaWriteSessionManager::PutResult::kOk) {
        DeleteItems(request_context, internal_instance_id, session_items, false, true);
        if (session_result == KvMetaWriteSessionManager::PutResult::kFull) {
            AddError(request_context, "KVMeta active write-session limit has been reached");
            return {EC_NOSPC, StartWriteResult{}};
        }
        if (session_result == KvMetaWriteSessionManager::PutResult::kStopped) {
            AddError(request_context, "KVMeta write-session manager is stopped");
            return {EC_SERVICE_NOT_LEADER, StartWriteResult{}};
        }
        AddError(request_context, "KVMeta could not generate a unique write-session id");
        return {EC_ERROR, StartWriteResult{}};
    }
    response.write_session_id = std::move(session_id);
    for (const auto &location : response.locations) {
        data_storage_manager->RecordWriteBytes(selected.name, location.value_size);
    }
    return {EC_OK, std::move(response)};
}

ErrorCode KvMetaManager::FinishWriteInternal(RequestContext *request_context,
                                             const std::string &internal_instance_id,
                                             const std::vector<bool> &success_keys,
                                             const std::vector<SessionItem> &items) {
    if (items.empty() || success_keys.size() != items.size()) {
        return EC_BADARGS;
    }
    // The protocol uses all-or-nothing failure handling. A single failed value
    // aborts the complete session, which also makes packed/remote backend
    // semantics unsurprising even though current allocations are singleton.
    if (std::any_of(success_keys.begin(), success_keys.end(), [](bool success) { return !success; })) {
        return DeleteItems(request_context, internal_instance_id, items, false, true);
    }

    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        return EC_INSTANCE_NOT_EXIST;
    }
    std::vector<CacheLocationConstPtr> committed_locations;
    std::vector<std::int64_t> item_keys;
    committed_locations.reserve(items.size());
    item_keys.reserve(items.size());
    for (const auto &item : items) {
        if (!item.metadata_location || item.metadata_location->status() != CLS_NEW ||
            item.metadata_location->create_time() <= 0) {
            return EC_CORRUPTION;
        }
        committed_locations.push_back(MakeCommittedLocation(*item.metadata_location));
        item_keys.push_back(item.internal_key);
    }

    std::vector<bool> committed(items.size(), false);
    ErrorCode commit_error = EC_OK;
    for (const auto &layer : MakeUniqueKeyLayers(item_keys)) {
        KeyVector keys;
        LocationIdsPerKey ids;
        std::vector<CacheLocationConstPtr> expected;
        std::vector<CacheLocationConstPtr> replacement;
        for (const std::size_t index : layer) {
            keys.push_back(items[index].internal_key);
            ids.push_back({items[index].location_id});
            expected.push_back(items[index].metadata_location);
            replacement.push_back(committed_locations[index]);
        }
        std::vector<bool> modifier_committed(layer.size(), false);
        auto modifier = [&expected, &replacement, &modifier_committed](const std::vector<ErrorCode> &get_ecs,
                                                                       const LocationIdVector &,
                                                                       std::size_t key_index,
                                                                       CacheLocationVector &locations,
                                                                       PropertyMap &) -> LocationModifierResult {
            if (get_ecs.size() != 1 || locations.size() != 1 || key_index >= expected.size()) {
                return {MA_FAIL, {EC_MISMATCH}};
            }
            if (get_ecs[0] != EC_OK) {
                return {MA_FAIL, {get_ecs[0]}};
            }
            if (!locations[0] || locations[0]->ToJsonString() != expected[key_index]->ToJsonString()) {
                return {MA_SKIP, {EC_MISMATCH}};
            }
            locations[0] = replacement[key_index];
            modifier_committed[key_index] = true;
            return {MA_OK, {EC_OK}};
        };
        const auto rmw = indexer->ReadModifyWriteLocation(request_context, keys, ids, modifier);
        if (rmw.per_location_error_codes.size() != layer.size()) {
            commit_error = FirstHardError(commit_error, EC_MISMATCH);
            break;
        }
        for (std::size_t i = 0; i < layer.size(); ++i) {
            if (rmw.per_location_error_codes[i].size() != 1) {
                commit_error = FirstHardError(commit_error, EC_MISMATCH);
                continue;
            }
            const ErrorCode ec = rmw.per_location_error_codes[i][0];
            if (ec == EC_OK && modifier_committed[i]) {
                committed[layer[i]] = true;
            } else {
                commit_error = FirstHardError(commit_error, ec == EC_OK ? EC_MISMATCH : ec);
            }
        }
        if (rmw.ec != EC_OK && rmw.ec != EC_PARTIAL_OK) {
            commit_error = FirstHardError(commit_error, rmw.ec);
        }
        if (commit_error != EC_OK) {
            break;
        }
    }
    if (commit_error == EC_OK && !indexer->Sync(item_keys)) {
        commit_error = EC_TIMEOUT;
    }
    if (commit_error == EC_OK) {
        return EC_OK;
    }

    std::vector<std::string> original_keys;
    original_keys.reserve(items.size());
    for (const auto &item : items) {
        original_keys.push_back(item.original_key);
    }
    std::vector<ExactLocation> current;
    indexer->Sync(item_keys);
    const ErrorCode reload_ec = LoadExactLocations(request_context, internal_instance_id, original_keys, current);
    std::vector<SessionItem> exact_deletes;
    std::vector<SessionItem> direct_deletes;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i < current.size() && current[i].ec == EC_OK && current[i].location) {
            const std::string current_value = current[i].location->ToJsonString();
            if (current_value == items[i].metadata_location->ToJsonString() ||
                current_value == committed_locations[i]->ToJsonString()) {
                SessionItem item = items[i];
                item.metadata_location = current[i].location;
                exact_deletes.push_back(std::move(item));
            } else {
                direct_deletes.push_back(items[i]);
            }
        } else if (i < current.size() && current[i].ec == EC_NOENT) {
            direct_deletes.push_back(items[i]);
        } else {
            SessionItem item = items[i];
            item.metadata_location = committed[i] ? committed_locations[i] : items[i].metadata_location;
            exact_deletes.push_back(std::move(item));
        }
    }
    const ErrorCode exact_ec = DeleteItems(request_context, internal_instance_id, exact_deletes, false, true);
    const ErrorCode direct_ec = DeleteAllocatedLocations(request_context, direct_deletes);
    if (reload_ec != EC_OK || exact_ec != EC_OK || direct_ec != EC_OK) {
        AddError(request_context,
                 "KVMeta commit rollback was incomplete; exact metadata guards prevented unsafe deletion");
    }
    return commit_error;
}

ErrorCode KvMetaManager::FinishWrite(RequestContext *request_context,
                                     const std::string &instance_id,
                                     const std::string &write_session_id,
                                     const std::vector<bool> &success_keys) {
    if (!initialized_.load(std::memory_order_acquire) ||
        ValidateInstanceId(request_context, instance_id) != EC_OK || write_session_id.empty() ||
        success_keys.empty() || !write_session_manager_) {
        AddError(request_context, "KVMeta FinishWrite requires a non-empty success mask");
        return EC_BADARGS;
    }
    KvMetaWriteSessionManager::Session session;
    const auto take_result = write_session_manager_->Take(write_session_id,
                                                           InternalInstanceId(instance_id),
                                                           std::optional<std::size_t>{success_keys.size()},
                                                           session);
    switch (take_result) {
    case KvMetaWriteSessionManager::TakeResult::kNotFound:
        AddError(request_context, "KVMeta write session does not exist or has expired");
        return EC_NOENT;
    case KvMetaWriteSessionManager::TakeResult::kInstanceMismatch:
        AddError(request_context, "KVMeta write session belongs to another instance");
        return EC_BADARGS;
    case KvMetaWriteSessionManager::TakeResult::kSizeMismatch:
        AddError(request_context, "KVMeta success mask size does not match the write session");
        return EC_MISMATCH;
    case KvMetaWriteSessionManager::TakeResult::kOk:
        break;
    }
    return FinishWriteInternal(request_context, session.internal_instance_id, success_keys, session.items);
}

ErrorCode KvMetaManager::Remove(RequestContext *request_context,
                                const std::string &instance_id,
                                const std::vector<std::string> &keys) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    const auto validated_instance = GetValidatedInstanceInfo(request_context, instance_id);
    if (validated_instance.first != EC_OK) {
        return validated_instance.first;
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
        if (const ErrorCode ec = ValidateOwnedLocation(request_context,
                                                       exact[i].internal_key,
                                                       exact[i].location_id,
                                                       *exact[i].location,
                                                       size);
            ec != EC_OK) {
            return ec;
        }
        // Do not let an independent Remove invalidate a writer's allocation
        // while its session can still commit or roll back that same URI.
        // Session timeout/PutFinish owns cleanup of active generations.
        if (!IsCommittedObject(*exact[i].location)) {
            AddError(request_context, "KVMeta cannot remove a value while its write session is active");
            return EC_EXIST;
        }
        items.push_back(SessionItem{i,
                                    keys[i],
                                    exact[i].internal_key,
                                    exact[i].location_id,
                                    exact[i].location,
                                    exact[i].location,
                                    size});
    }
    return DeleteItems(request_context, internal_instance_id, items, false, true);
}

ErrorCode KvMetaManager::TrimAll(RequestContext *request_context,
                                 const std::string &instance_id,
                                 bool metadata_only) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }
    const auto [instance_ec, instance_info] = GetValidatedInstanceInfo(request_context, instance_id);
    if (instance_ec != EC_OK || !instance_info) {
        return instance_ec == EC_OK ? EC_CORRUPTION : instance_ec;
    }
    const std::string internal_instance_id = InternalInstanceId(instance_id);
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
    if (!indexer) {
        return EC_INSTANCE_NOT_EXIST;
    }

    // Block only new KVMeta allocations in this dedicated group. Existing
    // KV-cache groups never take these locks, and each delete batch remains
    // bounded even for a very large embedding namespace.
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % quota_admission_mutexes_.size();
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[quota_shard]);
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }

    for (;;) {
        if (maintenance_cancelled_.load(std::memory_order_acquire)) {
            return EC_SERVICE_NOT_LEADER;
        }
        std::vector<SessionItem> batch;
        batch.reserve(kMaintenanceDeleteBatchSize);
        bool found_any = false;
        ErrorCode operation_ec = EC_OK;
        const auto flush = [&]() {
            if (batch.empty() || operation_ec != EC_OK) {
                return;
            }
            if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                operation_ec = EC_SERVICE_NOT_LEADER;
                return;
            }
            operation_ec = DeleteItems(request_context, internal_instance_id, batch, metadata_only, true);
            batch.clear();
        };

        std::string cursor = SCAN_BASE_CURSOR;
        do {
            if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                return EC_SERVICE_NOT_LEADER;
            }
            std::string next_cursor;
            KeyVector keys;
            const ErrorCode scan_ec =
                indexer->Scan(request_context, cursor, kRecoveryScanBatchSize, next_cursor, keys);
            if (scan_ec != EC_OK || next_cursor.empty()) {
                return scan_ec == EC_OK ? EC_CORRUPTION : scan_ec;
            }
            if (!keys.empty()) {
                CacheLocationMapVector locations;
                const auto get_result = indexer->GetLocations(request_context, keys, locations);
                if (get_result.ec != EC_OK || locations.size() != keys.size() ||
                    get_result.error_codes.size() != keys.size()) {
                    return get_result.ec == EC_OK ? EC_MISMATCH : get_result.ec;
                }
                for (std::size_t i = 0; i < keys.size() && operation_ec == EC_OK; ++i) {
                    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
                        return EC_SERVICE_NOT_LEADER;
                    }
                    if (get_result.error_codes[i] != EC_OK || locations[i].empty()) {
                        return get_result.error_codes[i] == EC_OK ? EC_CORRUPTION
                                                                  : get_result.error_codes[i];
                    }
                    for (const auto &[location_id, location] : locations[i]) {
                        if (!location) {
                            operation_ec = EC_CORRUPTION;
                            break;
                        }
                        std::uint64_t size = 0;
                        operation_ec =
                            ValidateOwnedLocation(request_context, keys[i], location_id, *location, size);
                        if (operation_ec != EC_OK) {
                            break;
                        }
                        found_any = true;
                        auto copy = std::make_shared<CacheLocation>(*location);
                        batch.push_back(SessionItem{0, {}, keys[i], location_id, copy, copy, size});
                        if (batch.size() == kMaintenanceDeleteBatchSize) {
                            flush();
                        }
                        if (operation_ec != EC_OK) {
                            break;
                        }
                    }
                }
            }
            if (operation_ec != EC_OK) {
                return operation_ec;
            }
            cursor = std::move(next_cursor);
        } while (cursor != SCAN_BASE_CURSOR);

        flush();
        if (operation_ec != EC_OK) {
            return operation_ec;
        }
        if (maintenance_cancelled_.load(std::memory_order_acquire)) {
            return EC_SERVICE_NOT_LEADER;
        }
        if (!found_any) {
            return EC_OK;
        }
        // Deleting during a cursor scan is safe for a single captured batch,
        // but some backends provide weak cursor guarantees under mutation.
        // Restart until a complete pass observes no remaining locations.
    }
}

ErrorCode KvMetaManager::DoRecover(std::function<bool()> should_abort) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    RequestContext request_context("kv_meta_recover");
    const auto [groups_ec, groups] = registry_manager_->ListInstanceGroup(&request_context);
    if (groups_ec != EC_OK) {
        return groups_ec;
    }
    ErrorCode overall = EC_OK;
    for (const auto &group : groups) {
        if (should_abort && should_abort()) {
            return EC_SERVICE_NOT_LEADER;
        }
        if (!group) {
            overall = FirstHardError(overall, EC_CORRUPTION);
            continue;
        }
        const auto [instances_ec, instances] = registry_manager_->ListInstanceInfo(&request_context, group->name());
        if (instances_ec != EC_OK) {
            overall = FirstHardError(overall, instances_ec);
            continue;
        }
        for (const auto &instance : instances) {
            if (!instance || !IsKvMetaInstance(*instance)) {
                continue;
            }
            if (should_abort && should_abort()) {
                return EC_SERVICE_NOT_LEADER;
            }
            auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                overall = FirstHardError(overall, EC_INSTANCE_NOT_EXIST);
                continue;
            }
            std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)>
                committed_usage_by_type{};
            ErrorCode instance_recovery_ec = EC_OK;
            bool removed_stale_in_pass = false;
            do {
                removed_stale_in_pass = false;
                committed_usage_by_type.fill(0);
                std::vector<SessionItem> stale_batch;
                stale_batch.reserve(kMaintenanceDeleteBatchSize);
                const auto flush_stale = [&]() {
                    if (stale_batch.empty()) {
                        return EC_OK;
                    }
                    const ErrorCode ec =
                        DeleteItems(&request_context, instance->instance_id(), stale_batch, false, true);
                    stale_batch.clear();
                    return ec;
                };

                std::string cursor = SCAN_BASE_CURSOR;
                do {
                    if (should_abort && should_abort()) {
                        return EC_SERVICE_NOT_LEADER;
                    }
                    std::string next_cursor;
                    KeyVector keys;
                    const ErrorCode scan_ec =
                        indexer->Scan(&request_context, cursor, kRecoveryScanBatchSize, next_cursor, keys);
                    if (scan_ec != EC_OK || next_cursor.empty()) {
                        instance_recovery_ec =
                            FirstHardError(instance_recovery_ec, scan_ec == EC_OK ? EC_CORRUPTION : scan_ec);
                        break;
                    }
                    if (!keys.empty()) {
                        CacheLocationMapVector locations;
                        const auto get_result = indexer->GetLocations(&request_context, keys, locations);
                        if (get_result.ec != EC_OK || locations.size() != keys.size() ||
                            get_result.error_codes.size() != keys.size()) {
                            instance_recovery_ec = FirstHardError(
                                instance_recovery_ec,
                                get_result.ec == EC_OK ? EC_MISMATCH : get_result.ec);
                            break;
                        }
                        for (std::size_t i = 0; i < keys.size() && instance_recovery_ec == EC_OK; ++i) {
                            if (get_result.error_codes[i] != EC_OK || locations[i].empty()) {
                                instance_recovery_ec = FirstHardError(
                                    instance_recovery_ec,
                                    get_result.error_codes[i] == EC_OK ? EC_CORRUPTION
                                                                       : get_result.error_codes[i]);
                                break;
                            }
                            for (const auto &[location_id, location] : locations[i]) {
                                if (!location) {
                                    instance_recovery_ec = FirstHardError(instance_recovery_ec, EC_CORRUPTION);
                                    break;
                                }
                                std::uint64_t size = 0;
                                const ErrorCode validate_ec = ValidateOwnedLocation(
                                    &request_context, keys[i], location_id, *location, size);
                                const DataStorageType base_type = ToBaseType(location->type());
                                const std::size_t type_index = ToIndex(base_type);
                                if (validate_ec != EC_OK || base_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN ||
                                    type_index >= committed_usage_by_type.size()) {
                                    instance_recovery_ec = FirstHardError(
                                        instance_recovery_ec, validate_ec == EC_OK ? EC_CORRUPTION : validate_ec);
                                    break;
                                }
                                if (location->create_time() < 0) {
                                    if (size > std::numeric_limits<std::uint64_t>::max() -
                                                   committed_usage_by_type[type_index]) {
                                        instance_recovery_ec =
                                            FirstHardError(instance_recovery_ec, EC_OUT_OF_LIMIT);
                                        break;
                                    }
                                    committed_usage_by_type[type_index] += size;
                                    continue;
                                }
                                removed_stale_in_pass = true;
                                auto copy = std::make_shared<CacheLocation>(*location);
                                stale_batch.push_back(
                                    SessionItem{0, {}, keys[i], location_id, copy, copy, size});
                                if (stale_batch.size() == kMaintenanceDeleteBatchSize) {
                                    if (should_abort && should_abort()) {
                                        return EC_SERVICE_NOT_LEADER;
                                    }
                                    instance_recovery_ec =
                                        FirstHardError(instance_recovery_ec, flush_stale());
                                    if (instance_recovery_ec != EC_OK) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (instance_recovery_ec != EC_OK) {
                        break;
                    }
                    cursor = std::move(next_cursor);
                } while (cursor != SCAN_BASE_CURSOR);
                if (instance_recovery_ec == EC_OK) {
                    if (should_abort && should_abort()) {
                        return EC_SERVICE_NOT_LEADER;
                    }
                    instance_recovery_ec = FirstHardError(instance_recovery_ec, flush_stale());
                }
                // If this pass removed anything, cursor semantics under
                // mutation may have skipped entries and the usage sum is no
                // longer authoritative. Repeat; the final no-delete pass is
                // stable and is the only one used below.
            } while (instance_recovery_ec == EC_OK && removed_stale_in_pass);
            if (should_abort && should_abort()) {
                return EC_SERVICE_NOT_LEADER;
            }
            if (instance_recovery_ec == EC_OK) {
                // Runtime mutations account exact bytes directly. A crash can
                // leave the periodically persisted usage snapshot behind the
                // durable locations, so rebuild it only after a complete,
                // error-free scan and stale-write cleanup. This is confined
                // to reserved KVMeta instances and runs after main recovery.
                for (std::size_t type_index = 1; type_index < committed_usage_by_type.size(); ++type_index) {
                    const auto type = static_cast<DataStorageType>(type_index);
                    if (ToBaseType(type) != type) {
                        continue;
                    }
                    indexer->SetStorageUsageByType(type, committed_usage_by_type[type_index]);
                }
                indexer->PersistMetaData();
            }
            overall = FirstHardError(overall, instance_recovery_ec);
        }
    }
    return overall;
}

} // namespace kv_cache_manager
