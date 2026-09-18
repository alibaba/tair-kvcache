#include "kv_cache_manager/manager/kv_meta_manager.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
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
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
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
constexpr std::int64_t kMicrosecondsPerSecond = 1'000'000;
constexpr std::int64_t kLeaseDeadlineTag = std::int64_t{1} << 62;
constexpr auto kRecoveryWaitPollInterval = std::chrono::milliseconds(100);

bool EncodeLeaseDeadline(std::int64_t now_us, std::int64_t timeout_seconds, std::int64_t &encoded_deadline) {
    encoded_deadline = 0;
    if (now_us <= 0 || timeout_seconds <= 0 || now_us >= kLeaseDeadlineTag ||
        timeout_seconds > (kLeaseDeadlineTag - 1 - now_us) / kMicrosecondsPerSecond) {
        return false;
    }
    encoded_deadline = kLeaseDeadlineTag + now_us + timeout_seconds * kMicrosecondsPerSecond;
    return true;
}

bool EncodeTaggedDeadlineUs(std::int64_t now_us, std::int64_t timeout_us, std::int64_t &encoded_deadline) {
    encoded_deadline = 0;
    if (now_us <= 0 || timeout_us < 0 || now_us >= kLeaseDeadlineTag || timeout_us > kLeaseDeadlineTag - 1 - now_us) {
        return false;
    }
    encoded_deadline = kLeaseDeadlineTag + now_us + timeout_us;
    return true;
}

bool DecodeLeaseDeadline(std::int64_t encoded_deadline, std::int64_t &deadline_us) {
    deadline_us = 0;
    if (encoded_deadline <= kLeaseDeadlineTag) {
        return false;
    }
    deadline_us = encoded_deadline - kLeaseDeadlineTag;
    return deadline_us > 0 && deadline_us < kLeaseDeadlineTag;
}

bool DecodeRecoveryLeaseDeadline(std::int64_t marker,
                                 std::int64_t max_write_timeout_seconds,
                                 std::int64_t &deadline_us) {
    if (DecodeLeaseDeadline(marker, deadline_us)) {
        return true;
    }
    // Rolling-upgrade compatibility: the first KVMeta implementation stored
    // the wall-clock allocation time directly as its positive active marker.
    // It did not persist the caller's timeout, so use the configured maximum
    // to avoid reclaiming an old leader's still-valid allocation early. The
    // recovery-wide steady-clock force deadline remains the hard upper bound
    // for corrupt or far-future values.
    deadline_us = 0;
    if (marker <= 0 || marker >= kLeaseDeadlineTag || max_write_timeout_seconds <= 0) {
        return false;
    }
    if (max_write_timeout_seconds > (kLeaseDeadlineTag - 1 - marker) / kMicrosecondsPerSecond) {
        deadline_us = kLeaseDeadlineTag - 1;
    } else {
        deadline_us = marker + max_write_timeout_seconds * kMicrosecondsPerSecond;
    }
    return true;
}

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

bool UriMatchesStorageBackend(const DataStorageUri &uri,
                              const std::string &storage_name,
                              DataStorageType storage_type) {
    if (!uri.Valid() || uri.GetHostName() != storage_name) {
        return false;
    }
    if (IsTairMempoolStorageType(storage_type)) {
        return uri.GetProtocol() == kTairMempoolUriScheme;
    }
    const DataStorageType uri_type = ToDataStorageType(uri.GetProtocol());
    return uri_type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN && ToBaseType(uri_type) == ToBaseType(storage_type);
}

bool HasSingletonAllocationShape(const DataStorageUri &uri, DataStorageType storage_type) {
    // NFS/HF3FS/Dummy backends can pack several logical blocks into one file
    // and encode the member offset as blkid. KVMeta deliberately calls Create
    // with one key at a time so accepting a non-zero blkid would reintroduce a
    // shared physical deletion boundary. Absence and the canonical value zero
    // are both produced by existing singleton implementations.
    switch (storage_type) {
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY:
        break;
    default:
        return true;
    }
    if (!uri.HasParam("blkid")) {
        return true;
    }
    const std::string block_id_text = uri.GetParam("blkid");
    std::uint64_t block_id = 0;
    const auto parsed = std::from_chars(block_id_text.data(), block_id_text.data() + block_id_text.size(), block_id);
    return !block_id_text.empty() && parsed.ec == std::errc{} &&
           parsed.ptr == block_id_text.data() + block_id_text.size() && block_id == 0;
}

bool HasMatchingStorageBackend(const CacheLocation &location,
                               const std::shared_ptr<DataStorageManager> &data_storage_manager) {
    if (!data_storage_manager || location.location_specs().size() != 1) {
        return false;
    }
    const DataStorageUri uri(location.location_specs().front().uri());
    const auto backend = data_storage_manager->GetDataStorageBackend(uri.GetHostName());
    return backend && backend->GetType() == location.type() &&
           UriMatchesStorageBackend(uri, uri.GetHostName(), location.type()) &&
           HasSingletonAllocationShape(uri, location.type());
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

bool IsRetiredObject(const CacheLocation &location) {
    return location.status() == CLS_DELETING && location.create_time() > kLeaseDeadlineTag;
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
    using Clock = std::chrono::steady_clock;

    enum class TakeResult {
        kOk,
        kNotFound,
        kInstanceMismatch,
        kSizeMismatch,
        kExpired
    };
    enum class PutResult {
        kOk,
        kDuplicate,
        kStopped,
        kFull,
        kExpired
    };

    struct Session {
        std::string internal_instance_id;
        std::size_t quota_shard = 0;
        Clock::time_point deadline;
        std::vector<KvMetaManager::SessionItem> items;
    };

    struct FinalizationState {
        std::mutex mutex;
        std::unordered_map<std::string, std::size_t> instances;
    };

    class FinalizationGuard {
    public:
        FinalizationGuard() = default;
        FinalizationGuard(std::shared_ptr<FinalizationState> state, std::string internal_instance_id)
            : state_(std::move(state)), internal_instance_id_(std::move(internal_instance_id)) {}
        ~FinalizationGuard() { Reset(); }

        FinalizationGuard(const FinalizationGuard &) = delete;
        FinalizationGuard &operator=(const FinalizationGuard &) = delete;
        FinalizationGuard(FinalizationGuard &&other) noexcept = default;
        FinalizationGuard &operator=(FinalizationGuard &&other) noexcept {
            if (this != &other) {
                Reset();
                state_ = std::move(other.state_);
                internal_instance_id_ = std::move(other.internal_instance_id_);
            }
            return *this;
        }

    private:
        void Reset() {
            // Keep an owning reference alive until after the mutex has been
            // unlocked. A guard is deliberately allowed to outlive the
            // session manager during shutdown; resetting the last shared_ptr
            // while lock_guard still referred to state_->mutex would destroy
            // a locked mutex and make the subsequent unlock undefined.
            auto state = std::move(state_);
            if (state) {
                std::lock_guard<std::mutex> lock(state->mutex);
                const auto it = state->instances.find(internal_instance_id_);
                if (it != state->instances.end()) {
                    if (it->second <= 1) {
                        state->instances.erase(it);
                    } else {
                        --it->second;
                    }
                }
            }
        }

        std::shared_ptr<FinalizationState> state_;
        std::string internal_instance_id_;
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
                  std::size_t quota_shard,
                  std::vector<KvMetaManager::SessionItem> items,
                  Clock::time_point deadline) {
        auto entry = std::make_shared<Entry>();
        entry->session_id = session_id;
        entry->deadline = deadline;
        entry->session.internal_instance_id = internal_instance_id;
        entry->session.quota_shard = quota_shard;
        entry->session.deadline = deadline;
        entry->session.items = std::move(items);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return PutResult::kStopped;
            }
            if (deadline <= Clock::now()) {
                return PutResult::kExpired;
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

    std::pair<TakeResult, FinalizationGuard> Take(const std::string &session_id,
                                                  const std::string &internal_instance_id,
                                                  std::optional<std::size_t> expected_size,
                                                  Session &out) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return {TakeResult::kNotFound, FinalizationGuard{}};
        }
        const auto &entry = it->second;
        if (entry->session.internal_instance_id != internal_instance_id) {
            return {TakeResult::kInstanceMismatch, FinalizationGuard{}};
        }
        if (expected_size && entry->session.items.size() != *expected_size) {
            return {TakeResult::kSizeMismatch, FinalizationGuard{}};
        }
        const bool expired = entry->deadline <= Clock::now();
        FinalizationGuard finalization = BeginFinalizationLocked(entry->session.internal_instance_id);
        deadlines_.erase(DeadlineKey{entry->deadline, entry->sequence});
        out = std::move(entry->session);
        sessions_.erase(it);
        condition_.notify_all();
        return {expired ? TakeResult::kExpired : TakeResult::kOk, std::move(finalization)};
    }

    bool HasSessionForInstance(const std::string &internal_instance_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool has_session = std::any_of(sessions_.begin(), sessions_.end(), [&](const auto &entry) {
            return entry.second && entry.second->session.internal_instance_id == internal_instance_id;
        });
        if (has_session) {
            return true;
        }
        std::lock_guard<std::mutex> finalization_lock(finalization_state_->mutex);
        return finalization_state_->instances.count(internal_instance_id) != 0;
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
            std::optional<FinalizationGuard> finalization;
            if (!deadlines_.empty() && deadlines_.begin()->first.first <= now) {
                auto entry = deadlines_.begin()->second;
                deadlines_.erase(deadlines_.begin());
                const auto session_it = sessions_.find(entry->session_id);
                if (session_it != sessions_.end() && session_it->second == entry) {
                    finalization.emplace(BeginFinalizationLocked(entry->session.internal_instance_id));
                    expired.emplace(std::move(entry->session));
                    sessions_.erase(session_it);
                }
            }
            if (expired) {
                lock.unlock();
                Expire(std::move(*expired));
                finalization.reset();
                lock.lock();
            }
        }
    }

    FinalizationGuard BeginFinalizationLocked(const std::string &internal_instance_id) {
        std::lock_guard<std::mutex> lock(finalization_state_->mutex);
        ++finalization_state_->instances[internal_instance_id];
        return FinalizationGuard(finalization_state_, internal_instance_id);
    }

    void Expire(Session session) {
        if (!owner_ || session.items.empty() || owner_->maintenance_cancelled_.load(std::memory_order_acquire)) {
            return;
        }
        if (session.quota_shard >= owner_->quota_admission_mutexes_.size()) {
            KVCM_LOG_ERROR("KVMeta write-session cleanup has an invalid quota shard");
            return;
        }
        // Serialize metadata removal and physical deletion with the next
        // StartWrite admission. Otherwise timeout cleanup can erase metadata,
        // expose the key as missing, and still be deleting the old allocation
        // while a new generation is admitted for the same group.
        std::unique_lock<std::mutex> quota_lock(owner_->quota_admission_mutexes_[session.quota_shard]);
        if (owner_->maintenance_cancelled_.load(std::memory_order_acquire)) {
            return;
        }
        RequestContext request_context("kv_meta_write_session_expired");
        const std::vector<bool> failed(session.items.size(), false);
        ErrorCode ec = EC_IO_ERROR;
        const char *failure_kind = "error_code";
        try {
            ec = owner_->FinishWriteInternal(&request_context, session.internal_instance_id, failed, session.items);
        } catch (const std::exception &) {
            failure_kind = "standard_exception";
        } catch (...) {
            failure_kind = "unknown_exception";
        }
        if (ec != EC_OK) {
            // Never replay an uncertain physical delete. Reusable-address
            // backends do not carry a generation token in the URI, so a retry
            // could delete an unrelated successor allocation. Metadata-first
            // cleanup makes the safe failure mode an orphan, not corruption.
            KVCM_LOG_WARN("KVMeta write-session cleanup did not complete; backend orphan cleanup may be required, "
                          "item_count[%zu], failure[%s], ec[%d]",
                          session.items.size(),
                          failure_kind,
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
    // A guard shares this small tracker instead of calling back through the
    // session manager, so its destructor remains safe across a concurrent
    // manager stop. The map contains only currently finalizing instances.
    std::shared_ptr<FinalizationState> finalization_state_ = std::make_shared<FinalizationState>();
    std::map<DeadlineKey, std::shared_ptr<Entry>> deadlines_;
    std::thread thread_;
};

// KVMeta deliberately does not enter CacheReclaimer's fixed-block state
// machine or its shared deletion executor. This worker reuses the same
// Instance Group watermark/LRU configuration, but samples and retires only
// reserved KVMeta instances. Keeping the worker here also lets it use the
// exact-value guards and group admission shards that protect KVMeta's stable
// location ids from ABA races.
class KvMetaReclaimer {
public:
    explicit KvMetaReclaimer(KvMetaManager *owner) : owner_(owner) { RegisterMetrics(); }
    ~KvMetaReclaimer() { StopAndJoin(); }

    KvMetaReclaimer(const KvMetaReclaimer &) = delete;
    KvMetaReclaimer &operator=(const KvMetaReclaimer &) = delete;

    bool Start() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (thread_.joinable()) {
                return !stopping_;
            }
            stopping_ = false;
            wake_requested_ = true;
        }
        try {
            thread_ = std::thread([this]() { Loop(); });
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            KVCM_LOG_ERROR("failed to start KVMeta reclaimer: %s", e.what());
            return false;
        }
        condition_.notify_all();
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
        std::thread worker;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            wake_requested_ = false;
            if (thread_.joinable()) {
                worker = std::move(thread_);
            }
        }
        condition_.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        pending_instances_.clear();
        pending_locations_.clear();
        pending_credits_.clear();
        pending_batches_.clear();
        sampling_rotation_by_group_.clear();
        pending_object_count_ = 0;
        pending_bytes_ = 0;
        UpdatePendingMetricsLocked();
    }

    void Wake() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            wake_requested_ = true;
        }
        condition_.notify_all();
    }

    bool HasPendingForInstance(const std::string &internal_instance_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_instances_.find(internal_instance_id);
        return it != pending_instances_.end() && it->second != 0;
    }

    bool HasPendingLocation(const std::string &internal_instance_id, const std::string &location_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_locations_.find({internal_instance_id, location_id});
        return it != pending_locations_.end() && it->second != 0;
    }

    bool IsAdmissionBlocked(const std::string &instance_group) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_credits_.find(instance_group);
        return it != pending_credits_.end() && it->second.blocked_batch_count != 0;
    }

private:
    void FailClosedMaintenance() noexcept {
        if (!owner_) {
            return;
        }
        owner_->maintenance_cancelled_.store(true, std::memory_order_release);
        if (owner_->write_session_manager_) {
            // Match demotion semantics: reject publication of any concurrently
            // allocating session, while sessions already published may still
            // Finish and release their own allocation safely.
            owner_->write_session_manager_->RequestStop();
        }
    }

    void RegisterMetrics() noexcept {
        try {
            const auto registry =
                owner_ && owner_->cache_manager_ ? owner_->cache_manager_->metrics_registry() : nullptr;
            if (!registry) {
                return;
            }
            round_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.round_count");
            retired_object_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.retired_object_count");
            reclaimed_object_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.reclaimed_object_count");
            reclaimed_bytes_metrics_ = registry->GetCounter("kv_meta_reclaimer.reclaimed_bytes");
            retry_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.retry_count");
            error_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.error_count");
            pending_limit_reject_count_metrics_ = registry->GetCounter("kv_meta_reclaimer.pending_limit_reject_count");
            physical_delete_attempted_object_count_metrics_ =
                registry->GetCounter("kv_meta_reclaimer.physical_delete_attempted_object_count");
            physical_delete_uncertain_object_count_metrics_ =
                registry->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_object_count");
            physical_delete_uncertain_bytes_metrics_ =
                registry->GetCounter("kv_meta_reclaimer.physical_delete_uncertain_bytes");
            pending_object_count_metrics_ = registry->GetGauge("kv_meta_reclaimer.pending_object_count");
            pending_bytes_metrics_ = registry->GetGauge("kv_meta_reclaimer.pending_bytes");
            blocked_group_count_metrics_ = registry->GetGauge("kv_meta_reclaimer.blocked_group_count");
            UpdatePendingMetricsLocked();
        } catch (const std::exception &e) {
            KVCM_LOG_WARN("failed to register KVMeta reclaimer metrics: %s", e.what());
        } catch (...) {
            KVCM_LOG_WARN("failed to register KVMeta reclaimer metrics with unknown exception");
        }
    }

    void UpdatePendingMetricsLocked() noexcept {
        pending_object_count_metrics_ = static_cast<double>(pending_object_count_);
        pending_bytes_metrics_ = static_cast<double>(pending_bytes_);
        const auto blocked_groups =
            std::count_if(pending_credits_.begin(), pending_credits_.end(), [](const auto &entry) {
                return entry.second.blocked_batch_count != 0;
            });
        blocked_group_count_metrics_ = static_cast<double>(blocked_groups);
    }

    struct Pressure {
        std::uint64_t group_bytes = 0;
        std::uint64_t keys = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> bytes_by_type{};

        bool Any() const noexcept {
            if (group_bytes != 0 || keys != 0) {
                return true;
            }
            return std::any_of(
                bytes_by_type.begin(), bytes_by_type.end(), [](std::uint64_t value) { return value != 0; });
        }

        bool Relevant(DataStorageType type, bool removes_key) const noexcept {
            const std::size_t type_index = ToIndex(ToBaseType(type));
            return group_bytes != 0 || (keys != 0 && removes_key) ||
                   (type_index < bytes_by_type.size() && bytes_by_type[type_index] != 0);
        }

        void Consume(DataStorageType type, std::uint64_t bytes) noexcept {
            group_bytes = bytes >= group_bytes ? 0 : group_bytes - bytes;
            const std::size_t type_index = ToIndex(ToBaseType(type));
            if (type_index < bytes_by_type.size()) {
                bytes_by_type[type_index] = bytes >= bytes_by_type[type_index] ? 0 : bytes_by_type[type_index] - bytes;
            }
        }
    };

    struct Candidate {
        std::string internal_instance_id;
        std::int64_t internal_key = 0;
        std::string location_id;
        CacheLocationConstPtr location;
        std::uint64_t value_size = 0;
        bool removes_metadata_key = false;
    };

    struct CandidateKey {
        std::string internal_instance_id;
        std::int64_t internal_key = 0;
        std::int64_t last_access_time_us = 0;
        bool all_locations_committed = false;
        std::vector<Candidate> objects;
    };

    struct RetiredItem {
        std::string internal_instance_id;
        KvMetaManager::SessionItem item;
        bool removes_metadata_key = false;
        bool metadata_durable = false;
    };

    struct PendingCredit {
        std::uint64_t bytes = 0;
        std::uint64_t keys = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> bytes_by_type{};
        // The entry is created when the batch is enqueued, before a metadata
        // cleanup can fail. Finalization can therefore close admission without
        // allocating memory on the error path.
        std::size_t blocked_batch_count = 0;
    };

    struct PendingBatch {
        std::string instance_group;
        std::size_t quota_shard = 0;
        std::chrono::steady_clock::time_point deadline;
        std::uint64_t sequence = 0;
        std::vector<RetiredItem> items;
        std::vector<std::string> instances;
        std::vector<std::pair<std::string, std::string>> locations;
        bool admission_blocked = false;
        std::uint32_t retry_count = 0;
    };

    using PendingDeadline = std::pair<std::chrono::steady_clock::time_point, std::uint64_t>;

    static constexpr std::uint64_t kPendingBatchLimit = 1024;
    static constexpr std::uint64_t kPendingObjectLimit = 20'000;
    static constexpr std::uint64_t kPendingBytesLimit = 4ULL * 1024 * 1024 * 1024 * 1024;
    static constexpr long double kWatermarkEpsilon = 1e-9L;

    static std::uint64_t SaturatingAdd(std::uint64_t lhs, std::uint64_t rhs) noexcept {
        return rhs > std::numeric_limits<std::uint64_t>::max() - lhs ? std::numeric_limits<std::uint64_t>::max()
                                                                     : lhs + rhs;
    }

    static std::uint64_t SaturatingSub(std::uint64_t lhs, std::uint64_t rhs) noexcept {
        return rhs >= lhs ? 0 : lhs - rhs;
    }

    static std::uint64_t BytesToFree(std::int64_t capacity, double threshold, std::uint64_t used) noexcept {
        if (used == 0) {
            return 0;
        }
        if (capacity <= 0) {
            return used;
        }
        // Match CacheReclaimer's ratio + 1e-9 > threshold trigger. A
        // successful round must settle at or below threshold - epsilon instead
        // of continuously firing at an equal (or epsilon-close) boundary.
        const long double adjusted_threshold =
            std::max<long double>(0.0L, static_cast<long double>(threshold) - kWatermarkEpsilon);
        const long double raw_allowed = static_cast<long double>(capacity) * adjusted_threshold;
        const std::uint64_t allowed =
            raw_allowed <= 0
                ? 0
                : static_cast<std::uint64_t>(std::min<long double>(
                      std::floor(raw_allowed), static_cast<long double>(std::numeric_limits<std::uint64_t>::max())));
        return used > allowed ? used - allowed : 0;
    }

    bool ShouldStop() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopping_ || !owner_ || owner_->maintenance_cancelled_.load(std::memory_order_acquire);
    }

    std::uint32_t IdleIntervalMs() const noexcept {
        try {
            if (!owner_ || !owner_->cache_manager_ || !owner_->cache_manager_->cache_reclaimer()) {
                return 100;
            }
            RequestContext request_context("kv_meta_reclaimer_config");
            return std::max<std::uint32_t>(
                1, owner_->cache_manager_->cache_reclaimer()->GetSleepIntervalMs(&request_context));
        } catch (...) {
            return 100;
        }
    }

    std::pair<std::size_t, std::size_t> SamplingAndBatchSize() const noexcept {
        try {
            if (!owner_ || !owner_->cache_manager_ || !owner_->cache_manager_->cache_reclaimer()) {
                return {100, 100};
            }
            RequestContext request_context("kv_meta_reclaimer_config");
            return {owner_->cache_manager_->cache_reclaimer()->GetSamplingSize(&request_context),
                    owner_->cache_manager_->cache_reclaimer()->GetBatchingSize(&request_context)};
        } catch (...) {
            return {100, 100};
        }
    }

    PendingCredit GetPendingCredit(const std::string &instance_group) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_credits_.find(instance_group);
        return it == pending_credits_.end() ? PendingCredit{} : it->second;
    }

    bool HasPendingCapacity(const std::vector<Candidate> &candidates) const {
        std::uint64_t candidate_bytes = 0;
        for (const auto &candidate : candidates) {
            candidate_bytes = SaturatingAdd(candidate_bytes, candidate.value_size);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_batches_.size() < kPendingBatchLimit && candidates.size() <= kPendingObjectLimit &&
               pending_object_count_ <= kPendingObjectLimit - candidates.size() &&
               candidate_bytes <= kPendingBytesLimit && pending_bytes_ <= kPendingBytesLimit - candidate_bytes;
    }

    bool ReadPressure(RequestContext *request_context,
                      const InstanceGroup &group,
                      const std::vector<InstanceInfoConstPtr> &instances,
                      double threshold,
                      Pressure &out) const {
        out = {};
        std::uint64_t group_usage = 0;
        std::uint64_t key_count = 0;
        std::uint64_t max_key_count = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> usage_by_type{};
        for (const auto &instance : instances) {
            if (!instance || !IsKvMetaInstance(*instance)) {
                AddError(request_context, "KVMeta reclaimer requires a dedicated generic-object instance group");
                return false;
            }
            const auto indexer =
                owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                AddError(request_context, "KVMeta reclaimer could not read an instance indexer");
                return false;
            }
            group_usage = SaturatingAdd(group_usage, indexer->GetStorageUsage());
            key_count = SaturatingAdd(key_count, static_cast<std::uint64_t>(indexer->GetKeyCount()));
            max_key_count = SaturatingAdd(max_key_count, static_cast<std::uint64_t>(indexer->GetMaxKeyCount()));
            for (std::size_t i = 1; i < usage_by_type.size(); ++i) {
                const auto type = static_cast<DataStorageType>(i);
                if (ToBaseType(type) != type) {
                    continue;
                }
                usage_by_type[i] = SaturatingAdd(usage_by_type[i], indexer->GetStorageUsageByType(type));
            }
        }

        const PendingCredit credit = GetPendingCredit(group.name());
        group_usage = SaturatingSub(group_usage, credit.bytes);
        key_count = SaturatingSub(key_count, credit.keys);
        for (std::size_t i = 0; i < usage_by_type.size(); ++i) {
            usage_by_type[i] = SaturatingSub(usage_by_type[i], credit.bytes_by_type[i]);
        }

        out.group_bytes = BytesToFree(group.quota().capacity(), threshold, group_usage);
        out.keys = BytesToFree(max_key_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                                   ? std::numeric_limits<std::int64_t>::max()
                                   : static_cast<std::int64_t>(max_key_count),
                               threshold,
                               key_count);
        for (const auto &quota : group.quota().quota_config()) {
            const auto base_type = ToBaseType(quota.storage_spec());
            const std::size_t type_index = ToIndex(base_type);
            if (base_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN || type_index >= usage_by_type.size()) {
                continue;
            }
            out.bytes_by_type[type_index] = std::max(
                out.bytes_by_type[type_index], BytesToFree(quota.capacity(), threshold, usage_by_type[type_index]));
        }
        return true;
    }

    bool CollectCandidates(RequestContext *request_context,
                           const std::string &instance_group,
                           const std::vector<InstanceInfoConstPtr> &instances,
                           std::size_t sampling_size,
                           std::vector<CandidateKey> &out) {
        out.clear();
        std::vector<std::pair<InstanceInfoConstPtr, std::size_t>> eligible;
        std::uint64_t total_key_count = 0;
        for (const auto &instance : instances) {
            if (!instance) {
                return false;
            }
            const auto indexer =
                owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                return false;
            }
            const std::size_t count = indexer->GetKeyCount();
            if (count != 0) {
                eligible.emplace_back(instance, count);
            }
        }
        if (eligible.empty() || sampling_size == 0) {
            return true;
        }

        std::sort(eligible.begin(), eligible.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.first->instance_id() < rhs.first->instance_id();
        });
        const std::size_t instance_budget = std::min(sampling_size, eligible.size());
        const std::size_t rotation = sampling_rotation_by_group_[instance_group] % eligible.size();
        std::vector<std::pair<InstanceInfoConstPtr, std::size_t>> selected_instances;
        selected_instances.reserve(instance_budget);
        for (std::size_t offset = 0; offset < instance_budget; ++offset) {
            selected_instances.push_back(eligible[(rotation + offset) % eligible.size()]);
            total_key_count = SaturatingAdd(total_key_count, selected_instances.back().second);
        }
        sampling_rotation_by_group_[instance_group] = (rotation + instance_budget) % eligible.size();

        // Give each selected instance one slot, then distribute the remaining
        // strict per-round budget by key count. Rotation prevents groups with
        // more instances than sampling slots from starving their tail.
        const std::size_t total_budget = sampling_size;
        std::size_t remaining_budget = total_budget;
        std::uint64_t remaining_weight = total_key_count;
        std::uint64_t observed_location_count = 0;
        std::set<std::tuple<std::string, std::int64_t>> sampled_keys;
        for (std::size_t instance_index = 0; instance_index < selected_instances.size(); ++instance_index) {
            const auto &[instance, key_count_for_instance] = selected_instances[instance_index];
            const std::size_t instances_left = selected_instances.size() - instance_index;
            std::size_t key_budget = 1;
            if (remaining_budget > instances_left && remaining_weight != 0) {
                const long double weighted = static_cast<long double>(remaining_budget) *
                                             static_cast<long double>(key_count_for_instance) /
                                             static_cast<long double>(remaining_weight);
                key_budget = std::max<std::size_t>(1, static_cast<std::size_t>(weighted));
            }
            key_budget = std::min({key_budget, key_count_for_instance, remaining_budget - instances_left + 1});
            remaining_budget -= key_budget;
            remaining_weight =
                key_count_for_instance >= remaining_weight ? 0 : remaining_weight - key_count_for_instance;

            const auto indexer =
                owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(instance->instance_id());
            ReclaimCandidateVector sampled;
            if (indexer->SampleReclaimCandidates(
                    request_context, static_cast<std::int64_t>(key_budget), sampled, true) != EC_OK) {
                return false;
            }
            if (sampled.size() > key_budget) {
                AddError(request_context, "KVMeta reclaim sampling exceeded its requested bound");
                ++error_count_metrics_;
                return false;
            }
            KeyVector keys;
            std::vector<std::int64_t> access_times;
            keys.reserve(sampled.size());
            access_times.reserve(sampled.size());
            for (const auto &candidate : sampled) {
                if (sampled_keys.emplace(instance->instance_id(), candidate.key).second) {
                    keys.push_back(candidate.key);
                    access_times.push_back(candidate.last_access_time_us);
                }
            }
            if (keys.empty()) {
                continue;
            }
            CacheLocationMapVector location_maps;
            const auto get_result = indexer->GetLocationMapsForMaintenance(request_context, keys, location_maps);
            if ((get_result.ec != EC_OK && get_result.ec != EC_PARTIAL_OK) || location_maps.size() != keys.size() ||
                get_result.error_codes.size() != keys.size()) {
                return false;
            }
            for (std::size_t key_index = 0; key_index < keys.size(); ++key_index) {
                if (get_result.error_codes[key_index] == EC_NOENT) {
                    continue;
                }
                if (get_result.error_codes[key_index] != EC_OK || location_maps[key_index].empty()) {
                    return false;
                }
                CandidateKey key_candidate;
                key_candidate.internal_instance_id = instance->instance_id();
                key_candidate.internal_key = keys[key_index];
                key_candidate.last_access_time_us = access_times[key_index];
                key_candidate.all_locations_committed = true;
                if (location_maps[key_index].size() > kPendingObjectLimit - observed_location_count) {
                    AddError(request_context, "KVMeta reclaim candidate locations exceeded the bounded scan limit");
                    ++error_count_metrics_;
                    return false;
                }
                observed_location_count += location_maps[key_index].size();
                for (const auto &[location_id, location] : location_maps[key_index]) {
                    if (!location) {
                        return false;
                    }
                    std::uint64_t value_size = 0;
                    if (owner_->ValidateOwnedLocation(
                            request_context, keys[key_index], location_id, *location, value_size) != EC_OK) {
                        return false;
                    }
                    if (!IsCommittedObject(*location)) {
                        key_candidate.all_locations_committed = false;
                        continue;
                    }
                    key_candidate.objects.push_back(
                        Candidate{instance->instance_id(), keys[key_index], location_id, location, value_size});
                }
                if (!key_candidate.objects.empty()) {
                    out.push_back(std::move(key_candidate));
                }
            }
        }
        return true;
    }

    static std::vector<Candidate>
    SelectCandidates(std::vector<CandidateKey> candidates, Pressure pressure, std::size_t batch_size) {
        std::sort(candidates.begin(), candidates.end(), [](const CandidateKey &lhs, const CandidateKey &rhs) {
            return std::tie(lhs.last_access_time_us, lhs.internal_instance_id, lhs.internal_key) <
                   std::tie(rhs.last_access_time_us, rhs.internal_instance_id, rhs.internal_key);
        });
        std::vector<Candidate> selected;
        selected.reserve(std::min(batch_size, candidates.size()));
        for (const auto &key_candidate : candidates) {
            if (!pressure.Any() || selected.size() >= batch_size) {
                break;
            }
            const bool reclaim_whole_key = pressure.keys != 0 && key_candidate.all_locations_committed &&
                                           key_candidate.objects.size() <= batch_size - selected.size();
            std::size_t selected_for_key = 0;
            for (const auto &candidate : key_candidate.objects) {
                if (selected.size() >= batch_size) {
                    break;
                }
                if (!reclaim_whole_key && !pressure.Relevant(candidate.location->type(), false)) {
                    continue;
                }
                selected.push_back(candidate);
                ++selected_for_key;
                pressure.Consume(candidate.location->type(), candidate.value_size);
            }
            if (reclaim_whole_key && selected_for_key == key_candidate.objects.size() && pressure.keys != 0) {
                selected.back().removes_metadata_key = true;
                --pressure.keys;
            } else if (key_candidate.all_locations_committed && selected_for_key == key_candidate.objects.size() &&
                       selected_for_key != 0) {
                // Byte pressure happened to select the complete metadata key;
                // credit that key so a concurrent key-count watermark does
                // not retire an unnecessary additional object.
                selected.back().removes_metadata_key = true;
            }
        }
        return selected;
    }

    std::vector<RetiredItem> RetireCandidates(RequestContext *request_context,
                                              const std::vector<Candidate> &candidates,
                                              std::int64_t retire_deadline) {
        std::vector<RetiredItem> retired_items;
        std::map<std::string, std::vector<std::size_t>> by_instance;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            by_instance[candidates[i].internal_instance_id].push_back(i);
        }
        for (const auto &[internal_instance_id, indices] : by_instance) {
            const auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
            if (!indexer) {
                continue;
            }
            std::vector<bool> retired(indices.size(), false);
            std::vector<std::int64_t> keys;
            keys.reserve(indices.size());
            for (const std::size_t index : indices) {
                keys.push_back(candidates[index].internal_key);
            }
            for (const auto &layer : MakeUniqueKeyLayers(keys)) {
                KeyVector layer_keys;
                LocationIdsPerKey layer_ids;
                std::vector<CacheLocationConstPtr> expected;
                std::vector<CacheLocationConstPtr> replacements;
                layer_keys.reserve(layer.size());
                layer_ids.reserve(layer.size());
                expected.reserve(layer.size());
                replacements.reserve(layer.size());
                for (const std::size_t relative_index : layer) {
                    const Candidate &candidate = candidates[indices[relative_index]];
                    layer_keys.push_back(candidate.internal_key);
                    layer_ids.push_back({candidate.location_id});
                    expected.push_back(candidate.location);
                    auto replacement = std::make_shared<CacheLocation>(*candidate.location);
                    replacement->set_status(CLS_DELETING);
                    replacement->set_create_time(retire_deadline);
                    replacements.push_back(std::move(replacement));
                }
                std::vector<bool> layer_retired(layer.size(), false);
                auto modifier = [&expected, &replacements, &layer_retired](const std::vector<ErrorCode> &get_ecs,
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
                    if (!locations[0] || locations[0]->ToJsonString() != expected[key_index]->ToJsonString() ||
                        !IsCommittedObject(*locations[0])) {
                        return {MA_SKIP, {EC_MISMATCH}};
                    }
                    locations[0] = replacements[key_index];
                    layer_retired[key_index] = true;
                    return {MA_OK, {EC_OK}};
                };
                const auto result = indexer->ReadModifyWriteLocationsForMaintenance(
                    request_context, layer_keys, layer_ids, modifier, false);
                if (result.per_location_error_codes.size() != layer.size()) {
                    continue;
                }
                for (std::size_t i = 0; i < layer.size(); ++i) {
                    if (result.per_location_error_codes[i].size() == 1 &&
                        result.per_location_error_codes[i][0] == EC_OK && layer_retired[i]) {
                        retired[layer[i]] = true;
                    }
                }
            }

            KeyVector sync_keys;
            for (std::size_t i = 0; i < indices.size(); ++i) {
                if (retired[i]) {
                    sync_keys.push_back(candidates[indices[i]].internal_key);
                }
            }
            if (sync_keys.empty()) {
                continue;
            }
            std::sort(sync_keys.begin(), sync_keys.end());
            sync_keys.erase(std::unique(sync_keys.begin(), sync_keys.end()), sync_keys.end());
            const bool metadata_durable = indexer->Sync(sync_keys);
            if (!metadata_durable) {
                // The in-memory view may already be retired, but without a
                // persistence barrier it is not safe to release the physical
                // allocation. Keep it in the pending queue and retry only the
                // persistence barrier; no physical Delete is issued first.
                KVCM_LOG_WARN("KVMeta reclaimer could not persist retired metadata for instance [%s]",
                              internal_instance_id.c_str());
            }
            std::vector<bool> removes_metadata_key(indices.size(), false);
            std::map<std::int64_t, std::vector<std::size_t>> retired_by_key;
            for (std::size_t i = 0; i < indices.size(); ++i) {
                retired_by_key[candidates[indices[i]].internal_key].push_back(i);
            }
            for (const auto &[_, relative_indices] : retired_by_key) {
                const bool selected_complete_key = std::any_of(
                    relative_indices.begin(), relative_indices.end(), [&](const std::size_t relative_index) {
                        return candidates[indices[relative_index]].removes_metadata_key;
                    });
                const bool all_retired =
                    std::all_of(relative_indices.begin(),
                                relative_indices.end(),
                                [&](const std::size_t relative_index) { return retired[relative_index]; });
                if (selected_complete_key && all_retired) {
                    removes_metadata_key[relative_indices.back()] = true;
                }
            }
            for (std::size_t i = 0; i < indices.size(); ++i) {
                if (!retired[i]) {
                    continue;
                }
                const Candidate &candidate = candidates[indices[i]];
                auto retired_location = std::make_shared<CacheLocation>(*candidate.location);
                retired_location->set_status(CLS_DELETING);
                retired_location->set_create_time(retire_deadline);
                retired_items.push_back(RetiredItem{internal_instance_id,
                                                    KvMetaManager::SessionItem{0,
                                                                               {},
                                                                               candidate.internal_key,
                                                                               candidate.location_id,
                                                                               retired_location,
                                                                               retired_location,
                                                                               candidate.value_size},
                                                    removes_metadata_key[i],
                                                    metadata_durable});
            }
        }
        return retired_items;
    }

    void AddPendingBatch(const std::string &instance_group,
                         std::size_t quota_shard,
                         std::chrono::steady_clock::time_point deadline,
                         std::vector<RetiredItem> items) {
        auto batch = std::make_shared<PendingBatch>();
        batch->instance_group = instance_group;
        batch->quota_shard = quota_shard;
        batch->deadline = deadline;
        batch->items = std::move(items);
        std::set<std::string> instances;
        for (const auto &item : batch->items) {
            instances.insert(item.internal_instance_id);
            batch->locations.emplace_back(item.internal_instance_id, item.item.location_id);
        }
        batch->instances.assign(instances.begin(), instances.end());
        std::uint64_t batch_bytes = 0;
        for (const auto &item : batch->items) {
            batch_bytes = SaturatingAdd(batch_bytes, item.item.value_size);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batch->sequence = next_pending_sequence_;
            // Allocate every map node before publishing counters. This gives
            // the in-memory indexes a strong exception guarantee: either the
            // complete pending batch is visible, or no zero/partial marker is
            // left behind.
            try {
                for (const auto &instance : batch->instances) {
                    pending_instances_.try_emplace(instance, 0);
                }
                for (const auto &location : batch->locations) {
                    pending_locations_.try_emplace(location, 0);
                }
                pending_credits_.try_emplace(instance_group, PendingCredit{});
                const auto [_, inserted] =
                    pending_batches_.emplace(PendingDeadline{batch->deadline, batch->sequence}, batch);
                if (!inserted) {
                    throw std::logic_error("duplicate KVMeta pending sequence");
                }
            } catch (...) {
                for (const auto &instance : batch->instances) {
                    const auto it = pending_instances_.find(instance);
                    if (it != pending_instances_.end() && it->second == 0) {
                        pending_instances_.erase(it);
                    }
                }
                for (const auto &location : batch->locations) {
                    const auto it = pending_locations_.find(location);
                    if (it != pending_locations_.end() && it->second == 0) {
                        pending_locations_.erase(it);
                    }
                }
                const auto credit_it = pending_credits_.find(instance_group);
                if (credit_it != pending_credits_.end() && credit_it->second.bytes == 0 &&
                    credit_it->second.keys == 0 && credit_it->second.blocked_batch_count == 0 &&
                    std::none_of(credit_it->second.bytes_by_type.begin(),
                                 credit_it->second.bytes_by_type.end(),
                                 [](std::uint64_t value) { return value != 0; })) {
                    pending_credits_.erase(credit_it);
                }
                FailClosedMaintenance();
                throw;
            }
            ++next_pending_sequence_;
            for (const auto &instance : batch->instances) {
                ++pending_instances_.find(instance)->second;
            }
            for (const auto &location : batch->locations) {
                ++pending_locations_.find(location)->second;
            }
            auto &credit = pending_credits_.find(instance_group)->second;
            for (const auto &item : batch->items) {
                credit.bytes = SaturatingAdd(credit.bytes, item.item.value_size);
                if (item.removes_metadata_key) {
                    credit.keys = SaturatingAdd(credit.keys, 1);
                }
                if (item.item.data_location) {
                    const std::size_t type_index = ToIndex(ToBaseType(item.item.data_location->type()));
                    if (type_index < credit.bytes_by_type.size()) {
                        credit.bytes_by_type[type_index] =
                            SaturatingAdd(credit.bytes_by_type[type_index], item.item.value_size);
                    }
                }
            }
            pending_object_count_ = SaturatingAdd(pending_object_count_, batch->items.size());
            pending_bytes_ = SaturatingAdd(pending_bytes_, batch_bytes);
            retired_object_count_metrics_ += batch->items.size();
            UpdatePendingMetricsLocked();
            wake_requested_ = true;
        }
        condition_.notify_all();
    }

    void CompletePending(const std::shared_ptr<PendingBatch> &batch) noexcept {
        if (!batch) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &instance : batch->instances) {
            const auto it = pending_instances_.find(instance);
            if (it == pending_instances_.end() || it->second <= 1) {
                pending_instances_.erase(instance);
            } else {
                --it->second;
            }
        }
        for (const auto &location : batch->locations) {
            const auto it = pending_locations_.find(location);
            if (it == pending_locations_.end() || it->second <= 1) {
                pending_locations_.erase(location);
            } else {
                --it->second;
            }
        }
        if (batch->admission_blocked) {
            const auto it = pending_credits_.find(batch->instance_group);
            if (it != pending_credits_.end() && it->second.blocked_batch_count != 0) {
                --it->second.blocked_batch_count;
            }
            batch->admission_blocked = false;
        }
        const auto credit_it = pending_credits_.find(batch->instance_group);
        if (credit_it != pending_credits_.end()) {
            auto &credit = credit_it->second;
            for (const auto &item : batch->items) {
                credit.bytes = SaturatingSub(credit.bytes, item.item.value_size);
                if (item.removes_metadata_key) {
                    credit.keys = SaturatingSub(credit.keys, 1);
                }
                if (item.item.data_location) {
                    const std::size_t type_index = ToIndex(ToBaseType(item.item.data_location->type()));
                    if (type_index < credit.bytes_by_type.size()) {
                        credit.bytes_by_type[type_index] =
                            SaturatingSub(credit.bytes_by_type[type_index], item.item.value_size);
                    }
                }
            }
            const bool has_type_credit = std::any_of(credit.bytes_by_type.begin(),
                                                     credit.bytes_by_type.end(),
                                                     [](std::uint64_t value) { return value != 0; });
            if (credit.bytes == 0 && credit.keys == 0 && !has_type_credit && credit.blocked_batch_count == 0) {
                pending_credits_.erase(credit_it);
            }
        }
        pending_object_count_ = SaturatingSub(pending_object_count_, batch->items.size());
        for (const auto &item : batch->items) {
            pending_bytes_ = SaturatingSub(pending_bytes_, item.item.value_size);
        }
        UpdatePendingMetricsLocked();
    }

    void BlockAdmission(const std::shared_ptr<PendingBatch> &batch) noexcept {
        if (!batch) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!batch->admission_blocked) {
            const auto credit_it = pending_credits_.find(batch->instance_group);
            if (credit_it == pending_credits_.end()) {
                // Losing this fence after an ambiguous metadata persistence
                // outcome could admit a successor allocation and create an ABA
                // delete race. Fail closed for KVMeta only; ordinary KV-cache
                // traffic does not consult maintenance_cancelled_.
                KVCM_LOG_ERROR("KVMeta reclaimer lost pending credit while blocking admission");
                ++error_count_metrics_;
                FailClosedMaintenance();
                return;
            }
            batch->admission_blocked = true;
            if (credit_it->second.blocked_batch_count != std::numeric_limits<std::size_t>::max()) {
                ++credit_it->second.blocked_batch_count;
            }
            UpdatePendingMetricsLocked();
        }
    }

    std::vector<std::shared_ptr<PendingBatch>> TakeDueBatches() {
        std::vector<std::shared_ptr<PendingBatch>> due;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        std::size_t due_count = 0;
        for (auto it = pending_batches_.begin(); it != pending_batches_.end() && it->first.first <= now; ++it) {
            ++due_count;
        }
        // Reserve before mutating the queue. If allocation fails, every batch
        // remains indexed and the outer loop can retry without losing its
        // pending markers or quota credit.
        due.reserve(due_count);
        while (!pending_batches_.empty() && pending_batches_.begin()->first.first <= now) {
            due.push_back(std::move(pending_batches_.begin()->second));
            pending_batches_.erase(pending_batches_.begin());
        }
        return due;
    }

    void ReschedulePending(const std::shared_ptr<PendingBatch> &batch) {
        if (!batch) {
            return;
        }
        constexpr std::uint64_t kMinimumRetryDelayMs = 100;
        constexpr std::uint64_t kMaximumRetryDelayMs = 30'000;
        const std::uint64_t base_delay_ms = std::max<std::uint64_t>(kMinimumRetryDelayMs, IdleIntervalMs());
        const std::uint32_t shift = std::min<std::uint32_t>(batch->retry_count, 8);
        const std::uint64_t retry_delay_ms = std::min<std::uint64_t>(kMaximumRetryDelayMs, base_delay_ms << shift);
        const auto retry_delay = std::chrono::milliseconds(retry_delay_ms);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            batch->deadline = std::chrono::steady_clock::now() + retry_delay;
            batch->sequence = next_pending_sequence_++;
            if (batch->retry_count != std::numeric_limits<std::uint32_t>::max()) {
                ++batch->retry_count;
            }
            try {
                const auto [_, inserted] =
                    pending_batches_.emplace(PendingDeadline{batch->deadline, batch->sequence}, batch);
                if (!inserted) {
                    throw std::logic_error("duplicate KVMeta retry sequence");
                }
            } catch (...) {
                // The durable retired marker must never be left without an
                // in-memory finalizer while this leader continues admitting
                // writes. Recovery on the next leader owns it after this
                // KVMeta-only fail-closed transition.
                FailClosedMaintenance();
                throw;
            }
            wake_requested_ = true;
        }
        condition_.notify_all();
    }

    bool EnsureRetiredMetadataDurable(const std::shared_ptr<PendingBatch> &batch) {
        std::map<std::string, KeyVector> keys_by_instance;
        for (const auto &retired : batch->items) {
            if (!retired.metadata_durable) {
                keys_by_instance[retired.internal_instance_id].push_back(retired.item.internal_key);
            }
        }
        for (auto &[internal_instance_id, keys] : keys_by_instance) {
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            const auto indexer = owner_->cache_manager_->meta_indexer_manager()->GetMetaIndexer(internal_instance_id);
            if (!indexer || !indexer->Sync(keys)) {
                return false;
            }
            for (auto &retired : batch->items) {
                if (retired.internal_instance_id == internal_instance_id) {
                    retired.metadata_durable = true;
                }
            }
        }
        return true;
    }

    void FinalizePending(const std::shared_ptr<PendingBatch> &batch) {
        if (!batch) {
            return;
        }
        if (ShouldStop()) {
            CompletePending(batch);
            return;
        }
        if (batch->quota_shard >= owner_->quota_admission_mutexes_.size()) {
            KVCM_LOG_ERROR("KVMeta reclaimer pending batch has an invalid quota shard");
            CompletePending(batch);
            return;
        }

        std::map<std::string, std::vector<KvMetaManager::SessionItem>> items_by_instance;
        std::vector<KvMetaManager::SessionItem> all_items;
        all_items.reserve(batch->items.size());
        for (const auto &retired : batch->items) {
            items_by_instance[retired.internal_instance_id].push_back(retired.item);
            all_items.push_back(retired.item);
        }

        std::unique_lock<std::mutex> quota_lock(owner_->quota_admission_mutexes_[batch->quota_shard]);
        if (ShouldStop()) {
            CompletePending(batch);
            return;
        }
        if (!EnsureRetiredMetadataDurable(batch)) {
            ++retry_count_metrics_;
            quota_lock.unlock();
            if (ShouldStop()) {
                CompletePending(batch);
            } else {
                ReschedulePending(batch);
            }
            return;
        }

        try {
            RequestContext request_context("kv_meta_reclaimer_finalize");
            for (const auto &[internal_instance_id, items] : items_by_instance) {
                const ErrorCode ec = owner_->DeleteRetiredMetadata(&request_context, internal_instance_id, items);
                if (ec != EC_OK) {
                    ++retry_count_metrics_;
                    KVCM_LOG_WARN("KVMeta reclaimer metadata cleanup will be retried for instance [%s], "
                                  "item_count[%zu], ec[%d]",
                                  internal_instance_id.c_str(),
                                  items.size(),
                                  ec);
                    // The exact delete may already have removed metadata from
                    // the in-memory view before its Sync failed. Close new
                    // admission for this KVMeta group before releasing the
                    // shard; existing sessions may still finish safely.
                    BlockAdmission(batch);
                    quota_lock.unlock();
                    if (ShouldStop()) {
                        CompletePending(batch);
                    } else {
                        ReschedulePending(batch);
                    }
                    return;
                }
            }
        } catch (...) {
            BlockAdmission(batch);
            throw;
        }

        // The metadata ownership change is durable for every item. Physical
        // deletion is intentionally attempted exactly once: a timeout or
        // provider exception has an uncertain outcome, and replay could delete
        // a successor allocation if a backend reuses addresses.
        std::uint64_t reclaimed_bytes = 0;
        for (const auto &item : batch->items) {
            reclaimed_bytes = SaturatingAdd(reclaimed_bytes, item.item.value_size);
        }
        physical_delete_attempted_object_count_metrics_ += batch->items.size();
        ErrorCode physical_ec = EC_IO_ERROR;
        const char *failure_kind = "error_code";
        try {
            RequestContext request_context("kv_meta_reclaimer_physical_delete");
            physical_ec = owner_->DeleteAllocatedLocations(&request_context, all_items);
        } catch (const std::exception &) {
            failure_kind = "standard_exception";
        } catch (...) {
            failure_kind = "unknown_exception";
        }
        if (physical_ec != EC_OK) {
            ++error_count_metrics_;
            physical_delete_uncertain_object_count_metrics_ += batch->items.size();
            physical_delete_uncertain_bytes_metrics_ += reclaimed_bytes;
            KVCM_LOG_WARN("KVMeta reclaimer left objects for backend orphan cleanup, item_count[%zu], "
                          "failure[%s], ec[%d]",
                          all_items.size(),
                          failure_kind,
                          physical_ec);
        }
        // These counters describe logical cache capacity reclaimed after the
        // durable metadata delete. The physical-attempt/uncertain counters
        // above separately expose backend cleanup health.
        reclaimed_object_count_metrics_ += batch->items.size();
        reclaimed_bytes_metrics_ += reclaimed_bytes;
        CompletePending(batch);
    }

    bool ReclaimGroup(RequestContext *request_context, const std::shared_ptr<const InstanceGroup> &group) {
        if (!group || !group->cache_config() || !group->cache_config()->reclaim_strategy()) {
            return false;
        }
        const auto &strategy = group->cache_config()->reclaim_strategy();
        const double threshold = strategy->trigger_strategy().used_percentage();
        const std::int64_t max_delete_delay_ms =
            owner_->limits_.max_write_timeout_seconds > std::numeric_limits<std::int64_t>::max() / 1000
                ? std::numeric_limits<std::int64_t>::max()
                : owner_->limits_.max_write_timeout_seconds * 1000;
        if (strategy->reclaim_policy() != ReclaimPolicy::POLICY_LRU) {
            KVCM_INTERVAL_LOG_WARN(10,
                                   "KVMeta reclaimer skipped group [%s]: only LRU is supported, policy[%d]",
                                   group->name().c_str(),
                                   static_cast<int>(strategy->reclaim_policy()));
            return false;
        }
        if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0 || strategy->delay_before_delete_ms() < 0 ||
            strategy->delay_before_delete_ms() > max_delete_delay_ms) {
            KVCM_INTERVAL_LOG_WARN(10,
                                   "KVMeta reclaimer skipped group [%s] with invalid watermark or delete delay",
                                   group->name().c_str());
            return false;
        }
        const auto [instances_ec, all_instances] =
            owner_->registry_manager_->ListInstanceInfo(request_context, group->name());
        if (instances_ec != EC_OK || all_instances.empty()) {
            return false;
        }
        std::vector<InstanceInfoConstPtr> instances;
        instances.reserve(all_instances.size());
        for (const auto &instance : all_instances) {
            if (!instance || !IsKvMetaInstance(*instance)) {
                // Never let the side-path worker scan or delete an ordinary KV
                // cache group, even if configuration was changed after KVMeta
                // registration.
                return false;
            }
            instances.push_back(instance);
        }

        Pressure pressure;
        if (!ReadPressure(request_context, *group, instances, threshold, pressure) || !pressure.Any()) {
            return false;
        }
        const auto [sampling_size, batch_size] = SamplingAndBatchSize();
        if (sampling_size == 0 || batch_size == 0) {
            KVCM_INTERVAL_LOG_WARN(10,
                                   "KVMeta reclaimer cannot make progress for group [%s]: sample[%zu], batch[%zu]",
                                   group->name().c_str(),
                                   sampling_size,
                                   batch_size);
            return false;
        }
        std::vector<CandidateKey> candidate_keys;
        if (!CollectCandidates(request_context, group->name(), instances, sampling_size, candidate_keys)) {
            KVCM_INTERVAL_LOG_WARN(
                10, "KVMeta reclaimer failed to collect exact LRU candidates for group [%s]", group->name().c_str());
            return false;
        }
        const auto delay = std::chrono::milliseconds(strategy->delay_before_delete_ms());
        const std::size_t quota_shard =
            std::hash<std::string>{}(group->name()) % owner_->quota_admission_mutexes_.size();
        std::vector<RetiredItem> retired;
        {
            std::unique_lock<std::mutex> quota_lock(owner_->quota_admission_mutexes_[quota_shard]);
            if (ShouldStop()) {
                return false;
            }
            const auto &trimming = owner_->trimming_instances_[quota_shard];
            if (std::any_of(instances.begin(), instances.end(), [&](const auto &instance) {
                    return instance && trimming.count(instance->instance_id()) != 0;
                })) {
                return false;
            }
            // Pressure and admission can change while sampling. Recheck under
            // the same group shard used by Put/Remove/Trim before changing any
            // metadata, so a completed concurrent delete cannot cause an
            // unnecessary retirement.
            Pressure current_pressure;
            if (!ReadPressure(request_context, *group, instances, threshold, current_pressure) ||
                !current_pressure.Any()) {
                return false;
            }
            auto selected = SelectCandidates(std::move(candidate_keys), current_pressure, batch_size);
            if (selected.empty()) {
                return false;
            }
            if (!HasPendingCapacity(selected)) {
                ++pending_limit_reject_count_metrics_;
                KVCM_INTERVAL_LOG_WARN(10,
                                       "KVMeta reclaimer pending limit reached for group [%s], selected[%zu]",
                                       group->name().c_str(),
                                       selected.size());
                return false;
            }

            const std::int64_t now_us = TimestampUtil::GetCurrentTimeUs();
            const auto delay_us = std::chrono::duration_cast<std::chrono::microseconds>(delay).count();
            std::int64_t retire_deadline = 0;
            if (!EncodeTaggedDeadlineUs(now_us, delay_us, retire_deadline)) {
                KVCM_LOG_WARN("KVMeta reclaimer could not encode retire deadline for group [%s]",
                              group->name().c_str());
                return false;
            }
            try {
                retired = RetireCandidates(request_context, selected, retire_deadline);
            } catch (...) {
                // Retirement may already be durable. Stop only the KVMeta side
                // before releasing the group shard so another round cannot
                // over-evict without pending credit; leader recovery owns any
                // persisted retired record.
                FailClosedMaintenance();
                throw;
            }
            if (!retired.empty()) {
                // Publish the pending marker before releasing the same group
                // shard observed by Trim. Otherwise Trim could see the
                // durable CLS_DELETING state in the tiny retire/marker window
                // and bypass the configured read grace period.
                const auto finalization_deadline = std::chrono::steady_clock::now() + delay;
                try {
                    AddPendingBatch(group->name(), quota_shard, finalization_deadline, std::move(retired));
                } catch (...) {
                    // AddPendingBatch can allocate before it reaches its
                    // internally guarded map publication. At this point the
                    // CAS+Sync retirement may already be durable, so every
                    // exception must close KVMeta admission before the group
                    // shard is released.
                    FailClosedMaintenance();
                    KVCM_LOG_ERROR("KVMeta reclaimer could not publish pending state for group [%s]; "
                                   "KVMeta maintenance is now fail-closed",
                                   group->name().c_str());
                    throw;
                }
                return true;
            }
        }
        return false;
    }

    bool ReclaimRound() {
        if (ShouldStop()) {
            return false;
        }
        // Share the existing operational pause switch. A pause stops new
        // retirements, while already-retired batches still pass through the
        // metadata/physical finalization path above, matching CacheReclaimer's
        // handling of accepted deletes.
        const auto cache_reclaimer = owner_->cache_manager_->cache_reclaimer();
        if (cache_reclaimer && cache_reclaimer->IsPaused()) {
            return false;
        }
        RequestContext request_context("kv_meta_reclaimer");
        ++round_count_metrics_;
        const auto [groups_ec, groups] = owner_->registry_manager_->ListInstanceGroup(&request_context);
        if (groups_ec != EC_OK) {
            KVCM_INTERVAL_LOG_WARN(10, "KVMeta reclaimer failed to list instance groups, ec[%d]", groups_ec);
            return false;
        }
        std::set<std::string> active_group_names;
        for (const auto &group : groups) {
            if (group) {
                active_group_names.insert(group->name());
            }
        }
        for (auto it = sampling_rotation_by_group_.begin(); it != sampling_rotation_by_group_.end();) {
            if (active_group_names.count(it->first) == 0) {
                it = sampling_rotation_by_group_.erase(it);
            } else {
                ++it;
            }
        }
        bool made_progress = false;
        for (const auto &group : groups) {
            if (ShouldStop()) {
                break;
            }
            try {
                made_progress = ReclaimGroup(&request_context, group) || made_progress;
            } catch (const std::exception &) {
                ++error_count_metrics_;
                KVCM_LOG_WARN("KVMeta reclaimer contained a standard provider exception for group [%s]",
                              group ? group->name().c_str() : "<null>");
            } catch (...) {
                ++error_count_metrics_;
                KVCM_LOG_WARN("KVMeta reclaimer contained an unknown provider exception for group [%s]",
                              group ? group->name().c_str() : "<null>");
            }
        }
        return made_progress;
    }

    void Loop() noexcept {
        for (;;) {
            try {
                const auto interval = std::chrono::milliseconds(IdleIntervalMs());
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    auto wake_deadline = std::chrono::steady_clock::now() + interval;
                    if (!pending_batches_.empty()) {
                        wake_deadline = std::min(wake_deadline, pending_batches_.begin()->first.first);
                    }
                    condition_.wait_until(lock, wake_deadline, [this]() { return stopping_ || wake_requested_; });
                    if (stopping_) {
                        break;
                    }
                    wake_requested_ = false;
                }
                for (const auto &batch : TakeDueBatches()) {
                    try {
                        FinalizePending(batch);
                    } catch (const std::exception &) {
                        ++error_count_metrics_;
                        KVCM_LOG_WARN("KVMeta reclaimer contained a standard finalization exception");
                        if (ShouldStop()) {
                            CompletePending(batch);
                        } else {
                            ReschedulePending(batch);
                        }
                    } catch (...) {
                        ++error_count_metrics_;
                        KVCM_LOG_WARN("KVMeta reclaimer contained an unknown finalization exception");
                        if (ShouldStop()) {
                            CompletePending(batch);
                        } else {
                            ReschedulePending(batch);
                        }
                    }
                }
                if (ShouldStop()) {
                    break;
                }
                const bool made_progress = ReclaimRound();
                if (made_progress) {
                    // Re-evaluate watermarks immediately after a bounded batch.
                    Wake();
                }
            } catch (const std::exception &) {
                ++error_count_metrics_;
                KVCM_LOG_WARN("KVMeta reclaimer loop contained a standard exception");
                try {
                    std::unique_lock<std::mutex> lock(mutex_);
                    wake_requested_ = false;
                    condition_.wait_for(lock, std::chrono::milliseconds(100), [this]() { return stopping_; });
                    if (stopping_) {
                        break;
                    }
                } catch (...) {
                    break;
                }
            } catch (...) {
                ++error_count_metrics_;
                KVCM_LOG_WARN("KVMeta reclaimer loop contained an unknown exception");
                try {
                    std::unique_lock<std::mutex> lock(mutex_);
                    wake_requested_ = false;
                    condition_.wait_for(lock, std::chrono::milliseconds(100), [this]() { return stopping_; });
                    if (stopping_) {
                        break;
                    }
                } catch (...) {
                    break;
                }
            }
        }
    }

    KvMetaManager *owner_ = nullptr;
    mutable std::mutex lifecycle_mutex_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = true;
    bool wake_requested_ = false;
    std::map<std::string, std::size_t> pending_instances_;
    std::map<std::pair<std::string, std::string>, std::size_t> pending_locations_;
    std::map<std::string, PendingCredit> pending_credits_;
    std::map<PendingDeadline, std::shared_ptr<PendingBatch>> pending_batches_;
    std::map<std::string, std::size_t> sampling_rotation_by_group_;
    std::uint64_t next_pending_sequence_ = 0;
    std::uint64_t pending_object_count_ = 0;
    std::uint64_t pending_bytes_ = 0;
    Counter round_count_metrics_;
    Counter retired_object_count_metrics_;
    Counter reclaimed_object_count_metrics_;
    Counter reclaimed_bytes_metrics_;
    Counter retry_count_metrics_;
    Counter error_count_metrics_;
    Counter pending_limit_reject_count_metrics_;
    Counter physical_delete_attempted_object_count_metrics_;
    Counter physical_delete_uncertain_object_count_metrics_;
    Counter physical_delete_uncertain_bytes_metrics_;
    Gauge pending_object_count_metrics_;
    Gauge pending_bytes_metrics_;
    Gauge blocked_group_count_metrics_;
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
        limits_.max_write_timeout_seconds <= 0 ||
        limits_.max_write_timeout_seconds > std::numeric_limits<std::int32_t>::max()) {
        KVCM_LOG_ERROR("KVMeta manager init failed: dependency or limits are invalid");
        return false;
    }
    data_storage_selector_ =
        std::make_unique<DataStorageSelector>(cache_manager_->meta_indexer_manager(), registry_manager_);
    reclaimer_ = std::make_unique<KvMetaReclaimer>(this);
    write_session_manager_ = std::make_unique<KvMetaWriteSessionManager>(this, limits_.max_active_write_sessions);
    if (!write_session_manager_->Start()) {
        write_session_manager_.reset();
        reclaimer_.reset();
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
    maintenance_cancelled_.store(true, std::memory_order_release);
    if (write_session_manager_) {
        write_session_manager_->RequestStop();
    }
    if (reclaimer_) {
        reclaimer_->RequestStop();
    }
}

bool KvMetaManager::ResumeMaintenance() {
    if (!initialized_.load(std::memory_order_acquire) || !write_session_manager_ || !reclaimer_) {
        return false;
    }
    if (!write_session_manager_->Start()) {
        return false;
    }
    maintenance_cancelled_.store(false, std::memory_order_release);
    if (!reclaimer_->Start()) {
        maintenance_cancelled_.store(true, std::memory_order_release);
        write_session_manager_->RequestStop();
        reclaimer_->RequestStop();
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

bool KvMetaManager::IsOwnedLocation(std::int64_t internal_key, const std::string &location_id) const {
    if (location_id.size() <= kKvMetaLocationIdPrefix.size() ||
        location_id.compare(0, kKvMetaLocationIdPrefix.size(), kKvMetaLocationIdPrefix) != 0) {
        return false;
    }
    const std::string_view encoded_key = std::string_view(location_id).substr(kKvMetaLocationIdPrefix.size());
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
    const bool known_state = (location.status() == CLS_NEW && location.create_time() != 0) || IsRetiredObject(location);
    if (!IsOwnedLocation(internal_key, location_id) || location.id() != location_id || !known_state ||
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
        sum = value > std::numeric_limits<std::uint64_t>::max() - sum ? std::numeric_limits<std::uint64_t>::max()
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

std::pair<ErrorCode, std::vector<KvMetaManager::GetResult>> KvMetaManager::Get(
    RequestContext *request_context, const std::string &instance_id, const std::vector<std::string> &keys) const {
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
        if (const ErrorCode ec = ValidateOwnedLocation(
                request_context, exact[i].internal_key, exact[i].location_id, *exact[i].location, value_size);
            ec != EC_OK) {
            return {ec, {}};
        }
        if (!IsCommittedObject(*exact[i].location)) {
            continue;
        }
        if (!ToValueLocation(*exact[i].location, result[i].location) || result[i].location.value_size != value_size) {
            AddError(request_context, "KVMeta committed location is malformed");
            return {EC_CORRUPTION, {}};
        }
        result[i].found = true;
    }
    return {EC_OK, std::move(result)};
}

ErrorCode KvMetaManager::DeleteStorageUris(RequestContext *request_context,
                                           const std::string &storage_name,
                                           const std::vector<DataStorageUri> &uris) const {
    if (uris.empty()) {
        return EC_OK;
    }
    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!data_storage_manager) {
        return EC_ERROR;
    }
    std::vector<ErrorCode> delete_results;
    const char *failure_kind = nullptr;
    try {
        delete_results = data_storage_manager->Delete(request_context, storage_name, uris, nullptr);
    } catch (const std::exception &) {
        failure_kind = "standard_exception";
    } catch (...) {
        failure_kind = "unknown_exception";
    }
    if (failure_kind) {
        KVCM_LOG_WARN("KVMeta storage delete caught a provider exception, item_count[%zu], failure[%s]",
                      uris.size(),
                      failure_kind);
        AddError(request_context, "KVMeta storage delete caught a provider exception");
        return EC_IO_ERROR;
    }
    if (delete_results.size() != uris.size()) {
        AddError(request_context, "KVMeta storage delete returned a mismatched result count");
        return EC_MISMATCH;
    }
    ErrorCode overall = EC_OK;
    for (const ErrorCode ec : delete_results) {
        if (ec != EC_OK && ec != EC_NOENT) {
            overall = FirstHardError(overall, ec);
        }
    }
    return overall;
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
        overall = FirstHardError(overall, DeleteStorageUris(request_context, storage_name, uris));
    }
    return overall;
}

ErrorCode KvMetaManager::DeleteItems(RequestContext *request_context,
                                     const std::string &internal_instance_id,
                                     const std::vector<SessionItem> &items,
                                     bool metadata_only,
                                     bool adjust_storage_usage,
                                     bool maintenance_no_touch,
                                     bool delete_if_metadata_absent,
                                     bool sync_metadata_absent,
                                     bool restore_usage_on_sync_failure) {
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
                                                                  adjust_storage_usage,
                                                                  true,
                                                                  maintenance_no_touch);
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

    KeyVector keys_to_sync;
    std::unordered_set<std::int64_t> unique_keys_to_sync;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if ((metadata_deleted[i] || (sync_metadata_absent && metadata_already_absent[i])) &&
            unique_keys_to_sync.insert(items[i].internal_key).second) {
            keys_to_sync.push_back(items[i].internal_key);
        }
    }
    const bool metadata_delete_is_durable = keys_to_sync.empty() || indexer->Sync(keys_to_sync);
    if (!metadata_delete_is_durable) {
        overall = FirstHardError(overall, EC_TIMEOUT);
        // BatchDeleteLocations adjusts the in-memory counter when the delete
        // is accepted. If its persistence barrier fails, restore a
        // conservative upper bound; the next KVMeta recovery rebuilds the
        // exact value from durable metadata.
        if (adjust_storage_usage && restore_usage_on_sync_failure) {
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
            const bool safe_to_delete = (delete_if_metadata_absent && metadata_already_absent[i]) ||
                                        (metadata_deleted[i] && metadata_delete_is_durable);
            if (safe_to_delete && items[i].data_location) {
                physical_items.push_back(items[i]);
            }
        }
        overall = FirstHardError(overall, DeleteAllocatedLocations(request_context, physical_items));
    }
    return overall;
}

ErrorCode KvMetaManager::DeleteRetiredMetadata(RequestContext *request_context,
                                               const std::string &internal_instance_id,
                                               const std::vector<SessionItem> &items) {
    // Reclaimer finalization deliberately separates metadata and physical
    // deletion. If Sync fails after the in-memory exact delete, a later retry
    // must Sync the now-absent key before releasing the allocation. Keeping
    // the already-decremented usage avoids double accounting on that retry.
    return DeleteItems(request_context, internal_instance_id, items, true, true, true, false, true, false);
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
    const auto write_deadline = KvMetaWriteSessionManager::Clock::now() + std::chrono::seconds(write_timeout_seconds);
    std::int64_t persistent_write_deadline = 0;
    if (!EncodeLeaseDeadline(TimestampUtil::GetCurrentTimeUs(), write_timeout_seconds, persistent_write_deadline)) {
        AddError(request_context, "KVMeta write lease deadline is outside the persistent timestamp range");
        return {EC_OUT_OF_LIMIT, StartWriteResult{}};
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
    if (trimming_instances_[quota_shard].count(internal_instance_id) != 0) {
        AddError(request_context, "KVMeta instance is being trimmed");
        return {EC_EXIST, StartWriteResult{}};
    }

    // A retired location normally remains visible as CLS_DELETING until its
    // final persistence barrier succeeds. If that barrier failed after the
    // in-memory delete, the exact metadata can temporarily be absent. Keep the
    // same logical key closed until the pending batch either completes or is
    // handed to leader recovery; otherwise a reusable backend URI could expose
    // an ABA window between old-object deletion and a new allocation.
    if (reclaimer_) {
        if (reclaimer_->IsAdmissionBlocked(instance_info->instance_group_name())) {
            AddError(request_context, "KVMeta admission is paused while reclaim metadata is being persisted");
            return {EC_EXIST, StartWriteResult{}};
        }
        for (const auto &key : keys) {
            if (reclaimer_->HasPendingLocation(internal_instance_id, StableLocationId(key))) {
                AddError(request_context, "KVMeta value is being reclaimed");
                return {EC_EXIST, StartWriteResult{}};
            }
        }
    }

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
            if (IsRetiredObject(*existing[i].location)) {
                AddError(request_context, "KVMeta value is being reclaimed");
                return {EC_EXIST, StartWriteResult{}};
            }
            if (existing_size != value_sizes[i]) {
                AddError(request_context, "KVMeta existing value size does not match PutStart value_sizes");
                return {EC_MISMATCH, StartWriteResult{}};
            }
            // Only a committed object is an idempotent cache hit. Treating an
            // active reservation as a hit can make a second object client
            // report SaveObjects success while the first writer later aborts,
            // leaving no readable value behind.
            if (!IsCommittedObject(*existing[i].location)) {
                AddError(request_context, "KVMeta value is still being written");
                return {EC_EXIST, StartWriteResult{}};
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
    case KvMetaWriteSessionManager::PutResult::kExpired:
        return {EC_TIMEOUT, StartWriteResult{}};
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
        if (ec == EC_NOSPC && reclaimer_) {
            reclaimer_->Wake();
        }
        return {ec, StartWriteResult{}};
    }
    const auto selected = data_storage_selector_->SelectCacheWriteDataStorageBackend(
        request_context, instance_info->instance_group_name(), missing_bytes);
    if (selected.ec != EC_OK || selected.name.empty() || selected.type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
        return {selected.ec == EC_OK ? EC_NOENT : selected.ec, StartWriteResult{}};
    }
    if (const ErrorCode ec = CheckDynamicByteAdmission(
            request_context, instance_info->instance_group_name(), selected.type, missing_bytes);
        ec != EC_OK) {
        if (ec == EC_NOSPC && reclaimer_) {
            reclaimer_->Wake();
        }
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
    std::unordered_set<std::string> candidate_allocation_uris;
    candidate_allocation_uris.reserve(missing_indices.size());
    for (const std::size_t request_index : missing_indices) {
        const bool cancelled = maintenance_cancelled_.load(std::memory_order_acquire);
        const bool expired = KvMetaWriteSessionManager::Clock::now() >= write_deadline;
        if (cancelled || expired) {
            const ErrorCode cleanup_ec = DeleteAllocatedLocations(request_context, candidates);
            if (cleanup_ec != EC_OK) {
                KVCM_LOG_WARN("KVMeta admission stop could not release all uncommitted allocations, ec[%d]",
                              cleanup_ec);
            }
            return {cancelled ? EC_SERVICE_NOT_LEADER : EC_TIMEOUT, StartWriteResult{}};
        }
        const std::uint64_t instance_path_hash =
            Hash64(internal_instance_id.data(), internal_instance_id.size(), kInstancePathHashSeed);
        const std::string object_key =
            "kvmeta/" + StringUtil::Uint64ToHex(instance_path_hash) + "/" +
            StringUtil::Uint64ToHex(static_cast<std::uint64_t>(existing[request_index].internal_key)) + "/" +
            StringUtil::GenerateRandomString(32);
        std::vector<std::pair<ErrorCode, DataStorageUri>> create_result;
        try {
            create_result = data_storage_manager->Create(request_context,
                                                         selected.name,
                                                         {object_key},
                                                         static_cast<std::size_t>(value_sizes[request_index]),
                                                         nullptr);
        } catch (const std::exception &) {
            KVCM_LOG_WARN("KVMeta storage create caught a standard provider exception; "
                          "backend orphan cleanup may be required");
            DeleteAllocatedLocations(request_context, candidates);
            AddError(request_context, "KVMeta storage create failed; backend orphan cleanup may be required");
            return {EC_IO_ERROR, StartWriteResult{}};
        } catch (...) {
            KVCM_LOG_WARN("KVMeta storage create caught an unknown provider exception; "
                          "backend orphan cleanup may be required");
            DeleteAllocatedLocations(request_context, candidates);
            AddError(request_context, "KVMeta storage create failed; backend orphan cleanup may be required");
            return {EC_IO_ERROR, StartWriteResult{}};
        }
        if (create_result.size() != 1 || create_result[0].first != EC_OK ||
            !UriMatchesStorageBackend(create_result[0].second, selected.name, selected.type) ||
            !HasSingletonAllocationShape(create_result[0].second, selected.type)) {
            const ErrorCode create_ec = create_result.size() == 1 ? create_result[0].first : EC_MISMATCH;
            // A backend contract violation can still carry successful
            // allocations. Release every safely attributable singleton, not
            // just the first result, or an overlong response would orphan its
            // physical objects before any metadata/session is recorded.
            std::vector<DataStorageUri> malformed_allocations;
            std::unordered_set<std::string> seen_allocations;
            for (const auto &[ec, uri] : create_result) {
                // A malformed response is untrusted. Only send locations back
                // to this backend when both its identity and URI scheme prove
                // that the backend owns them.
                if (ec == EC_OK && UriMatchesStorageBackend(uri, selected.name, selected.type) &&
                    HasSingletonAllocationShape(uri, selected.type) &&
                    seen_allocations.insert(uri.ToUriString()).second) {
                    malformed_allocations.push_back(uri);
                }
            }
            if (!malformed_allocations.empty()) {
                if (DeleteStorageUris(request_context, selected.name, malformed_allocations) != EC_OK) {
                    KVCM_LOG_WARN("KVMeta could not release every malformed new allocation");
                }
            }
            DeleteAllocatedLocations(request_context, candidates);
            AddError(request_context, "KVMeta singleton storage allocation failed");
            return {create_ec == EC_OK ? EC_CORRUPTION : create_ec, StartWriteResult{}};
        }
        const std::string allocation_uri = create_result[0].second.ToUriString();
        if (!candidate_allocation_uris.insert(allocation_uri).second) {
            // A singleton call must still return a distinct physical object
            // for every key. The duplicate URI is already owned by an earlier
            // candidate, so deleting the deduplicated candidate set releases
            // it exactly once and no metadata is published for either key.
            const ErrorCode cleanup_ec = DeleteAllocatedLocations(request_context, candidates);
            if (cleanup_ec != EC_OK) {
                KVCM_LOG_WARN("KVMeta duplicate allocation cleanup failed, ec[%d]", cleanup_ec);
            }
            AddError(request_context, "KVMeta storage reused one singleton allocation for multiple keys");
            return {EC_CORRUPTION, StartWriteResult{}};
        }

        auto location = std::make_shared<CacheLocation>();
        location->set_id(existing[request_index].location_id);
        location->set_status(CLS_NEW);
        location->set_type(selected.type);
        location->set_spec_size(1);
        // Positive create_time is KVMeta-private for active objects. Persist a
        // tagged wall-clock lease deadline, rather than only the allocation
        // time, so a new leader cannot reclaim this URI while the client is
        // still inside its advertised write window. Older untagged positive
        // markers remain readable and receive the configured maximum lease
        // during rolling-upgrade recovery.
        location->set_create_time(persistent_write_deadline);
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
            if (DeleteStorageUris(request_context, selected.name, {create_result[0].second}) != EC_OK) {
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

    const bool cancelled_after_allocation = maintenance_cancelled_.load(std::memory_order_acquire);
    if (cancelled_after_allocation || KvMetaWriteSessionManager::Clock::now() >= write_deadline) {
        const ErrorCode cleanup_ec = DeleteAllocatedLocations(request_context, candidates);
        if (cleanup_ec != EC_OK) {
            KVCM_LOG_WARN("KVMeta stopped allocation cleanup failed, ec[%d]", cleanup_ec);
        }
        return {cancelled_after_allocation ? EC_SERVICE_NOT_LEADER : EC_TIMEOUT, StartWriteResult{}};
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
            return EC_OUTCOME_UNKNOWN;
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
            // The conditional insert can lose to a writer whose metadata is
            // valid but not committed yet. Do not turn that transient state
            // into a successful cache hit: our own candidate allocations must
            // be rolled back and the caller must retry explicitly.
            if (!IsCommittedObject(*race_winners[i].location)) {
                AddError(request_context, "KVMeta concurrent winner is still writing");
                return {rollback_start(EC_EXIST), StartWriteResult{}};
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
    for (int attempt = 0; attempt < 8 && session_result == KvMetaWriteSessionManager::PutResult::kDuplicate;
         ++attempt) {
        session_id = StringUtil::GenerateRandomString(32);
        auto items_for_attempt = session_items;
        session_result =
            write_session_manager_
                ? write_session_manager_->Put(
                      session_id, internal_instance_id, quota_shard, std::move(items_for_attempt), write_deadline)
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
        if (session_result == KvMetaWriteSessionManager::PutResult::kExpired) {
            AddError(request_context, "KVMeta write lease expired before PutStart completed");
            return {EC_TIMEOUT, StartWriteResult{}};
        }
        AddError(request_context, "KVMeta could not generate a unique write-session id");
        return {EC_ERROR, StartWriteResult{}};
    }
    response.write_session_id = std::move(session_id);
    response.session_item_count = session_items.size();
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
                // A reservation disappearing before commit is a failed CAS,
                // not an idempotent absence. FirstHardError intentionally
                // ignores EC_NOENT for read/remove aggregation, so normalize
                // it here or a partially committed batch could be reported
                // as successful without running rollback.
                commit_error = FirstHardError(commit_error, ec == EC_OK || ec == EC_NOENT ? EC_MISMATCH : ec);
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
        return EC_OUTCOME_UNKNOWN;
    }
    return commit_error;
}

ErrorCode KvMetaManager::FinishWrite(RequestContext *request_context,
                                     const std::string &instance_id,
                                     const std::string &write_session_id,
                                     const std::vector<bool> &success_keys) {
    if (!initialized_.load(std::memory_order_acquire) || ValidateInstanceId(request_context, instance_id) != EC_OK ||
        write_session_id.empty() || write_session_id.size() > limits_.max_write_session_id_bytes ||
        success_keys.empty() || !write_session_manager_) {
        AddError(request_context, "KVMeta FinishWrite has an invalid session id or success mask");
        return EC_BADARGS;
    }
    KvMetaWriteSessionManager::Session session;
    auto [take_result, finalization] = write_session_manager_->Take(
        write_session_id, InternalInstanceId(instance_id), std::optional<std::size_t>{success_keys.size()}, session);
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
    case KvMetaWriteSessionManager::TakeResult::kExpired:
    case KvMetaWriteSessionManager::TakeResult::kOk:
        break;
    }
    if (session.quota_shard >= quota_admission_mutexes_.size()) {
        AddError(request_context, "KVMeta write session has an invalid quota shard");
        return EC_CORRUPTION;
    }
    // Keep commit/rollback finalization in the same KVMeta-only group critical
    // section as allocation, Remove and Trim. In particular, rollback must
    // finish deleting the old physical object before a new generation can be
    // allocated for the key.
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[session.quota_shard]);
    // Take removes the session before waiting for the group shard. Another
    // KVMeta operation in the same group can hold that shard across backend
    // I/O, so the lease may expire while this finalizer is queued. Recheck
    // after acquiring the lock; otherwise a value could become visible after
    // the timeout promised to both the client and leader-recovery logic.
    const auto cleanup_active_session = [&](ErrorCode completed_result) {
        ErrorCode cleanup_ec = EC_IO_ERROR;
        const char *failure_kind = "error_code";
        try {
            const std::vector<bool> failed(session.items.size(), false);
            cleanup_ec = FinishWriteInternal(request_context, session.internal_instance_id, failed, session.items);
        } catch (const std::exception &) {
            failure_kind = "standard_exception";
        } catch (...) {
            failure_kind = "unknown_exception";
        }
        if (cleanup_ec == EC_OK) {
            return completed_result;
        }
        // The physical-delete outcome may be ambiguous. Never replay it
        // without a generation-bearing URI: an orphan is safer than deleting
        // a successor allocation that reused the same backend address.
        KVCM_LOG_WARN("KVMeta active write cleanup did not complete; backend orphan cleanup may be required, "
                      "item_count[%zu], failure[%s], ec[%d]",
                      session.items.size(),
                      failure_kind,
                      cleanup_ec);
        AddError(request_context, "KVMeta active write cleanup failed; backend orphan cleanup may be required");
        // Preserve the lease-expired result: cleanup failure can create an
        // orphan, but it must not hide the fact that the session was already
        // ineligible to commit. Explicit caller aborts still surface their
        // cleanup error because completed_result is EC_OK in that path.
        return completed_result == EC_OK ? cleanup_ec : completed_result;
    };

    const bool expired = take_result == KvMetaWriteSessionManager::TakeResult::kExpired ||
                         KvMetaWriteSessionManager::Clock::now() >= session.deadline;
    if (expired) {
        AddError(request_context, "KVMeta write session expired before FinishWrite");
        return cleanup_active_session(EC_TIMEOUT);
    }
    if (std::any_of(success_keys.begin(), success_keys.end(), [](bool success) { return !success; })) {
        return cleanup_active_session(EC_OK);
    }
    // Successful commits are observed by the bounded periodic reclaim round.
    // Waking on every PutFinish would turn sustained write QPS into registry
    // scan QPS even when the group is far below its watermark. EC_NOSPC still
    // wakes the worker immediately from StartWrite's admission path.
    return FinishWriteInternal(request_context, session.internal_instance_id, success_keys, session.items);
}

ErrorCode KvMetaManager::Remove(RequestContext *request_context,
                                const std::string &instance_id,
                                const std::vector<std::string> &keys) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return EC_ERROR;
    }
    const auto [instance_ec, instance_info] = GetValidatedInstanceInfo(request_context, instance_id);
    if (instance_ec != EC_OK || !instance_info) {
        return instance_ec == EC_OK ? EC_CORRUPTION : instance_ec;
    }
    if (const ErrorCode ec = ValidateKeys(request_context, keys); ec != EC_OK) {
        return ec;
    }

    // Keep metadata removal, its persistence barrier, and physical deletion
    // in the same KVMeta-only group critical section as StartWrite. Without
    // this ordering, a new generation of the same key could allocate after
    // the old metadata disappears but before its physical delete completes.
    // A backend that quickly reuses allocation URIs would then expose an ABA
    // window in which the old Remove can delete the new generation's object.
    // Regular fixed-block KV-cache operations never take these shard locks.
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % quota_admission_mutexes_.size();
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[quota_shard]);

    const std::string internal_instance_id = InternalInstanceId(instance_id);
    if (trimming_instances_[quota_shard].count(internal_instance_id) != 0) {
        AddError(request_context, "KVMeta instance is being trimmed");
        return EC_EXIST;
    }
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
        if (const ErrorCode ec = ValidateOwnedLocation(
                request_context, exact[i].internal_key, exact[i].location_id, *exact[i].location, size);
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
        items.push_back(SessionItem{
            i, keys[i], exact[i].internal_key, exact[i].location_id, exact[i].location, exact[i].location, size});
    }
    return DeleteItems(request_context, internal_instance_id, items, false, true);
}

ErrorCode KvMetaManager::TrimAll(RequestContext *request_context, const std::string &instance_id, bool metadata_only) {
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

    // Fail fast when a session is already known to be active/finalizing. A
    // finalizer can hold the group shard across slow physical I/O, and Trim
    // should report WRITE_IN_PROGRESS rather than wait behind that I/O. The
    // second check below remains mandatory to close the check-to-lock race.
    if (write_session_manager_ && write_session_manager_->HasSessionForInstance(internal_instance_id)) {
        AddError(request_context, "KVMeta cannot trim an instance while a write session is active or finalizing");
        return EC_EXIST;
    }
    if (reclaimer_ && reclaimer_->HasPendingForInstance(internal_instance_id)) {
        AddError(request_context, "KVMeta cannot trim an instance while automatic reclaim is pending");
        return EC_EXIST;
    }

    // Publish a per-instance fence under the same shard used by StartWrite,
    // Remove and reclaim retirement. The fence closes the check-to-lock race,
    // while releasing the shard below prevents an unbounded metadata scan or
    // slow physical delete from stalling unrelated instances in this group.
    const std::size_t quota_shard =
        std::hash<std::string>{}(instance_info->instance_group_name()) % quota_admission_mutexes_.size();
    std::unique_lock<std::mutex> quota_lock(quota_admission_mutexes_[quota_shard]);
    if (maintenance_cancelled_.load(std::memory_order_acquire)) {
        return EC_SERVICE_NOT_LEADER;
    }
    // StartWrite holds the same group shard until its active metadata and
    // session are both installed (or fully rolled back), so no new writer for
    // this instance can appear after this check. Never let Trim free a remote
    // allocation while its owner can still be writing into it. A session stays
    // protected after Take removes it from the lookup map until FinishWrite or
    // timeout cleanup has completed all metadata and physical-storage I/O.
    if (write_session_manager_ && write_session_manager_->HasSessionForInstance(internal_instance_id)) {
        AddError(request_context, "KVMeta cannot trim an instance while a write session is active or finalizing");
        return EC_EXIST;
    }
    if (reclaimer_ && reclaimer_->HasPendingForInstance(internal_instance_id)) {
        AddError(request_context, "KVMeta cannot trim an instance while automatic reclaim is pending");
        return EC_EXIST;
    }
    try {
        if (!trimming_instances_[quota_shard].insert(internal_instance_id).second) {
            AddError(request_context, "KVMeta instance is already being trimmed");
            return EC_EXIST;
        }
    } catch (...) {
        AddError(request_context, "KVMeta could not publish the trim admission fence");
        return EC_ERROR;
    }

    struct TrimMarkerGuard {
        std::mutex *mutex = nullptr;
        std::unordered_set<std::string> *instances = nullptr;
        const std::string *instance_id = nullptr;

        ~TrimMarkerGuard() noexcept {
            if (!mutex || !instances || !instance_id) {
                return;
            }
            std::lock_guard<std::mutex> lock(*mutex);
            instances->erase(*instance_id);
        }
    } trim_marker{&quota_admission_mutexes_[quota_shard], &trimming_instances_[quota_shard], &internal_instance_id};
    quota_lock.unlock();

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
            const ErrorCode scan_ec = indexer->Scan(request_context, cursor, kRecoveryScanBatchSize, next_cursor, keys);
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
                        return get_result.error_codes[i] == EC_OK ? EC_CORRUPTION : get_result.error_codes[i];
                    }
                    for (const auto &[location_id, location] : locations[i]) {
                        if (!location) {
                            operation_ec = EC_CORRUPTION;
                            break;
                        }
                        std::uint64_t size = 0;
                        operation_ec = ValidateOwnedLocation(request_context, keys[i], location_id, *location, size);
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
    // Even corrupted/far-future persisted deadlines cannot keep this isolated
    // recovery blocked forever. A legitimate active lease is bounded by the
    // same configured maximum, so waiting at most that long from promotion
    // preserves its data-I/O window without affecting main KV-cache recovery.
    const auto recovery_force_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(limits_.max_write_timeout_seconds);
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
            std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> committed_usage_by_type{};
            ErrorCode instance_recovery_ec = EC_OK;
            bool removed_stale_in_pass = false;
            bool deferred_active_in_pass = false;
            do {
                removed_stale_in_pass = false;
                deferred_active_in_pass = false;
                auto earliest_deferred_deadline = std::chrono::steady_clock::time_point::max();
                const auto scan_steady_time = std::chrono::steady_clock::now();
                const std::int64_t scan_wall_time_us = TimestampUtil::GetCurrentTimeUs();
                if (scan_wall_time_us <= 0) {
                    instance_recovery_ec = FirstHardError(instance_recovery_ec, EC_ERROR);
                    break;
                }
                committed_usage_by_type.fill(0);
                std::vector<SessionItem> stale_batch;
                stale_batch.reserve(kMaintenanceDeleteBatchSize);
                const auto flush_stale = [&]() {
                    if (stale_batch.empty()) {
                        return EC_OK;
                    }
                    // Complete and persist the ownership change before any
                    // physical release. A metadata failure must keep the gate
                    // closed because the active record is still authoritative.
                    ErrorCode metadata_ec = EC_IO_ERROR;
                    try {
                        metadata_ec = DeleteItems(&request_context, instance->instance_id(), stale_batch, true, false);
                    } catch (const std::exception &) {
                        KVCM_LOG_WARN("KVMeta recovery metadata cleanup caught a standard exception; "
                                      "service remains disabled");
                    } catch (...) {
                        KVCM_LOG_WARN("KVMeta recovery metadata cleanup caught an unknown exception; "
                                      "service remains disabled");
                    }
                    if (metadata_ec != EC_OK) {
                        stale_batch.clear();
                        return metadata_ec;
                    }

                    ErrorCode physical_ec = EC_IO_ERROR;
                    const char *failure_kind = "error_code";
                    try {
                        physical_ec = DeleteAllocatedLocations(&request_context, stale_batch);
                    } catch (const std::exception &) {
                        failure_kind = "standard_exception";
                    } catch (...) {
                        failure_kind = "unknown_exception";
                    }
                    if (physical_ec != EC_OK) {
                        // Reusable-address backends do not expose a generation
                        // token. Once metadata is gone, replaying an uncertain
                        // delete could remove an unrelated successor object.
                        // Keep recovery available and leave this unreachable
                        // allocation to backend-level orphan reclamation.
                        KVCM_LOG_WARN("KVMeta recovery left objects for backend orphan cleanup, "
                                      "item_count[%zu], failure[%s], ec[%d]",
                                      stale_batch.size(),
                                      failure_kind,
                                      physical_ec);
                    }
                    stale_batch.clear();
                    return EC_OK;
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
                            instance_recovery_ec = FirstHardError(instance_recovery_ec,
                                                                  get_result.ec == EC_OK ? EC_MISMATCH : get_result.ec);
                            break;
                        }
                        for (std::size_t i = 0; i < keys.size() && instance_recovery_ec == EC_OK; ++i) {
                            if (get_result.error_codes[i] != EC_OK || locations[i].empty()) {
                                instance_recovery_ec = FirstHardError(
                                    instance_recovery_ec,
                                    get_result.error_codes[i] == EC_OK ? EC_CORRUPTION : get_result.error_codes[i]);
                                break;
                            }
                            for (const auto &[location_id, location] : locations[i]) {
                                if (!location) {
                                    instance_recovery_ec = FirstHardError(instance_recovery_ec, EC_CORRUPTION);
                                    break;
                                }
                                std::uint64_t size = 0;
                                const ErrorCode validate_ec =
                                    ValidateOwnedLocation(&request_context, keys[i], location_id, *location, size);
                                const DataStorageType base_type = ToBaseType(location->type());
                                const std::size_t type_index = ToIndex(base_type);
                                if (validate_ec != EC_OK || base_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN ||
                                    type_index >= committed_usage_by_type.size()) {
                                    instance_recovery_ec = FirstHardError(
                                        instance_recovery_ec, validate_ec == EC_OK ? EC_CORRUPTION : validate_ec);
                                    break;
                                }
                                if (IsRetiredObject(*location)) {
                                    std::int64_t retire_deadline_us = 0;
                                    if (DecodeLeaseDeadline(location->create_time(), retire_deadline_us) &&
                                        retire_deadline_us > scan_wall_time_us &&
                                        scan_steady_time < recovery_force_deadline) {
                                        const auto force_remaining_us =
                                            std::chrono::duration_cast<std::chrono::microseconds>(
                                                recovery_force_deadline - scan_steady_time)
                                                .count();
                                        if (force_remaining_us > 0) {
                                            const std::int64_t wait_us =
                                                std::min(retire_deadline_us - scan_wall_time_us, force_remaining_us);
                                            deferred_active_in_pass = true;
                                            earliest_deferred_deadline =
                                                std::min(earliest_deferred_deadline,
                                                         scan_steady_time + std::chrono::microseconds(wait_us));
                                            continue;
                                        }
                                    }
                                    removed_stale_in_pass = true;
                                    auto copy = std::make_shared<CacheLocation>(*location);
                                    stale_batch.push_back(SessionItem{0, {}, keys[i], location_id, copy, copy, size});
                                    if (stale_batch.size() == kMaintenanceDeleteBatchSize) {
                                        if (should_abort && should_abort()) {
                                            return EC_SERVICE_NOT_LEADER;
                                        }
                                        instance_recovery_ec = FirstHardError(instance_recovery_ec, flush_stale());
                                        if (instance_recovery_ec != EC_OK) {
                                            break;
                                        }
                                    }
                                    continue;
                                }
                                if (location->create_time() < 0) {
                                    if (size > std::numeric_limits<std::uint64_t>::max() -
                                                   committed_usage_by_type[type_index]) {
                                        instance_recovery_ec = FirstHardError(instance_recovery_ec, EC_OUT_OF_LIMIT);
                                        break;
                                    }
                                    committed_usage_by_type[type_index] += size;
                                    continue;
                                }
                                std::int64_t lease_deadline_us = 0;
                                if (DecodeRecoveryLeaseDeadline(location->create_time(),
                                                                limits_.max_write_timeout_seconds,
                                                                lease_deadline_us) &&
                                    lease_deadline_us > scan_wall_time_us &&
                                    scan_steady_time < recovery_force_deadline) {
                                    const auto force_remaining_us =
                                        std::chrono::duration_cast<std::chrono::microseconds>(recovery_force_deadline -
                                                                                              scan_steady_time)
                                            .count();
                                    if (force_remaining_us > 0) {
                                        const std::int64_t wait_us =
                                            std::min(lease_deadline_us - scan_wall_time_us, force_remaining_us);
                                        deferred_active_in_pass = true;
                                        earliest_deferred_deadline =
                                            std::min(earliest_deferred_deadline,
                                                     scan_steady_time + std::chrono::microseconds(wait_us));
                                        continue;
                                    }
                                }
                                removed_stale_in_pass = true;
                                auto copy = std::make_shared<CacheLocation>(*location);
                                stale_batch.push_back(SessionItem{0, {}, keys[i], location_id, copy, copy, size});
                                if (stale_batch.size() == kMaintenanceDeleteBatchSize) {
                                    if (should_abort && should_abort()) {
                                        return EC_SERVICE_NOT_LEADER;
                                    }
                                    instance_recovery_ec = FirstHardError(instance_recovery_ec, flush_stale());
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
                if (instance_recovery_ec == EC_OK && deferred_active_in_pass) {
                    // Keep the KVMeta gate closed while an old leader's client
                    // may still be writing. Poll in short intervals so
                    // demotion/Stop cancellation never waits for the lease.
                    const auto poll_interval =
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(kRecoveryWaitPollInterval);
                    while (std::chrono::steady_clock::now() < earliest_deferred_deadline) {
                        if (should_abort && should_abort()) {
                            return EC_SERVICE_NOT_LEADER;
                        }
                        const auto remaining = earliest_deferred_deadline - std::chrono::steady_clock::now();
                        if (remaining <= std::chrono::steady_clock::duration::zero()) {
                            break;
                        }
                        std::this_thread::sleep_for(std::min(remaining, poll_interval));
                    }
                }
                // If this pass removed anything, cursor semantics under
                // mutation may have skipped entries and the usage sum is no
                // longer authoritative. A deferred live lease also requires a
                // rescan after its deadline. The final no-delete/no-defer pass
                // is stable and is the only one used below.
            } while (instance_recovery_ec == EC_OK && (removed_stale_in_pass || deferred_active_in_pass));
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
