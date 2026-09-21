#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/data_storage/common_define.h"

namespace kv_cache_manager {

// A request may report the same stable location id for tens of thousands of
// block keys. CacheLocation can borrow shared ownership of one canonical
// string so immutable location copies do not allocate/copy that id again.
using InternedLocationId = std::shared_ptr<const std::string>;

class LocationSpec : public Jsonizable {
public:
    LocationSpec();

    LocationSpec(const std::string &name, const std::string &uri) : name_(name), uri_(uri) {}

    LocationSpec(const LocationSpec &) = default;
    LocationSpec &operator=(const LocationSpec &) = default;
    LocationSpec(LocationSpec &&) noexcept = default;
    LocationSpec &operator=(LocationSpec &&) noexcept = default;

    ~LocationSpec() override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "name", name_);
        Put(writer, "uri", uri_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "name", name_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "uri", uri_, std::string(""));
        return true;
    }

    void set_name(const std::string &name) { name_ = name; }
    void set_name(std::string &&name) noexcept { name_ = std::move(name); }
    void set_name_view(std::string_view name) { name_.assign(name.data(), name.size()); }
    void set_uri(const std::string &uri) { uri_ = uri; }
    void set_uri(std::string &&uri) noexcept { uri_ = std::move(uri); }

    inline const std::string &name() const { return name_; }
    inline const std::string &uri() const { return uri_; }

private:
    std::string name_; // 对应LocationSpecInfo中的name
    std::string uri_;  // URI
};

enum CacheLocationStatus : int32_t {
    CLS_NOT_FOUND = 0,
    CLS_NEW = 1,
    CLS_WRITING = 2,
    CLS_SERVING = 3,
    CLS_DELETING = 4,
};

// Persistent ownership fence for an asynchronous migration target.  The guard lives
// in the CacheLocation JSON so both Reclaimer and the background GC can make the
// correct decision after a Manager restart, when the process-local active task table
// is no longer available.
enum class MigrationCopyGuardState : int32_t {
    MCGS_NONE = 0,
    MCGS_SUBMITTING = 1,
    MCGS_ACTIVE = 2,
    MCGS_CANCELLING = 3,
    MCGS_UNKNOWN = 4,
};

class MigrationCopyGuard : public Jsonizable {
public:
    static constexpr uint32_t kCurrentSchemaVersion = 1;

    MigrationCopyGuard() = default;
    ~MigrationCopyGuard() override = default;

    // A serialized guard is a safety fence, not an optional best-effort hint.
    // Only the exact schema understood by this binary and a known state may be
    // interpreted.  CacheLocation::FromRapidValue rejects an explicitly
    // present but invalid/future guard, so callers fail closed instead of
    // silently treating the target as an ordinary WRITING location.
    bool present() const {
        return schema_version_ == kCurrentSchemaVersion && IsKnownState(state_) && !operation_id_.empty();
    }
    uint32_t schema_version() const { return schema_version_; }
    MigrationCopyGuardState state() const { return state_; }
    const std::string &operation_id() const { return operation_id_; }
    const std::string &source_location_id() const { return source_location_id_; }
    int64_t source_location_create_time() const { return source_location_create_time_; }
    const std::string &source_storage_name() const { return source_storage_name_; }
    const std::string &target_storage_name() const { return target_storage_name_; }
    int32_t migration_retention() const { return migration_retention_; }
    const std::string &mark_target() const { return mark_target_; }
    int64_t mark_deadline_ms() const { return mark_deadline_ms_; }
    uint64_t total_bytes() const { return total_bytes_; }
    const std::vector<std::string> &backend_task_ids() const { return backend_task_ids_; }
    int64_t create_time_us() const { return create_time_us_; }
    int64_t update_time_us() const { return update_time_us_; }
    const std::string &last_error() const { return last_error_; }

    void set_schema_version(uint32_t value) { schema_version_ = value; }
    void set_state(MigrationCopyGuardState value) { state_ = value; }
    void set_operation_id(std::string value) { operation_id_ = std::move(value); }
    void set_source_location_id(std::string value) { source_location_id_ = std::move(value); }
    void set_source_location_create_time(int64_t value) { source_location_create_time_ = value; }
    void set_source_storage_name(std::string value) { source_storage_name_ = std::move(value); }
    void set_target_storage_name(std::string value) { target_storage_name_ = std::move(value); }
    void set_migration_retention(int32_t value) { migration_retention_ = value; }
    void set_mark_target(std::string value) { mark_target_ = std::move(value); }
    void set_mark_deadline_ms(int64_t value) { mark_deadline_ms_ = value; }
    void set_total_bytes(uint64_t value) { total_bytes_ = value; }
    void set_backend_task_ids(std::vector<std::string> value) { backend_task_ids_ = std::move(value); }
    void set_create_time_us(int64_t value) { create_time_us_ = value; }
    void set_update_time_us(int64_t value) { update_time_us_ = value; }
    void set_last_error(std::string value) { last_error_ = std::move(value); }

    static bool IsKnownState(MigrationCopyGuardState state) {
        return state == MigrationCopyGuardState::MCGS_SUBMITTING ||
               state == MigrationCopyGuardState::MCGS_ACTIVE ||
               state == MigrationCopyGuardState::MCGS_CANCELLING ||
               state == MigrationCopyGuardState::MCGS_UNKNOWN;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "schema_version", schema_version_);
        Put(writer, "state", state_);
        Put(writer, "operation_id", operation_id_);
        Put(writer, "source_location_id", source_location_id_);
        Put(writer, "source_location_create_time", source_location_create_time_);
        Put(writer, "source_storage_name", source_storage_name_);
        Put(writer, "target_storage_name", target_storage_name_);
        Put(writer, "migration_retention", migration_retention_);
        Put(writer, "mark_target", mark_target_);
        Put(writer, "mark_deadline_ms", mark_deadline_ms_);
        Put(writer, "total_bytes", total_bytes_);
        Put(writer, "backend_task_ids", backend_task_ids_);
        Put(writer, "create_time_us", create_time_us_);
        Put(writer, "update_time_us", update_time_us_);
        Put(writer, "last_error", last_error_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "schema_version", schema_version_, uint32_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "state", state_, MigrationCopyGuardState::MCGS_NONE);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "operation_id", operation_id_, std::string());
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "source_location_id", source_location_id_, std::string());
        KVCM_JSON_GET_DEFAULT_MACRO(
            rapid_value, "source_location_create_time", source_location_create_time_, int64_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "source_storage_name", source_storage_name_, std::string());
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "target_storage_name", target_storage_name_, std::string());
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "migration_retention", migration_retention_, int32_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "mark_target", mark_target_, std::string());
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "mark_deadline_ms", mark_deadline_ms_, int64_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "total_bytes", total_bytes_, uint64_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(
            rapid_value, "backend_task_ids", backend_task_ids_, std::vector<std::string>());
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "create_time_us", create_time_us_, int64_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "update_time_us", update_time_us_, int64_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "last_error", last_error_, std::string());
        return true;
    }

private:
    uint32_t schema_version_ = 0;
    MigrationCopyGuardState state_ = MigrationCopyGuardState::MCGS_NONE;
    std::string operation_id_;
    std::string source_location_id_;
    int64_t source_location_create_time_ = 0;
    std::string source_storage_name_;
    std::string target_storage_name_;
    // Stored as the public enum's numeric value to keep meta independent from
    // config headers.  Unknown/missing values recover conservatively as
    // KEEP_BOTH.
    int32_t migration_retention_ = 0;
    std::string mark_target_;
    int64_t mark_deadline_ms_ = 0;
    uint64_t total_bytes_ = 0;
    std::vector<std::string> backend_task_ids_;
    int64_t create_time_us_ = 0;
    int64_t update_time_us_ = 0;
    std::string last_error_;
};

using BlockMaskVector = std::vector<bool>;
using BlockMaskOffset = size_t;

using BlockMask = std::variant<BlockMaskVector, BlockMaskOffset>;
bool IsIndexInMaskRange(const BlockMask &mask, size_t index);
bool IsBlockMaskValid(const BlockMask &mask, size_t size);
inline void
PutBlockMask(rapidjson::Writer<rapidjson::StringBuffer> &writer, const std::string &key, BlockMask block_mask) {
    if (std::holds_alternative<BlockMaskVector>(block_mask)) {
        const auto &mask_vector = std::get<BlockMaskVector>(block_mask);
        writer.Key(key.c_str(), key.size(), false);
        writer.StartArray();
        for (const auto &val : mask_vector) {
            writer.Bool(val);
        }
        writer.EndArray();
    } else if (std::holds_alternative<BlockMaskOffset>(block_mask)) {
        const auto &mask_offset = std::get<BlockMaskOffset>(block_mask);
        writer.Key(key.c_str(), key.size(), false);
        writer.Int64(mask_offset);
    }
}
class CacheLocation : public Jsonizable {
public:
    CacheLocation();
    CacheLocation(const CacheLocation &other);
    CacheLocation &operator=(const CacheLocation &) = default;
    CacheLocation(CacheLocation &&) noexcept = default;
    CacheLocation &operator=(CacheLocation &&) noexcept = default;
    CacheLocation(DataStorageType type, size_t spec_size, const std::vector<LocationSpec> &location_specs);
    CacheLocation(const std::string &id,
                  CacheLocationStatus status,
                  DataStorageType type,
                  size_t spec_size,
                  const std::vector<LocationSpec> &location_specs);
    ~CacheLocation() override;

    static std::string CacheLocationStatusToString(CacheLocationStatus status) {
        switch (status) {
        case CLS_NOT_FOUND:
            return "CLS_NOT_FOUND";
        case CLS_NEW:
            return "CLS_NEW";
        case CLS_WRITING:
            return "CLS_WRITING";
        case CLS_SERVING:
            return "CLS_SERVING";
        case CLS_DELETING:
            return "CLS_DELETING";
        default:
            return "CLS_INVALID";
        }
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "id", id());
        Put(writer, "status", status_);
        Put(writer, "type", type_);
        Put(writer, "spec_size", spec_size_);
        Put(writer, "create_time", create_time_);
        Put(writer, "location_specs", location_specs_);
        if (migration_copy_guard_.present()) {
            Put(writer, "migration_copy_guard", migration_copy_guard_);
        }
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        std::string id;
        validated_total_size_ = kUnknownValidatedTotalSize;
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "id", id, std::string(""));
        id_ = std::move(id);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "status", status_, CacheLocationStatus::CLS_NOT_FOUND);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "type", type_, DataStorageType::DATA_STORAGE_TYPE_UNKNOWN);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "spec_size", spec_size_, size_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "create_time", create_time_, int64_t{0});
        KVCM_JSON_GET_MACRO(rapid_value, "location_specs", location_specs_);
        const bool serialized_guard_present =
            rapid_value.IsObject() && rapid_value.HasMember("migration_copy_guard");
        KVCM_JSON_GET_DEFAULT_MACRO(
            rapid_value, "migration_copy_guard", migration_copy_guard_, MigrationCopyGuard());
        if (serialized_guard_present && !migration_copy_guard_.present()) {
            // Reject malformed and future-schema fences.  Returning a normal
            // unguarded CacheLocation here would let old maintenance paths
            // reclaim a target whose remote Copy outcome is still unknown.
            return false;
        }
        return true;
    }

    void set_status(CacheLocationStatus status) { status_ = status; }
    void set_type(DataStorageType type) { type_ = type; }
    void set_id(const std::string &id) { id_ = id; }
    void set_id(std::string &&id) noexcept { id_ = std::move(id); }
    void set_id(InternedLocationId id) noexcept {
        if (id) {
            id_ = std::move(id);
        } else {
            id_ = std::string{};
        }
    }
    void set_spec_size(size_t spec_size) { spec_size_ = spec_size; }
    void set_create_time(int64_t create_time) { create_time_ = create_time; }
    void push_location_spec(LocationSpec &&location_spec) {
        validated_total_size_ = kUnknownValidatedTotalSize;
        location_specs_.push_back(std::move(location_spec));
    }
    void set_location_specs(std::vector<LocationSpec> &&location_specs) {
        validated_total_size_ = kUnknownValidatedTotalSize;
        location_specs_ = std::move(location_specs);
    }
    void set_validated_total_size(std::uint64_t size) noexcept { validated_total_size_ = size; }
    [[nodiscard]] bool GetValidatedTotalSize(std::uint64_t &size) const noexcept {
        if (validated_total_size_ == kUnknownValidatedTotalSize) {
            return false;
        }
        size = validated_total_size_;
        return true;
    }
    [[nodiscard]] bool HasValidatedLocationSpecs() const noexcept {
        return validated_total_size_ != kUnknownValidatedTotalSize;
    }
    void set_migration_copy_guard(const MigrationCopyGuard &guard) { migration_copy_guard_ = guard; }
    void clear_migration_copy_guard() { migration_copy_guard_ = MigrationCopyGuard(); }

    [[nodiscard]] const std::vector<LocationSpec> &location_specs() const { return location_specs_; }
    [[nodiscard]] std::vector<LocationSpec> &mutable_location_specs() {
        validated_total_size_ = kUnknownValidatedTotalSize;
        return location_specs_;
    }
    [[nodiscard]] const std::string &id() const {
        if (const auto *owned = std::get_if<std::string>(&id_)) {
            return *owned;
        }
        const auto &interned = std::get<InternedLocationId>(id_);
        if (interned) {
            return *interned;
        }
        static const std::string empty;
        return empty;
    }
    [[nodiscard]] CacheLocationStatus status() const { return status_; }
    [[nodiscard]] DataStorageType type() const { return type_; }
    [[nodiscard]] size_t spec_size() const { return spec_size_; }
    [[nodiscard]] int64_t create_time() const { return create_time_; }
    [[nodiscard]] bool has_migration_copy_guard() const { return migration_copy_guard_.present(); }
    [[nodiscard]] const MigrationCopyGuard &migration_copy_guard() const { return migration_copy_guard_; }
    [[nodiscard]] size_t EstimateMemUsage() const {
        size_t usage = sizeof(CacheLocation) + id().size();
        for (const auto &spec : location_specs_) {
            usage += sizeof(LocationSpec) + spec.name().size() + spec.uri().size();
        }
        if (migration_copy_guard_.present()) {
            // sizeof(CacheLocation) already includes the value member
            // MigrationCopyGuard (including its vector object).  Count only
            // heap-owned strings and the vector's dynamically allocated string
            // elements here.
            usage += migration_copy_guard_.operation_id().size() + migration_copy_guard_.source_location_id().size() +
                     migration_copy_guard_.source_storage_name().size() +
                     migration_copy_guard_.target_storage_name().size() +
                     migration_copy_guard_.mark_target().size() + migration_copy_guard_.last_error().size();
            usage += migration_copy_guard_.backend_task_ids().size() * sizeof(std::string);
            for (const auto &task_id : migration_copy_guard_.backend_task_ids()) {
                usage += task_id.size();
            }
        }
        return usage;
    }

private:
    static constexpr std::uint64_t kUnknownValidatedTotalSize = std::numeric_limits<std::uint64_t>::max();

    std::variant<std::string, InternedLocationId> id_;
    CacheLocationStatus status_ = CacheLocationStatus::CLS_NEW;
    DataStorageType type_ = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    size_t spec_size_ = 0;
    int64_t create_time_ = 0;
    std::vector<LocationSpec> location_specs_;
    // Pure-local ReportEvent has already validated every URI and summed its
    // sizes. Keep that proof and aggregate beside the immutable specs so the
    // writer and query paths can skip repeated structural URI parsing. Every
    // specs mutator clears it; merge/delete code may restore it only when all
    // retained specs are also proven valid. It is intentionally not
    // serialized, so recovered values fail closed and validate again.
    std::uint64_t validated_total_size_ = kUnknownValidatedTotalSize;
    MigrationCopyGuard migration_copy_guard_;
};

using CacheLocationConstPtr = std::shared_ptr<const CacheLocation>;
using CacheLocationVector = std::vector<CacheLocationConstPtr>;
using CacheLocationMap = std::unordered_map<std::string, CacheLocationConstPtr>;
using CacheLocationMapVector = std::vector<CacheLocationMap>;

// Compact positional representation for large all-location reads. Keeping one
// vector object per key is disproportionately expensive for GetHostCacheState:
// a million-key request otherwise constructs a million small vectors and, for
// the common one-location case, performs a million heap allocations. Offsets
// keep the same positional contract while all shared_ptr values live in one
// contiguous allocation per backend chunk.
class CacheLocationValueView {
public:
    using const_iterator = CacheLocationVector::const_iterator;

    CacheLocationValueView() = default;
    CacheLocationValueView(const_iterator begin, const_iterator end) : begin_(begin), end_(end) {}

    [[nodiscard]] const_iterator begin() const { return begin_; }
    [[nodiscard]] const_iterator end() const { return end_; }
    [[nodiscard]] bool empty() const { return begin_ == end_; }
    [[nodiscard]] size_t size() const { return static_cast<size_t>(end_ - begin_); }

private:
    const_iterator begin_{};
    const_iterator end_{};
};

struct CompactLocationsPerKey {
    std::vector<size_t> offsets{0};
    CacheLocationVector values;

    void Clear(size_t key_capacity = 0, size_t location_capacity = 0) {
        offsets.clear();
        offsets.reserve(key_capacity + 1);
        offsets.push_back(0);
        values.clear();
        values.reserve(location_capacity);
    }

    [[nodiscard]] size_t size() const { return offsets.empty() ? 0 : offsets.size() - 1; }
    [[nodiscard]] bool empty() const { return size() == 0; }
    [[nodiscard]] bool IsValid(size_t expected_key_count) const {
        if (offsets.size() != expected_key_count + 1 || offsets.empty() || offsets.front() != 0 ||
            offsets.back() != values.size()) {
            return false;
        }
        for (size_t i = 1; i < offsets.size(); ++i) {
            if (offsets[i] < offsets[i - 1]) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] CacheLocationValueView operator[](size_t index) const {
        return CacheLocationValueView(values.begin() + static_cast<ptrdiff_t>(offsets[index]),
                                      values.begin() + static_cast<ptrdiff_t>(offsets[index + 1]));
    }

    void FinishKey() { offsets.push_back(values.size()); }
};

// A target guard pins the exact source identity used by the asynchronous
// operation.  Legacy/malformed schema-v1 guards may have no create_time; fail
// closed in that case and pin by location id rather than allowing one deletion
// path to disagree with another.
inline const MigrationCopyGuard *FindPersistentMigrationSourcePin(const CacheLocation &candidate,
                                                                  const CacheLocationMap &location_map) {
    for (const auto &[_, target] : location_map) {
        if (!target || !target->has_migration_copy_guard()) {
            continue;
        }
        const auto &guard = target->migration_copy_guard();
        if (guard.source_location_id() == candidate.id() &&
            (guard.source_location_create_time() <= 0 ||
             guard.source_location_create_time() == candidate.create_time())) {
            return &guard;
        }
    }
    return nullptr;
}

} // namespace kv_cache_manager
