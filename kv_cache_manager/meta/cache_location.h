#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
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

// A bounded per-backend dictionary for locations reconstructed from Redis.
// Live entries share their id; weak references and the entry limit avoid
// retaining every distinct id for the backend's lifetime.
class LocationIdInterner {
public:
    InternedLocationId Intern(const std::string &id, bool *retained = nullptr);

private:
    static constexpr size_t kShardCount = 16;
    static constexpr size_t kMaxIdsPerShard = 65536 / kShardCount;
    struct Shard {
        std::mutex mutex;
        std::unordered_map<std::string, std::weak_ptr<const std::string>> ids;
    };
    std::array<Shard, kShardCount> shards_;
};

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
    CacheLocation(const CacheLocation &other) = default;
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
        Put(writer, "spec_size", spec_size());
        Put(writer, "create_time", create_time_);
        Put(writer, "location_specs", location_specs_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override { return FromRapidValueImpl(rapid_value, nullptr); }

    bool FromJsonStringInterned(const std::string &json, LocationIdInterner &interner);
    void set_status(CacheLocationStatus status) { status_ = status; }
    void set_type(DataStorageType type) { type_ = type; }
    void set_id(const std::string &id) {
        id_ = std::make_shared<const std::string>(id);
        id_is_interned_ = false;
    }
    void set_id(std::string &&id) {
        id_ = std::make_shared<const std::string>(std::move(id));
        id_is_interned_ = false;
    }
    void set_id(InternedLocationId id, bool is_interned = true) noexcept {
        id_ = std::move(id);
        id_is_interned_ = is_interned;
    }
    void set_spec_size(size_t spec_size) {
        spec_size_overflow_ = spec_size > kMaxStoredSpecSize;
        const uint64_t stored = std::min(spec_size, static_cast<size_t>(kMaxStoredSpecSize));
        packed_sizes_ = (packed_sizes_ & kValidatedTotalSizeMask) | (stored << kSpecSizeShift);
    }
    void set_create_time(int64_t create_time) { create_time_ = create_time; }
    void push_location_spec(LocationSpec &&location_spec) {
        ClearValidatedTotalSize();
        location_specs_.push_back(std::move(location_spec));
    }
    void set_location_specs(std::vector<LocationSpec> &&location_specs) {
        ClearValidatedTotalSize();
        location_specs_ = std::move(location_specs);
    }
    void set_validated_total_size(std::uint64_t size) noexcept {
        // Large locations are still valid; they simply use the existing URI
        // parse path instead of a lossy cached 48-bit aggregate.
        packed_sizes_ = (packed_sizes_ & ~kValidatedTotalSizeMask) |
                        (size < kUnknownValidatedTotalSize ? size : kUnknownValidatedTotalSize);
    }
    [[nodiscard]] bool GetValidatedTotalSize(std::uint64_t &size) const noexcept {
        if ((packed_sizes_ & kValidatedTotalSizeMask) == kUnknownValidatedTotalSize) {
            return false;
        }
        size = packed_sizes_ & kValidatedTotalSizeMask;
        return true;
    }
    [[nodiscard]] bool HasValidatedLocationSpecs() const noexcept {
        return (packed_sizes_ & kValidatedTotalSizeMask) != kUnknownValidatedTotalSize;
    }

    [[nodiscard]] const std::vector<LocationSpec> &location_specs() const { return location_specs_; }
    [[nodiscard]] std::vector<LocationSpec> &mutable_location_specs() {
        ClearValidatedTotalSize();
        return location_specs_;
    }
    [[nodiscard]] const std::string &id() const {
        if (id_) {
            return *id_;
        }
        static const std::string empty;
        return empty;
    }
    [[nodiscard]] CacheLocationStatus status() const { return status_; }
    [[nodiscard]] DataStorageType type() const { return type_; }
    [[nodiscard]] size_t spec_size() const {
        return spec_size_overflow_ ? location_specs_.size() : packed_sizes_ >> kSpecSizeShift;
    }
    [[nodiscard]] int64_t create_time() const { return create_time_; }
    [[nodiscard]] size_t EstimateMemUsage() const {
        size_t usage = sizeof(CacheLocation) + id().size();
        if (id_ && !id_is_interned_) {
            usage += sizeof(std::string) + sizeof(void *) * 2;
        }
        for (const auto &spec : location_specs_) {
            usage += sizeof(LocationSpec) + spec.name().size() + spec.uri().size();
        }
        return usage;
    }

private:
    bool FromRapidValueImpl(const rapidjson::Value &rapid_value, LocationIdInterner *interner) {
        std::string id;
        ClearValidatedTotalSize();
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "id", id, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "type", type_, DataStorageType::DATA_STORAGE_TYPE_UNKNOWN);
        id_is_interned_ = false;
        id_ = interner && IsEventReportStorageType(type_) ? interner->Intern(id, &id_is_interned_)
                                                         : std::make_shared<const std::string>(std::move(id));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "status", status_, CacheLocationStatus::CLS_NOT_FOUND);
        size_t spec_size = 0;
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "spec_size", spec_size, size_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "create_time", create_time_, int64_t{0});
        KVCM_JSON_GET_MACRO(rapid_value, "location_specs", location_specs_);
        if (spec_size > kMaxStoredSpecSize && spec_size != location_specs_.size()) {
            return false;
        }
        set_spec_size(spec_size);
        return true;
    }

    static constexpr uint64_t kSpecSizeShift = 48;
    static constexpr uint64_t kValidatedTotalSizeMask = (uint64_t{1} << kSpecSizeShift) - 1;
    static constexpr uint64_t kUnknownValidatedTotalSize = kValidatedTotalSizeMask;
    static constexpr uint64_t kMaxStoredSpecSize = std::numeric_limits<uint16_t>::max();
    void ClearValidatedTotalSize() { packed_sizes_ |= kValidatedTotalSizeMask; }

    InternedLocationId id_;
    int64_t create_time_ = 0;
    std::vector<LocationSpec> location_specs_;
    // Pure-local ReportEvent has already validated every URI and summed its
    // sizes. Keep that proof and aggregate with the 16-bit spec count so the
    // writer and query paths can skip repeated structural URI parsing. Every
    // specs mutator clears it; merge/delete code may restore it only when all
    // retained specs are also proven valid. It is intentionally not
    // serialized, so recovered values fail closed and validate again.
    uint64_t packed_sizes_ = kUnknownValidatedTotalSize;
    CacheLocationStatus status_ = CacheLocationStatus::CLS_NEW;
    DataStorageType type_ = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    bool id_is_interned_ = false;
    bool spec_size_overflow_ = false;
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

} // namespace kv_cache_manager
