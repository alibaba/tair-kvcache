#pragma once

#include <atomic>
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
    CacheLocation &operator=(const CacheLocation &other);
    CacheLocation(CacheLocation &&other) noexcept;
    CacheLocation &operator=(CacheLocation &&other) noexcept;
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
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        std::string id;
        InvalidateEstimatedMemUsage();
        validated_total_size_ = kUnknownValidatedTotalSize;
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "id", id, std::string(""));
        id_ = std::move(id);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "status", status_, CacheLocationStatus::CLS_NOT_FOUND);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "type", type_, DataStorageType::DATA_STORAGE_TYPE_UNKNOWN);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "spec_size", spec_size_, size_t{0});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "create_time", create_time_, int64_t{0});
        KVCM_JSON_GET_MACRO(rapid_value, "location_specs", location_specs_);
        return true;
    }

    void set_status(CacheLocationStatus status) { status_ = status; }
    void set_type(DataStorageType type) { type_ = type; }
    void set_id(const std::string &id) {
        InvalidateEstimatedMemUsage();
        id_ = id;
    }
    void set_id(std::string &&id) noexcept {
        InvalidateEstimatedMemUsage();
        id_ = std::move(id);
    }
    void set_id(InternedLocationId id) noexcept {
        InvalidateEstimatedMemUsage();
        if (id) {
            id_ = std::move(id);
        } else {
            id_ = std::string{};
        }
    }
    void set_spec_size(size_t spec_size) { spec_size_ = spec_size; }
    void set_create_time(int64_t create_time) { create_time_ = create_time; }
    void push_location_spec(LocationSpec &&location_spec) {
        InvalidateEstimatedMemUsage();
        validated_total_size_ = kUnknownValidatedTotalSize;
        location_specs_.push_back(std::move(location_spec));
    }
    void set_location_specs(std::vector<LocationSpec> &&location_specs) {
        InvalidateEstimatedMemUsage();
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

    [[nodiscard]] const std::vector<LocationSpec> &location_specs() const { return location_specs_; }
    [[nodiscard]] std::vector<LocationSpec> &mutable_location_specs() {
        estimated_mem_usage_state_.store(kUncacheableEstimatedMemUsage, std::memory_order_relaxed);
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
    [[nodiscard]] size_t EstimateMemUsage() const {
        const size_t cached = estimated_mem_usage_state_.load(std::memory_order_relaxed);
        if (cached > kUncacheableEstimatedMemUsage) {
            return cached;
        }
        size_t usage = sizeof(CacheLocation) + id().size();
        for (const auto &spec : location_specs_) {
            usage += sizeof(LocationSpec) + spec.name().size() + spec.uri().size();
        }
        if (cached == kUnknownEstimatedMemUsage) {
            estimated_mem_usage_state_.store(usage, std::memory_order_relaxed);
        }
        return usage;
    }

private:
    static constexpr std::uint64_t kUnknownValidatedTotalSize = std::numeric_limits<std::uint64_t>::max();
    static constexpr size_t kUnknownEstimatedMemUsage = 0;
    static constexpr size_t kUncacheableEstimatedMemUsage = 1;

    void InvalidateEstimatedMemUsage() noexcept {
        if (estimated_mem_usage_state_.load(std::memory_order_relaxed) != kUncacheableEstimatedMemUsage) {
            estimated_mem_usage_state_.store(kUnknownEstimatedMemUsage, std::memory_order_relaxed);
        }
    }

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
    mutable std::atomic<size_t> estimated_mem_usage_state_{kUnknownEstimatedMemUsage};
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
