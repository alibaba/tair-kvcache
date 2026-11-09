#include "kv_cache_manager/meta/cache_location.h"

#include <cstring>
#include <stdexcept>

#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/meta/common.h"
namespace kv_cache_manager {

namespace {
const SharedLocationSpecName &EmptySpecName() noexcept {
    static const std::string empty;
    static const SharedLocationSpecName handle(std::shared_ptr<const std::string>{}, &empty);
    return handle;
}
} // namespace

LocationSpec::LocationSpec() : name_(EmptySpecName()) {}
LocationSpec::LocationSpec(const LocationSpec &other) : name_(other.name_), uri_(other.uri_) {
    set_checksum(other.checksum());
}

LocationSpec &LocationSpec::operator=(const LocationSpec &other) {
    if (this != &other) {
        LocationSpec copy(other);
        *this = std::move(copy);
    }
    return *this;
}

LocationSpec::LocationSpec(LocationSpec &&other) noexcept
    : name_(std::move(other.name_))
    , uri_(std::move(other.uri_))
    , checksum_(std::move(other.checksum_))
    , checksum_size_(std::exchange(other.checksum_size_, 0)) {
    other.name_ = EmptySpecName();
}

LocationSpec &LocationSpec::operator=(LocationSpec &&other) noexcept {
    if (this != &other) {
        name_ = std::move(other.name_);
        uri_ = std::move(other.uri_);
        checksum_ = std::move(other.checksum_);
        checksum_size_ = std::exchange(other.checksum_size_, 0);
        other.name_ = EmptySpecName();
    }
    return *this;
}

LocationSpec::~LocationSpec() = default;

bool LocationSpec::FromRapidValueInterned(const rapidjson::Value &rapid_value, LocationSpecNameInterner &interner) {
    std::string name;
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "name", name, std::string(""));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "uri", uri_, std::string(""));
    name_ = interner.Intern(name);
    return ParseChecksum(rapid_value);
}

SharedLocationSpecName LocationSpecNameInterner::Intern(const std::string &name) {
    {
        std::shared_lock lock(mutex_);
        auto it = names_.find(name);
        if (it != names_.end()) {
            if (auto live = it->second.lock()) {
                return live;
            }
        }
    }
    std::unique_lock lock(mutex_);
    auto it = names_.find(name);
    if (it != names_.end()) {
        if (auto live = it->second.lock()) {
            return live;
        }
        names_.erase(it);
    }
    if (names_.size() >= kMaxEntries) {
        lock.unlock();
        return std::make_shared<const std::string>(name);
    }
    auto shared = std::make_shared<const std::string>(name);
    names_.emplace(*shared, shared);
    return shared;
}

void LocationSpec::set_checksum(std::string_view checksum) {
    if (checksum.empty()) {
        clear_checksum();
        return;
    }
    if (checksum.size() == std::numeric_limits<size_t>::max()) {
        throw std::length_error("checksum is too large");
    }
    auto buffer = std::make_unique<char[]>(checksum.size() + 1);
    std::memcpy(buffer.get(), checksum.data(), checksum.size());
    buffer[checksum.size()] = '\0';
    // Allocate and copy first: checksum may refer to our existing buffer.
    checksum_ = std::move(buffer);
    checksum_size_ = checksum.size();
}

bool IsIndexInMaskRange(const BlockMask &mask, size_t index) {
    if (std::holds_alternative<BlockMaskVector>(mask)) {
        const auto &mask_vector = std::get<BlockMaskVector>(mask);
        if (index < mask_vector.size()) {
            return mask_vector[index];
        }
        return false;
    } else if (std::holds_alternative<BlockMaskOffset>(mask)) {
        const auto &mask_offset = std::get<BlockMaskOffset>(mask);
        return index < mask_offset;
    }
    return false;
}
bool IsBlockMaskValid(const BlockMask &mask, size_t size) {
    if (std::holds_alternative<BlockMaskVector>(mask)) {
        const auto &mask_vector = std::get<BlockMaskVector>(mask);
        return mask_vector.size() == size;
    } else if (std::holds_alternative<BlockMaskOffset>(mask)) {
        const auto &mask_offset = std::get<BlockMaskOffset>(mask);
        return mask_offset <= size;
    }
    return false;
}

CacheLocation::CacheLocation() = default;

bool CacheLocation::FromJsonStringInterned(const std::string &json,
                                           LocationIdInterner &id_interner,
                                           LocationSpecNameInterner *name_interner) {
    rapidjson::Document doc;
    return Parse(doc, json) && FromRapidValueImpl(doc, &id_interner, name_interner);
}

InternedLocationId LocationIdInterner::Intern(const std::string &id, bool *retained) {
    auto &shard = shards_[std::hash<std::string>{}(id) & (kShardCount - 1)];
    std::lock_guard lock(shard.mutex);
    auto it = shard.ids.find(id);
    if (it != shard.ids.end()) {
        if (auto live = it->second.lock()) {
            if (retained) {
                *retained = true;
            }
            return live;
        }
    }
    auto shared = std::make_shared<const std::string>(id);
    const bool keep = it != shard.ids.end() || shard.ids.size() < kMaxIdsPerShard;
    if (it != shard.ids.end()) {
        it->second = shared;
    } else if (keep) {
        shard.ids.emplace(*shared, shared);
    }
    if (retained) {
        *retained = keep;
    }
    return shared;
}

CacheLocation::CacheLocation(DataStorageType type, size_t spec_size, const std::vector<LocationSpec> &location_specs)
    : location_specs_(location_specs), type_(type) {
    set_spec_size(spec_size);
}

CacheLocation::CacheLocation(const std::string &id,
                             CacheLocationStatus status,
                             DataStorageType type,
                             size_t spec_size,
                             const std::vector<LocationSpec> &location_specs)
    : id_(std::make_shared<const std::string>(id)), location_specs_(location_specs), status_(status), type_(type) {
    set_spec_size(spec_size);
}

CacheLocation::~CacheLocation() = default;

} // namespace kv_cache_manager
