#include "kv_cache_manager/meta/simple/simple_kv_cache_item.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <new>
#include <string_view>

#include "kv_cache_manager/common/cache/allocation_size.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"

namespace kv_cache_manager {
namespace {

ErrorCode ValidateLocations(const CacheLocationMap &locations,
                            const SharedLocationSpecName &spec_name,
                            const std::unordered_map<LocationId, InternedLocationId> &interned_ids) {
    if (!spec_name || spec_name->empty()) {
        return EC_BADARGS;
    }
    for (const CacheLocationMap::value_type &entry : locations) {
        const CacheLocationConstPtr &location = entry.second;
        if (!location || entry.first != location->id() || location->status() != CLS_SERVING ||
            location->spec_size() != 1 || location->location_specs().size() != 1) {
            return EC_BADARGS;
        }
        std::string_view storage_type;
        std::string_view medium;
        std::string_view host;
        if (!SnapshotUriUtils::ParseEventReportLocationIdView(entry.first, storage_type, medium, host)) {
            return EC_BADARGS;
        }
        const DataStorageType type = storage_type == "event_report_l1p5"
                                         ? DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5
                                         : DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2;
        if (location->type() != type) {
            return EC_BADARGS;
        }
        const std::unordered_map<LocationId, InternedLocationId>::const_iterator id = interned_ids.find(entry.first);
        if (id == interned_ids.end() || !id->second || *id->second != entry.first) {
            return EC_BADARGS;
        }
        const LocationSpec &spec = location->location_specs().front();
        if (!spec.name_handle() || spec.name() != *spec_name) {
            return EC_BADARGS;
        }
        if (spec.uri().size() > std::numeric_limits<uint32_t>::max()) {
            return EC_OUT_OF_RANGE;
        }
        if (!location->HasValidatedLocationSpecs()) {
            StandardUri parsed;
            if (!parsed.Parse(spec.uri()) || !parsed.Valid()) {
                return EC_BADARGS;
            }
        }
    }
    return EC_OK;
}

} // namespace

SimpleKVCacheItem::SimpleKVCacheItem() noexcept = default;

SimpleKVCacheItem::~SimpleKVCacheItem() {
    for (size_t i = 0; i < location_count_; ++i) {
        Record(i).~LocationRecord();
    }
}

ErrorCode SimpleKVCacheItem::Allocate(size_t location_count, Cache::ObjectPtr &out_item) {
    static_assert(sizeof(SimpleKVCacheItem) % alignof(LocationRecord) == 0, "Trailing records must be aligned");
    constexpr size_t kMaxBytes = std::numeric_limits<uint32_t>::max();
    if (location_count > (kMaxBytes - sizeof(SimpleKVCacheItem)) / sizeof(LocationRecord)) {
        return EC_OUT_OF_RANGE;
    }
    const size_t bytes = sizeof(SimpleKVCacheItem) + location_count * sizeof(LocationRecord);
    const size_t charge = EstimateAllocationSize(bytes);
    if (charge > kMaxBytes) {
        return EC_OUT_OF_RANGE;
    }
    SimpleKVCacheItem *item = new (::operator new(bytes)) SimpleKVCacheItem();
    item->charge_bytes_ = static_cast<uint32_t>(charge);
    out_item = item;
    return EC_OK;
}

ErrorCode SimpleKVCacheItem::Create(const CacheLocationMap &locations,
                                    const SharedLocationSpecName &spec_name,
                                    const std::unordered_map<LocationId, InternedLocationId> &interned_ids,
                                    Cache::ObjectPtr &out_item) {
    if (out_item != nullptr || locations.empty()) {
        return EC_BADARGS;
    }
    const ErrorCode validation_ec = ValidateLocations(locations, spec_name, interned_ids);
    if (validation_ec != EC_OK) {
        return validation_ec;
    }
    Cache::ObjectPtr value = nullptr;
    const ErrorCode allocation_ec = Allocate(locations.size(), value);
    if (allocation_ec != EC_OK) {
        return allocation_ec;
    }
    // 原 Cache 接收裸块；移交前用局部 RAII 接管，异常或校验失败均释放已构造记录。
    const auto destroy = [](SimpleKVCacheItem *item) { Deleter(item, nullptr); };
    std::unique_ptr<SimpleKVCacheItem, decltype(destroy)> item(static_cast<SimpleKVCacheItem *>(value), destroy);
    for (const CacheLocationMap::value_type &entry : locations) {
        auto data = std::make_shared<const LocationData>(
            LocationData{interned_ids.find(entry.first)->second, entry.second->location_specs().front().uri()});
        const ErrorCode ec = item->AppendLocation(data, entry.second->create_time());
        if (ec != EC_OK) {
            return ec;
        }
    }
    out_item = item.release();
    return EC_OK;
}

ErrorCode SimpleKVCacheItem::Merge(const SimpleKVCacheItem &current,
                                   const CacheLocationMap &updates,
                                   const SharedLocationSpecName &spec_name,
                                   const std::unordered_map<LocationId, InternedLocationId> &interned_ids,
                                   Cache::ObjectPtr &out_item) {
    if (out_item != nullptr) {
        return EC_BADARGS;
    }
    const ErrorCode validation_ec = ValidateLocations(updates, spec_name, interned_ids);
    if (validation_ec != EC_OK) {
        return validation_ec;
    }
    size_t retained_count = 0;
    for (size_t i = 0; i < current.location_count_; ++i) {
        if (updates.find(*current.Record(i).data->id) == updates.end()) {
            ++retained_count;
        }
    }
    constexpr size_t kMaxCount =
        (std::numeric_limits<uint32_t>::max() - sizeof(SimpleKVCacheItem)) / sizeof(LocationRecord);
    if (updates.size() > kMaxCount - retained_count) {
        return EC_OUT_OF_RANGE;
    }
    Cache::ObjectPtr value = nullptr;
    const ErrorCode allocation_ec = Allocate(retained_count + updates.size(), value);
    if (allocation_ec != EC_OK) {
        return allocation_ec;
    }
    const auto destroy = [](SimpleKVCacheItem *item) { Deleter(item, nullptr); };
    std::unique_ptr<SimpleKVCacheItem, decltype(destroy)> item(static_cast<SimpleKVCacheItem *>(value), destroy);
    for (size_t i = 0; i < current.location_count_; ++i) {
        const LocationRecord &record = current.Record(i);
        if (updates.find(*record.data->id) != updates.end()) {
            continue;
        }
        const ErrorCode ec = item->AppendLocation(record.data, record.create_time);
        if (ec != EC_OK) {
            return ec;
        }
    }
    for (const CacheLocationMap::value_type &entry : updates) {
        const std::string &uri = entry.second->location_specs().front().uri();
        std::shared_ptr<const LocationData> data;
        size_t index = 0;
        if (current.FindLocationIndex(entry.first, index) && current.Record(index).data->uri == uri) {
            data = current.Record(index).data;
        } else {
            data = std::make_shared<const LocationData>(LocationData{interned_ids.find(entry.first)->second, uri});
        }
        const ErrorCode ec = item->AppendLocation(data, entry.second->create_time());
        if (ec != EC_OK) {
            return ec;
        }
    }
    out_item = item.release();
    return EC_OK;
}

ErrorCode SimpleKVCacheItem::Remove(const SimpleKVCacheItem &current,
                                    const LocationIdVector &location_ids,
                                    Cache::ObjectPtr &out_item) {
    if (out_item != nullptr) {
        return EC_BADARGS;
    }
    size_t retained_count = 0;
    for (size_t i = 0; i < current.location_count_; ++i) {
        const LocationId &id = *current.Record(i).data->id;
        if (std::find(location_ids.begin(), location_ids.end(), id) == location_ids.end()) {
            ++retained_count;
        }
    }
    Cache::ObjectPtr value = nullptr;
    const ErrorCode allocation_ec = Allocate(retained_count, value);
    if (allocation_ec != EC_OK) {
        return allocation_ec;
    }
    const auto destroy = [](SimpleKVCacheItem *item) { Deleter(item, nullptr); };
    std::unique_ptr<SimpleKVCacheItem, decltype(destroy)> item(static_cast<SimpleKVCacheItem *>(value), destroy);
    for (size_t i = 0; i < current.location_count_; ++i) {
        const LocationRecord &record = current.Record(i);
        if (std::find(location_ids.begin(), location_ids.end(), *record.data->id) != location_ids.end()) {
            continue;
        }
        const ErrorCode ec = item->AppendLocation(record.data, record.create_time);
        if (ec != EC_OK) {
            return ec;
        }
    }
    out_item = item.release();
    return EC_OK;
}

void SimpleKVCacheItem::Deleter(Cache::ObjectPtr value, MemoryAllocator * /*allocator*/) {
    SimpleKVCacheItem *item = static_cast<SimpleKVCacheItem *>(value);
    if (item != nullptr) {
        item->~SimpleKVCacheItem();
        // Create 使用全局 operator new，与 LRU 自身的 allocator 无关。
        ::operator delete(item);
    }
}

const SimpleKVCacheItem::LocationRecord &SimpleKVCacheItem::Record(size_t index) const noexcept {
    const char *address =
        reinterpret_cast<const char *>(this) + sizeof(SimpleKVCacheItem) + index * sizeof(LocationRecord);
    return *std::launder(reinterpret_cast<const LocationRecord *>(address));
}

bool SimpleKVCacheItem::FindLocationIndex(const LocationId &location_id, size_t &out_index) const noexcept {
    for (size_t i = 0; i < location_count_; ++i) {
        if (*Record(i).data->id == location_id) {
            out_index = i;
            return true;
        }
    }
    return false;
}

size_t SimpleKVCacheItem::EstimateDataCharge(size_t uri_capacity) {
    static const size_t kInlineStringCapacity = std::string().capacity();
    return EstimateAllocationSize(sizeof(LocationData) + sizeof(void *) * 2) +
           (uri_capacity > kInlineStringCapacity ? EstimateAllocationSize(uri_capacity + 1) : 0);
}

ErrorCode SimpleKVCacheItem::EstimateWriteGrowth(const std::shared_ptr<SimpleKVCacheItem> &current,
                                                 const CacheLocationMap &updates,
                                                 bool replace,
                                                 size_t &out_growth) {
    size_t next_count = current && !replace ? current->LocationCount() : 0;
    size_t after =
        current && !replace
            ? current->Size() - EstimateAllocationSize(sizeof(SimpleKVCacheItem) + next_count * sizeof(LocationRecord))
            : 0;
    for (const CacheLocationMap::value_type &entry : updates) {
        if (!entry.second || entry.second->location_specs().size() != 1) {
            return EC_BADARGS;
        }
        const std::string &uri = entry.second->location_specs().front().uri();
        size_t index = 0;
        if (current && !replace && current->FindLocationIndex(entry.first, index)) {
            const std::string &previous = current->Record(index).data->uri;
            if (previous == uri) {
                continue;
            }
            after -= EstimateDataCharge(previous.capacity());
        } else {
            ++next_count;
        }
        after += EstimateDataCharge(uri.size());
    }
    after += EstimateAllocationSize(sizeof(SimpleKVCacheItem) + next_count * sizeof(LocationRecord));
    const size_t before = current ? current->Size() : 0;
    out_growth = after > before ? after - before : 0;
    return EC_OK;
}

ErrorCode SimpleKVCacheItem::AppendLocation(const std::shared_ptr<const LocationData> &data,
                                            int64_t create_time) noexcept {
    constexpr size_t kMaxBytes = std::numeric_limits<uint32_t>::max();
    // 共享数据及控制块按每条引用保守计费，避免外置 URI 后 LRU charge 漏掉大部分内存。
    const size_t data_charge = EstimateDataCharge(data->uri.capacity());
    if (data_charge > kMaxBytes - charge_bytes_) {
        return EC_OUT_OF_RANGE;
    }
    void *address =
        reinterpret_cast<char *>(this) + sizeof(SimpleKVCacheItem) + location_count_ * sizeof(LocationRecord);
    new (address) LocationRecord{data, create_time};
    ++location_count_;
    charge_bytes_ += static_cast<uint32_t>(data_charge);
    return EC_OK;
}

CacheLocationConstPtr SimpleKVCacheItem::ProjectLocation(const LocationRecord &record,
                                                         const SharedLocationSpecName &spec_name) const {
    auto location = std::make_shared<CacheLocation>();
    location->set_id(record.data->id);
    // 所有记录入块前已校验完整 ID，类型可由其固定前缀还原。
    constexpr std::string_view kL1p5Prefix = "kvs#event_report_l1p5#";
    location->set_type(record.data->id->compare(0, kL1p5Prefix.size(), kL1p5Prefix) == 0
                           ? DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5
                           : DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    location->set_status(CLS_SERVING);
    location->set_spec_size(1);
    location->set_create_time(record.create_time);
    location->push_location_spec(LocationSpec(spec_name, record.data->uri));
    return location;
}

ErrorCode SimpleKVCacheItem::GetLocationValues(const SharedLocationSpecName &spec_name,
                                               CacheLocationVector &out_locations) const {
    CacheLocationVector locations;
    locations.reserve(location_count_);
    const ErrorCode ec = AppendLocationValues(spec_name, locations);
    if (ec == EC_OK) {
        out_locations.swap(locations);
    }
    return ec;
}

ErrorCode SimpleKVCacheItem::AppendLocationValues(const SharedLocationSpecName &spec_name,
                                                  CacheLocationVector &out_locations) const {
    if (!spec_name || spec_name->empty()) {
        return EC_BADARGS;
    }
    for (size_t i = 0; i < location_count_; ++i) {
        out_locations.push_back(ProjectLocation(Record(i), spec_name));
    }
    return EC_OK;
}

ErrorCode SimpleKVCacheItem::GetLocations(const SharedLocationSpecName &spec_name,
                                          CacheLocationMap &out_locations) const {
    if (!spec_name || spec_name->empty()) {
        return EC_BADARGS;
    }
    CacheLocationMap locations;
    locations.reserve(location_count_);
    for (size_t i = 0; i < location_count_; ++i) {
        const LocationRecord &record = Record(i);
        [[maybe_unused]] const bool inserted =
            locations.emplace(*record.data->id, ProjectLocation(record, spec_name)).second;
        assert(inserted);
    }
    out_locations.swap(locations);
    return EC_OK;
}

ErrorCode SimpleKVCacheItem::GetLocation(const SharedLocationSpecName &spec_name,
                                         const LocationId &location_id,
                                         CacheLocationConstPtr &out_location) const {
    if (!spec_name || spec_name->empty()) {
        return EC_BADARGS;
    }
    size_t index = 0;
    if (!FindLocationIndex(location_id, index)) {
        return EC_NOENT;
    }
    out_location = ProjectLocation(Record(index), spec_name);
    return EC_OK;
}

ErrorCode SimpleKVCacheItem::GetLocationIds(LocationIdVector &out_location_ids) const {
    LocationIdVector ids;
    ids.reserve(location_count_);
    for (size_t i = 0; i < location_count_; ++i) {
        ids.push_back(*Record(i).data->id);
    }
    out_location_ids.swap(ids);
    return EC_OK;
}

size_t SimpleKVCacheItem::LocationCount() const noexcept { return location_count_; }

size_t SimpleKVCacheItem::Size() const noexcept { return charge_bytes_; }

} // namespace kv_cache_manager
