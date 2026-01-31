#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/data_storage/common_define.h"

namespace kv_cache_manager {

class LocationSpec : public Jsonizable {
public:
    LocationSpec();

    LocationSpec(const std::string &name, const std::string &uri) : name_(name), uri_(uri) {}

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
    void set_uri(const std::string &uri) { uri_ = uri; }

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
        Put(writer, "id", id_);
        Put(writer, "status", status_);
        Put(writer, "type", type_);
        Put(writer, "spec_size", spec_size_);
        Put(writer, "location_specs", location_specs_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "id", id_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "status", status_, CacheLocationStatus::CLS_NOT_FOUND);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "type", type_, DataStorageType::DATA_STORAGE_TYPE_UNKNOWN);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "spec_size", spec_size_, size_t{0});
        KVCM_JSON_GET_MACRO(rapid_value, "location_specs", location_specs_);
        return true;
    }

    void set_status(CacheLocationStatus status) { status_ = status; }
    void set_type(DataStorageType type) { type_ = type; }
    void set_id(const std::string &id) { id_ = id; }
    void set_spec_size(size_t spec_size) { spec_size_ = spec_size; }
    void push_location_spec(LocationSpec &&location_spec) { location_specs_.push_back(std::move(location_spec)); }
    void set_location_specs(std::vector<LocationSpec> &&location_specs) { location_specs_ = location_specs; }

    [[nodiscard]] const std::vector<LocationSpec> &location_specs() const { return location_specs_; }
    [[nodiscard]] const std::string &id() const { return id_; }
    [[nodiscard]] CacheLocationStatus status() const { return status_; }
    [[nodiscard]] DataStorageType type() const { return type_; }
    [[nodiscard]] size_t spec_size() const { return spec_size_; }

private:
    std::string id_;
    CacheLocationStatus status_ = CacheLocationStatus::CLS_NEW;
    DataStorageType type_;
    size_t spec_size_ = 0;
    std::vector<LocationSpec> location_specs_;
};

using CacheLocationVector = std::vector<CacheLocation>;
using CacheLocationMap = std::unordered_map<std::string, CacheLocation>;

class BlockCacheLocationsMeta : public Jsonizable {
public:
    BlockCacheLocationsMeta();
    ~BlockCacheLocationsMeta() override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        for (auto &location_kv : location_map_) {
            Put(writer, location_kv.first, location_kv.second);
        }
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        if (!rapid_value.IsObject()) {
            return false;
        }
        for (auto itr = rapid_value.MemberBegin(); itr != rapid_value.MemberEnd(); ++itr) {
            const std::string key = itr->name.GetString();
            CacheLocation location;
            if (location.FromRapidValue(itr->value)) {
                location_map_[key] = location;
            } else {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] CacheLocationMap &location_map() { return location_map_; }

    void AddNewLocation(const CacheLocation &location, std::string &out_location_id);
    ErrorCode UpdateLocationStatus(const std::string &location_id, CacheLocationStatus status);
    ErrorCode DeleteLocation(const std::string &location_id);
    ErrorCode GetLocationStatus(const std::string &location_id, CacheLocationStatus &out_status);
    size_t GetLocationCount() const;

private:
    CacheLocationMap location_map_;
};

} // namespace kv_cache_manager