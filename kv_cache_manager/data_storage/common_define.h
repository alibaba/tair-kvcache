#pragma once

#include <map>
#include <vector>

#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

static const std::string kDefaultStorageName = "nfs_01";
static const DataStorageType kDefaultStorageType = DataStorageType::DATA_STORAGE_TYPE_NFS;

struct Hf3fsDataStorageItem {
    std::string file_path;
    static Hf3fsDataStorageItem FromUri(const DataStorageUri &uri);
};

struct VcnsHf3fsDataStorageItem {
    std::string file_path;
    static VcnsHf3fsDataStorageItem FromUri(const DataStorageUri &uri);
};

struct MooncakeDataStorageItem {
    std::string key;
    static MooncakeDataStorageItem FromUri(const DataStorageUri &uri);
};

struct TairMempoolDataStorageItem {
    uint64_t offset{0}; // offset node内地址区间
    uint64_t size{0};
    uint16_t node_id{0};    // 表示资源归属的节点，该地址是由当前节点完成分配
    uint16_t media_type{0}; // 要感知
    uint16_t range_id{0};   // 多租需求
    static TairMempoolDataStorageItem FromUri(const DataStorageUri &uri);
};

struct LocalDataStorageItem {
    std::string key;
    static LocalDataStorageItem FromUri(const DataStorageUri &uri);
};

} // namespace kv_cache_manager
