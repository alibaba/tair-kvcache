#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
namespace kv_cache_manager {

enum class SdkType : uint8_t {
    HF3FS = 0,
    MOONCAKE = 1,
    TAIR_MEMPOOL = 2,
    LOCAL_FILE = 3,
    SDK_TYPE_MAX = 255,
};

// 一组block group对应一个remote path，支持一个远端文件/内存保存多个block的数据
struct BlockGroup {
    std::vector<DataStorageUri> remote_uris;
    BlockBuffers local_buffers;
};

struct Hf3fsRemoteItem {
    std::string file_path;
    uint64_t blkid{0};
    static Hf3fsRemoteItem FromUri(const DataStorageUri &uri);
};

struct MooncakeRemoteItem {
    std::string key;
    static MooncakeRemoteItem FromUri(const DataStorageUri &uri);
};

struct TairMempoolRemoteItem {
    uint64_t offset{0}; // offset node内地址区间
    uint64_t size{0};
    uint16_t node_id{0};    // 表示资源归属的节点，该地址是由当前节点完成分配
    uint16_t media_type{0}; // 要感知
    uint16_t range_id{0};   // 多租需求
    static TairMempoolRemoteItem FromUri(const DataStorageUri &uri);
};

struct LocalFileItem {
    std::string file_path;
    uint64_t blkid{0};
    size_t size{0};
    static LocalFileItem FromUri(const DataStorageUri &uri);
};

} // namespace kv_cache_manager