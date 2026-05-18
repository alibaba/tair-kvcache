#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {
class MetaStorageBackendConfig;
class RequestContext;

// MetaStorageBackend — MetaIndexer 后端存储抽象基类
//
// 数据模型：
//   每个 key (int64_t) 对应一组 CacheLocation（按 location_id 索引）和
//   一组 property（string→string KV 对）。
//   后端自行决定内部存储格式：Local 后端直接存结构体，Redis 后端存 JSON string。

class MetaStorageBackend {
public:
    using KeyType = ::kv_cache_manager::KeyType;
    using KeyTypeVec = ::kv_cache_manager::KeyTypeVec;
    using FieldMap = ::kv_cache_manager::FieldMap;
    using FieldMapVec = ::kv_cache_manager::FieldMapVec;

    virtual ~MetaStorageBackend() = default;

    virtual std::string GetStorageType() noexcept = 0;

    // 初始化后端。必须在 Open() 之前调用。
    // @param instance_id  实例标识，用于 key 前缀隔离
    // @param config       后端配置（含 storage_uri 等）
    // @return EC_OK 成功；EC_BADARGS 参数非法；EC_ERROR 内部错误
    virtual ErrorCode Init(const std::string &instance_id,
                           const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept = 0;

    // 打开后端连接/资源。Init 成功后调用。
    // @return EC_OK 成功；EC_ERROR 连接/资源获取失败
    virtual ErrorCode Open() noexcept = 0;

    // 关闭后端，释放连接和资源。
    // @return EC_OK 成功；EC_ERROR 关闭时出错
    virtual ErrorCode Close() noexcept = 0;

    // =====================================================================
    // Write APIs
    // =====================================================================

    // 整 key 覆盖写入 locations + properties。若 key 已存在则全量替换
    // @param request_context  请求上下文，用于 metrics 上报；可为 nullptr
    // @param keys             待写入的 key 列表
    // @param locations        每个 key 对应的 CacheLocationMap，大小必须等于 keys.size()；
    //                         可以为空 map 表示该 key 无 location
    // @param properties       每个 key 对应的 PropertyMap，大小必须等于 keys.size()；
    //                         可以为空 map 表示该 key 无 property
    // @return 每个 key 的错误码：
    //   - EC_OK:    写入成功
    //   - EC_ERROR: 写入失败（网络/IO 错误等）
    virtual std::vector<ErrorCode> Put(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       const CacheLocationMapVector &locations,
                                       const PropertyMapVector &properties) noexcept = 0;

    // 若 key 存在则合并 不存在则创建 key 并写入。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待操作的 key 列表
    // @param locations        每个 key 要合并的 CacheLocationMap，大小必须等于 keys.size()；
    //                         空 map 表示不更新该 key 的 location
    // @param properties       每个 key 要合并的 PropertyMap，大小必须等于 keys.size()；
    //                         空 map 表示不更新该 key 的 property
    // @return 每个 key 的错误码：
    //   - EC_OK:    操作成功（无论是新建还是合并）
    //   - EC_ERROR: 操作失败
    virtual std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          const CacheLocationMapVector &locations,
                                          const PropertyMapVector &properties) noexcept = 0;

    // 删除整个 key 及其所有 locations 和 properties。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys  待删除的 key 列表
    // @return 每个 key 的错误码：
    //   - EC_OK:    删除成功（key 确实存在并被删除）
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 删除失败
    virtual std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept = 0;

    // 删除指定 key 下的指定 location。幂等语义：删除不存在的 location 视为成功。
    // 不影响 key 中的 properties 和其他 location。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待操作的 key 列表
    // @param location_ids     每个 key 要删除的 location id 列表，大小必须等于 keys.size()；
    //                         若 location_ids[i] 为空，该 key 为 no-op（返回 EC_OK）
    // @return 每个 key 的错误码：
    //   - EC_OK:    删除成功（含 no-op 和 location 原本不存在的情况）
    //   - EC_NOENT: key 本身不存在
    //   - EC_ERROR: 操作失败
    virtual std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                                   const KeyTypeVec &keys,
                                                   const LocationIdsPerKey &location_ids) noexcept = 0;

    // =====================================================================
    // Read APIs
    // =====================================================================

    // 读取 key 的完整元数据（locations + properties 一起返回）。
    // @param request_context   请求上下文；可为 nullptr
    // @param keys              待查询的 key 列表
    // @param out_locations     [out] 每个 key 的 CacheLocationMap，大小等于 keys.size()
    // @param out_properties    [out] 每个 key 的 PropertyMap，大小等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:        key 存在
    //   - EC_NOENT:     key 不存在
    //   - EC_CORRUPTION: location 数据损坏（反序列化失败）
    //   - EC_ERROR:     查询失败
    virtual std::vector<ErrorCode> Get(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       CacheLocationMapVector &out_locations,
                                       PropertyMapVector &out_properties) noexcept = 0;

    // 读取 key 的全量 CacheLocationMap（所有 location）。
    // @param request_context   请求上下文；可为 nullptr
    // @param keys              待查询的 key 列表
    // @param out_locations     [out] 每个 key 的 CacheLocationMap，大小等于 keys.size()。
    //                          key 存在但无 location 时为空 map。
    // @return 每个 key 的错误码：
    //   - EC_OK:        key 存在（out_locations[i] 可能为空 map）
    //   - EC_NOENT:     key 不存在
    //   - EC_CORRUPTION: location 数据损坏（反序列化失败，仅 Redis 后端）
    //   - EC_ERROR:     查询失败
    virtual std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                                const KeyTypeVec &keys,
                                                CacheLocationMapVector &out_locations) noexcept = 0;

    // 读取指定 location id 对应的 CacheLocation。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待查询的 key 列表
    // @param location_ids     每个 key 要查询的 location id 列表，大小必须等于 keys.size()
    // @param out_locations    [out] 每个 key 的 CacheLocation 列表，out_locations[i] 大小
    //                         等于 location_ids[i].size()。未找到的 location 为默认构造。
    // @return 二维错误码，results[i][j] 对应 keys[i] 的 location_ids[i][j]：
    //   - EC_OK:         location 存在
    //   - EC_NOENT:      key 不存在或该 location 不存在
    //   - EC_CORRUPTION: location 数据损坏（反序列化失败）
    //   - EC_ERROR:      查询失败
    virtual std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                             const KeyTypeVec &keys,
                                                             const LocationIdsPerKey &location_ids,
                                                             LocationsPerKey &out_locations) noexcept = 0;

    // 仅获取 key 的 location id 列表（不读取 location body）。
    // @param request_context    请求上下文；可为 nullptr
    // @param keys               待查询的 key 列表
    // @param out_location_ids   [out] 每个 key 的 location id 列表，大小等于 keys.size()。
    //                           key 存在但无 location 时为空 vector。
    // @return 每个 key 的错误码：
    //   - EC_OK:    key 存在
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                                  const KeyTypeVec &keys,
                                                  LocationIdsPerKey &out_location_ids) noexcept = 0;

    // 读取指定字段名的 properties。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待查询的 key 列表
    // @param field_names      要读取的 property 字段名列表（所有 key 使用同一组字段名）
    // @param out_properties   [out] 每个 key 的 PropertyMap（仅包含请求的字段），
    //                         大小等于 keys.size()。字段不存在时不出现在 map 中。
    // @return 每个 key 的错误码：
    //   - EC_OK:    key 存在
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode> GetProperties(RequestContext *request_context,
                                                 const KeyTypeVec &keys,
                                                 const std::vector<std::string> &field_names,
                                                 PropertyMapVector &out_properties) noexcept = 0;

    // 检查 key 是否存在。
    // @param request_context   请求上下文；可为 nullptr
    // @param keys              待查询的 key 列表
    // @param out_is_exist_vec  [out] 每个 key 是否存在，大小等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:    查询成功（out_is_exist_vec[i] 反映结果）
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode>
    Exists(RequestContext *request_context, const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept = 0;

    // 判断 key 是否存在至少一个有效 location。
    // @param request_context  请求上下文；可为 nullptr
    // @param keys             待查询的 key 列表
    // @param out_exists       [out] 每个 key 是否拥有至少一个 location，大小等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:    查询成功（out_exists[i] 反映结果）
    //   - EC_NOENT: key 不存在（out_exists[i] 为 false）
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode>
    ExistsLocation(RequestContext *request_context, const KeyTypeVec &keys, std::vector<bool> &out_exists) noexcept = 0;

    // 基于 cursor 分页扫描所有 key。
    // @param request_context 请求上下文；可为 nullptr
    // @param cursor          游标（首次传 SCAN_BASE_CURSOR = "0"）
    // @param limit           本次最多返回的 key 数量（hint，实际可能多于或少于此值）
    // @param out_next_cursor [out] 下一次扫描的 cursor；等于 SCAN_BASE_CURSOR 时表示扫描结束
    // @param out_keys        [out] 本次扫描到的 key 列表
    // @return EC_OK 成功；EC_BADARGS cursor 非法；EC_ERROR 扫描失败
    virtual ErrorCode ListKeys(RequestContext *request_context,
                               const std::string &cursor,
                               const int64_t limit,
                               std::string &out_next_cursor,
                               KeyTypeVec &out_keys) noexcept = 0;

    // 随机采样 key。
    // @param request_context 请求上下文；可为 nullptr
    // @param count    期望采样数量
    // @param out_keys [out] 采样结果（实际数量可能小于 count）
    // @return EC_OK 成功；EC_ERROR 采样失败
    virtual ErrorCode
    RandomSample(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept = 0;

    // 采样适合回收的 key（通常按 LRU 或 access time 排序）。
    // @param request_context 请求上下文；可为 nullptr
    // @param count    期望采样数量
    // @param out_keys [out] 采样结果
    // @return EC_OK 成功；EC_ERROR 采样失败
    virtual ErrorCode
    SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyTypeVec &out_keys) noexcept = 0;

    // =====================================================================
    // Metadata APIs — 用于持久化 MetaIndexer 自身的元信息（key_count、storage_usage 等）
    // =====================================================================

    // 写入/更新 MetaIndexer 元数据。
    // @param field_maps  要写入的元数据 field→value
    // @return EC_OK 成功；EC_ERROR 写入失败
    virtual ErrorCode PutMetaData(const FieldMap &field_maps) noexcept = 0;

    // 读取 MetaIndexer 元数据。
    // @param field_maps [out] 读到的元数据
    // @return EC_OK 成功；EC_NOENT 无元数据；EC_ERROR 读取失败
    virtual ErrorCode GetMetaData(FieldMap &field_maps) noexcept = 0;

    // Synchronously flush pending writes for the given keys.
    // Returns true if all writes persisted successfully, false on failure/timeout.
    // Default: no-op (sync backends have no pending writes).
    virtual bool Sync(const KeyTypeVec & /*keys*/) noexcept { return true; }

    // Returns per-queue pending key sizes. Default: empty (no async queues).
    virtual std::vector<int64_t> GetAsyncQueueSizes() const noexcept { return {}; }
};
} // namespace kv_cache_manager
