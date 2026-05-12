#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {
class MetaStorageBackendConfig;

// MetaStorageBackend — MetaIndexer 后端存储抽象基类
//
// 数据模型：
//   每个 key (int64_t) 对应一个 hash 结构，内含多个 field→value 映射。
//   field 用于存储 location（前缀 "__loc__"）和 property。

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

    // 创建 key 并写入所有 field。若 key 已存在，行为等同于全量覆盖。
    // @param keys        待写入的 key 列表
    // @param field_maps  每个 key 对应的 field→value 映射，大小必须等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:    写入成功
    //   - EC_ERROR: 写入失败（网络/IO 错误等）
    virtual std::vector<ErrorCode> Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;

    // 更新已存在 key 的部分 field。不会创建新 key。
    // @param keys        待更新的 key 列表
    // @param field_maps  每个 key 要更新的 field→value，大小必须等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:    更新成功
    //   - EC_NOENT: key 不存在（仅 Local/Dummy 后端；Redis HMSET 对不存在的 key 会隐式创建）
    //   - EC_ERROR: 更新失败
    virtual std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;

    // 若 key 存在则更新 field，不存在则创建 key 并写入。语义 = Put + UpdateFields 的合体。
    // @param keys        待操作的 key 列表
    // @param field_maps  每个 key 对应的 field→value，大小必须等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:    操作成功（无论是新建还是更新）
    //   - EC_ERROR: 操作失败
    virtual std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;

    // 删除整个 key 及其所有 field。
    // @param keys  待删除的 key 列表
    // @return 每个 key 的错误码：
    //   - EC_OK:    删除成功（key 确实存在并被删除）
    //   - EC_NOENT: key 不存在（Local/Dummy 后端）；Redis DEL 返回 0
    //   - EC_ERROR: 删除失败
    virtual std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept = 0;

    // 删除 key 中的指定 field（部分删除）。幂等语义：删除不存在的 field 视为成功。
    // @param keys             待操作的 key 列表
    // @param field_names_vec  每个 key 要删除的 field name 列表，大小必须等于 keys.size()
    //                         若 field_names_vec[i] 为空，该 key 跳过操作（no-op，返回 EC_OK）
    // @return 每个 key 的错误码：
    //   - EC_OK:    删除成功（包括 field 原本就不存在的情况，以及 field_names 为空的 no-op）
    //   - EC_NOENT: key 本身不存在（仅当 field_names 非空时；Local/Dummy 后端）
    //   - EC_ERROR: 操作失败
    virtual std::vector<ErrorCode>
    DeleteFields(const KeyTypeVec &keys, const std::vector<std::vector<std::string>> &field_names_vec) noexcept = 0;

    // =====================================================================
    // Read APIs
    // =====================================================================

    // 按统一的 field name 列表，批量获取多个 key 的指定 field。
    // @param keys           待查询的 key 列表
    // @param field_names    所有 key 共用的 field name 列表（不能为空）
    // @param out_field_maps [out] 每个 key 的结果 map，大小等于 keys.size()。
    //                       仅包含存在且 value 非空的 field。
    // @return 每个 key 的错误码：
    //   - EC_OK:    key 存在（out_field_maps[i] 可能为空——表示请求的 field 都不存在）
    //   - EC_NOENT: key 不存在
    //              注意：Redis 后端当所有 field 返回 nil 时统一返回 EC_NOENT。
    //   - EC_BADARGS: field_names 为空
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode>
    Get(const KeyTypeVec &keys, const std::vector<std::string> &field_names, FieldMapVec &out_field_maps) noexcept = 0;

    // 按每个 key 独立的 field name 列表，批量获取 field。
    // @param keys             待查询的 key 列表
    // @param field_names_vec  每个 key 对应的 field name 列表，大小必须等于 keys.size()
    //                         每个子列表不能为空（空列表返回 EC_BADARGS）
    // @param out_field_maps   [out] 每个 key 的结果 map。仅含存在且 value 非空的 field。
    // @return 每个 key 的错误码：
    //   - EC_OK:     key 存在（out_field_maps[i] 可能为空——表示请求的 field 都不存在）
    //   - EC_NOENT:  key 不存在
    //              注意：Redis 后端当所有 field 返回 nil 时统一返回 EC_NOENT。
    //   - EC_BADARGS: field_names_vec[i] 为空
    //   - EC_ERROR:  查询失败
    virtual std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                                       const std::vector<std::vector<std::string>> &field_names_vec,
                                       FieldMapVec &out_field_maps) noexcept = 0;

    // 获取 key 的所有 field（含 tombstone 空 value）。
    // 注意：此接口不过滤 tombstone，调用方需自行判断空 value。
    // @param keys           待查询的 key 列表
    // @param out_field_maps [out] 每个 key 的完整 field map（包含空 value 的 field）
    // @return 每个 key 的错误码：
    //   - EC_OK:    key 存在（即使所有 field value 为空）
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode> GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept = 0;

    // 获取 key 中以指定前缀开头的 field name 列表。跳过 value 为空的 tombstone field。
    // @param keys                 待查询的 key 列表
    // @param field_prefix         field name 前缀（如 "__loc__"）
    // @param out_field_names_vec  [out] 每个 key 匹配的 field name 列表（仅含 value 非空的）
    // @return 每个 key 的错误码：
    //   - EC_OK:    key 存在（结果可能为空——key 存在但无匹配的非空 field）
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode>
    GetFieldNamesWithPrefix(const KeyTypeVec &keys,
                            const std::string &field_prefix,
                            std::vector<std::vector<std::string>> &out_field_names_vec) noexcept = 0;

    // 检查 key 是否存在。
    // @param keys              待查询的 key 列表
    // @param out_is_exist_vec  [out] 每个 key 是否存在，大小等于 keys.size()
    // @return 每个 key 的错误码：
    //   - EC_OK:    查询成功（out_is_exist_vec[i] 反映结果）
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode> Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept = 0;

    // 检查 key 中是否存在以指定前缀开头且 value 非空的 field。
    // 空 value (tombstone) 不被视为有效存在。
    // @param keys            待查询的 key 列表
    // @param field_prefix    field name 前缀
    // @param out_exists_vec  [out] 每个 key 是否有匹配的非空 field
    // @return 每个 key 的错误码：
    //   - EC_OK:    key 存在（out_exists_vec[i] 反映结果，可能为 false）
    //   - EC_NOENT: key 不存在
    //   - EC_ERROR: 查询失败
    virtual std::vector<ErrorCode> ExistsFieldWithPrefix(const KeyTypeVec &keys,
                                                         const std::string &field_prefix,
                                                         std::vector<bool> &out_exists_vec) noexcept = 0;

    // 基于 cursor 分页扫描所有 key。
    // @param cursor          游标（首次传 SCAN_BASE_CURSOR = "0"）
    // @param limit           本次最多返回的 key 数量（hint，实际可能多于或少于此值）
    // @param out_next_cursor [out] 下一次扫描的 cursor；等于 SCAN_BASE_CURSOR 时表示扫描结束
    // @param out_keys        [out] 本次扫描到的 key 列表
    // @return EC_OK 成功；EC_BADARGS cursor 非法；EC_ERROR 扫描失败
    virtual ErrorCode ListKeys(const std::string &cursor,
                               const int64_t limit,
                               std::string &out_next_cursor,
                               KeyTypeVec &out_keys) noexcept = 0;

    // 随机采样 key。
    // @param count    期望采样数量
    // @param out_keys [out] 采样结果（实际数量可能小于 count）
    // @return EC_OK 成功；EC_ERROR 采样失败
    virtual ErrorCode RandomSample(const int64_t count, KeyTypeVec &out_keys) noexcept = 0;

    // 采样适合回收的 key（通常按 LRU 或 access time 排序）。
    // @param count    期望采样数量
    // @param out_keys [out] 采样结果
    // @return EC_OK 成功；EC_ERROR 采样失败
    virtual ErrorCode SampleReclaimKeys(const int64_t count, KeyTypeVec &out_keys) noexcept = 0;

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
};
} // namespace kv_cache_manager
