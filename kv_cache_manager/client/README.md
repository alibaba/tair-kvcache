## 目录结构
* include: 头文件
* src: 源文件

## 元数据接口

`MetaClient::StartWrite` 的 `min_replica_count` 默认 0，由服务端按 1 解释；负数会返回参数错误。其后可选的 `storage_replica_requirement` 用于按指定存储类型补写，例如：

```cpp
StorageReplicaRequirement requirement{StorageType::ST_TAIRMEMPOOL, 1};
auto result = meta_client->StartWrite(trace_id, keys, tokens, spec_groups, 30, 0, requirement);
```

使用新参数时，旧 `min_replica_count` 必须为 0，新数量 `min_location_count` 必须大于 0。上述指定模式的类型不能是未指定、未知或 EventReport 类型。未传新参数时沿用旧路径。每次每个待写 key 最多返回一份新 Location，调用方继续按原流程写入并 Finish；数量是写入门槛，不保证并发上限或分散到不同后端。指定类型不能分配时返回既有服务错误，不跨类型回退。完整语义见[设计说明](../../docs/design/start_write_storage_replica.md)。

日常分层场景也可使用排除模式，保留范围内的 Mark 引导。指定类型和非空排除列表必须二选一，排除项不能是未指定或未知类型：

```cpp
StorageReplicaRequirement requirement{
    StorageType::ST_UNSPECIFIED, 1,
    {StorageType::ST_EVENT_REPORT_L1P5, StorageType::ST_EVENT_REPORT_L2}};
auto result = meta_client->StartWrite(trace_id, keys, tokens, spec_groups, 30, 0, requirement);
```

上述请求让非 EventReport 类型合计满足 1 份。数量已够但可执行的 Mark 目标尚未覆盖时，仍补到 Mark 后端；Mark 目标类型被排除时只忽略本次引导。SSD 已有副本时不会仅因 DRAM 被回收而补回热层。普通分配沿用 Group 策略并过滤排除类型，Finish 仍按实际成功后端判断是否清标。

指定或排除的类型指后端注册配置类型。VCNS 与普通 3FS 分别计数和筛选，但返回的 Location 仍沿用原协议类型，VCNS 可以返回 `ST_3FS`；客户端继续按原数据传输流程处理，不以请求类型与响应 `type` 字面相等作为分配成功的条件。

`MetaClient` 新增 `GetCacheLocationsByBackend` 和 `GetHostCacheState`，经 `GrpcStub` 调用真实 MetaService。
前者每次选择一个 backend，固定 `QT_BATCH_GET`，返回与输入 key 对齐的嵌套位置数组，保留 miss／mask 的空项及 storage type、spec size、URI；spec-name 筛选若非空，须每个 key 一个名称。
后者通过 `top_k_host_count` 限制计算远端命中的逻辑引擎数量，通过 `backend_types` 选择 TairMempool 和／或 Vineyard P2P；返回主机标识、本地连续命中块数及包含本地命中的全局连续命中块数。

新增或变更虚接口必须重新产出匹配的头文件和动态库，不能将新头文件覆盖到旧 RPM。

## C++ memory registration

`TransferClient::Create` accepts either a `SharedMemoryRegistration` for host
shared memory or `ClientMemoryRegistrations` for host and GPU memory. The
registration object does not own GPU allocations; callers must keep them alive
until the `TransferClient` is destroyed.

Because both registration types are accepted by overloads, brace-only calls
that previously selected the host registration overload are ambiguous in
C++17. Name the intended type explicitly:

```cpp
auto disabled_host_client = TransferClient::Create(
    client_config, init_params, SharedMemoryRegistration{});

auto host_client = TransferClient::Create(
    client_config,
    init_params,
    SharedMemoryRegistration{host_base, host_size, host_fd});

ClientMemoryRegistrations registrations;
registrations.gpu.push_back(GpuMemorySpan{gpu_base, gpu_size, device_id});
auto gpu_client = TransferClient::Create(client_config, init_params, registrations);
```

Do not pass an untyped `{}` or `{base, size, fd}` as the third argument. This
source-compatibility limitation does not affect calls that already pass a named
`SharedMemoryRegistration` object.
