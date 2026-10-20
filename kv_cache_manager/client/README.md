## 目录结构
* include: 头文件
* src: 源文件

## 元数据接口

`MetaClient::StartWrite` 增加末尾参数 `min_replica_count`，默认 0，由服务端按 1 解释；负数会返回参数错误。
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
