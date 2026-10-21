## 目录结构
* include: 头文件
* src: 源文件

## 元数据接口

`MetaClient::StartWrite` 增加末尾参数 `min_replica_count`，默认 0，由服务端按 1 解释；负数会返回参数错误。
`MetaClient` 新增 `GetCacheLocationsByBackend` 和 `GetHostCacheState`，经 `GrpcStub` 调用真实 MetaService。
前者每次选择一个 backend，固定 `QT_BATCH_GET`，返回与输入 key 对齐的嵌套位置数组，保留 miss／mask 的空项及 storage type、spec size、URI；spec-name 筛选若非空，须每个 key 一个名称。
后者返回主机标识、本地可复用长度、P2P 拉取块数和最终命中长度；查询模式由服务端限制为 prefix／Mamba。

新增或变更虚接口必须重新产出匹配的头文件和动态库，不能将新头文件覆盖到旧 RPM。

## C++ memory registration

`TransferClient::Create` accepts a `SharedMemoryRegistration` for one host
shared-memory segment, `ClientMemoryRegistrations` for that segment plus GPU
memory, or a `std::vector<SharedMemoryRegistration>` (`+ optional
std::vector<GpuMemorySpan>`) for several host segments. The vector overload
registers every segment in submission order; an empty vector registers no host
memory. `ClientMemoryRegistrations` keeps its single `host` field because it is
part of the published `include/` ABI — do not add fields to it, add an overload.

The registration object owns neither the host mappings nor the GPU allocations;
callers must keep them alive until the `TransferClient` is destroyed.

Because several registration types are accepted by overloads, brace-only calls
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
registrations.host = SharedMemoryRegistration{host_base, host_size, host_fd};
registrations.gpu.push_back(GpuMemorySpan{gpu_base, gpu_size, device_id});
auto gpu_client = TransferClient::Create(client_config, init_params, registrations);

std::vector<SharedMemoryRegistration> host_registrations{
    {host_base, host_size, host_fd}, {other_base, other_size, other_fd}};
auto multi_host_client = TransferClient::Create(client_config, init_params, host_registrations);
```

Do not pass an untyped `{}` or `{base, size, fd}` as the third argument. This
source-compatibility limitation does not affect calls that already pass a named
`SharedMemoryRegistration` object.
