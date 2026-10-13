## 目录结构
* include: 头文件
* src: 源文件

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
