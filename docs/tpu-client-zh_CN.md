# TPU 客户端 (TpuClient)

TpuClient 是 KVCacheManager 访问 TPU 设备的底层 SDK 封装，基于
PJRT C API（Pretty Much Just a Runtime —— OpenXLA 的统一设备抽象层）
实现。它通过 `dlopen` 动态加载 `libtpu.so`，等价于 JAX 在 Python 中
加载 TPU 插件的方式。

源文件：`kv_cache_manager/client/src/internal/sdk/tpu_client.{h,cc}`

## 为什么使用 dlopen 而非静态链接

libtpu.so 内部依赖 Google 基础设施（flags、logging、filesystem 等），
在独立 C++ 进程中直接链接会导致 `CheckInitGoogleIsDone()` 致命断言
失败。dlopen 方式允许我们在调用 `PJRT_Client_Create` 之前，手动调用
libtpu 内部的 `RealInitGoogle()` 完成 Google 子系统初始化。

详见「初始化流程」一节。

## 核心能力

| 能力 | 接口 | 说明 |
|---|---|---|
| 客户端初始化 | `Init()` | dlopen libtpu → RealInitGoogle → GetPjrtApi → PJRT_Client_Create → 枚举设备 |
| 资源释放 | `Destroy()` | PJRT_Client_Destroy；共享 dlopen handle 不 dlclose |
| DMA 注册 | `DmaMap()` / `DmaUnmap()` | 主机内存 DMA 映射（TPU 当前为 no-op） |
| 同步 H2D 传输 | `BufferFromHost()` | 主机 → TPU 设备，创建新的 PJRT_Buffer，阻塞等待完成 |
| 同步 D2H 传输 | `BufferToHost()` | TPU 设备 → 主机，从现有 PJRT_Buffer 读出，阻塞等待完成 |
| 异步 H2D 传输 | `BufferFromHostAsync()` | 主机 → TPU 设备，返回 PJRT_Event*，非阻塞 |
| 异步 D2H 传输 | `BufferToHostAsync()` | TPU 设备 → 主机，返回 PJRT_Event*，非阻塞 |
| 单事件等待 | `WaitEvent()` | 阻塞等待单个 PJRT_Event 完成并销毁 |
| 批量事件等待 | `WaitEvents()` | 阻塞等待一批 PJRT_Event 全部完成并销毁 |
| 事件销毁 | `DestroyEvent()` | 手动销毁单个 PJRT_Event |
| Buffer 销毁 | `DestroyBuffer()` | 释放 PJRT_Buffer |
| RawBuffer 创建 | `CreateRawAlias()` | 从 PJRT_Buffer 创建 RawBuffer 别名（无并发保护） |
| RawBuffer H2D | `RawBufferFromHost()` | 原始 H2D 拷贝，支持偏移，返回 Event |
| RawBuffer D2H | `RawBufferToHost()` | 原始 D2H 拷贝，支持偏移，返回 Event |
| RawBuffer 大小查询 | `RawBufferGetDeviceSize()` | 查询 RawBuffer 的设备物理字节大小 |
| RawBuffer 销毁 | `DestroyRawBuffer()` | 释放 RawBuffer（不影响原 Buffer） |
| 错误信息 | `GetErrorMessage()` | 静态方法，提取 PJRT_Error 中的文本 |

注意：`CreateViewOfDeviceBuffer` 在 TPU 平台上**不受支持**，所有数据
传输均通过 `BufferFromHostBuffer` / `ToHostBuffer` 实现。

## 初始化流程

`Init()` 内部执行以下步骤：

```
1. dlopen(libtpu.so, RTLD_NOW | RTLD_GLOBAL)
2. 计算 RealInitGoogle 地址 = dlopen_base + kRealInitGoogleOffset
3. 调用 RealInitGoogle("kvcm_tpu", ...) —— 仅首次（static 保护）
4. dlsym("GetPjrtApi") → 获取 PJRT_Api 函数指针表
5. PJRT_Client_Create() → 创建 PJRT 客户端
6. PJRT_Client_AddressableDevices() → 枚举可寻址 TPU 设备
7. LogDeviceInfo() → 查询并打印平台/设备/内存信息
8. ProbeApiCapabilities() → 探测并打印所有 API 支持情况
```

### RealInitGoogle

libtpu 内部的 `RealInitGoogle()` 等价于 Google 标准的 `InitGoogle()`，
负责初始化 flags、logging、filesystem 等子系统。它**不是幂等的**——
重复调用会导致 abort。TpuClient 通过 `static bool google_initialized`
确保全进程仅调用一次。

### 偏移量

`kRealInitGoogleOffset` 与 libtpu 版本绑定（当前 v0.0.41）。升级
libtpu 后需要重新计算此偏移量（通过 `nm`/`readelf` 分析 .so）。

## 设备信息查询 (LogDeviceInfo)

初始化时自动调用 `LogDeviceInfo()`，通过只读 PJRT API 查询并打印：

### 平台级信息

| API | 输出示例 |
|---|---|
| `PJRT_Client_PlatformName` | `tpu` |
| `PJRT_Client_PlatformVersion` | `TFRT TPU7x Built on May 13 2026 ...` |
| `PJRT_Client_ProcessIndex` | `0`（单进程始终为 0） |
| `PJRT_Client_Devices` | 设备总数（含不可寻址） |
| `PJRT_Client_AddressableMemories` | 客户端级可寻址内存数 |

### 设备级信息（每台设备）

| API | 输出示例 |
|---|---|
| `PJRT_Device_IsAddressable` | `true` |
| `PJRT_Device_LocalHardwareId` | `0`（本地硬件编号） |
| `PJRT_DeviceDescription_Id` | `0`（全局唯一设备 ID） |
| `PJRT_DeviceDescription_ProcessIndex` | `0` |
| `PJRT_DeviceDescription_Kind` | `TPU7x` |
| `PJRT_DeviceDescription_ToString` | `TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)` |
| `PJRT_DeviceDescription_DebugString` | `TPU_0(process=0,(0,0,0,0))` |

### 内存级信息（默认内存）

| API | 输出示例 |
|---|---|
| `PJRT_Memory_Id` | `0` |
| `PJRT_Memory_Kind` | `device` |
| `PJRT_Memory_Kind_Id` | 平台相关 ID |
| `PJRT_Memory_DebugString` | `TpuHbmMemory(id=0, process_index=0, client=tpu)` |
| `PJRT_Device_AddressableMemories` | 每设备可寻址内存数（如 3） |
| `PJRT_Device_MemoryStats` | `bytes_in_use`、`bytes_limit`、`peak_bytes` 等统计 |

MemoryStats 可能返回 `UNIMPLEMENTED`，此时日志会显示 `unavailable`。

## API 能力探测 (ProbeApiCapabilities)

初始化时自动调用 `ProbeApiCapabilities()`，遍历 `PJRT_Api` 函数指针表
中全部 138 个 API，分为 `[OK]` 和 `[NULL]` 两类输出。

此外还遍历 PJRT 扩展链（extension chain），输出每个扩展的：

| 字段 | 含义 |
|---|---|
| `type` | 扩展类型编号及可读名称（如 `[8] RawBuffer`） |
| `struct_size` | 扩展结构体的字节大小，用于 PJRT ABI 版本控制 |

### RawBuffer 扩展

`PJRT_RawBuffer_*` API 不在主 `PJRT_Api` 结构体中，而是通过
`PJRT_RawBuffer_Extension` 扩展结构体提供。ProbeApiCapabilities 会
单独探测这些 API：

| API | 说明 |
|---|---|
| `PJRT_RawBuffer_CreateRawAliasOfBuffer` | 创建 alias 原始缓冲区 |
| `PJRT_RawBuffer_Destroy` | 销毁原始缓冲区 |
| `PJRT_RawBuffer_GetOnDeviceSizeInBytes` | 获取设备上的字节数 |
| `PJRT_RawBuffer_GetMemorySpace` | 获取内存空间 |
| `PJRT_RawBuffer_CopyRawHostToDevice` | 原始 H2D 拷贝 |
| `PJRT_RawBuffer_CopyRawDeviceToHost` | 原始 D2H 拷贝 |
| `PJRT_RawBuffer_GetHostPointer` | 获取主机指针 |

### RawBuffer 已知限制

#### 主机内存必须 32 字节对齐

TPU 的 `PJRT_RawBuffer_CopyRawHostToDevice` / `CopyRawDeviceToHost`
要求传入的主机内存指针**必须 32 字节对齐**，否则 PJRT 运行时会在
Event 中返回错误 `"Host buffer is not aligned to 32 bytes"`。

`std::vector`、`std::string` 等标准容器**不保证** 32 字节对齐，
调用方应使用 `std::aligned_alloc(32, size)` 或等效方式分配主机缓冲区：

```cpp
size_t aligned_size = (size + 31) & ~static_cast<size_t>(31);
void* host_buf = std::aligned_alloc(32, aligned_size);
// ... 使用 host_buf 做 RawBuffer 拷贝 ...
free(host_buf);
```

#### 物理字节布局与逻辑数据不同

TPU 使用 tile 布局存储设备内存，RawBuffer D2H 读回的是**物理字节**，
其顺序可能与通过 `BufferFromHost` 写入的逻辑数据不同。例如：

```
逻辑写入: [0, 1, 2, 3, 4, 5, ...]  (row-major)
物理读出: [0, 1, 2, ..., 7, 128, 129, ...]  (tile 8×128 重排)
```

因此不能将 RawBuffer D2H 的结果直接与逻辑数据做 `memcmp` 比对。
如需验证数据正确性，应使用逻辑接口 `BufferToHost()` 读回后比对。

> **后续计划**：上层（如 tpu-raiden）需要根据 tile 布局计算物理偏移，
> 按 tile 为单位解析/填充 RawBuffer 数据。TpuClient 层面暂不做自动转换。

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `TPU_LIBRARY_PATH` | libtpu.so 的完整路径 | 虚拟环境内置路径 |

## 线程安全

- `Init()` / `Destroy()` **不是**线程安全的，调用方需自行加锁
- 内部的 `google_initialized` 静态变量保证了 `RealInitGoogle` 只执行
  一次，但并发 `Init()` 仍存在 data race
- 同一个 `TpuClient` 实例不应被多线程并发使用

## 约束

- KVCache 仅在同一个 `instance_id` 内复用，跨 Instance 不匹配
- TPU 不支持 `CreateViewOfDeviceBuffer`，所有传输走 BufferFromHost/ToHost
- libtpu 版本变更时需更新 `kRealInitGoogleOffset`
- RawBuffer 操作要求主机内存 32 字节对齐（见 RawBuffer 已知限制）
- RawBuffer D2H/H2D 传输的是物理字节，tile 布局下与逻辑数据顺序不同

## 测试

```bash
bazelisk test //kv_cache_manager/client/src/internal/sdk/test:TpuClientTest \
    --config=client_with_tpu \
    --config=debug \
    --test_env=TPU_LIBRARY_PATH=<path-to-libtpu.so>
```

测试覆盖：InitSucceeds、DoubleInit、DestroyClearsState、DmaMapUnmap、
DmaMapUninitReturnsError、BufferRoundTrip、BufferFromHostZeroSize、
BufferToHostZeroSize、BufferOpsUninitReturnsError、DestroyBufferNullSafe、
GetErrorMessageNullSafe、LargeBufferRoundTrip、
AsyncBufferFromHostRoundTrip、AsyncBufferToHostRoundTrip、
WaitEventsBatch、DestroyEventNullSafe、WaitEventNullSafe、
AsyncBufferOpsUninitReturnsError、HasRawBufferExtensionAfterInit、
RawBufferRoundTrip、RawBufferOpsUninitReturnsError、
DestroyRawBufferNullSafe、RawBufferBatchAsyncD2H。
