# KVCM + TairMempool 压测

一个 C++ 多线程压测进程，无 Python 热路径、不需要 GPU。元数据经 MetaClient 的 **gRPC** 连接 KVCM，数据经 TransferClient 的 **PACE SDK** 真正写入/读出 TairMempool。

每个 worker 独立持有 gRPC 客户端和可复用 CPU buffer，进程共享一个 TransferClient/SDK 线程池。写请求生成新 key，依次执行 `StartWrite → SaveKvCaches → FinishWrite`；只有 FinishWrite 成功的 key 才进入有界的近期读取池。读请求执行 `GetCacheLocation(QT_BATCH_GET)`，再按配置概率加载命中数据。支持 SIGINT/SIGTERM，停止发新请求后等待在途操作完成。

## 构建

在内源仓库根目录切换依赖视图，再进入外源子仓构建。真实 PACE SDK 只在内源可用；外源 stub 无法用于 TairMempool 压测。

```bash
bash internal_source/scripts/switch_to_internal_source.sh
cd github-opensource
bazelisk --output_base=/tmp/kvcm-stress-is build --config=release \
  --define ENABLE_TAIR_MEMPOOL=true --copt=-DENABLE_TAIR_MEMPOOL \
  //tools/kvcm_stress:kvcm_stress
```

产物：`bazel-bin/tools/kvcm_stress/kvcm_stress`。以上仅启用 CPU PACE 客户端；不需要 `--config=client_with_cuda`。运行环境须具有链接所需的动态库（例如 libibverbs、librdmacm）（跨机器运行时一并携带构建产物的 `_solib_*` / runfiles，或使用已有客户端运行镜像）。

## 运行

先编辑本目录 `client.json`：

- `address`：KVCM **Meta gRPC** 地址，不能填 HTTP 管理端口。
- `instance_group`：服务端已配置好 TairMempool storage 的 group。程序会注册 `instance_id`；建议压测使用独立 instance。
- `location_spec_infos.tp0`：每块真实数据字节数，例子为 64 KiB；须是 TairMempool 支持的规格。`block_size` 是 token 数，与字节数不同。
- 仅支持一个 spec，省去 TP/多 spec 配置；不要配置 `location_spec_groups`。
- `meta_channel_config.call_timeout`：gRPC 超时毫秒。保持 `retry_time=1` 禁用重试，防止失败统计被重试掩盖或重复创建写会话。
- `sdk_config.thread_num` / `queue_size`：数据面并发和队列容量；SDK 超时为毫秒，应大于 PACE 内部超时。`--write-timeout` 是服务端写会话 TTL，应留出数据写入和 FinishWrite 的时间。

```bash
mkdir -p logs
./bazel-bin/tools/kvcm_stress/kvcm_stress \
  --config tools/kvcm_stress/client.json \
  --threads 64 --batch 16 --seconds 300 \
  --query-percent 30 --read-percent 100
```

这个例子约 70% 的**逻辑操作**是完整写事务，30% 是查询，所有命中的查询都真实读数据。比例按操作选择次数计算，不是按 RPC 数或字节数；一次写包含两个元数据 RPC。缺少读取样本时 worker 会先写入，不空转。

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--threads` | 32 | 并发 worker 数；先调到 64/128，再结合客户端 CPU、SDK 队列和服务端负载增加 |
| `--batch` | 16 | 每个逻辑操作的 block 数 |
| `--seconds` | 60 | 正式压测秒数；0 表示持续运行至收到信号 |
| `--query-percent` | 30 | 查询操作比例，0 为纯写，100 为预填充后的纯读 |
| `--read-percent` | 100 | 查询后真实读数据的概率；0 为只查元数据 |
| `--pool` | 4096 | 每个 worker 保留的近期成功 key 数，不是服务端容量上限 |
| `--prefill` | 1 | 每个 worker 在计时前写入的 batch 数；不计入正式统计 |
| `--write-timeout` | 60 | 写会话 TTL 秒数，范围 1～1800 |
| `--report` | 5 | 统计输出间隔秒数 |
| `--key-base` | 随机 | 首个 key，之后全进程递增；多进程指定时须使用不重叠区间 |
| `--verify` | 关闭 | 写入按 key 生成的内容并逐字节验证读回数据，建议先小并发开启验通 |

例如先用 `--threads 2 --batch 4 --seconds 10 --verify` 验通，再关闭校验提高压力。纯查询压测可以用 `--query-percent 100 --read-percent 0 --prefill 256`；纯数据读取用 `--query-percent 100 --read-percent 100 --prefill 256`。预填充按 worker 顺序执行，发生失败即退出，不将无样本的测试当作成功。

payload 内存约 `threads × batch × block_bytes`，读取池另占 `threads × pool × 8` 字节。写入持续产生新 key，**不会自动删除服务端数据**；服务端回收由配置的容量、配额和回收策略决定。读取池是近期写成功的样本集合，服务端回收后的 miss 属于正常结果，会体现在命中率中；查询命中后读失败则计为错误。

PACE 目标默认来自 KVCM 返回的 storage 配置；需要覆盖时按现有 SDK 约定设置 `PACE_META_SERVER_ADDR`。不要把另一个 TairMempool 集群设为覆盖目标。程序使用普通 CPU buffer 和 SDK 默认搬运路径，不要求外部共享内存/零拷贝注册。

## 输出与判断

每次输出区间读写 MiB/s、累计命中块数/查询块数、成功写入/读出块数、跳过块数、校验失败块数，以及 `start / put / finish / query / get / write_e2e` 的区间调用 QPS、累计调用数、错误数、最近错误码、累计平均延迟和累计近似 p99 上界（2 的幂次微秒桶）。QPS 包括失败调用，吞吐仅计成功数据；`write_e2e` 包括整个写事务，其他阶段是单次客户端调用延迟。最后报告包含在途请求排空时间，QPS/吞吐为整段平均。

退出码：0 表示没有请求/校验错误，1 表示压测中出现错误，2 表示参数、初始化或预填充失败。客户端详细错误在日志中。程序不声称某个固定 QPS；实际瓶颈取决于批量、块大小、SDK 并发、网络和服务端能力。

此分支的 KVCM 在 StartWrite 时预分配地址，FinishWrite 服务端忽略 `locations` 字段。程序检查 SDK 返回 URI 必须与分配地址一致，出现变化按写失败处理；PUT 失败仍发送表示全部失败的 FinishWrite（offset=0），避免发布半写入数据。StartWrite 的 skip mask 与 FinishWrite 的紧凑 success mask 分别处理，兼容部分 key 被跳过的情况。

## 验证

```bash
# 在 github-opensource 下执行；内外源使用独立 output_base。
bazelisk --output_base=/tmp/kvcm-stress-is test --config=debug --config=asan \
  --test_env ASAN_OPTIONS=detect_odr_violation=0 \
  //tools/kvcm_stress:WorkloadTest
```

`WorkloadTest` 使用内存中的数据面和元数据面验证完整字节读写、近期池轮转、mask 对齐、跳过全部 key、失败收尾、地址变化拒绝、数据损坏识别、只查元数据和 miss。真实服务连通性及最终吞吐需在部署好的 KVCM + TairMempool 上用上述验通命令测量。
