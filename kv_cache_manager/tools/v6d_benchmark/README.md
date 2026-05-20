# V6D Benchmark Tool

V6D压测工具，用于压测V6D通过KVCM的完整链路(节点注册、Block增删查)。

## 功能特性

- ✅ 自动注册V6D节点(支持自动获取本机IP)
- ✅ 压测BlockAdd/BlockDelete/GetCacheLocation的增删查链路
- ✅ 支持多线程并发压测，带明确的QPS限流控制
- ✅ 自动验证查询结果的正确性
- ✅ 使用Kmonitor汇报详细指标(QPS、延迟、带宽、错误率等)
- ✅ 所有参数通过环境变量配置
- ✅ 进程持续运行直到被杀死，无固定持续时间
- ✅ 支持多机部署，每台机器自动使用不同的IP

## 构建

```bash
bazel build //kv_cache_manager/tools/v6d_benchmark:v6d_benchmark
```

构建产物位于:
```
bazel-bin/kv_cache_manager/tools/v6d_benchmark/v6d_benchmark
```

## 配置参数

所有配置通过环境变量设置：

### KVCM连接配置
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| KVCM_BASE_URL | http://127.0.0.1:8080 | KVCM基础URL |
| KVCM_ADMIN_URL | 同KVCM_BASE_URL | KVCM Admin URL |
| INSTANCE_ID | v6d_benchmark_0 | 实例ID |
| INSTANCE_GROUP | default | 实例组 |

### V6D节点配置
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| AUTO_DETECT_HOST | true | 自动获取本机IP |
| V6D_HOST | (空) | 固定IP(当AUTO_DETECT_HOST=false时) |
| V6D_PORT | 8080 | V6D端口 |
| V6D_MEDIUMS | mem,disk | 支持的介质类型(逗号分隔) |

### 压测参数
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| NUM_BLOCKS | 10000 | 测试数据集大小 |
| BLOCK_SIZE | 128 | 每个block大小(字节) |
| NUM_THREADS | 1 | 并发线程数 |

### QPS限流配置
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| TARGET_QPS | 1000.0 | 目标QPS |
| ENABLE_QPS_LIMIT | true | 启用QPS限流 |

### 压测模式
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| TEST_MODE | full | 压测模式: full/add/query/delete |
| QUERY_BATCH_SIZE | 50 | 每次查询的key数量 |
| ADD_RATIO | 0.7 | Add操作比例(仅full模式) |
| QUERY_RATIO | 0.2 | Query操作比例(仅full模式) |
| DELETE_RATIO | 0.1 | Delete操作比例(仅full模式) |

### Kmonitor配置
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| ENABLE_KMONITOR | true | 启用Kmonitor |
| KMONITOR_CONFIG | (空) | Kmonitor JSON配置 |
| REPORT_INTERVAL_MS | 10000 | 指标汇报间隔(毫秒) |

### 结果验证
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| ENABLE_VERIFICATION | true | 启用结果验证 |

## 使用示例

### 单机压测

```bash
# 设置环境变量
export KVCM_BASE_URL="http://192.168.1.200:8080"
export KVCM_ADMIN_URL="http://192.168.1.200:6492"
export INSTANCE_ID="v6d_test_cluster"
export AUTO_DETECT_HOST=true
export V6D_PORT="8080"
export V6D_MEDIUMS="mem,disk"
export NUM_BLOCKS=50000
export NUM_THREADS=16
export TARGET_QPS=5000
export TEST_MODE="full"
export ENABLE_VERIFICATION=true
export ENABLE_KMONITOR=true

# 运行压测
./bazel-bin/kv_cache_manager/tools/v6d_benchmark/v6d_benchmark
```

### 多机压测

在多台机器上使用相同配置运行，每台机器会自动使用自己的IP：

**机器1**:
```bash
export KVCM_BASE_URL="http://192.168.1.200:8080"
export KVCM_ADMIN_URL="http://192.168.1.200:6492"
export INSTANCE_ID="v6d_test_cluster"  # 相同instance_id
export AUTO_DETECT_HOST=true            # 自动获取不同的IP
export V6D_PORT="8080"
export V6D_MEDIUMS="mem,disk"
export NUM_BLOCKS=50000
export NUM_THREADS=16
export TARGET_QPS=5000
export TEST_MODE="full"

./bazel-bin/kv_cache_manager/tools/v6d_benchmark/v6d_benchmark
```

**机器2**:
```bash
# 相同的配置，只需确保KVCM_ACCESSIBLE
export KVCM_BASE_URL="http://192.168.1.200:8080"
export KVCM_ADMIN_URL="http://192.168.1.200:6492"
export INSTANCE_ID="v6d_test_cluster"
export AUTO_DETECT_HOST=true
export V6D_PORT="8080"
export V6D_MEDIUMS="mem,disk"
export NUM_BLOCKS=50000
export NUM_THREADS=16
export TARGET_QPS=5000
export TEST_MODE="full"

./bazel-bin/kv_cache_manager/tools/v6d_benchmark/v6d_benchmark
```

两台机器会使用各自的IP注册为不同的V6D节点，但操作同一个instance_id的数据。

### 停止压测

```bash
# 发送SIGTERM信号，优雅退出
kill <pid>

# 或强制杀死
kill -9 <pid>
```

## 压测模式说明

### full模式
按配置的比例混合执行Add/Query/Delete操作：
- 70% AddBlock (默认)
- 20% Query (默认)
- 10% DeleteBlock (默认)

可通过`ADD_RATIO`, `QUERY_RATIO`, `DELETE_RATIO`调整比例。

### add模式
只执行BlockAdd操作，将所有block添加到数据集中。

### query模式
只执行GetCacheLocation查询操作，每次查询`QUERY_BATCH_SIZE`个key。

### delete模式
只执行BlockDelete操作，从数据集中删除block。

## 指标汇报

### Kmonitor指标

程序会定期(默认10秒)向Kmonitor汇报以下指标：

- `v6d_benchmark.qps` - 当前QPS
- `v6d_benchmark.avg_latency_us` - 平均延迟(微秒)
- `v6d_benchmark.p50_latency_us` - P50延迟
- `v6d_benchmark.p99_latency_us` - P99延迟
- `v6d_benchmark.p999_latency_us` - P999延迟
- `v6d_benchmark.success_rate` - 成功率(%)
- `v6d_benchmark.bandwidth_mbps` - 带宽(MB/s)
- `v6d_benchmark.add_block_qps` - BlockAdd QPS
- `v6d_benchmark.query_qps` - Query QPS
- `v6d_benchmark.delete_block_qps` - BlockDelete QPS
- `v6d_benchmark.verification_passed` - 验证通过次数
- `v6d_benchmark.verification_failed` - 验证失败次数
- `v6d_benchmark.target_qps` - 目标QPS配置值

### 日志输出示例

```
[INFO] === Benchmark Configuration ===
[INFO] KVCM Base URL: http://192.168.1.200:8080
[INFO] KVCM Admin URL: http://192.168.1.200:6492
[INFO] Instance ID: v6d_test_cluster
[INFO] Auto Detect Host: true
[INFO] V6D Port: 8080
[INFO] V6D Mediums: [mem, disk]
[INFO] Num Blocks: 50000
[INFO] Block Size: 128 bytes
[INFO] Num Threads: 16
[INFO] Target QPS: 5000.00
[INFO] Test Mode: full
[INFO] Enable Verification: true
[INFO] Enable Kmonitor: true
[INFO] ================================

[INFO] V6D Benchmark starting...
[INFO] Local IP:Port: 10.0.1.100:8080
[INFO] V6D storage registered: v6d_v6d_test_cluster
[INFO] Instance registered: v6d_test_cluster
[INFO] V6D node registered: 10.0.1.100:8080 with mediums [mem, disk]
[INFO] Generated dataset: 50000 blocks
[INFO] Starting 16 worker threads with QPS limit: 5000.00
[INFO] Running benchmark (press Ctrl+C to stop)...

[INFO] [Metrics] QPS=4985.20, AvgLatency=450us, SuccessRate=99.80%, Bandwidth=45.20Mbps, AddQPS=3489.64, QueryQPS=997.04, DeleteQPS=498.52, Verification: Passed=12500, Failed=2

[INFO] [Metrics] QPS=5002.00, AvgLatency=448us, SuccessRate=99.85%, Bandwidth=46.00Mbps, AddQPS=3501.40, QueryQPS=1000.40, DeleteQPS=500.20, Verification: Passed=25000, Failed=3

# 收到SIGTERM信号后:
[INFO] Received signal 15, initiating shutdown...
[INFO] Worker thread 0 stopped
[INFO] Worker thread 1 stopped
...
[INFO] Benchmark finished.
[INFO] Total requests: 600000, Success: 598800, Failed: 1200
[INFO] Verification: Passed=120000, Failed=15, MissingKeys=42, UnexpectedKeys=8
[INFO] Benchmark finished. Final metrics logged above.
```

## 结果验证

工具会自动维护数据集的期望状态，并验证查询结果的正确性：

1. **AddBlock成功后**：记录该block_key应该在KVCM中存在
2. **DeleteBlock成功后**：从期望状态中移除该block_key
3. **Query后**：对比查询结果与期望状态：
   - 检查期望存在的key是否在查询结果中返回
   - 检查期望已删除的key是否不再返回
   - 统计missing_keys和unexpected_keys

验证结果会汇报到Kmonitor并在日志中打印。

## QPS限流

工具使用滑动时间窗口算法实现QPS限流：
- 维护当前1秒窗口内的请求计数
- 当计数超过`TARGET_QPS`时，线程会等待到下一个窗口
- 确保整体QPS不会超过配置的目标值

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    V6D Benchmark                         │
├─────────────────────────────────────────────────────────┤
│  Main Thread                                             │
│  ├─ 解析环境变量配置                                      │
│  ├─ 初始化Kmonitor                                       │
│  ├─ 注册V6D节点和实例                                     │
│  └─ 等待工作线程完成                                     │
├─────────────────────────────────────────────────────────┤
│  Worker Threads (NUM_THREADS)                            │
│  ├─ QPS限流控制                                          │
│  ├─ 执行压测操作(Add/Query/Delete)                        │
│  ├─ 记录延迟和成功/失败状态                               │
│  └─ 验证查询结果                                         │
├─────────────────────────────────────────────────────────┤
│  Metrics Reporter Thread                                 │
│  ├─ 每10秒采集一次指标                                    │
│  ├─ 计算QPS、延迟百分位、成功率、带宽                      │
│  └─ 汇报到Kmonitor                                       │
├─────────────────────────────────────────────────────────┤
│  Result Verifier                                         │
│  ├─ 维护期望状态映射                                      │
│  ├─ 验证查询结果正确性                                    │
│  └─ 统计missing/unexpected keys                          │
└─────────────────────────────────────────────────────────┘
```

## 文件结构

```
kv_cache_manager/tools/v6d_benchmark/
├── BUILD                          # Bazel构建文件
├── v6d_benchmark_main.cc          # main函数入口
├── v6d_benchmark.h                # 压测工具类定义
├── v6d_benchmark.cc               # 压测工具实现
├── http_client.h                  # HTTP客户端头文件
├── http_client.cc                 # HTTP客户端实现
├── metrics_reporter.h             # 指标汇报头文件
├── metrics_reporter.cc            # 指标汇报实现
├── result_verifier.h              # 结果验证器
├── config.h                       # 环境变量配置解析
└── README.md                      # 使用说明
```

## 注意事项

1. **多机部署**：确保所有机器的`INSTANCE_ID`相同，但`AUTO_DETECT_HOST=true`以使用各自的IP
2. **QPS限流**：`TARGET_QPS`是每台机器的目标QPS，多台机器的总QPS = `TARGET_QPS` × 机器数量
3. **数据集大小**：`NUM_BLOCKS`应足够大以避免重复操作同一个block
4. **Kmonitor配置**：如果不使用Kmonitor，设置`ENABLE_KMONITOR=false`
5. **优雅退出**：建议使用`kill <pid>`发送SIGTERM信号，而不是`kill -9`

## 故障排查

### 无法连接KVCM
- 检查`KVCM_BASE_URL`和`KVCM_ADMIN_URL`是否正确
- 确认网络连通性：`curl http://<kvcm-host>:8080/api/clusterInfo`

### 节点注册失败
- 检查`V6D_MEDIUMS`格式是否正确(逗号分隔，无空格)
- 确认`INSTANCE_ID`在KVCM中已创建对应的存储后端

### 指标未上报
- 检查`ENABLE_KMONITOR=true`
- 验证`KMONITOR_CONFIG`格式是否正确
- 查看日志中是否有Kmonitor初始化失败的错误

### QPS达不到预期
- 检查`TARGET_QPS`设置是否合理
- 增加`NUM_THREADS`提高并发度
- 检查KVCM服务端是否有性能瓶颈
