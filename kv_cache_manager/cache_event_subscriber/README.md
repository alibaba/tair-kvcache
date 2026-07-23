# KVCM Cache Event Subscriber

独立进程消费 RTP-LLM/vLLM 的 KVCache 事件，并通过 KVCM HTTP `ReportEvent` 同步元数据。设计与一致性语义见 [设计文档](../../docs/design/cache_event_subscriber.md)。

## 运行依赖

- 公共依赖：Python 3.10+、仓库内 `KvCacheManagerClient`、`requests`、`protobuf`、`grpcio`。
- RTP 路径无需额外包。
- vLLM 路径额外安装 `requirements-vllm.txt` 中的 `msgspec` 与 `pyzmq`。

Subscriber 需从 tair-kvcache 源码树运行，或将 `kv_cache_manager` Python 包安装/挂载到 RTP 容器；RTP Launcher 只管理命令生命周期，不复制另一仓库的运行时。

KVCM 侧必须先配置专用元数据后端，并绑定到 instance group：

```json
{
  "global_unique_name": "engine_cache_events",
  "event_report": {
    "heartbeat_timeout_ms": 30000,
    "cleanup_grace_ms": 300000,
    "liveness_check_interval_ms": 5000
  }
}
```

对应 instance group 的 `event_reporting_storage_candidates` 需包含 `engine_cache_events`。Subscriber 固定以 `ST_EVENT_REPORT` 上报，并生成 `event-report://<host>/<medium>` 位置；不要配置为 Vineyard storage。

该 storage 只保存引擎缓存的可发现元数据，不属于 KVCM 容量池，也不能放入普通 `storage_candidates`；缓存淘汰由引擎负责，KVCM 仅依据事件、快照和节点失活清理位置。

RTP 示例：

```bash
python -m kv_cache_manager.cache_event_subscriber \
  --engine rtp \
  --manager-uri http://manager:8080 \
  --instance-group default \
  --instance-id model-a \
  --host-ip-port worker-a:9000 \
  --rtp-endpoints worker-a:9001 \
  --block-size 64 \
  --model-name model-a \
  --cache-group-count 1
```

vLLM 运行时还需安装 `requirements-vllm.txt`，并配置 publisher 的 PUB 与 replay endpoint。冷启动 replay 已裁剪时 Subscriber 会拒绝上报，必须与引擎共同重启或等到明确的 `AllBlocksCleared`。

RTP 要求 `--rtp-endpoints` 恰好提供 `--dp-size` 个互不重复的 endpoint，只有全部拉取成功才提交聚合状态。当前 vLLM 路径显式限制 `--dp-size=1`，避免多 DP 时静默只消费 rank 0；多 publisher 聚合需后续作为独立能力扩展。

vLLM 示例：

```bash
python -m kv_cache_manager.cache_event_subscriber \
  --engine vllm \
  --manager-uri http://manager:8080 \
  --instance-group default \
  --instance-id model-a \
  --host-ip-port worker-a:9000 \
  --vllm-pub-endpoint tcp://worker-a:5557 \
  --vllm-replay-endpoint tcp://worker-a:5558 \
  --engine-health-url http://worker-a:8000/health \
  --block-size 16 \
  --model-name model-a
```

由 RTP-LLM 托管时，将完整 argv 放入 `KVCM_CACHE_EVENT_SUBSCRIBER_COMMAND`。命令中的 `{rtp_endpoint}` 会在 backend 健康后替换为本机 gRPC endpoint；`KVCM_CACHE_EVENT_SUBSCRIBER_OWNER_RANK` 默认是 `0`，`KVCM_CACHE_EVENT_SUBSCRIBER_REQUIRED=false` 时异常退出会限频重启，设为 `true` 时失败由 RTP `ProcessManager` 向整组进程传播。

Subscriber 只有在 Source 已准备好首个权威快照后才注册节点，并在快照得到 KVCM ACK 后启动 HEARTBEAT。首快照失败会先上报 `HOST_DOWN` 取消该次注册，再用未提交的同一快照重新注册重试。
