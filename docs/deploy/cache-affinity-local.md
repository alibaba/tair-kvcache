# 本地运行亲和性：Manager + SDK

本例在同一台 Linux 机器上启动真实 Manager，使用 NFS 文件后端和 C++ SDK，走完 **申请位置 → 写入数据 → 发布 → 匹配 → 读回校验**。随后用 HTTP 检查远端读的复制提示。数据面是实际文件 I/O；HTTP 中的远端 caller 仅用于验证控制面，不代表另一台物理机器。

整体设计见[亲和性设计](../design-cache-affinity-v1.md)，完整参数见[配置与使用](../cache-affinity-zh_CN.md)。PACE 双节点复制、源节点停机和物理回收需要[双机环境](../../integration_test/affinity/README.md)单独验收。

## 1. 构建

使用仓库开发环境（Bazelisk、C++17 工具链、Python 3.10+），在源码根目录执行；内源仓库对应 `github-opensource` 目录。以下命令在同一个 Bash 会话中运行。

```bash
bazelisk build --config=debug \
  //kv_cache_manager:kv_cache_manager_bin \
  //integration_test/affinity:local_client
repo_dir="$PWD"
export demo_dir="$(mktemp -d /tmp/kvcm-affinity-local-XXXXXX)"
```

## 2. 准备配置并启动

示例使用 RPC `46381`、HTTP `46380`、Admin HTTP `46382`。需要改端口时，同时修改服务启动参数和客户端 `address`。

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["demo_dir"])
(root / "data").mkdir()
startup = json.loads(Path("package/etc/default_startup_config.json").read_text())
startup["storage_config"]["storage_spec"].update(
    root_path=str(root / "data") + "/", key_count_per_file=1)
strategy = {
    "type": "local_replica",
    "write": {"ops": {"prefer_local": {"on_miss": "passthrough"}}},
    "read": {"on_miss": {
        "replication_hot_threshold": 2, "suppression_window_ms": 60000}},
    "replica_limits": {"max_replicas_per_key": 2, "max_instance_bytes": 1048576},
}
client = {
    "instance_group": "default",
    "instance_id": "affinity_local_demo",
    "address": ["127.0.0.1:46381"],
    "block_size": 128,
    "location_spec_infos": {"spec_0": 4096},
    "meta_channel_config": {"call_timeout": 10000},
    "sdk_config": {},
    "model_deployment": {
        "model_name": "local_demo", "dtype": "FP16", "use_mla": False,
        "tp_size": 1, "dp_size": 1, "pp_size": 1},
    "auto_replicate": False,
}
for name, value in [("startup", startup), ("affinity", strategy), ("client", client)]:
    (root / (name + ".json")).write_text(json.dumps(value, indent=2))
print(root)
PY

(
  cd "$demo_dir"
  exec "$repo_dir/bazel-bin/kv_cache_manager/kv_cache_manager_bin" \
    --env kvcm.service.rpc_port=46381 \
    --env kvcm.service.http_port=46380 \
    --env kvcm.service.admin_rpc_port=46381 \
    --env kvcm.service.admin_http_port=46382 \
    --env kvcm.startup_config="$demo_dir/startup.json" \
    --env kvcm.affinity.enabled=true \
    --env kvcm.affinity.strategy_file="$demo_dir/affinity.json"
) >"$demo_dir/server.stdout" 2>"$demo_dir/server.stderr" &
manager_pid=$!
trap 'kill -TERM "$manager_pid" 2>/dev/null; wait "$manager_pid" 2>/dev/null' EXIT
```

确认服务已选出 Leader，再运行 SDK；启动期间可重试：

```bash
curl --noproxy '*' --fail --silent --show-error \
  http://127.0.0.1:46380/api/getClusterInfo \
  -H 'Content-Type: application/json' -d '{"trace_id":"local-readiness"}'
```

响应应包含 `header.status.code: "OK"` 和非空 `leader_node_id`。启动失败时查看 `$demo_dir/server.stderr` 和 `$demo_dir/logs/`。策略文件仅在启动时加载，修改后需重启。

## 3. 执行真实 SDK 读写

```bash
set -o pipefail
"$repo_dir/bazel-bin/integration_test/affinity/local_client" \
  "$demo_dir/client.json" | tee "$demo_dir/client.out"
```

成功退出时会打印：

```text
KEY=<本次生成的 block key>
CALLER=<本机 IP>
URI=<Save 实际返回的文件 URI>
LOCAL_SDK_OK
```

[示例源码](../../integration_test/affinity/local_client.cc)固定使用 `spec_0` 和 4096 字节 CPU buffer，会自动注册实例并校验：

1. `StartWrite` 返回的位置属于 SDK 的 `GetCallerNode()`。
2. `SaveKvCaches` 写入真实数据后，未 `FinishWrite` 时仍不可读。
3. `FinishWrite` 提交 Save 实际返回的 URI；之后 Match 命中同一位置，且本地读无复制提示。
4. `LoadKvCaches` 读回全部 4096 字节，与原始数据逐字节一致。

Save 失败时示例调用失败 Finish 回滚；Finish 返回异常后不会再发相反的 Finish。每次运行生成不同 key，可重复执行；本例容量预算为 1 MiB。

## 4. 检查远端读的复制提示

从 SDK 输出取 key。连续两次从同一个虚拟远端 caller 查询；第一次无 hint，第二次达到本例热度门限，应返回以 `remote_demo` 为目标、包含原始 URI 的 hint：

```bash
block_key="$(sed -n 's/^KEY=//p' "$demo_dir/client.out")"
for attempt in 1 2; do
  curl --noproxy '*' --fail --silent --show-error \
    http://127.0.0.1:46380/api/getCacheLocation \
    -H 'Content-Type: application/json' \
    -d "{\"trace_id\":\"remote-$attempt\",\"instance_id\":\"affinity_local_demo\",
         \"query_type\":\"QT_PREFIX_MATCH\",\"block_keys\":[$block_key],
         \"caller\":{\"node_id\":\"remote_demo\",\"replication_capabilities\":1}}"
  echo
done
```

两次远端查询连续执行，避免热度衰减影响门限。响应中的 `hints` 是复制建议，不代表已产生副本；示例关闭了 `auto_replicate`。同机 NFS 的 caller 身份相同，增加 SDK 进程也不会变成两个物理节点；NFS 无法按 `remote_demo` 强制分配，严格目标复制应失败。

## 5. 自动回归与结束

```bash
bazelisk test --config=debug --cache_test_results=no --test_output=errors \
  //integration_test/client_test:manager_client_test \
  //integration_test/client_test:client_scheduler_test \
  //integration_test/affinity:affinity_replication_test
```

`manager_client_test` 覆盖 3 个 block × 2 个 spec 的实际写入、部分发布和读回：Save 后仍 miss，失败发布的 block 仍 miss，成功发布的 block 每个 spec 都逐字节一致。`affinity_replication_test` 用真实服务覆盖 25 个控制面场景，包括 strict 目标失败、hint、预算回滚、实例隔离和重启恢复；恢复用例使用持久化文件元数据，无需 Redis。ASAN 回归增加 `--config=asan`。

结束手动示例：

```bash
kill -TERM "$manager_pid"
wait "$manager_pid"
trap - EXIT
```

数据、配置和日志保留在 `$demo_dir` 供排查。本例使用本地元数据，重启不保留实例与缓存索引；它不验收 NFS 的物理删除、PACE 复制或持久化恢复能力。
