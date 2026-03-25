# sglang Connector 本地调试指南

本文档介绍如何在本地环境中构建 KVCM sglang connector，配合真实 sglang 服务进行调试。

## 前置条件

- 已安装 sglang 的 Python venv 环境
- 可用的模型文件
- GPU 显存需满足：模型权重 + KV Cache（由 `--mem-fraction-static` 控制）
- 可用内存需满足：GPU 上 KV Cache 大小 × `--hicache-ratio`

## 1. 构建并安装 wheel 包

在项目主目录下构建 connector wheel：

```bash
bazel build //kv_cache_manager/py_connector/sglang:kvcm_sglang_connector_wheel
```

构建产物在 `bazel-bin/kv_cache_manager/py_connector/sglang/` 下。安装到 sglang venv 中：

```bash
<sglang-venv>/bin/pip install --force-reinstall --no-deps \
    bazel-bin/kv_cache_manager/py_connector/sglang/kvcm_sglang_connector-*.whl
```

> **注意**：如果 bazel build 因网络问题失败（如无法拉取 rules_python），
> 需确保 bazel 缓存可用或网络畅通。

## 2. 启动 KVCM Manager

准备 startup config：

```bash
mkdir -p /tmp/kvcm_debug/nfs

cat > /tmp/kvcm_debug/startup_config.json << 'EOF'
{
  "instance_groups": [
    {
      "name": "default",
      "storage_backend": {
        "type": "file",
        "file_root": "/tmp/kvcm_debug/nfs/"
      }
    }
  ]
}
EOF
```

启动 manager（默认监听 6382 端口）：

```bash
bazel-bin/kv_cache_manager/manager/kv_cache_manager_bin \
    --startup_config_file=/tmp/kvcm_debug/startup_config.json
```

验证 manager 就绪：

```bash
curl -s http://127.0.0.1:6382/health
# 应返回 "OK"
```

## 3. 启动 sglang

关键参数说明：

| 参数 | 说明 |
|------|------|
| `--mem-fraction-static` | GPU 显存中用于 KV Cache 的比例，需确保模型权重 + KV Cache 不超显存 |
| `--hicache-ratio` | 主机内存 HiCache 大小 = KV Cache 大小 × ratio |
| `--page-size 64` | 每个 KV Cache page 的 token 数 |
| `--enable-cache-report` | 在 Prefill 日志中显示 `#cached-token`，调试时建议开启 |
| `--hicache-storage-backend dynamic` | 使用动态加载的存储后端 |

启动命令参考：

```bash
EXTRA_CONFIG='{
    "backend_name": "kvcm",
    "module_path": "kv_cache_manager.py_connector.sglang.connector",
    "class_name": "HiCacheKVCM",
    "instance_group": "default",
    "instance_id": "sglang-debug-01",
    "manager_uri": "http://127.0.0.1:6382",
    "hicache_storage_pass_prefix_keys": true
}'

python -m sglang.launch_server \
    --model-path <模型路径> \
    --host 127.0.0.1 --port 30000 \
    --enable-hierarchical-cache \
    --mem-fraction-static 0.80 \
    --hicache-ratio 1.2 \
    --page-size 64 \
    --enable-cache-report \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend dynamic \
    --hicache-storage-backend-extra-config "$EXTRA_CONFIG"
```

等待日志出现 `Application startup complete` 后即可发送请求。

> **注意**：`interface_v1` 无需在 `EXTRA_CONFIG` 中手动配置，connector 会在初始化时
> 自动声明。

## 4. 调试示例

### 4.1 验证 Backup + Prefetch 流程

发送一个足够长的请求（prompt 需超过 1 个 page = 64 tokens），触发 backup 写入 KVCM，
然后 flush GPU cache 后重复请求，观察是否从 KVCM prefetch 回来。

```bash
# 1. 发送请求（prompt 需 > 64 tokens 以凑满至少 1 个 page）
curl -s http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<模型名>",
        "messages": [
            {"role": "system", "content": "<长 system prompt, 200+ tokens>"},
            {"role": "user", "content": "你的问题"}
        ],
        "max_tokens": 16
    }'

# 2. 等待 backup 线程完成写入
sleep 5

# 3. flush GPU radix cache，确保下次请求不从 GPU/host 内存命中
curl -s http://127.0.0.1:30000/flush_cache

# 4. 重复相同请求
curl -s http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{...同上...}'
```

**预期**：第二次请求的 sglang Prefill 日志显示 `#cached-token` > 0。

### 4.2 验证 Partial Write（部分 block 已存在）

当请求 B 与请求 A 共享相同的 prefix 时，B 的 backup 阶段 KVCM 会告知 prefix 对应的
block 已存在不需要写入，只有新增 block 需要写入。

```bash
# 1. 清空缓存
curl -s http://127.0.0.1:30000/flush_cache

# 2. 请求 A：较短问题 + 长 system prompt，建立 prefix cache
curl ... -d '{"messages": [{"role":"system","content":"<长 prefix>"}, {"role":"user","content":"短问题"}], ...}'
sleep 5

# 3. flush GPU cache
curl -s http://127.0.0.1:30000/flush_cache

# 4. 请求 B：同 system prompt + 更长的 user message（触发 partial write）
curl ... -d '{"messages": [{"role":"system","content":"<同一个长 prefix>"}, {"role":"user","content":"更长的问题..."}], ...}'
sleep 5

# 5. flush 后重复请求 B，验证 prefetch 能取回全部 page（包括 A 写入的和 B 新写入的）
curl -s http://127.0.0.1:30000/flush_cache
curl ... -d '{...同请求 B...}'
```

**预期**：
- 请求 B 的 partial write 不报错
- 重复请求 B 的 Prefill 日志 `#cached-token` 等于全部完整 page 数 × 64

### 4.3 日志检查

调试时关注 sglang server 日志中的关键信息：

```bash
# 正常的 Prefill 日志，关注 #cached-token 字段
Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 256, ...

# 以下关键词不应出现，出现则表示有问题
Exception | NotImplementedError | Error | failed | Write page
```

## 5. 常见问题

### sglang OOM

`--mem-fraction-static` 设太小导致模型权重 + KV Cache 超过 GPU 显存。增大该值（如 0.8），
并确保可用内存 >= GPU KV Cache 大小 × `--hicache-ratio`。

### connector import 失败

如果 sglang 启动时报 `Failed to import backend 'kvcm'`，检查 wheel 是否已正确安装：

```bash
<sglang-venv>/bin/python -c "from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM; print('OK')"
```

## 6. 清理

```bash
# 停止 sglang
kill $(lsof -ti :30000)

# 停止 KVCM manager
kill $(lsof -ti :6382)

# 清理调试数据
rm -rf /tmp/kvcm_debug
```
