# KVCM Raft 集群部署（3 节点）

基于 Raft 共识协议的 3 节点 KVCM 高可用集群，**零外部依赖**（不需要 Redis / Nacos）。

## 快速开始

```bash
cd deploy/raft-cluster
docker compose up -d
```

首次启动会编译 `kv_cache_manager_bin`（约 5 分钟，Bazel 缓存会被持久化到 volume，后续秒启）。

## 集群拓扑

| 节点 | server_id | 宿主机端口 | 容器内端口 |
|------|-----------|-----------|-----------|
| kvcm-node1 | 1 | 6381/6382/6491/6492/9001 | 6381/6382/6491/6492/9001 |
| kvcm-node2 | 2 | 6383/6384/6493/6494/9002 | 6381/6382/6491/6492/9002 |
| kvcm-node3 | 3 | 6385/6386/6495/6496/9003 | 6381/6382/6491/6492/9003 |

端口说明：
- `6381` / `6382`：MetaService gRPC / HTTP
- `6491` / `6492`：AdminService gRPC / HTTP
- `9001-9003`：Raft 内部通信

## 常用操作

```bash
# 查看日志
docker compose logs -f

# 查看单个节点日志
docker compose logs -f kvcm-node1

# 停止一个节点（模拟故障）
docker compose stop kvcm-node1

# 恢复节点
docker compose start kvcm-node1

# 完全清理（包括数据卷）
docker compose down -v
```

## 自定义配置

### 使用不同的开发镜像

```bash
KVCM_DEV_IMAGE=my-registry/kvcm-dev:latest docker compose up -d
```

### 指定源码目录

默认挂载 `../..`（即仓库根目录）。如需更改：

```bash
KVCM_SRC_DIR=/path/to/KVCacheManager/github-opensource docker compose up -d
```

## 配置文件说明

- `conf/node{1,2,3}.conf` — 每个节点的 KVCM 配置，核心是 `kvcm.raft.*` 参数
- `conf/logger.conf` — 日志配置
- `conf/startup_config.json` — 启动时自动注册的 StorageConfig 和 InstanceGroup

### Raft 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kvcm.raft.server_id` | - | 节点唯一 ID（正整数） |
| `kvcm.raft.host` | - | Raft 通信地址（容器 hostname） |
| `kvcm.raft.port` | - | Raft 通信端口 |
| `kvcm.raft.peers` | - | 集群成员列表，格式：`id:host:port,...` |
| `kvcm.raft.data_dir` | - | 持久化目录（非空即启用 Raft 模式） |
| `kvcm.raft.election_timeout_lower` | 500 | 选举超时下界（ms） |
| `kvcm.raft.election_timeout_upper` | 1000 | 选举超时上界（ms） |
| `kvcm.raft.heart_beat_interval` | 200 | 心跳间隔（ms） |
| `kvcm.raft.snapshot_distance` | 10000 | 每多少条日志触发一次 snapshot |

## 架构

```
                    ┌──────────────┐
                    │   Client     │
                    └──────┬───────┘
                           │ gRPC / HTTP
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌───────────┐┌───────────┐┌───────────┐
        │  Node 1   ││  Node 2   ││  Node 3   │
        │  (Raft)   ││  (Raft)   ││  (Raft)   │
        │           ││           ││           │
        │ MetaRaft  ││ MetaRaft  ││ MetaRaft  │
        │ Backend   ││ Backend   ││ Backend   │
        │     │     ││     │     ││     │     │
        │ RaftCoord ││ RaftCoord ││ RaftCoord │
        │  ┌──┴──┐  ││  ┌──┴──┐  ││  ┌──┴──┐  │
        │  │LMDB │  ││  │LMDB │  ││  │LMDB │  │
        │  │ Log  │  ││  │ Log  │  ││  │ Log  │  │
        │  └─────┘  ││  └─────┘  ││  └─────┘  │
        └───────────┘└───────────┘└───────────┘
              ◄──── Raft Consensus ────►
```

- 只有 Leader 节点处理写请求，Follower 自动转发
- 节点故障后 Raft 自动选出新 Leader（秒级）
- 重启节点从 LMDB 恢复日志，增量追赶（无需全量 snapshot）
