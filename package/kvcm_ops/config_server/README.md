# kvcm_ops · config_server

`kvcm_ops` 中针对 **ConfigServer** 的运维工具集合。所有 config_server 子命令统一通过 `python3 -m kvcm_ops config_server <subcommand>` 调用。

ConfigServer 使用 **instance_pin** 路由模式（显式映射表：instance → cell URL）。

可通过 `server_capability` 探测当前服务器的路由模式：

```bash
python3 -m kvcm_ops config_server server_capability --url http://127.0.0.1:9101
# routing_mode   = INSTANCE_PIN
# server_version = ...
```

---

## 安装

`kvcm_ops` 已经随 ConfigServer 内源安装包一起发布（`config_server.tar.gz` 里带 `kvcm_ops-0.1.0-py3-none-any.whl`），
`start_server.sh` 启动 server 时会自动 `pip install`。如果你想在本地直接跑：

```bash
pip install kvcm_ops-0.1.0-py3-none-any.whl
# 或者在仓库内直接 -m
cd KVCacheManager/github-opensource/package
python3 -m kvcm_ops config_server --help
```

之后下文统一用 `python3 -m kvcm_ops config_server <subcommand> ...` 来演示。

所有在线子命令默认连本机 `http://127.0.0.1:9101`，可用 `--url` 指向其它实例。

---

## hpnzone 生命周期管理

```bash
# 列出所有 hpnzone
python3 -m kvcm_ops config_server list-hpnzones
python3 -m kvcm_ops config_server list-hpnzones --url http://10.0.0.1:9101

# 创建 hpnzone
python3 -m kvcm_ops config_server create-hpnzone --hpnzone prod_zone_a

# 删除 hpnzone（交互确认，--yes 跳过）
python3 -m kvcm_ops config_server delete-hpnzone --hpnzone prod_zone_a
python3 -m kvcm_ops config_server delete-hpnzone --hpnzone prod_zone_a --yes
```

---

## server_capability（探测路由模式）

```bash
python3 -m kvcm_ops config_server server_capability
python3 -m kvcm_ops config_server server_capability --url http://10.0.0.1:9101
python3 -m kvcm_ops config_server server_capability --url http://10.0.0.1:9101 --json
```

---

## instance_pin 模式

> 概念速查：
> - **cell**：一个 kvcm 服务发现 URL（一主一备，对外原子）
> - **group pin**：(hpnzone, instance_group) → cell；动态回退，不物化为 instance pin
> - **instance pin**：(hpnzone, instance) → cell；最高优先级，持久化
> - **优先级**：instance pin > group pin > 兜底分配（选最轻 cell 并写入 instance pin）

### instance_pin 工具

```bash
python3 -m kvcm_ops config_server instance_pin --help
```

#### Cell 管理

```bash
# 列出 hpnzone 下所有 cell
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a list-cells

# 注册 cell（幂等）
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    register-cell --cell-url url://kvcm-cell-0

# 注销 cell（必须先清空该 cell 上的 instance pin 和 group pin）
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    unregister-cell --cell-url url://kvcm-cell-0
```

#### Group Pin 管理

```bash
# 列出所有 group pin
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a list-group-pins

# 把 groupA 钉到某 cell
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    register-group-pin --group groupA --cell-url url://kvcm-cell-1

# 迁移 group pin 到另一 cell（仅影响"未自己 pin"的 instance）
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    reassign-group-pin --group groupA --cell-url url://kvcm-cell-2

# 解除 group pin
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    unregister-group-pin --group groupA
```

#### Instance 管理

```bash
# 注册 instance（通过 group pin 或兜底分配自动选 cell）
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    register-instance --instance model-x:dep-007 --group groupA

# 显式指定 cell
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    register-instance --instance model-x:dep-007 --group groupA \
    --pin-url url://kvcm-cell-1

# 查询 instance 当前映射到哪个 cell
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    resolve-instance --instance model-x:dep-007 --group groupA

# 迁移 instance 到另一 cell
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    reassign-instance --instance model-x:dep-007 --cell-url url://kvcm-cell-2

# 解除 instance pin（之后退回 group pin 或兜底分配）
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    unregister-instance --instance model-x:dep-007
```

#### 通用开关

| 开关 | 作用 |
|---|---|
| `--url URL`     | ConfigServer HTTP base，默认 `http://127.0.0.1:9101` |
| `--hpnzone ID`  | 目标 hpnzone_id，必填 |
| `--timeout SEC` | HTTP 超时，默认 5s |
| `--dry-run`     | 只打印请求 body，不发送 |
| `-v / --verbose`| 打印 HTTP 请求细节 |

---

## 典型工作流

### A. 搭建一个 hpnzone

```bash
# 0. 创建 hpnzone
python3 -m kvcm_ops config_server create-hpnzone --hpnzone prod_zone_a

# 1. 注册 cell
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    register-cell --cell-url url://kvcm-cell-0
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    register-cell --cell-url url://kvcm-cell-1

# 2. 规划 group pin
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    register-group-pin --group groupA --cell-url url://kvcm-cell-0
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    register-group-pin --group groupB --cell-url url://kvcm-cell-1

# 3. 验证
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a list-cells
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a list-group-pins
```

### B. instance_pin 迁移

```bash
# 查看当前状态
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    resolve-instance --instance model-x:dep-007 --group groupA

# 迁移到另一 cell
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    reassign-instance --instance model-x:dep-007 --cell-url url://kvcm-cell-1

# 验证
python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \
    resolve-instance --instance model-x:dep-007 --group groupA
```

### C. 下线 hpnzone

```bash
# 需先清空所有 instance pin 和 group pin，再注销所有 cell，最后 delete-hpnzone
python3 -m kvcm_ops config_server delete-hpnzone --hpnzone prod_zone_a
```
