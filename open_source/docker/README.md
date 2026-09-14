# Docker镜像构建说明

## 开发镜像（Dockerfile.dev）

开发镜像用于Tair KVCache Manager的构建和开发环境。

### 构建与发布（GitHub Actions）

正式镜像由 [build-dev-image.yml](../../.github/workflows/build-dev-image.yml) 构建和发布。
在 GitHub Actions 中选择 **Create and publish a KVCM Docker image**，点击 **Run workflow**，
选择包含 Dockerfile 改动的分支，并将 `flavor` 设为 `dev`。

工作流分别构建 `linux/amd64` 和 `linux/arm64`，合并 manifest 后发布到
`ghcr.io/alibaba/tair-kvcache-kvcm-dev`，标签包含构建时间与 commit；从默认分支运行时还会更新 `latest`。
`flavor=integration` 用于构建服务端二进制包和集成镜像，本次开发依赖更新应选择 `dev`。

### 本地试构建（可选）

```bash
# 构建开发镜像
cd open_source/docker
docker build -f Dockerfile.dev -t kv_cache_manager_dev:latest .
```

### 开发依赖清单

以 `24230b154`（2026-02-13，上一版开发镜像）为基线，补齐后续开发流程使用的系统依赖和 Python 工具：

| 依赖 | Alibaba Cloud Linux 3 包名 | 用途与引入依据 |
| --- | --- | --- |
| jemalloc | `jemalloc` | `df8d3e72c`（2026-08-06）的 `package/script/start_server.sh` 默认查找并预加载 `libjemalloc.so.2`；缺失时会退回系统 allocator，影响性能复现。无需 `jemalloc-devel`，当前没有编译期链接。 |
| Valkey 服务端/命令行客户端 | `valkey` | 为 `8483a0516`（2026-07-11）引入的 Redis 协议集成测试提供独立服务。`integration_test/testlib/redis_server.py` 优先查找 `valkey-server`，兼容已有 `REDIS_SERVER_BIN` 覆盖及 `redis-server` 回退。镜像只安装程序，不自动启动服务。 |
| core dump 调试工具 | `gdb`、`file` | `f2cea041b`（2026-03-10）引入 CI core dump 分析，原先在出现 core 后临时安装。 |
| HTTP / JSON 命令行工具 | `curl`、`jq` | `5246532b3`（2026-07-08）引入的 `tools/scripts/get_host_cache_state_stress.sh` 直接调用。 |
| Python 构建工具 | `packaging==25.0` | 显式预装 `3rdparty/py/python_configure.bzl` 已有的构建期安装依赖，与 Ubuntu setup 脚本保持一致。 |
| Connector 静态检查 | `ruff`、`ty`、`pre-commit`（Python 包） | `f1ec6ccea`（2026-09-14）引入；与已有 `autopep8` 一并预装，Git hook 仍由开发者自行启用。 |
| Python 维护脚本依赖 | `requests`、`redis`（Python 包） | `7146e3658`（2026-03-19）的 stale cache 清理脚本及后续 ReportEvent 压测脚本使用。 |

Python 开发依赖统一维护在 [requirements-dev.txt](requirements-dev.txt)，固定直接依赖版本，开发镜像通过 Python 3.11 安装。
Ubuntu 环境的系统依赖也同步到 [setup_env_ubuntu.sh](../../tools/scripts/setup_env_ubuntu.sh)；该脚本不安装 Connector 静态检查工具，可按下方通用镜像示例补充。
Python 的 `redis` 包保留为现有维护脚本的协议客户端，连接的是 Valkey 服务。

近期新增的 msgpack-cxx、xxHash、zstd 和 Mooncake 升级由 `open_source/deps/*.bzl` 下载和构建；
`pydantic`、`orjson`、`grpcio-tools` 及 kvcm_ops 的运行依赖由 `open_source/deps/requirements_lock_cpu.txt` 管理。
它们继续使用 Bazel 的依赖声明。CUDA/MUSA 和推理引擎仍由对应的 Connector 开发环境提供；
`ty check` 所需的引擎 Python 环境见 [Connector 开发说明](../../kv_cache_manager/py_connector/README.md#type-environment)。

jemalloc 由服务启动脚本按进程启用。直接运行 Bazel 二进制进行性能测试时，可在该命令上设置
`LD_PRELOAD=/usr/lib64/libjemalloc.so.2`；不要在开发镜像中全局设置 `LD_PRELOAD`，以免影响 ASAN 等开发工具。

### 镜像验证

```bash
docker run --rm kv_cache_manager_dev:latest bash -lc '
  set -e
  test -r /usr/lib64/libjemalloc.so.2
  LD_PRELOAD=/usr/lib64/libjemalloc.so.2 python3.11 -c "import ctypes; assert ctypes.CDLL(None).mallctl"
  valkey-server --version
  valkey-cli --version
  gdb --version
  file --version
  curl --version
  jq --version
  python3.11 -m pip check
  python3.11 -c "import packaging, requests, redis"
  autopep8 --version
  ruff --version
  ty --version
  pre-commit --version
  buildifier --version
  USE_BAZEL_VERSION=6.4.0 bazelisk version
'
```

发布新镜像后，还需将 CI 中固定的 `2026_02_13_12_03_24230b1` 镜像标签更新为实际发布的新标签，已有固定标签不会自动获得这些依赖。

### 基于已有推理引擎开发镜像构建通用开发镜像（Manager+Connector）

通用开发镜像用于推理引擎（vllm/sglang） + Tair KVCache Manager + Tair KVCache Manager Connector的构建和开发环境。
参考 `open_source/docker/Dockerfile.dev` 补充相关依赖即可。
具体请结合推理引擎镜像的实际情况。
以下 apt 示例要求软件源提供 `valkey-server` 和 `valkey-tools`（例如 [Ubuntu 24.04 的 universe 仓库](https://packages.ubuntu.com/noble/valkey-server)）；
软件源不提供这两个包的旧基础镜像（如 Ubuntu 22.04），需先按 [Valkey 安装说明](https://valkey.io/topics/installation/)准备 Valkey，并从下面的 apt 安装列表中移除这两个包名。

<details>

<summary>示例Dockerfile</summary>

```dockerfile
ARG BASE_OS_IMAGE=vllm/vllm-openai:v0.11.2
ARG BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64"
ARG BUILDIFER_URL="https://github.com/bazelbuild/buildtools/releases/download/v8.2.1/buildifier-linux-amd64"
ARG BAZELISK_BASE_URL=https://mirrors.huaweicloud.com/bazel/

FROM $BASE_OS_IMAGE

ARG BAZELISK_URL
ARG BUILDIFER_URL
ARG BAZELISK_BASE_URL

USER root

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # 基础开发工具
    vim gcc g++ git openssh-client wget curl jq procps iproute2 tar gdb file \
    # RDMA相关依赖
    librdmacm-dev libibverbs-dev libnuma-dev \
    # 构建工具
    cpio rpm2cpio patchelf libaio-dev pigz libjemalloc2 valkey-server valkey-tools \
    # Python开发环境（Ubuntu 22.04默认使用Python 3.10）
    python3 python3-pip python3-dev \
    # 代码格式化工具
    clang-format \
    # ICU库（用于CLion IDE）
    libicu-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 可选：如果需要Python 3.11，可以添加PPA源并安装
# RUN apt-get install -y software-properties-common && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.11 python3.11-dev python3.11-distutils && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 将本仓库的 open_source/docker/requirements-dev.txt 放到构建上下文中
COPY requirements-dev.txt /tmp/requirements-dev.txt
# 配置Python包源并安装Python工具
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip3 install --no-cache-dir -r /tmp/requirements-dev.txt && \
    rm /tmp/requirements-dev.txt

# 安装Bazelisk和Buildifier
RUN wget "$BAZELISK_URL" -O /usr/local/bin/bazelisk && chmod a+x /usr/local/bin/bazelisk && \
    BAZELISK_BASE_URL=$BAZELISK_BASE_URL USE_BAZEL_VERSION=6.4.0 bazelisk && \
    wget "$BUILDIFER_URL" -O /usr/local/bin/buildifier && chmod a+x /usr/local/bin/buildifier

# 如果是sglang容器，需要移除冲突的jsoncpp库
# RUN apt-get remove -y libjsoncpp-dev
```

</details>

如果只需要在现有Ubuntu推理容器中补充依赖，可以使用以下命令：

<details>

<summary>快速依赖补充脚本</summary>

```bash
# 更新包列表并安装依赖
apt-get update && \
apt-get install -y --no-install-recommends \
    librdmacm-dev libibverbs-dev libnuma-dev \
    cpio rpm2cpio patchelf libaio-dev pigz libjemalloc2 valkey-server valkey-tools \
    python3 python3-pip python3-dev \
    clang-format libicu-dev curl jq gdb file

# 安装Python工具和Bazel
pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip3 install -r open_source/docker/requirements-dev.txt && \
    wget "https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64" -O /usr/local/bin/bazelisk && \
    chmod a+x /usr/local/bin/bazelisk && \
    BAZELISK_BASE_URL=https://mirrors.huaweicloud.com/bazel/ USE_BAZEL_VERSION=6.4.0 bazelisk && \
    wget "https://github.com/bazelbuild/buildtools/releases/download/v8.2.1/buildifier-linux-amd64" -O /usr/local/bin/buildifier && \
    chmod a+x /usr/local/bin/buildifier

#（仅sglang需要执行下一条）
apt-get remove -y libjsoncpp-dev # sglang容器内jsoncpp的和KVCM自带的有冲突
```

</details>

## 生产镜像 (Dockerfile.prod)

生产镜像用于部署 Tair KVCache Manager 服务。

### 构建命令

```bash
# 需要先构建二进制包。建议使用上述开发镜像作为构建环境。
sh open_source/package/build_for_image.sh
cp bazel-bin/package/kv_cache_manager_server.tar.gz open_source/package/
cd open_source/docker/
# 然后构建生产镜像
docker build -f Dockerfile.prod \
  --build-arg BINARY_PACKAGE_TAR=../package/kv_cache_manager.tar.gz \
  -t kv_cache_manager_prod:latest .
```

### 运行容器

```bash
# 运行生产容器
docker run -d --name kv_cache_manager \
    -p 6381:6381 -p 6382:6382 -p 6491:6491 -p 6492:6492 \
    kv_cache_manager_prod:latest

# 设置启动参数。可配置参数请参考docs/configuration.md
docker run -d --name kv_cache_manager \
    -p 3000:3000 -p 6382:6382 -p 6491:6491 -p 6492:6492 \
    -e kvcm.service.rpc_port=3000 \
    kv_cache_manager_prod:latest
```

注意对于NFS和HF3FS等KVCache存储后端，可能需要将文件系统中的相关目录挂载到容器中。
具体请参照存储后端配置要求。

### 默认端口说明

- 6381: MetaService gRPC 端口
- 6491: AdminService gRPC 端口
- 6382: MetaService HTTP 端口
- 6492: AdminService HTTP 端口

## 推理容器（Dockerfile.vllm/sglang）

推理容器镜像用于部署推理服务。

### 构建命令

基于已有推理容器镜像，构造包含Tair KVCache Manager Connector的推理容器镜像。

```bash
# 需要先构建wheel包，建议使用上述基于推理引擎开发镜像构建的通用开发镜像作为构建环境。
bazelisk build //kv_cache_manager/py_connector/vllm:kvcm_vllm_connector_wheel
# 如果需要指定编译器和CUDA架构要求等可以添加相关编译参数
# --action_env=CC=gcc-11 --action_env=CXX=g++-11 --@rules_cuda//cuda:archs="compute_75:sm_75;compute_80:sm_80;compute_86:sm_86;compute_89:sm_89;compute_90:sm_90,compute_90"
cp bazel-bin/kv_cache_manager/py_connector/vllm/*.whl ./
# 然后构建包含Tair KVCache Manager Connector的推理容器镜像。
docker build -f open_source/docker/Dockerfile.xxxx -t xxxx_with_tair_kvcm:latest .
```
