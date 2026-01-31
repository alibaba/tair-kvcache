# Docker 镜像构建说明

## 开发镜像 (Dockerfile.dev)

开发镜像用于构建和开发环境。

### 构建命令
```bash
# 构建开发镜像
docker build -f Dockerfile.dev -t kv_cache_manager_dev:latest .
```


### 基于已有推理引擎开发镜像构建通用开发镜像(Manager+Connector)
参考open_source/docker/Dockerfile.dev补充相关依赖即可。具体请结合推理引擎镜像的实际情况

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
    vim gcc g++ git openssh-client wget procps iproute2 tar \
    # RDMA相关依赖
    librdmacm-dev libibverbs-dev libnuma-dev \
    # 构建工具
    cpio patchelf libaio-dev pigz \
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

# 配置Python包源并安装Python工具
RUN pip3 config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip3 install autopep8

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
    cpio patchelf libaio-dev pigz \
    python3 python3-pip python3-dev \
    clang-format libicu-dev

# 安装Python工具和Bazel
pip3 config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
pip3 install autopep8 && \
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


## 推理容器 (Dockerfile.vllm&sglang)
推理容器镜像用于部署推理服务。
### 构建命令
基于已有推理容器镜像，构造包含Tair KVCache Manager Connector的推理容器镜像。
```bash
# 需要先构建wheel包，建议使用上述基于推理引擎开发镜像构建的通用开发镜像作为构建环境。
bazelisk build //kv_cache_manager/py_connector/vllm:kvcm_vllm_connector_wheel --config=client_with_cuda
# 如果需要指定编译器和CUDA架构要求等可以添加相关编译参数
# --action_env=CC=gcc-11 --action_env=CXX=g++-11 --@rules_cuda//cuda:archs="compute_75:sm_75;compute_80:sm_80;compute_86:sm_86;compute_89:sm_89;compute_90:sm_90,compute_90"
cp bazel-bin/kv_cache_manager/py_connector/vllm/*.whl ./
# 然后构建包含Tair KVCache Manager Connector的推理容器镜像。
docker build -f open_source/docker/Dockerfile.xxxx -t xxxx_with_tair_kvcm:latest .
```