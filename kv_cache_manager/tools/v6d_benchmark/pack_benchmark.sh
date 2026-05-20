#!/bin/bash
#
# v6d_benchmark 打包脚本
# 将编译产物、配置文件、运行脚本打包为自包含的 tar.gz，方便分发到线上压测环境
#
# 用法:
#   先构建:  bazel build //kv_cache_manager/tools/v6d_benchmark:v6d_benchmark
#   本地包:  bash kv_cache_manager/tools/v6d_benchmark/pack_benchmark.sh
#   Hippo包: bash kv_cache_manager/tools/v6d_benchmark/pack_benchmark.sh --hippo
#
#   本地包: 结构化目录 (bin/ conf/ run.sh env.sh logs/)
#   Hippo包: 扁平结构 (v6d_benchmark benchmark_alog.conf run.sh env.sh)，符合Hippo部署约定
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCHMARK_DIR="${SCRIPT_DIR}"
BAZEL_BIN="${PROJECT_ROOT}/bazel-bin/kv_cache_manager/tools/v6d_benchmark/v6d_benchmark"

# 解析参数
HIPPO_MODE=false
if [[ "${1:-}" == "--hippo" ]]; then
    HIPPO_MODE=true
fi

# 输出包名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if $HIPPO_MODE; then
    PACKAGE_NAME="v6d_benchmark_hippo_${TIMESTAMP}"
else
    PACKAGE_NAME="v6d_benchmark_${TIMESTAMP}"
fi
PACKAGE_DIR="${PROJECT_ROOT}/${PACKAGE_NAME}"
OUTPUT_TAR="${PROJECT_ROOT}/${PACKAGE_NAME}.tar.gz"

echo "============================================"
echo " V6D Benchmark 打包工具"
if $HIPPO_MODE; then
    echo " 模式: Hippo (扁平结构)"
else
    echo " 模式: 本地 (结构化目录)"
fi
echo "============================================"
echo ""

# 1. 检查二进制是否存在
if [ ! -f "${BAZEL_BIN}" ]; then
    echo "[ERROR] 找不到压测二进制: ${BAZEL_BIN}"
    echo "请先运行: bazel build //kv_cache_manager/tools/v6d_benchmark:v6d_benchmark"
    exit 1
fi

echo "[INFO] 二进制: ${BAZEL_BIN}"
echo "[INFO] 大小:   $(du -h "${BAZEL_BIN}" | cut -f1)"
echo ""

# 2. 检查动态库依赖（仅做信息展示）
echo "[INFO] 动态库依赖检查:"
ldd "${BAZEL_BIN}" 2>&1 | grep -v "vdso" || true
echo ""

# 3. 创建打包目录结构
rm -rf "${PACKAGE_DIR}"
if $HIPPO_MODE; then
    # Hippo模式: 扁平结构，所有文件在根目录
    mkdir -p "${PACKAGE_DIR}"
else
    # 本地模式: 结构化目录
    mkdir -p "${PACKAGE_DIR}/bin"
    mkdir -p "${PACKAGE_DIR}/conf"
    mkdir -p "${PACKAGE_DIR}/logs"
fi

# 4. 复制二进制
if $HIPPO_MODE; then
    cp -v "${BAZEL_BIN}" "${PACKAGE_DIR}/v6d_benchmark"
    chmod +x "${PACKAGE_DIR}/v6d_benchmark"
else
    cp -v "${BAZEL_BIN}" "${PACKAGE_DIR}/bin/v6d_benchmark"
    chmod +x "${PACKAGE_DIR}/bin/v6d_benchmark"
fi

# 5. 复制配置文件
if $HIPPO_MODE; then
    cp -v "${BENCHMARK_DIR}/benchmark_alog.conf" "${PACKAGE_DIR}/"
else
    cp -v "${BENCHMARK_DIR}/benchmark_alog.conf" "${PACKAGE_DIR}/conf/"
fi

# 6. 生成线上运行脚本 (run.sh)
if $HIPPO_MODE; then
    cat > "${PACKAGE_DIR}/run.sh" << 'RUNEOF'
#!/bin/bash
#
# V6D Benchmark Hippo线上运行脚本
# Hippo会通过processInfos.envs设置环境变量，此脚本仅做路径检测和启动
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/v6d_benchmark"

# 日志配置（Hippo扁平结构：conf文件在包根目录）
export ALOG_CONF="${ALOG_CONF:-${SCRIPT_DIR}/benchmark_alog.conf}"

# 切到包根目录，确保 benchmark_alog.conf 中 logs/ 相对路径落在这里
cd "${SCRIPT_DIR}"

echo "=========================================="
echo " V6D Benchmark 线上压测 (Hippo)"
echo "=========================================="
echo "Binary:       ${BIN}"
echo "KVCM Base:    ${KVCM_BASE_URL:-<from hippo env>}"
echo "Instance:     ${INSTANCE_ID:-<from hippo env>}"
echo "Threads:      ${NUM_THREADS:-<from hippo env>}"
echo "Target QPS:   ${TARGET_QPS:-<from hippo env>}"
echo "Test Mode:    ${TEST_MODE:-<from hippo env>}"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止压测"
echo ""

exec "${BIN}"
RUNEOF
else
    cat > "${PACKAGE_DIR}/run.sh" << 'RUNEOF'
#!/bin/bash
#
# V6D Benchmark 线上运行脚本
# 用法:
#   1. 按需修改下方环境变量
#   2. 执行: bash run.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/bin/v6d_benchmark"

# ==================== 配置区（按线上环境修改） ====================

# --- KVCM 连接 ---
export KVCM_BASE_URL="${KVCM_BASE_URL:-http://127.0.0.1:56020}"
export KVCM_ADMIN_URL="${KVCM_ADMIN_URL:-http://127.0.0.1:56040}"
export INSTANCE_ID="${INSTANCE_ID:-v6d_benchmark_online}"
export INSTANCE_GROUP="${INSTANCE_GROUP:-default}"

# --- V6D 节点 ---
export AUTO_DETECT_HOST="${AUTO_DETECT_HOST:-true}"
export V6D_PORT="${V6D_PORT:-8080}"
export V6D_MEDIUMS="${V6D_MEDIUMS:-mem}"

# --- 压测参数 ---
export NUM_BLOCKS="${NUM_BLOCKS:-10000}"
export BLOCK_SIZE="${BLOCK_SIZE:-128}"
export NUM_THREADS="${NUM_THREADS:-8}"
export TARGET_QPS="${TARGET_QPS:-1000}"
export ENABLE_QPS_LIMIT="${ENABLE_QPS_LIMIT:-true}"

# --- 压测模式 ---
export TEST_MODE="${TEST_MODE:-full}"
export QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-50}"
export ADD_RATIO="${ADD_RATIO:-0.7}"
export QUERY_RATIO="${QUERY_RATIO:-0.2}"
export DELETE_RATIO="${DELETE_RATIO:-0.1}"

# --- Kmonitor 上报 ---
export ENABLE_KMONITOR="${ENABLE_KMONITOR:-true}"
export KMONITOR_CONFIG="${KMONITOR_CONFIG:-{\"domain\":\"v6d_benchmark\"}}"
export REPORT_INTERVAL_MS="${REPORT_INTERVAL_MS:-10000}"

# --- 结果验证 ---
export ENABLE_VERIFICATION="${ENABLE_VERIFICATION:-true}"

# --- 日志配置 ---
export ALOG_CONF="${ALOG_CONF:-${SCRIPT_DIR}/conf/benchmark_alog.conf}"

# ==================== 运行 ====================

echo "=========================================="
echo " V6D Benchmark 线上压测"
echo "=========================================="
echo "Binary:       ${BIN}"
echo "KVCM Base:    ${KVCM_BASE_URL}"
echo "KVCM Admin:   ${KVCM_ADMIN_URL}"
echo "Instance:     ${INSTANCE_ID}"
echo "Threads:      ${NUM_THREADS}"
echo "Target QPS:   ${TARGET_QPS}"
echo "Test Mode:    ${TEST_MODE}"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止压测"
echo ""

exec "${BIN}"
RUNEOF
fi
chmod +x "${PACKAGE_DIR}/run.sh"

# 7. 生成配置文件模板 (env.sh)，方便 source 后使用
cat > "${PACKAGE_DIR}/env.sh" << 'ENVEOF'
#!/bin/bash
#
# 环境变量配置文件
# 使用方式: source env.sh && bash run.sh
#

# KVCM 连接
export KVCM_BASE_URL="http://127.0.0.1:56020"
export KVCM_ADMIN_URL="http://127.0.0.1:56040"
export INSTANCE_ID="v6d_benchmark_online"
export INSTANCE_GROUP="default"

# V6D 节点
export AUTO_DETECT_HOST=true
export V6D_PORT="8080"
export V6D_MEDIUMS="mem"

# 压测参数
export NUM_BLOCKS=10000
export BLOCK_SIZE=128
export NUM_THREADS=8
export TARGET_QPS=1000
export ENABLE_QPS_LIMIT=true

# 压测模式 (full / add / query / delete)
export TEST_MODE="full"
export QUERY_BATCH_SIZE=50
export ADD_RATIO=0.7
export QUERY_RATIO=0.2
export DELETE_RATIO=0.1

# Kmonitor 上报
export ENABLE_KMONITOR=true
export KMONITOR_CONFIG='{"domain":"v6d_benchmark"}'
export REPORT_INTERVAL_MS=10000

# 结果验证
export ENABLE_VERIFICATION=true

# 日志
export ALOG_CONF="conf/benchmark_alog.conf"
ENVEOF

# 8. 打包为 tar.gz
echo ""
echo "[INFO] 正在打包..."
if $HIPPO_MODE; then
    # Hippo模式: cd到临时目录内打包，解压后文件直接在根目录，不包含父目录
    cd "${PACKAGE_DIR}"
    tar czf "${OUTPUT_TAR}" .
    cd "${PROJECT_ROOT}"
else
    # 本地模式: 保留顶层目录名
    cd "${PROJECT_ROOT}"
    tar czf "${OUTPUT_TAR}" "${PACKAGE_NAME}"
fi

# 9. 清理临时目录
rm -rf "${PACKAGE_DIR}"

echo ""
echo "============================================"
echo " 打包完成!"
echo "============================================"
echo " 文件: ${OUTPUT_TAR}"
echo " 大小: $(du -h "${OUTPUT_TAR}" | cut -f1)"
echo ""
echo " 使用方法:"
if $HIPPO_MODE; then
    echo "   (Hippo部署) 上传tar.gz到OSS，更新JSON中packageURI后通过Hippo平台部署"
    echo "   解压后文件直接在根目录: v6d_benchmark benchmark_alog.conf run.sh env.sh"
else
    echo "   1. 上传到目标机器:  scp ${OUTPUT_TAR} user@host:/path/"
    echo "   2. 解压:           tar xzf ${OUTPUT_TAR##*/}"
    echo "   3. 进入目录:       cd ${PACKAGE_NAME}"
    echo "   4. 修改配置:       vim env.sh"
    echo "   5. (推荐) 加载配置: source env.sh"
    echo "   6. 启动压测:       bash run.sh"
fi
echo ""
