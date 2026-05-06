#!/bin/bash

# V6D Benchmark 运行示例脚本
# 使用方法:
#   1. 修改下面的配置参数
#   2. 运行: bash run_benchmark.sh

# ==================== 配置区 ====================

# KVCM连接配置
export KVCM_BASE_URL="http://127.0.0.1:56020"
export KVCM_ADMIN_URL="http://127.0.0.1:56040"
export INSTANCE_ID="v6d_benchmark_local"
export INSTANCE_GROUP="default"

# V6D节点配置
export AUTO_DETECT_HOST=true          # 自动获取本机IP
export V6D_PORT="8080"
export V6D_MEDIUMS="mem"              # 支持的介质类型

# 压测参数
export NUM_BLOCKS=1000                # 数据集大小
export BLOCK_SIZE=128                 # 每个block大小(字节)
export NUM_THREADS=4                  # 并发线程数

# QPS限流配置
export TARGET_QPS=100                 # 目标QPS
export ENABLE_QPS_LIMIT=true          # 启用QPS限流

# 压测模式
export TEST_MODE="full"               # full/add/query/delete
export QUERY_BATCH_SIZE=10            # 每次查询的key数量

# 操作比例 (仅full模式)
export ADD_RATIO=0.7                  # Add操作占70%
export QUERY_RATIO=0.2                # Query操作占20%
export DELETE_RATIO=0.1               # Delete操作占10%

# Kmonitor配置
export ENABLE_KMONITOR=false          # 本地测试关闭Kmonitor
export KMONITOR_CONFIG='{"domain":"v6d_benchmark"}'  # Kmonitor配置
export REPORT_INTERVAL_MS=10000       # 指标汇报间隔(10秒)

# 结果验证
export ENABLE_VERIFICATION=true       # 启用结果验证

# ==================== 运行区 ====================

echo "=========================================="
echo "V6D Benchmark 配置"
echo "=========================================="
echo "KVCM Base URL: $KVCM_BASE_URL"
echo "Instance ID:   $INSTANCE_ID"
echo "Test Mode:     $TEST_MODE"
echo "Num Threads:   $NUM_THREADS"
echo "Target QPS:    $TARGET_QPS"
echo "Num Blocks:    $NUM_BLOCKS"
echo "Auto Host:     $AUTO_DETECT_HOST"
echo "=========================================="
echo ""

# 检查二进制文件是否存在
BENCHMARK_BIN="./bazel-bin/kv_cache_manager/tools/v6d_benchmark/v6d_benchmark"

if [ ! -f "$BENCHMARK_BIN" ]; then
    echo "错误: 找不到压测程序 $BENCHMARK_BIN"
    echo "请先运行: bazel build //kv_cache_manager/tools/v6d_benchmark:v6d_benchmark"
    exit 1
fi

# 运行压测
echo "启动压测程序..."
echo "按 Ctrl+C 停止压测"
echo ""

$BENCHMARK_BIN

# 获取退出码
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "压测正常结束"
else
    echo "压测异常退出，退出码: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
