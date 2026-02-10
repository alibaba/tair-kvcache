[English Version](README.md)

# 用户请求数据集采集

* `hisim` 支持基于用户请求重放的仿真方式，本工具帮助您实现低侵入无感知的请求数据采集，用以后续仿真分析
* 本工具通过动态注入的方式，拦截 SGLang 关键函数的参数，获取到请求信息

## 步骤
* 首先通过环境变量`HISIM_BENCHMARK_OUT_DIR`指定生成的数据集路径
* 使用我们提供的脚本启动推理服务，动态注入使能数据集捕获，方式有两种：
  1. (推荐) 使用自定义的启动脚本，如 `python ./serving_hook/sglang_launch_server.py --model Qwen/Qwen3-8B`
  2. 利用 PYTHON 的机制(`usercustomize.py`自动加载)，`PYTHONPATH=`pwd`/serving_hook python -m sglang.launch_server --model Qwen/Qwen3-8B`
* 用户发起若干请求，或通过sglang框架提供的bench_serving打流，例如
```python
python3 -m sglang.launch_server \
    --warmup-requests 0 \
    --dataset-name random \
    --request-rate 4 \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts 50 
```
* 采集结束后，向推理服务发送如下收集指令，即可在`HISIM_BENCHMARK_OUT_DIR`环境变量指定的路径下，得到用户请求数据集
```shell
curl http://localhost:30000/start_profile
```
* 每条请求数据包括：
```json
{"rid": "21e5xx", "timestamp":732.31, "queue_end":732.39, "output_length": 1024, "input_length": 1024, "final_prefix_cache_len":679, "input_ids": [92, 911, ...], "output_ids": [24, 19, ...]}
```
* 注意事项：本采集步骤需要使用GPU

