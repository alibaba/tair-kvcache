[中文版本](README_zh.md)

# User Request Dataset Collection

* `hisim` supports simulation based on user request replay. This tool helps you achieve low-intrusion and seamless request data collection for subsequent simulation analysis
* This tool intercepts parameters of key SGLang functions through dynamic injection to capture request information

## Steps
* First, specify the generated dataset path through the environment variable `HISIM_BENCHMARK_OUT_DIR`
* Use our provided script to start the inference service and enable dataset capture through dynamic injection. There are two ways:
  1. (Recommended) Use the custom startup script, e.g., `python ./serving_hook/sglang_launch_server.py --model Qwen/Qwen3-8B`
  2. Utilize Python's mechanism (`usercustomize.py` auto-loading), `PYTHONPATH=`pwd`/serving_hook python -m sglang.launch_server --model Qwen/Qwen3-8B`
* Users initiate several requests, or use the bench_serving provided by the sglang framework for load testing, for example:
```python
python3 -m sglang.launch_server \
    --warmup-requests 0 \
    --dataset-name random \
    --request-rate 4 \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts 50 
```
* After collection is complete, send the following collection command to the inference service, and you will get the user request dataset in the path specified by the `HISIM_BENCHMARK_OUT_DIR` environment variable:
```shell
curl http://localhost:30000/start_profile
```
* Each request data includes:
```json
{"rid": "21e5xx", "timestamp":732.31, "queue_end":732.39, "output_length": 1024, "input_length": 1024, "final_prefix_cache_len":679, "input_ids": [92, 911, ...], "output_ids": [24, 19, ...]}
```
* Note: This collection step requires GPU usage

