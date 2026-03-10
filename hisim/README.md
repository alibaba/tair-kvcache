# Hisim

---
[中文版本](README_zh.md)

---

## Background

As large language models (LLMs) are rapidly deployed at scale for inference services, inference performance directly impacts user experience, service cost, and resource efficiency. Key metrics such as Time to First Token (TTFT), Time Per Output Token (TPOT), and system throughput are highly dependent on the complex interplay among model architecture, hardware platforms (e.g., A100/H100), inference engines (e.g., SGLang, vLLM, TensorRT-LLM), and runtime configurations (e.g., quantization, batching, parallelism strategies).

Traditional end-to-end stress testing on real GPU clusters is expensive and time-consuming, making it impractical to efficiently explore the vast space of configuration combinations. To address this, we propose **Tair-KVCache Hisim**, a high-performance CPU-based simulation system. Hisim enables fast, low-cost, and high-fidelity prediction of key performance metrics across different models, target hardware, inference engines, and configurations by replaying real-world inference workload traces collected from production or representative scenarios—thereby accelerating the design and optimization of inference systems.

---

## Introduction

Hisim is a simulation tool that provides a command-line interface compatible with SGLang. It launches a mock inference service that accepts user requests—either via real-world trace replay or synthetic load generation—using standard benchmarking scripts. Hisim outputs performance metrics identical to those produced by `sglang bench_serving`. See **Quick Start** for usage examples.

---

## Installation

```bash
cd hisim
pip install .
```

---

## Support Matrix

* **Inference Engine**: SGLang v0.5.6.post2  
* **Models**: Qwen3-32B-FP8, Qwen3-8B  
* **GPU**: H20-96GB  

---

## Quick Start

### Step 1 (Optional): Request Replay Mode – Prepare a Real Request Dataset

* If you already have a collected dataset, convert it into the following format and proceed to Step 2:
  ```json
  {"rid": "21e5xx", "timestamp":732.31, "output_length": 1024, "input_length": 1024, "input_ids": [925, 3911, ...], "output_ids": [244, 129, ...], "queue_end": 7522.4524386, "final_prefix_cache_len": 0}
  {"rid": "21e6xx", "timestamp":736.39, "output_length": 512, "input_length": 256, "input_ids": [192, 9171, ...], "output_ids": [254, 149, ...], "queue_end": 7523.4524386, "final_prefix_cache_len": 0}
  {"rid": "21e7xx", "timestamp":739.35, "output_length": 20, "input_length": 100, "input_ids": [92, 911, ...], "output_ids": [24, 19, ...], "queue_end": 7524.4524386, "final_prefix_cache_len": 10}
  ...
  ```

* If you don’t have a dataset, Hisim provides data collection utilities. Refer to [README](benchmark/collection/README.md) for instructions on gathering real request traces.

---

### Step 2: Mock Simulation

This example runs inference simulation using real-world trace data. You may also use synthetic random workloads (see below).

#### Launch the Simulation Server

Run the following command from the project root directory (the folder containing this `README.md`):

```bash
python3 -m hisim.simulation.sglang.launch_server \
  --model-path "Qwen/Qwen3-32B-FP8" \
  --sim-config-path test/assets/mock/config.json \
  --skip-server-warmup
```

> **Notes**:
> - Parameters prefixed with `--sim-` are simulation-specific; others are passed to the SGLang framework.
> - Use `--sim-config-path` to specify the simulation configuration file.
> - In pure CPU simulation environments, you may need to install the CPU-compatible version of vLLM (a dependency of SGLang CPU mode) and set the following environment variables:
>   ```bash
>   export SGLANG_USE_CPU_ENGINE=1
>   export FLASHINFER_DISABLE_VERSION_CHECK=1
>   ```
> - The provided [config file](test/assets/mock/config.json) is for testing. Adjust hardware bandwidth and other parameters to match your actual deployment scenario for higher fidelity. If using Hisim-collected traces, use the corresponding config file. See **Configuration File Format** below for details.

#### Run the Simulation Benchmark

Open a new terminal and run the benchmark client. **Important**: Use `--bench-mode simulation` and set `--warmup-request=0`.

- **Replay from user dataset** (`user_replay_requests.jsonl`):
  ```bash
  python3 -m hisim.simulation.bench_serving \
      --warmup-requests 0 \
      --bench-mode simulation \
      --model "Qwen/Qwen3-32B-FP8" \
      --backend sglang \
      --dataset-name hisim-collection \
      --dataset-path user_replay_requests.jsonl
  ```

- **Synthetic random workload**:
  ```bash
  python3 -m hisim.simulation.bench_serving \
      --warmup-requests 0 \
      --model "Qwen/Qwen3-32B-FP8" \
      --bench-mode simulation \
      --dataset-name random \
      --request-rate 4 \
      --random-input-len 1024 \
      --random-output-len 1024 \
      --random-range-ratio 1 \
      --num-prompts 50
  ```

You have now completed a framework-intercepted inference simulation.

#### Example Output

```bash
============ Serving Benchmark Result ============
Benchmark Mode:                          simulation
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     50        
Benchmark duration (s):                  20.36     
Elapsed time (s):                        4.40      
Total input tokens:                      51513     
Total input text tokens:                 -1        
Total input vision tokens:               -1        
Total generated tokens:                  51200     
Total generated tokens (retokenized):    -1        
Request throughput (req/s):              2.46      
Input token throughput (tok/s):          2530.02   
Output token throughput (tok/s):         2514.64   
Peak output token throughput (tok/s):    -1.00     
Peak concurrent requests:                -1        
Total token throughput (tok/s):          5044.66   
Concurrency:                             -1.00     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   9509.61   
Median E2E Latency (ms):                 9572.55   
---------------Time to First Token----------------
Mean TTFT (ms):                          10.97     
Median TTFT (ms):                        11.08     
P99 TTFT (ms):                           16.71     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.29      
Median TPOT (ms):                        9.35      
P99 TPOT (ms):                           9.74      
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.29      
Median ITL (ms):                         9.27      
P95 ITL (ms):                            10.31     
P99 ITL (ms):                            16.47     
Max ITL (ms):                            21.47     
==================================================
```

> Fields marked with `-1` indicate unsupported metrics in simulation mode.

---

## Usage

### Mock Simulation (Inference Simulation)

Hisim uses **dynamic interception** to hijack the execution flow of the inference framework, bypassing actual LLM computation. Currently supports **SGLang v0.5.6.post2**.

- For supported launch options, run:  
  ```bash
  python -m hisim.simulation.sglang.launch_server --help
  ```

---

### Configuration File Format (`HISIM_CONFIG_PATH`)

The config file is a JSON with three main sections:

- **`platform`**: Hardware and bandwidth settings  
  - `accelerator.name`: GPU model (e.g., `"H20"`)
  - `disk_*_bandwidth_gb`: L3 (disk) read/write bandwidth (GB/s)
  - `memory_*_bandwidth_gb`: L2 (memory) read/write bandwidth (GB/s)

- **`predictor`**: Time prediction module  
  - `name`: predictor type (`"aiconfigurator"` or `"schedule_replay"`)
  - See **TimePredictor** section below for details

- **`scheduler`**: Parallelism and backend metadata  
  > ⚠️ **Note**: Multi-GPU parallelism (TP/EP) should **not** be configured at the framework level during server launch. Instead, specify `tp_size` and `ep_size` here—the predictor will simulate parallel execution overhead accordingly.
  - When `predictor.name = "aiconfigurator"`, `backend_version` must be provided.

**Example Config**:
```json
{
  "platform": {
    "accelerator": { "name": "H20" },
    "disk_read_bandwidth_gb": 4,
    "disk_write_bandwidth_gb": 4,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64
  },
  "predictor": {
    "name": "aiconfigurator",
    "database_path": "path/to/aiconfigurator/data",
    "device_name": "h20_sxm",
    "prefill_scale_factor": 1.02040816,
    "decode_scale_factor": 1.01010101
  },
  "scheduler": {
    "tp_size": 1,
    "ep_size": 1,
    "data_type": "FP16",
    "kv_cache_data_type": "FP16",
    "backend_name": "sglang",
    "backend_version": "0.5.6.post2"
  }
}
```

---

## TimePredictor

### AIConfigurator

- Project: <https://github.com/ai-dynamo/aiconfigurator>
- Parameters:
  - `database_path`: (optional) path to custom operator profiling data
  - `device_name`: device/system name defined internally in AIConfigurator
  - `prefill_scale_factor` / `decode_scale_factor`: optional calibration factors to adjust predicted latency

**Example**:
```json
{
  "name": "aiconfigurator",
  "database_path": "path/to/aiconfigurator/data",
  "device_name": "h20_sxm",
  "prefill_scale_factor": 1.02040816,
  "decode_scale_factor": 1.01010101
}
```

Here is the English translation of your Markdown content, polished for clarity and technical accuracy:

---

## Examples

### Qwen3 Models

**Configuration**

In the `hisim/test/assets/mock` directory, we provide simulation configuration files for running Qwen3-8B and Qwen3-32B-FP8 on H20:  
- `config.qwen8b.aic.json`  
- `config.qwen32b.aic.json`

The base hardware specifications and operator interpolation data package for **aiconfigurator** can be obtained from the `Hisim` directory in the [LatencyPrism/hisim](https://github.com/kunluninsight/LatencyPrism/tree/hisim) repository.  
To use these configurations, download `H20_AIC.zip`, extract it to your `download_path`, and you're ready to run simulations.

**Accuracy**

We evaluated both models under three KV cache hit scenarios across varying request rates. All prediction errors are below **5%**.

***Qwen3-8B***

![alt text](docs/image/H20-Qwen3-8B-L2.png)

| Predictor        | Case       | Mean TTFT | Mean TPOT | Mean ITL | Input Throughput | Duration | Prefix Hit Ratio |
|:---------------:|:----------:|:--------:|:--------:|:--------:|:----------------:|:--------:|:----------------:|
| aiconfigurator  | no_cache   | 3.42%    | 1.64%    | 1.78%    | 1.41%            | 1.39%    | 0.0%             |
| aiconfigurator  | L1         | 4.15%    | 3.47%    | 3.54%    | 2.4%             | 2.33%    | 0.04%            |
| aiconfigurator  | L2         | 2.38%    | 4.05%    | 4.07%    | 2.35%            | 2.29%    | 0.0%             |

***Qwen3-32B-FP8***

![alt text](docs/image/H20-Qwen3-32B-FP8-L2.png)

| Predictor        | Case       | Mean TTFT | Mean TPOT | Mean ITL | Input Throughput | Duration | Prefix Hit Ratio |
|:---------------:|:----------:|:--------:|:--------:|:--------:|:----------------:|:--------:|:----------------:|
| aiconfigurator  | no_cache   | 2.77%    | 0.58%    | 0.52%    | 0.52%            | 0.51%    | 0.00%            |
| aiconfigurator  | L1         | 2.40%    | 1.03%    | 1.02%    | 1.04%            | 1.03%    | 0.04%            |
| aiconfigurator  | L2         | 3.05%    | 1.11%    | 1.02%    | 1.13%            | 1.12%    | 0.00%            |

**Case Definitions**:  
- `no_cache`: No KV cache hits.  
- `L1`: Hits only in GPU HBM (Level-1 cache).  
- `L2`: Two-level cache hits — both HBM and host DRAM (Level-2 cache).

**Error Metrics**:  
Results are reported using **MAPE** (Mean Absolute Percentage Error).