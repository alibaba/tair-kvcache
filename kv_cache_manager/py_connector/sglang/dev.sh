# SGLang container — requires version after PR #20457 for hybrid cache support
sudo docker run -d -it \
    --ulimit memlock=-1  --ulimit stack=67108864  --ulimit core=-1 \
    --ipc=host --network=host --privileged \
    --shm-size=64g --gpus all \
    -v /home/zhikuan.psc/data:/data-home \
    -v /mnt/vdb1:/data-mnt \
    --name sglang \
    <sglang-image> bash

sudo docker run -d -it \
    --ulimit memlock=-1  --ulimit stack=67108864  --ulimit core=-1 \
    --ipc=host --network=host --privileged \
    --shm-size=64g --gpus all \
    -v /home/zhikuan.psc/data:/data-home \
    -v /mnt/vdb1:/data-mnt \
    --name kvcm-251109 \
    --entrypoint /bin/bash \
    hub.docker.alibaba-inc.com/isearch/kv_cache_manager_prod:2025_11_09_21_00_e42bd6e

apt install libaio-dev
pip install kvcm_py_client-0.0.1-cp312-cp312-manylinux_2_35_x86_64.whl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/_solib_k8/_U_S_S3rdparty_S3fs_Chf3fs___U3rdparty_S3fs/
python3 kv_cache_manager/client/test/transfer_client_py_test.py

##############################
# Integration tests
#
# test.py / test_linear.py support KVCM_HOME env to override binary paths.
# Default: /home/admin/kv_cache_manager
# Override: export KVCM_HOME=/path/to/kvcm  (must contain bin/ and etc/)
#
# Tests auto-start/stop KVCM, no manual launch needed.
##############################

# KV-only integration test (in sglang container)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/_solib_k8/_U_S_S3rdparty_S3fs_Chf3fs___U3rdparty_S3fs/
python3 kv_cache_manager/py_connector/sglang/test.py

# Hybrid (KV + Mamba) integration test (in sglang container)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/_solib_k8/_U_S_S3rdparty_S3fs_Chf3fs___U3rdparty_S3fs/
python3 kv_cache_manager/py_connector/sglang/test_linear.py

##############################
# End-to-end SGLang serving test
##############################
export CUDA_HOME=/etc/alternatives/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/_solib_k8/_U_S_S3rdparty_S3fs_Chf3fs___U3rdparty_S3fs
export INSTANCE_ID=$(date +%s)

# Start KVCM (in kvcm container or same container if binary accessible)
# KVCM_HOME defaults to /home/admin/kv_cache_manager
rm -rf /tmp/nfs/*
${KVCM_HOME:-/home/admin/kv_cache_manager}/bin/kv_cache_manager_bin \
    -c ${KVCM_HOME:-/home/admin/kv_cache_manager}/etc/default_server_config.conf \
    -l ${KVCM_HOME:-/home/admin/kv_cache_manager}/etc/default_logger_config.conf &

# Start SGLang with KVCM storage backend (hybrid cache)
# Generic hybrid cache launch (file backend):
#   python3 -m sglang.launch_server --model-path <model> \
#       --mamba-scheduler-strategy extra_buffer --page-size 64 --port 30000 --tp 4 \
#       --enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0 \
#       --hicache-write-policy write_through --hicache-storage-backend file \
#       --hicache-storage-prefetch-policy wait_complete \
#       --hicache-io-backend direct --hicache-mem-layout page_first_direct
#
# With KVCM as storage backend, replace --hicache-storage-backend file
# with --hicache-storage-backend dynamic + extra config:
nohup python3 -m sglang.launch_server \
    --model-path /data-mnt/Qwen3.5-9B \
    --mamba-scheduler-strategy extra_buffer \
    --port 30000 \
    --tp 2 \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    --hicache-size 0 \
    --hicache-mem-layout page_first_direct \
    --hicache-io-backend direct \
    --hicache-write-policy write_through \
    --hicache-storage-backend dynamic \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend-extra-config '{
        "prefetch_threshold":0,
        "backend_name":"kvcm",
        "module_path": "kv_cache_manager.py_connector.sglang.connector",
        "class_name": "HiCacheKVCM",
        "interface_v1":1,
        "instance_group": "default",
        "instance_id": "'${INSTANCE_ID}'",
        "manager_uri": "http://127.0.0.1:6382"
    }' > sglang.out &

# Send a long prompt (must > 1 page = 64 tokens to trigger backup)
curl -w "\n"  -X POST 'http://localhost:30000/v1/completions' \
-H 'Content-Type: application/json' -H 'Accept: application/json' \
--data-raw '{
    "model": "DeepSeek-R1",
    "prompt": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, \"and what is the use of a book,\" thought Alice \"without pictures or conversations?\"",
    "max_tokens": 64,
    "stream": true,
    "stream_options": {
        "include_usage": true
    }
}'

# Flush local radix cache to force prefetch from KVCM on next request
curl 127.0.0.1:30000/flush_cache

# Re-send same prompt — should show #cached-token > 0 (prefetch hit)
curl -w "\n"  -X POST 'http://localhost:30000/v1/completions' \
-H 'Content-Type: application/json' -H 'Accept: application/json' \
--data-raw '{
    "model": "DeepSeek-R1",
    "prompt": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, \"and what is the use of a book,\" thought Alice \"without pictures or conversations?\"",
    "max_tokens": 64,
    "stream": true,
    "stream_options": {
        "include_usage": true
    }
}'
