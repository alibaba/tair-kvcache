sudo docker run -d -it \
    --ulimit memlock=-1  --ulimit stack=67108864  --ulimit core=-1 \
    --ipc=host --network=host --privileged \
    --shm-size=64g --gpus all \
    -v /home/zhikuan.psc/data:/data-home \
    -v /mnt/vdb1:/data-mnt \
    --name sglang-0.5.5.p1 \
    mirrors-ssl.aliyuncs.com/lmsysorg/sglang:v0.5.5.post1 bash

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

# in kvcm container
rm -rf /root/KVCacheManager/logs/*
exec /home/admin/kv_cache_manager/bin/kv_cache_manager_bin \
    -c /home/admin/kv_cache_manager/etc/default_server_config.conf \
    -l /home/admin/kv_cache_manager/etc/default_logger_config.conf

# in sglang container
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/_solib_k8/_U_S_S3rdparty_S3fs_Chf3fs___U3rdparty_S3fs/
PYTHONPATH=.:${PYTHONPATH} python3 kv_cache_manager/py_connector/sglang/test.py

##############################
export CUDA_HOME=/etc/alternatives/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/_solib_k8/_U_S_S3rdparty_S3fs_Chf3fs___U3rdparty_S3fs
export INSTANCE_ID=$(date +%s)
PYTHONPATH=/sgl-workspace/KVCacheManager:${PYTHONPATH} \
python3 -m sglang.launch_server \
    --log-level debug \
    --model-path /data-mnt/Qwen3-8B \
    --tp 2 \
    --page-size 64 \
    --enable-metrics --enable-cache-report \
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
    }'

curl -w "\n"  -X POST 'http://localhost:30000/v1/completions' \
-H 'Content-Type: application/json' -H 'Accept: application/json' \
--data-raw '{
    "model": "DeepSeek-R1",
    "prompt": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”",
    "max_tokens": 64,
    "stream": true,
    "stream_options": {
        "include_usage": true
    }
}'

curl 127.0.0.1:30000/flush_cache
