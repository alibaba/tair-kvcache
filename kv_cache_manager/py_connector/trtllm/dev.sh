apt install libaio-dev
apt install libaio1t64
ln -s /usr/lib/x86_64-linux-gnu/libaio.so.1t64 /usr/lib/x86_64-linux-gnu/libaio.so.1
pip install kvcm_py_client-0.0.1-cp312-cp312-manylinux_2_35_x86_64.whl

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/_solib_k8/_U_S_S3rdparty_S3fs_Chf3fs___U3rdparty_S3fs/
PYTHONPATH=/workspace/KVCacheManager:${PYTHONPATH} \
KVCM_CONFIG_PATH="/tmp/kvcm_config.json" \
python test.py /data-mnt/Qwen3-0.6B/

# in kvcm container
rm -rf /root/KVCacheManager/logs/*
exec /home/admin/kv_cache_manager/bin/kv_cache_manager_bin \
    -c /home/admin/kv_cache_manager/etc/default_server_config.conf \
    -l /home/admin/kv_cache_manager/etc/default_logger_config.conf
