FROM openmind-registry.cn-hangzhou.cr.aliyuncs.com/openmind/kv-cache-manager-dashllm:2026_08_18_17_42_d7f472d

USER root

COPY kv_cache_manager_bin /home/admin/kv_cache_manager/bin/kv_cache_manager_bin
COPY kvcm_dashtrace.whl /home/admin/dashtrace/kvcm_dashtrace.whl

RUN chmod 0755 /home/admin/kv_cache_manager/bin/kv_cache_manager_bin && \
    python3 -m pip install --no-cache-dir \
      "grpcio==1.62.1" \
      "protobuf==4.25.8" \
      "prometheus-client==0.20.0" \
      /home/admin/dashtrace/kvcm_dashtrace.whl

LABEL com.alibaba.owner="mxc" \
      com.alibaba.kvcm.base_commit="02c903f174e8c20bbc217385ab199ae4783133cf" \
      com.alibaba.kvcm.purpose="dashtrace-online-mrc-batch"
