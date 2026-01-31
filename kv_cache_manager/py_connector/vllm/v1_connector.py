import copy
import json
import math
import time
import typing
import inspect
import threading

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple

from concurrent.futures import ThreadPoolExecutor
from kv_cache_manager.client.pybind import kvcm_py_client

import torch
import typing_extensions
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

try:
    # vllm >= v0.11.1
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.utils.network_utils import get_ip
except ImportError:
    # vllm <= v0.11.0
    from vllm.utils import get_kv_cache_torch_dtype, get_ip

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.py_connector.common.tp_coordinator import CoordinateMsgSerializer, TpCoordinatorServer, \
    TpCoordinatorClient, SendBlockStartEvent, CoordinateMessage, SaveContext
from kv_cache_manager.py_connector.common.logger import logger

from kv_cache_manager.py_connector.common.types import KVCacheInfo
from kv_cache_manager.py_connector.kernel.gather_scatter_helper import CopyBufferAllocator
from kv_cache_manager.py_connector.vllm.metadata import SaveRequest, LoadRequest, FinishRequest, ReqStateToWorker, \
    TairKvCacheConnectorMetadata
from kv_cache_manager.py_connector.vllm.config import TairKvCacheConnectorExtraConfig
from kv_cache_manager.py_connector.vllm.location_query_manager import LocationQueryManager
from kv_cache_manager.py_connector.vllm.data_transfer import MultiResult, DataTransferManager

if typing_extensions.TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.attention import AttentionMetadata
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks


@dataclass
class ReqState:
    """请求状态类，跟踪单个请求的状态信息"""

    # TODO: split this class to ReqStateInScheduler and ReqStateInWorker
    req_id: str
    token_ids: list[int]
    local_block_ids: list[int]
    has_saved_block_num: int
    local_matched_token_num: int
    remote_matched_token_num: int

    # vllm_request only avail in scheduler
    vllm_request: Optional["Request"]

    # scheduled_saving_count, sent_saving_count, need_report_after_saving_finished:
    # not sync between scheduler and worker and have different meaning
    # only available in scheduler and tp0 worker
    scheduled_saving_count: int = 0
    sent_saving_count: int = 0
    need_report_after_saving_finished: bool = False

    @staticmethod
    def create_from_delta(req_state_delta: 'ReqStateToWorker'):
        """从ReqStateToWorker创建ReqState实例"""
        return ReqState(
            req_id=req_state_delta.req_id,
            token_ids=req_state_delta.new_tokens_ids,
            local_block_ids=req_state_delta.new_local_block_ids,
            has_saved_block_num=req_state_delta.has_saved_block_num,
            local_matched_token_num=0,
            remote_matched_token_num=0,
            vllm_request=None
        )

    def update_from_delta(self, req_state_delta: 'ReqStateToWorker'):
        """使用ReqStateToWorker更新当前状态"""
        self.token_ids.extend(req_state_delta.new_tokens_ids)

        if req_state_delta.resumed_from_preemption:
            self.local_block_ids = req_state_delta.new_local_block_ids
        else:
            self.local_block_ids.extend(req_state_delta.new_local_block_ids)


@dataclass
class TransferTaskArgs:
    blocks_idx: List[List[int]] = field(default_factory=list)
    remote_uris: List[str] = field(default_factory=list)


class TairKvCacheConnector(KVConnectorBase_V1):
    def _tp_rank_to_spec_name(self, tp_rank: int) -> str:
        """Convert TP rank to location spec name."""
        return f"tp{tp_rank}"

    def __init__(self,
                 vllm_config: "VllmConfig",
                 role: KVConnectorRole,
                 kv_cache_config: Optional["KVCacheConfig"] = None,
                 ):

        init_params = inspect.signature(KVConnectorBase_V1.__init__).parameters
        if len(init_params) == 3:
            # vllm <= 0.11.0
            super().__init__(vllm_config, role)
        else:
            # vllm >= 0.11.1
            super().__init__(vllm_config, role, kv_cache_config)

        self._extra_config = TairKvCacheConnectorExtraConfig(vllm_config.kv_transfer_config.kv_connector_extra_config)
        self._kv_caches: Optional[dict[str, torch.Tensor]] = None
        self._local_block_size = vllm_config.cache_config.block_size

        model_config = vllm_config.model_config

        self._use_mla = (hasattr(model_config, "use_mla") and
                         isinstance(model_config.use_mla, bool) and
                         model_config.use_mla)

        manager_block_size = self._local_block_size
        if self._extra_config.preferred_block_size != 0:
            manager_block_size = self._extra_config.preferred_block_size

        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        kv_dtype = get_kv_cache_torch_dtype(vllm_config.cache_config.cache_dtype, model_config.dtype)
        num_layer = model_config.get_num_layers(vllm_config.parallel_config)
        per_tp_rank_kv_head_num = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        per_manager_location_spec_shape = [num_layer, 1 if self._use_mla else 2, manager_block_size,
                                           per_tp_rank_kv_head_num,
                                           head_size]

        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        deployment = {
            "model_name": model_config.served_model_name,
            "dtype": str(kv_dtype)[6:],  # remove "torch."
            "use_mla": self._use_mla,
            "tp_size": vllm_config.parallel_config.tensor_parallel_size,
            "dp_size": vllm_config.parallel_config.data_parallel_size,
            "pp_size": vllm_config.parallel_config.pipeline_parallel_size,
        }
        logger.info(deployment)

        self._manager_client = KvCacheManagerClient(self._extra_config.manager_uri)
        self._manager_block_size = manager_block_size

        self._alive_requests: dict[str, ReqState] = {}
        self._waiting_to_load_requests: List[LoadRequest] = []
        self._waiting_to_save_requests_lock = threading.Lock()
        self._waiting_to_save_requests: List[SaveRequest] = []
        self._waiting_to_finish_requests: List[FinishRequest] = []

        self._canceled_save_request_ids_lock = threading.Lock()
        self._canceled_save_request_ids: List[str] = []

        # TODO: add coordinator host auto detection, maybe use data parallel host
        # TODO: add DP support
        self._host_ip = get_ip()
        port = self._extra_config.coordinator_base_port

        register_response = self._manager_client.register_instance({
            "trace_id": "trace_trace",
            "instance_group": self._extra_config.instance_group,
            "instance_id": self._extra_config.instance_id,
            "model_deployment": deployment,
            "block_size": manager_block_size,
            "location_spec_infos": [{
                "name": self._tp_rank_to_spec_name(rank),
                "size": math.prod(per_manager_location_spec_shape) * kv_dtype.itemsize
            } for rank in range(self._tp_size)],
        })
        # TODO: check conflict and update
        self._iov_size = math.prod(
            per_manager_location_spec_shape) * kv_dtype.itemsize * self._extra_config.hf3fs_concurrent_io_block_count

        if role == KVConnectorRole.SCHEDULER:
            self._epoch = 0
            self._coordinator_client = TpCoordinatorClient(self._host_ip, port)
            self._http_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kvcm_http_")
            self._location_query_manager = LocationQueryManager(self._manager_client, self._http_executor,
                                                                self._extra_config.instance_id,
                                                                self._extra_config.async_get_cache_location)

            logger.warning(
                "TairKvCacheConnector in scheduler inited, kv_connector_extra_config: %r,"
                " server block size: %d, vllm block size: %d,",
                self._extra_config.__dict__,
                self._manager_block_size,
                self._local_block_size,
            )

        elif role == KVConnectorRole.WORKER:
            self._tp_rank = get_tensor_model_parallel_rank()
            self._save_stream = torch.cuda.Stream()
            self._load_stream = torch.cuda.Stream()

            logger.warning(
                "TairKvCacheConnector in worker inited, tp rank: %d, tp size: %d, host_ip: %s, port: %d" % (
                    self._tp_rank, self._tp_size, self._host_ip, port)
            )

            if self._tp_rank == 0:
                # start coordinator
                self._coordinator_server = TpCoordinatorServer(self._host_ip, port, self._tp_size,
                                                               self.on_save_finished)

            self._coordinator_client = TpCoordinatorClient(self._host_ip, port)

            self._storage_configs = register_response["storage_configs"]
            # data transfer setup
            self._location_spec_name = self._tp_rank_to_spec_name(self._tp_rank)
            self._write_timeout_seconds = self._extra_config.write_timeout_seconds

            sdk_backend_configs = []

            hf3fs_configs = self.parse_hf3fs_configs(self._storage_configs)
            sdk_backend_configs.extend(hf3fs_configs)
            logger.debug(sdk_backend_configs)
            transfer_client_json = {
                "instance_group": self._extra_config.instance_group,
                "instance_id": self._extra_config.instance_id,
                "block_size": self._manager_block_size,
                "sdk_config": {
                    "thread_num": self._extra_config.sdk_thread_num,
                    "queue_size": self._extra_config.sdk_queue_size,
                    "sdk_backend_configs": sdk_backend_configs,
                    "timeout_config": {
                        "get_timeout_ms": self._extra_config.sdk_get_timeout_ms,
                        "put_timeout_ms": self._extra_config.sdk_put_timeout_ms,
                    },
                },
                "location_spec_infos": {
                    self._location_spec_name: math.prod(per_manager_location_spec_shape) * kv_dtype.itemsize,
                },
            }
            self._transfer_client_config = json.dumps(transfer_client_json)

            self._init_params = kvcm_py_client.InitParams()
            self._init_params.role_type = kvcm_py_client.RoleType.WORKER
            self._init_params.self_location_spec_name = self._location_spec_name
            self._init_params.storage_configs = f"{self._storage_configs}"

            logger.info("_transfer_client_config:%s, _init_params:%s", self._transfer_client_config, self._init_params)

            self._transfer_client = kvcm_py_client.TransferClient.Create(
                self._transfer_client_config, self._init_params
            )
            assert self._transfer_client is not None, "kvcm_py_client.TransferClient.Create failed"

    def shutdown(self):
        # TODO: stop background threads and cleanup transfer client
        return None

    def parse_hf3fs_configs(self, storage_configs):
        hf3fs_configs = []
        storage_configs_json = json.loads(storage_configs)
        for storage_config in storage_configs_json:
            if storage_config["type"] == "vcns_hf3fs":
                storage_config["type"] = "hf3fs"
            if storage_config["type"] == "hf3fs" and storage_config["is_available"]:
                hf3fs_config = {
                    "type": storage_config["type"],
                    "mountpoint": storage_config["storage_spec"]["mountpoint"],
                    "root_dir": storage_config["storage_spec"]["root_dir"],
                    "read_iov_block_size": self._extra_config.read_iov_block_size,
                    "read_iov_size": self._iov_size,
                    "write_iov_block_size": self._extra_config.write_iov_block_size,
                    "write_iov_size": self._iov_size,
                }
                hf3fs_configs.append(hf3fs_config)
        self._storage_configs = json.dumps(storage_configs_json)
        return hf3fs_configs

    def generate_blocks(self, token_ids, block_size, max_token_length) -> list[dict[str, Any]]:
        results = []
        token_length = min(len(token_ids), max_token_length)
        # token_length = len(token_ids)
        for i in range(0, token_length, block_size):
            if i + block_size > token_length:
                break
            results.append({
                "token_ids": token_ids[i:i + block_size],
                "unique_id": None,
                "location": None
            })
        return results

    # ==============================
    # Worker-side methods
    # ==============================

    def generate_blocks_idx(self, manager_block_idxes, local_block_ids):
        blocks_idx = []
        for manager_block_idx in manager_block_idxes:
            # get kvcache index list
            block_idx = []
            for i in range(self._manager_block_size):
                now_token_idx = manager_block_idx * self._manager_block_size + i
                assert now_token_idx // self._local_block_size < len(local_block_ids)
                local_block_id = local_block_ids[now_token_idx // self._local_block_size]
                token_offset = now_token_idx % self._local_block_size
                block_idx.append(local_block_id * self._local_block_size + token_offset)
            blocks_idx.append(block_idx)
        return blocks_idx

    def on_save_finished(self, write_session_id: str, save_context: SaveContext):
        logger.debug(save_context.result_per_rank)
        for block_idx in range(len(save_context.locations)):
            # TODO: report uri when enable local alloc
            # location_specs = []
            is_fully_saved = True
            for rank in range(self._tp_size):
                is_success = save_context.result_per_rank[rank][block_idx]
                if not is_success:
                    # this spec is not fully saved, report failed
                    is_fully_saved = False
                # else:
                #     # Convert the spec to include name field instead of tp_rank
                #     location_specs.append({
                #         "name": self._tp_rank_to_spec_name(rank),
                #         "uri": spec
                #     })
            if is_fully_saved:
                # save_context.locations[block_idx]["location_specs"] = location_specs
                save_context.success_mask.append(True)
            else:
                save_context.success_mask.append(False)
        logger.debug("finish_write_cache blocks:%s mask:%s write_session_id:%s", save_context.locations,
                     save_context.success_mask, write_session_id)
        try:
            self._manager_client.finish_write_cache({
                "trace_id": "test_test",
                "instance_id": self._extra_config.instance_id,
                "write_session_id": write_session_id,
                "success_blocks": {
                    "bool_masks": {
                        "values": save_context.success_mask
                    }
                }
            })
        except Exception as e:
            logger.warning("finish_write_cache failed, write_session_id: %s, error: %s", write_session_id, e)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_layer_kvcache = next(iter(kv_caches.items()))
        self._kv_caches = kv_caches
        # TODO: support MLA

        assert self._local_block_size == first_layer_kvcache.shape[2], "kv cache shape error"
        for layer_name, kvcache in kv_caches.items():
            assert kvcache.is_contiguous(), "kv cache must be contiguous"

        # torch.Size([2, block_num, block_size, kv_head_num, kv_dim])
        # 2 -> key, value
        self._local_block_num = first_layer_kvcache.shape[1]
        self._local_token_num = self._local_block_num * self._local_block_size

        self._dtype = first_layer_kvcache.dtype
        self._device = first_layer_kvcache.device
        self._per_manager_location_spec_layer_shape = [first_layer_kvcache.shape[0],
                                                       self._manager_block_size,
                                                       first_layer_kvcache.shape[3] * first_layer_kvcache.shape[4]]
        self._per_manager_location_spec_layer_byte_size = math.prod(
            self._per_manager_location_spec_layer_shape) * self._dtype.itemsize
        self._per_layer_token_key_dim_size = first_layer_kvcache.shape[3] * first_layer_kvcache.shape[4]
        self._per_layer_token_key_byte_size = (first_layer_kvcache.shape[3] *
                                               first_layer_kvcache.shape[4] * self._dtype.itemsize)
        assert self._per_layer_token_key_byte_size == first_layer_kvcache[0][0][1].data_ptr() - \
               first_layer_kvcache[0][0][0].data_ptr(), "kv cache shape error"
        assert self._per_manager_location_spec_layer_byte_size == 2 * self._manager_block_size * self._per_layer_token_key_byte_size

        self._per_manager_location_spec_shape = [len(self._kv_caches)] + self._per_manager_location_spec_layer_shape
        self._per_manager_location_spec_byte_size = math.prod(
            self._per_manager_location_spec_shape) * self._dtype.itemsize

        self._kvcache_ptr_tensor_cpu = torch.tensor(
            [self._kv_caches[name].data_ptr() for name in self._kv_caches],
            dtype=torch.int64,
            device="cpu"
        )
        self._kvcache_ptr_tensor_gpu = self._kvcache_ptr_tensor_cpu.to(self._device)
        if self._use_mla:
            self._all_kvcache_ptr_tensor_cpu = torch.tensor(
                [self._kv_caches[name].data_ptr() for name in self._kv_caches],
                dtype=torch.int64,
                device="cpu"
            )
        else:
            kvcache_ptrs = []
            for name in self._kv_caches:
                kvcache_ptrs.append(self._kv_caches[name][0].data_ptr())
                kvcache_ptrs.append(self._kv_caches[name][1].data_ptr())
            self._all_kvcache_ptr_tensor_cpu = torch.tensor(
                kvcache_ptrs,
                dtype=torch.int64,
                device="cpu"
            )
        self._all_kvcache_ptr_tensor_gpu = self._all_kvcache_ptr_tensor_cpu.to(self._device)

        self._kvcache_info = KVCacheInfo(
            self._tp_rank,
            self._tp_size,
            self._kv_caches,
            self._kvcache_ptr_tensor_cpu,
            self._kvcache_ptr_tensor_gpu,
            self._all_kvcache_ptr_tensor_gpu,
            len(self._kv_caches),
            self._local_token_num,
            tuple(self._per_manager_location_spec_shape),
            self._per_manager_location_spec_byte_size,
            self._per_layer_token_key_dim_size,
            self._device,
            self._dtype
        )
        self._copy_buffer_allocator = CopyBufferAllocator(torch.device("cpu"), self._dtype,
                                                          self._per_manager_location_spec_shape, 1024)

        # 初始化DataTransferManager实例
        self._data_transfer = DataTransferManager(
            self._kvcache_info,
            self._manager_block_size,
            self._copy_buffer_allocator,
            self._transfer_client,
            self._coordinator_client,
            self._extra_config,
        )

        logger.warning("register_kv_caches, _per_manager_location_spec_layer_shape: %s",
                       self._per_manager_location_spec_layer_shape)


    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())

        for load_req in meta.to_load_requests:
            if len(load_req.need_load_locations) == 0:
                continue

            block_token_indices = self.generate_blocks_idx(load_req.manager_block_idxes, load_req.local_block_ids)
            all_remote_uris = self.get_self_uris(load_req.need_load_locations)

            per_task_size = self._extra_config.block_per_load_task
            task_num = math.ceil(len(block_token_indices) / per_task_size)
            done_callback = self._data_transfer.create_load_done_callback(
                load_req.req_id,
                self._kvcache_info.tp_rank,
                meta.epoch,
                copy.copy(load_req.local_block_ids)
            )
            multi_result = MultiResult(task_num, done_callback)

            task_idx = 0
            for i in range(0, len(block_token_indices), per_task_size):
                end_idx = min(len(block_token_indices), i + per_task_size)
                task_remote_uris = all_remote_uris[i:end_idx]
                task_block_token_indices = block_token_indices[i:end_idx]
                self._data_transfer.submit_task(self._data_transfer.load_task, multi_result, task_idx, task_remote_uris,
                                                task_block_token_indices)
                task_idx += 1

    def wait_for_layer_load(self, layer_name: str) -> None:
        # logger.warning("wait_for_layer_load, layer_name: %s", layer_name)
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata",
                      **kwargs) -> None:
        # logger.warning("save_kv_layer, layer_name: %s", layer_name)
        pass

    def wait_for_save(self):
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())
        # logger.warning("wait_for_save, meta: %r", meta)

        kvcache_ready_event = None
        if len(meta.to_save_requests) > 0:
            kvcache_ready_event = torch.cuda.Event()
            kvcache_ready_event.record(torch.cuda.current_stream())

        for req_save in meta.to_save_requests:
            req = self._alive_requests[req_save.req_id]

            # get idx
            blocks_idx = self.generate_blocks_idx(req_save.manager_block_idxes, req.local_block_ids)
            all_remote_uris = self.get_self_uris(req_save.target_locations)

            per_task_size = self._extra_config.block_per_save_task
            task_num = math.ceil(len(blocks_idx) / per_task_size)
            done_callback = self._data_transfer.create_save_done_callback(
                req.req_id,
                self._kvcache_info.tp_rank,
                req_save.write_session_id
            )
            multi_result = MultiResult(task_num, done_callback)

            task_idx = 0
            for i in range(0, len(blocks_idx), per_task_size):
                end_idx = min(len(blocks_idx), i + per_task_size)
                task_remote_uris = all_remote_uris[i:end_idx]
                task_block_token_indices = blocks_idx[i:end_idx]
                self._data_transfer.submit_task(self._data_transfer.save_task, multi_result, task_idx, task_remote_uris,
                                                task_block_token_indices,
                                                kvcache_ready_event)
                task_idx += 1
            if self._tp_rank == 0:
                req.scheduled_saving_count += 1

    def get_self_uris(self, locations):
        all_remote_uris = []
        for idx, location in enumerate(locations):
            for location_spec in location["location_specs"]:
                # Match by location spec name instead of tp_rank
                if self._tp_rank_to_spec_name(self._kvcache_info.tp_rank) == location_spec["name"]:
                    all_remote_uris.append(location_spec["uri"])
        return all_remote_uris

    def get_finished(
            self, finished_req_ids: set[str]
    ) -> Tuple[Optional[set[str]], Optional[set[str]]]:
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())

        if self._tp_rank != 0:
            for finish_req in meta.to_finish_requests:
                req_id = finish_req.req_id
                if req_id in self._alive_requests:
                    self._alive_requests.pop(req_id)
            return None, None

        # self._tp_rank == 0
        finished_saving_reqs = []
        # check if any request is saving kvcache
        (finished_saving_tasks, finished_loading_tasks) = self._coordinator_server.get_finished_tasks()
        for req_id in finished_saving_tasks:
            req = self._alive_requests[req_id]
            req.sent_saving_count += 1

            assert req.sent_saving_count <= req.scheduled_saving_count
            if (req.need_report_after_saving_finished and
                    req.sent_saving_count == req.scheduled_saving_count):
                finished_saving_reqs.append(req_id)
                self._alive_requests.pop(req_id)

        for finish_req in meta.to_finish_requests:
            req_id = finish_req.req_id
            if req_id not in self._alive_requests:
                # called get_num_new_matched_tokens but never scheduled
                continue
            req = self._alive_requests[req_id]
            if req.sent_saving_count == req.scheduled_saving_count:
                finished_saving_reqs.append(req_id)
                self._alive_requests.pop(req_id)
            else:
                self._alive_requests[req_id].need_report_after_saving_finished = True
        return set(finished_saving_reqs), set(finished_loading_tasks)

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self._tp_rank != 0:
            return set()
        failed_set = self._coordinator_server.get_failed_loading_block_idxs()
        if len(failed_set) > 0:
            logger.warning("block_ids_with_load_errors: %s", failed_set)
        return failed_set

    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        self._connector_metadata = connector_metadata
        meta = typing.cast(TairKvCacheConnectorMetadata, self._get_connector_metadata())

        for req_state_delta in meta.requests:
            if req_state_delta.req_id not in self._alive_requests:
                assert not req_state_delta.is_delta
            if not req_state_delta.is_delta:
                self._alive_requests[req_state_delta.req_id] = ReqState.create_from_delta(req_state_delta)
            else:
                self._alive_requests[req_state_delta.req_id].update_from_delta(req_state_delta)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> Tuple[int, bool]:
        # logger.warning("get matched token ids: %s, id: %s", request.prompt_token_ids, request.request_id)

        bypass_match = False
        # TODO: add arrival_time to req_id in order to handle same request id
        if request.request_id in self._alive_requests:
            # bypass remote match for alive requests
            # possible cases:
            # 1. reschedule when all kvcache loading failed
            # 2. TODO: no enough hbm to schedule the request
            # bypass_match = True
            # logger.warning("bypass match for alive request, req_id: %s", request.request_id)
            pass

        computed_manager_block_size = num_computed_tokens // self._manager_block_size
        all_calced_remote_block_num = computed_manager_block_size
        new_matched_count = 0

        if not bypass_match:
            is_query_done, need_load_locations = (
                self._location_query_manager.get_locations_for_query(request, computed_manager_block_size))
            if not is_query_done:
                # async get_cache_location
                return None, False
            new_matched_count = len(need_load_locations) * self._manager_block_size
            logger.info("req:%s, new_matched_count:%d", request.request_id, new_matched_count)

            all_calced_remote_block_num = computed_manager_block_size + len(need_load_locations)

            if new_matched_count != 0:
                self._waiting_to_load_requests.append(LoadRequest(
                    req_id=request.request_id,
                    manager_block_idxes=[i for i in range(computed_manager_block_size, all_calced_remote_block_num)],
                    need_load_locations=need_load_locations,
                ))

        new_req_meta = ReqState(request.request_id, copy.copy(request.prompt_token_ids), [],
                                all_calced_remote_block_num,
                                num_computed_tokens,
                                new_matched_count,
                                request)

        self._alive_requests[request.request_id] = new_req_meta
        return new_matched_count, new_matched_count > 0

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        if request.request_id not in self._alive_requests:
            return
        req_state = self._alive_requests[request.request_id]
        # blocks_ids[0]: only one KV cache groups for now
        # refer to vllm/v1/core/kv_cache_manager.py:35
        req_state.local_block_ids = copy.copy(blocks.get_block_ids()[0])

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = TairKvCacheConnectorMetadata(self._epoch)
        self._epoch += 1

        for load_req in self._waiting_to_load_requests:
            request = self._alive_requests[load_req.req_id]
            if len(request.local_block_ids) == 0:
                # ignore load_req if vllm has not called update_state_after_alloc,
                # vllm will call get_num_new_matched_tokens again
                continue
            load_req.local_block_ids = request.local_block_ids
            meta.add_load_request(load_req)
        self._waiting_to_load_requests = []

        for vllm_req in scheduler_output.scheduled_new_reqs:
            request = self._alive_requests[vllm_req.req_id]
            request.local_block_ids = copy.copy(vllm_req.block_ids[0])

            state_to_worker = ReqStateToWorker(req_id=request.req_id,
                                               has_saved_block_num=request.has_saved_block_num,
                                               new_tokens_ids=request.token_ids,
                                               new_local_block_ids=request.local_block_ids,
                                               is_delta=False
                                               )
            meta.add_req_state_to_worker(state_to_worker)
            logger.info("new request: %s, block_ids_len: %d", vllm_req.req_id, len(vllm_req.block_ids[0]))

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            request = self._alive_requests[req_id]
            vllm_req = request.vllm_request
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_current_tokens = len(request.token_ids)

            new_token_ids = vllm_req.all_token_ids[
                num_current_tokens: num_current_tokens + num_new_tokens
            ]
            state_to_worker = ReqStateToWorker(req_id=request.req_id,
                                               has_saved_block_num=request.has_saved_block_num)

            request.token_ids.extend(new_token_ids)
            state_to_worker.new_tokens_ids = new_token_ids

            resumed_from_preemption = False
            if hasattr(cached_reqs, "resumed_req_ids"):
                # vllm >= 0.11.1
                resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            else:
                # vllm <= 0.11.0
                resumed_from_preemption = cached_reqs.resumed_from_preemption[idx]

            if resumed_from_preemption:
                request.local_block_ids = copy.copy(cached_reqs.new_block_ids[idx][0])
                state_to_worker.resumed_from_preemption = True
                state_to_worker.new_local_block_ids = request.local_block_ids
            else:
                if cached_reqs.new_block_ids[idx] is None:
                    # https://github.com/vllm-project/vllm/pull/23262
                    continue
                new_block_ids = cached_reqs.new_block_ids[idx][0]
                request.local_block_ids.extend(new_block_ids)
                state_to_worker.new_local_block_ids = new_block_ids
            meta.add_req_state_to_worker(state_to_worker)

        for req in self._alive_requests.values():
            target_save_num = min(len(req.token_ids),
                                  len(req.local_block_ids) * self._local_block_size) // self._manager_block_size
            if target_save_num > req.has_saved_block_num:
                req.scheduled_saving_count += 1
                self._http_executor.submit(
                    self.start_save_kvcache_async,
                    req.req_id,
                    req.token_ids[:target_save_num * self._manager_block_size],
                    target_save_num
                )
            req.has_saved_block_num = target_save_num

        new_save_reqs: List[SaveRequest] = []
        with self._waiting_to_save_requests_lock:
            new_save_reqs = self._waiting_to_save_requests
            self._waiting_to_save_requests = []
        for save_req in new_save_reqs:
            if save_req.req_id not in self._alive_requests:
                # TODO: should not happen anymore
                logger.warning("request %s is not alive, skip saving", save_req.req_id)
                continue
            req = self._alive_requests[save_req.req_id]
            meta.add_save_request(save_req)

            req.sent_saving_count += 1
            if (req.need_report_after_saving_finished and
                    req.scheduled_saving_count == req.sent_saving_count):
                self._waiting_to_finish_requests.append(FinishRequest(req.req_id))
                self._alive_requests.pop(req.req_id)

        self.handle_canceled_save_req()

        for finish_req in self._waiting_to_finish_requests:
            meta.add_finish_request(finish_req)
        self._waiting_to_finish_requests = []

        # logger.warning("build_connector_meta: %r", meta)
        return meta

    def start_save_kvcache_async(self, req_id, token_ids, target_save_num):
        request = {
            "trace_id": "%s_%d" % (req_id, self._epoch),
            "instance_id": self._extra_config.instance_id,
            "block_keys": [],
            "token_ids": token_ids,
            "write_timeout_seconds": 30
        }
        logger.debug("start_write_cache req: %s", request)
        try:
            response = self._manager_client.start_write_cache(request)
        except Exception as e:
            logger.warning("start_write_cache error, skip this saving, exception: %s", e)
            with self._canceled_save_request_ids_lock:
                self._canceled_save_request_ids.append(req_id)
            return
        # call manager start write
        logger.debug("start_write_cache resp: %s", response)
        locations = response["locations"]
        write_session_id = response["write_session_id"]
        # check if success

        if len(locations) == 0:
            try:
                self._manager_client.finish_write_cache({
                    "trace_id": "test_test",
                    "instance_id": self._extra_config.instance_id,
                    "write_session_id": write_session_id,
                    "success_blocks": {
                        "bool_masks": {
                            "offset": 0
                        }
                    }
                })
            except Exception as e:
                logger.warning("finish_write_cache failed, write_session_id: %s, error: %s", write_session_id, e)
            with self._canceled_save_request_ids_lock:
                self._canceled_save_request_ids.append(req_id)
            return

        need_block_idx = self.parse_block_mask_to_save_indices(response, target_save_num)
        logger.debug("target_save_num: %s, need_block_idx: %s", target_save_num, need_block_idx)
        message = CoordinateMessage(time.time(), SendBlockStartEvent(request_id=req_id,
                                                                     write_session_id=write_session_id,
                                                                     locations=locations))
        self._coordinator_client.send(CoordinateMsgSerializer.dumps(message))

        with self._waiting_to_save_requests_lock:
            self._waiting_to_save_requests.append(SaveRequest(
                req_id,
                locations,
                need_block_idx,
                write_session_id
            ))

    def handle_canceled_save_req(self):
        canceled_save_req_ids = []
        with self._canceled_save_request_ids_lock:
            canceled_save_req_ids = self._canceled_save_request_ids
            self._canceled_save_request_ids = []
        for canceled_req_id in canceled_save_req_ids:
            req = self._alive_requests[canceled_req_id]
            req.sent_saving_count += 1
            if (req.need_report_after_saving_finished and
                    req.scheduled_saving_count == req.sent_saving_count):
                self._waiting_to_finish_requests.append(FinishRequest(req.req_id))
                self._alive_requests.pop(req.req_id)

    def get_finished_count(self):
        # only rank0 will return finished
        return 1

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """

        return

    def parse_block_mask_to_save_indices(self, response: dict, target_save_num: int) -> list[int]:
        # 从response中提取block_mask
        block_mask = response.get("block_mask", {})
        save_indices = []
        if "offset" in block_mask:
            offset = block_mask["offset"]
            for idx in range(offset, target_save_num):
                save_indices.append(idx)
        else:
            bool_masks = block_mask.get("bool_masks", {}).get("values", [])
            # 找出所有为False的索引（需要保存的block）
            for idx, is_saved in enumerate(bool_masks):
                if not is_saved:  # False表示需要保存
                    save_indices.append(idx)

        return save_indices

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
    ) -> Tuple[bool, Optional[dict[str, Any]]]:
        if request.request_id not in self._alive_requests:
            logger.info("request_finished not alive request: %s", request.request_id)
            return False, {}

        req = self._alive_requests[request.request_id]
        extra_info = {"local_matched_token_num": req.local_matched_token_num,
                      "remote_matched_token_num": req.remote_matched_token_num}

        if req.scheduled_saving_count == req.sent_saving_count:
            self._waiting_to_finish_requests.append(FinishRequest(req.req_id))
            self._alive_requests.pop(req.req_id)
            return True, extra_info

        # This request still has some save requests waiting to be issued or canceled,
        # delay finishing this request
        req.need_report_after_saving_finished = True

        return True, extra_info
