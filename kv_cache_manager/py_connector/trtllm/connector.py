# https://github.com/NVIDIA/TensorRT-LLM/blob/v1.2.0rc0/examples/llm-api/llm_kv_cache_connector.py

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import logging
import uuid
import math

import click
import torch

from tensorrt_llm import LLM, SamplingParams, logger
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler, KvCacheConnectorWorker, SchedulerOutput)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, TorchLlmArgs
import torch.distributed as dist

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.client.pybind import kvcm_py_client

logger = logging.getLogger(__name__)


# This is a simple example of the use of the KV cache connector.
# It persists KV cache contents into a folder, and can load them back on subsequent runs.
# See tensorrt_llm/_torch/pyexecutor/connector.py for details about the KV cache connector interface.
# NOTE: This example connector implementation is NOT suitable for production use.

KVCM_CONFIG_PATH_KEY = "KVCM_CONFIG_PATH"


def init_kvcm_config():
    kvcm_config_path = os.environ.get(KVCM_CONFIG_PATH_KEY)
    assert kvcm_config_path is not None, f"env {KVCM_CONFIG_PATH_KEY} is needed"
    logger.info(f"{kvcm_config_path=}")

    try:
        with open(kvcm_config_path, 'r', encoding='utf-8') as f:
            kvcm_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"KVCM config file not found at {kvcm_config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in KVCM config file {kvcm_config_path}: {e}")

    return kvcm_config


def init_kvcm_meta_client(llm_args: TorchLlmArgs):
    kvcm_config = init_kvcm_config()
    manager_client = KvCacheManagerClient(kvcm_config["manager_uri"])
    logger.info("kvcm manager_client initialized")
    return kvcm_config, manager_client


def init_kvcm_transfer_client(llm_args, kvcm_config, extra_config):
    location_spec_name = extra_config["location_spec_name"]
    location_spec_size = extra_config["location_spec_size"]
    register_response = extra_config["register_response"]

    storage_configs = register_response["storage_configs"]

    instance_group = kvcm_config.get("instance_group")
    instance_id = kvcm_config.get("instance_id")

    tokens_per_block = llm_args.kv_cache_config.tokens_per_block

    # sdk
    sdk_thread_num = kvcm_config.get("sdk_thread_num", 4)
    sdk_queue_size = kvcm_config.get("sdk_queue_size", 1000)
    sdk_get_timeout_ms = kvcm_config.get("sdk_get_timeout_ms", 5000)
    sdk_put_timeout_ms = kvcm_config.get("sdk_put_timeout_ms", 10000)

    # data transfer setup
    # TODO: sdk_config
    transfer_client_config = f'''{{
            "instance_group": "{instance_group}",
            "instance_id": "{instance_id}",
            "block_size": {tokens_per_block},
            "sdk_config": {{
                "thread_num": {sdk_thread_num},
                "queue_size": {sdk_queue_size},
                "sdk_config": [],
                "timeout_config": {{
                    "get_timeout_ms": {sdk_get_timeout_ms},
                    "put_timeout_ms": {sdk_put_timeout_ms}
                }}
            }},
            "location_spec_infos": {{
                "{location_spec_name}": {location_spec_size}
            }}
        }}'''

    init_params = kvcm_py_client.InitParams()
    init_params.role_type = kvcm_py_client.RoleType.WORKER
    init_params.self_location_spec_name = location_spec_name
    init_params.storage_configs = f"{storage_configs}"

    transfer_client = kvcm_py_client.TransferClient.Create(
        transfer_client_config, init_params
    )
    assert transfer_client is not None, "kvcm_py_client.TransferClient.Create failed"
    logger.info("kvcm transfer_client initialized")

    return transfer_client


def get_trace_id() -> str:
    return str(uuid.uuid1())


def tp_rank_to_spec_name(tp_rank: int) -> str:
    return f"tp_{tp_rank}"


@dataclass
class KVCMKvCacheConnectorMetadata:
    # [locations, block_ids]
    load: list[tuple[list[str], list[int]]] = field(default_factory=list)
    # [locations, block_ids, write_session_id]
    save: list[tuple[list[str], list[int], str]] = field(default_factory=list)


class KVCMKvCacheConnectorWorker(KvCacheConnectorWorker):

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        self.kv_cache_tensor = None

        self.kvcm_config, self.manager_client = init_kvcm_meta_client(self._llm_args)

        self.instance_group = self.kvcm_config["instance_group"]
        self.instance_id = self.kvcm_config["instance_id"]

        self.write_timeout_seconds = self.kvcm_config.get("write_timeout_seconds", 30)

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        assert self.kv_cache_tensor is None, "KV cache tensor already registered"
        self.kv_cache_tensor = kv_cache_tensor
        extra_config = self._register_kvcm()
        self._init_kvcm_transfer_client(extra_config)

    def _register_kvcm(self):
        mapping = self._llm_args.parallel_config.to_mapping()
        if not dist.is_initialized():
            master_ip = os.getenv("KVCM_CONNECTOR_ADDR", "localhost")
            master_port = os.getenv("KVCM_CONNECTOR_PORT", "6688")
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="nccl",
                                    init_method=init_method,
                                    world_size=mapping.world_size,
                                    rank=mapping.rank)
        self.cpu_tp_group = dist.new_group(mapping.tp_group, backend="gloo")

        self.tp_rank = mapping.tp_rank
        self.tp_world_size = mapping.tp_size
        self.location_spec_name = tp_rank_to_spec_name(self.tp_rank)

        deployment = {
            "model_name": self._llm_args.model,
            "tp_size": mapping.tp_size,
            "dp_size": mapping.dp_size,
            "pp_size": mapping.pp_size,
            "use_mla": self.kvcm_config["use_mla"],
            "dtype": str(self.kv_cache_tensor.dtype)[6:],
        }

        # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.2.0rc0/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp#L828
        # self.kv_cache_tensor.shape = {mNumPrimaryBlocks, pool.numLayers, mKVFactor, blockSize}
        assert self.kv_cache_tensor.shape[2] == (
            1 if self.kvcm_config["use_mla"] else 2)  # mKVFactor == (1 if use_mla else 2)
        # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.2.0rc0/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h#L543
        # blockSize = (numKvHeads * sizePerHead * tokensPerBlock)
        location_spec_size = math.prod(self.kv_cache_tensor.shape[1:]) * self.kv_cache_tensor.dtype.itemsize
        location_spec_infos = [{
            "name": tp_rank_to_spec_name(rank),
            "size": location_spec_size,
        } for rank in range(mapping.tp_size)]

        register_request = {
            "trace_id": get_trace_id(),
            "instance_group": self.kvcm_config["instance_group"],
            "instance_id": self.kvcm_config["instance_id"],
            "model_deployment": deployment,
            "block_size": self._llm_args.kv_cache_config.tokens_per_block,
            "location_spec_infos": location_spec_infos,
        }
        logger.debug(f"{register_request=}")
        # TODO: check conflict and update
        register_response = self.manager_client.register_instance(register_request)
        logger.debug(f"register_instance {register_response=}")

        extra_config = {
            "location_spec_name": self.location_spec_name,
            "location_spec_size": location_spec_size,
            "register_response": register_response,
        }
        return extra_config

    def _init_kvcm_transfer_client(self, extra_config):
        self.transfer_client = init_kvcm_transfer_client(
            self._llm_args, self.kvcm_config, extra_config)

    def start_load_kv(self, stream: torch.cuda.Stream):
        for locations, block_ids in self._metadata.load:
            uris = self._extract_uris(locations)
            buffers, cpu_tensors = self._prepare_buffers(block_ids)
            result = self.transfer_client.LoadKvCaches(uris, buffers)
            logger.debug(f"LoadKvCaches {result=}")
            for block_id, cpu_tensor in zip(block_ids, cpu_tensors):
                self.kv_cache_tensor[block_id].copy_(cpu_tensor, non_blocking=False)

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def wait_for_save(self, stream: torch.cuda.Stream):

        # Make sure the forward pass is complete before beginning our save.
        stream.synchronize()

        for locations, block_ids, write_session_id in self._metadata.save:
            uris = self._extract_uris(locations)
            buffers, _ = self._prepare_buffers(block_ids)
            result = self.transfer_client.SaveKvCaches(uris, buffers)
            logger.debug(f"SaveKvCaches {result=}")
            flag = (result[0] == kvcm_py_client.ClientErrorCode.ER_OK)
            if self.tp_world_size > 1:
                flag_tensor = torch.tensor(flag, dtype=torch.int)
                dist.all_reduce(
                    flag_tensor,
                    op=dist.ReduceOp.MIN,
                    group=self.cpu_tp_group,
                )
                flag = bool(flag_tensor.item())

            if self.tp_rank == 0:
                finish_mask = [flag] * len(locations)
                self.manager_client.finish_write_cache({
                    "trace_id": get_trace_id(),
                    "instance_id": self.instance_id,
                    "write_session_id": write_session_id,
                    "success_blocks": {"bool_masks": {"values": finish_mask}},
                })

    def _extract_uris(self, locations: list[dict]) -> list[str]:
        uris = []
        for location in locations:
            for location_spec in location["location_specs"]:
                if location_spec["name"] == self.location_spec_name:
                    uris.append(location_spec["uri"])
        return uris

    def _prepare_buffers(self, block_ids: list[int]) -> list[kvcm_py_client.BlockBuffer]:
        buffers = []
        cpu_tensors = []
        for block_id in block_ids:
            cpu_tensor = self.kv_cache_tensor[block_id].cpu()

            iov = kvcm_py_client.Iov()
            iov.type = kvcm_py_client.MemoryType.CPU
            iov.base = cpu_tensor.data_ptr()
            iov.size = cpu_tensor.element_size() * cpu_tensor.numel()
            iov.ignore = False
            iovs = [iov]

            buffer = kvcm_py_client.BlockBuffer()
            buffer.iovs = iovs

            buffers.append(buffer)
            cpu_tensors.append(cpu_tensor)
        return buffers, cpu_tensors

    def get_finished(
            self, finished_gen_req_ids: list[int],
            started_loading_req_ids: list[int]) -> tuple[list[int], list[int]]:

        return [], []


class KVCMKvCacheConnectorLeader(KvCacheConnectorScheduler):

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        self.block_size = self._llm_args.kv_cache_config.tokens_per_block
        # req_id -> (locations, computed_hashes, remaining_hashes)
        self.pending_loads = {}

        self.kvcm_config, self.manager_client = init_kvcm_meta_client(self._llm_args)
        self.instance_id = self.kvcm_config["instance_id"]

        self.write_timeout_seconds = self.kvcm_config.get("write_timeout_seconds", 30)

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        # NOTE: This is a simplified implementation, and does not work with chunked prefill.

        metadata = KVCMKvCacheConnectorMetadata()

        for req in scheduler_output.new_requests:
            # If we don't have any pending loads for this request, we can skip it.
            if req.request_id not in self.pending_loads:
                continue

            num_computed_blocks = req.computed_position // self.block_size
            block_ids = req.new_block_ids

            load_locations, computed_hashes, remaining_hashes = self.pending_loads[req.request_id]
            assert num_computed_blocks == len(computed_hashes)
            assert len(load_locations) == len(remaining_hashes)

            metadata.load.append(
                (load_locations, [
                    block_ids[block_pos] for block_pos in range(
                        num_computed_blocks, num_computed_blocks + len(load_locations))]))

            # Break up the remainder of the token sequence into chunks.
            chunks = self._chunk_tokens(req.new_tokens)

            new_chunks = [chunk for chunk in chunks[len(computed_hashes) +
                                                    len(remaining_hashes):] if len(chunk) == self.block_size]
            new_hashes = [self._hash_tokens(chunk) for chunk in new_chunks]
            block_keys = computed_hashes + remaining_hashes + new_hashes
            request = {
                "trace_id": get_trace_id(),
                "instance_id": self.instance_id,
                "block_keys": block_keys,
                "write_timeout_seconds": self.write_timeout_seconds,
            }
            logger.debug(f"start_write_cache {request=}")
            try:
                result = self.manager_client.start_write_cache(request)
                logger.debug(f"start_write_cache {result=}")
                store_locations = result["locations"]
                write_session_id = result["write_session_id"]
                block_mask = result["block_mask"]
            except Exception as e:
                logger.error(f"get_cache_location {e=}, {get_request=}")
                len_locations = 0

            save_indices = self._parse_block_mask(block_mask, len(block_keys))
            assert len(store_locations) == len(save_indices)

            metadata.save.append((store_locations, [block_ids[block_pos]
                                 for block_pos in save_indices], write_session_id))
        
        logger.info(f"{metadata=}")

        self.pending_loads = {}

        return metadata

    def _hash_tokens(self, tokens: list[int]) -> int:
        return abs(hash(tuple(tokens)))

    def _chunk_tokens(self, tokens: list[int]) -> list[list[int]]:
        return [
            tokens[i:i + self.block_size]
            for i in range(0, len(tokens), self.block_size)
        ]

    def _parse_block_mask(self, block_mask: dict, len_block_keys: int) -> int:
        save_indices = None
        if "offset" in block_mask:
            offset = block_mask["offset"]
            save_indices = list(range(offset, len_block_keys))
        else:
            # False: need to store
            bool_masks = block_mask.get("bool_masks", {}).get("values", [])
            save_indices = [idx for idx, is_saved in enumerate(bool_masks) if not is_saved]
        return save_indices

    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> tuple[int, bool]:
        self.pending_loads[request.request_id] = [[], [], []]

        # Don't bother with sequences with partial matches.
        if (num_computed_tokens % self.block_size) != 0:
            return 0, False

        computed_blocks = num_computed_tokens // self.block_size

        computed_tokens = request.get_tokens(0)[:computed_blocks * self.block_size]
        # Get all the tokens that don't have a cache hit on device.
        remaining_tokens = request.get_tokens(0)[computed_blocks *
                                                 self.block_size:]

        computed_chunks = self._chunk_tokens(computed_tokens)
        remaining_chunks = self._chunk_tokens(remaining_tokens)

        computed_hashes = [self._hash_tokens(chunk) for chunk in computed_chunks]
        remaining_hashes = [self._hash_tokens(chunk) for chunk in remaining_chunks]
        block_keys = computed_hashes + remaining_hashes

        get_request = {
            "trace_id": get_trace_id(),
            "block_keys": block_keys,
            "instance_id": self.instance_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_mask": {"offset": len(computed_hashes)},
        }

        locations = []
        try:
            result = self.manager_client.get_cache_location(get_request)
            logger.debug(f"get_cache_location {result=}")
            locations = result["locations"]
        except Exception as e:
            logger.error(f"get_cache_location {e=}, {get_request=}")

        len_locations = len(locations)

        self.pending_loads[request.request_id][1].extend(computed_hashes)
        if len_locations > 0:
            self.pending_loads[request.request_id][0].extend(locations)
            self.pending_loads[request.request_id][2].extend(remaining_hashes[:len(locations)])

        logger.info(
            f"KV CONNECTOR: Matched {len_locations} blocks for request {request.request_id}"
        )
        logger.debug(f"{self.pending_loads[request.request_id]=}")

        return len_locations * self.block_size, False

    def request_finished(self, request: LlmRequest,
                         cache_block_ids: list[int]) -> bool:
        # We don't do any asynchronous saving, so always return False
        return False

    def update_state_after_alloc(self, request: LlmRequest,
                                 block_ids: list[int]):
        pass
