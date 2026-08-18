"""KVCM vLLM connector (v1): a thin shell over per-role cores.

vLLM instantiates one connector per role (a scheduler instance and one
worker instance per TP rank) from a single registered class, so this shell
is the plugin entry point: it performs the role-agnostic setup (config
parsing, manager registration) and delegates every hook to the role's core:

* ``SchedulerCore`` -- matching, saving orchestration, request finishing
  (scheduler_core.py);
* ``WorkerCore`` -- block translation and data-plane transfer
  (worker_core.py).

Shared vocabulary (GroupMeta / spec naming / KV layout
normalization / hybrid gate) lives in vllm_common.py. Compatibility
re-exports below keep external import paths (tests, e2e harness) stable.
"""

import json
import typing

from typing import Optional

from kv_cache_manager.client.pybind import kvcm_py_client

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

try:
    # vllm >= v0.11.1
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.utils.network_utils import get_ip
except ImportError:
    # vllm <= v0.11.0
    from vllm.utils import get_kv_cache_torch_dtype, get_ip

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import KVConnectorOutput

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.py_connector.common.tp_coordinator import TpCoordinatorClient
from kv_cache_manager.py_connector.common.logger import logger, configure_log_level
from kv_cache_manager.py_connector.common._version_info import FULL_VERSION, GIT_COMMIT, BUILD_TIME

from kv_cache_manager.py_connector.vllm.config import TairKvCacheConnectorExtraConfig
from kv_cache_manager.py_connector.vllm.metadata import TairKvCacheConnectorMetadata
from kv_cache_manager.py_connector.vllm.scheduler_core import SchedulerCore
from kv_cache_manager.py_connector.vllm.worker_core import WorkerCore
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta, GroupMeta, StateGroupMeta, attn_kv_views,
    build_spec_groups, ensure_hybrid_supported, parse_groups, spec_name)

if typing.TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.attention import AttentionMetadata
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig

# Compatibility re-exports: tests and the e2e harness import these from
# v1_connector (their original home before the core split).
__all__ = [
    "TairKvCacheConnector", "attn_kv_views", "ensure_hybrid_supported",
    "GroupMeta", "spec_name", "build_spec_groups", "parse_groups",
]


class TairKvCacheConnector(KVConnectorBase_V1, SupportsHMA):

    # ------------------------------------------------------------------ #
    # Init / registration (role-agnostic)
    # ------------------------------------------------------------------ #
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: Optional["KVCacheConfig"] = None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert kv_cache_config is not None, \
            "TairKvCacheConnector requires vLLM to pass kv_cache_config (vllm >= 0.11.1)"

        logger.warning("KVCM vllm connector version: %s (commit: %s, build: %s)",
                       FULL_VERSION, GIT_COMMIT, BUILD_TIME)

        self._extra_config = TairKvCacheConnectorExtraConfig(
            vllm_config.kv_transfer_config.kv_connector_extra_config)
        configure_log_level(self._extra_config.log_level)

        model_config = vllm_config.model_config
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        if getattr(model_config, "use_mla", False):
            raise NotImplementedError("MLA models are not supported by TairKvCacheConnector")

        self._role = role
        self._vllm_block_size = vllm_config.cache_config.block_size
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._kv_dtype = get_kv_cache_torch_dtype(
            vllm_config.cache_config.cache_dtype, model_config.dtype)

        # Manager block size: attention KV is token-granular and can be re-blocked,
        # but mamba state exists once per scheduler block, so hybrid models must
        # keep manager block == scheduler block.
        manager_block_size = self._vllm_block_size
        self._has_state_groups = any(
            isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_config.kv_cache_groups)
        if self._has_state_groups:
            ensure_hybrid_supported(force=self._extra_config.force_hybrid_support)
        if self._extra_config.preferred_block_size != 0:
            if self._has_state_groups:
                if self._extra_config.preferred_block_size != self._vllm_block_size:
                    logger.warning(
                        "preferred_block_size=%d ignored for hybrid model: mamba state is "
                        "per scheduler block (%d)", self._extra_config.preferred_block_size,
                        self._vllm_block_size)
            else:
                manager_block_size = self._extra_config.preferred_block_size
        self._manager_block_size = manager_block_size

        self._group_metas = parse_groups(kv_cache_config, manager_block_size)
        self._num_groups = len(self._group_metas)
        self._state_group_idxs = [m.group_idx for m in self._group_metas
                                  if isinstance(m, StateGroupMeta)]

        deployment = {
            "model_name": model_config.served_model_name,
            "dtype": str(self._kv_dtype)[6:],  # strip "torch."
            "use_mla": False,
            "tp_size": self._tp_size,
            "dp_size": vllm_config.parallel_config.data_parallel_size,
            "pp_size": vllm_config.parallel_config.pipeline_parallel_size,
        }
        logger.info("deployment: %s, groups: %s", deployment, self._group_metas)

        self._manager_client = KvCacheManagerClient.from_connector_config(vars(self._extra_config))
        self._host_ip = get_ip()

        register_request = {
            "trace_id": "register_%s" % self._extra_config.instance_id,
            "instance_group": self._extra_config.instance_group,
            "instance_id": self._extra_config.instance_id,
            "model_deployment": deployment,
            "block_size": manager_block_size,
            "location_spec_infos": [
                {"name": spec_name(rank, meta.group_idx), "size": meta.per_block_bytes}
                for rank in range(self._tp_size) for meta in self._group_metas
            ],
        }
        spec_groups = build_spec_groups(self._group_metas, self._tp_size)
        if spec_groups:
            # Hybrid models publish per-block spec coverage
            # (see vllm_common.build_spec_groups).
            register_request["location_spec_groups"] = spec_groups
        register_response = self._manager_client.register_instance(register_request)

        if role == KVConnectorRole.SCHEDULER:
            self._core = SchedulerCore(
                self._extra_config, self._group_metas, self._manager_block_size,
                self._vllm_block_size, self._tp_size,
                self._manager_client,
                TpCoordinatorClient(self._host_ip, self._extra_config.coordinator_base_port))
            logger.warning(
                "TairKvCacheConnector scheduler inited, extra_config: %r, manager block size: %d, "
                "vllm block size: %d, groups: %d",
                self._extra_config.__dict__, self._manager_block_size,
                self._vllm_block_size, self._num_groups)
        else:
            self._core = WorkerCore(
                self._extra_config, self._group_metas, self._manager_block_size,
                self._tp_size, self._host_ip, self._manager_client,
                TpCoordinatorClient(self._host_ip, self._extra_config.coordinator_base_port),
                register_response)

    def shutdown(self):
        self._core.shutdown()
        self._manager_client.close()
        return None

    # ------------------------------------------------------------------ #
    # vLLM hooks -- thin delegation to the role's core
    # ------------------------------------------------------------------ #
    # Scheduler hooks (called on the scheduler-role instance)
    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int):
        return self._core.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        self._core.update_state_after_alloc(request, blocks, num_external_tokens)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        self._core.update_connector_output(connector_output)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        return self._core.build_connector_meta(scheduler_output)

    def request_finished_all_groups(self, request: "Request",
                                    block_ids) -> tuple:
        return self._core.request_finished_all_groups(request, block_ids)

    def request_finished(self, request: "Request", block_ids) -> tuple:
        return self._core.request_finished(request, block_ids)

    def get_finished_count(self):
        return self._core.get_finished_count()

    # Worker hooks (called on each worker-role instance)
    def register_kv_caches(self, kv_caches: dict):
        self._core.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # The worker consumes each instruction directly from the metadata
        # (start_load_kv / wait_for_save receive it explicitly); no mirror
        # to replay here.
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._core.start_load_kv(forward_context, self._get_connector_metadata(), **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._core.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        self._core.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        self._core.wait_for_save(self._get_connector_metadata())

    def get_finished(self, finished_req_ids: set):
        return self._core.get_finished(finished_req_ids, self._get_connector_metadata())

    def get_block_ids_with_load_errors(self) -> set:
        return self._core.get_block_ids_with_load_errors()
