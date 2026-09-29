"""Role views of the same KV cache config, and who registers what.

vLLM hands the scheduler and the worker two different views of one
``UniformTypeKVCacheSpecs`` group: the scheduler gets a *folded* spec
(``generate_scheduler_kv_cache_config`` keeps one sub spec per group), the
worker gets the wrapper with every sub spec. Splitting the wrapper into
transfer buckets is therefore only meaningful on the worker side, and the
registered ``location_spec_infos`` (name -> bytes) must come from there --
registering the folded view as well is rejected by the manager as a duplicate
instance with mismatched fields. These tests pin both halves:

* the two views parse into different bucket sets for the same group;
* only the worker role calls register_instance, with the worker payload.

They also reconcile the frozen group facts with the local tiny-model golden
when it is readable (it is generated outside this repository).
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest import mock

from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)
from kv_cache_manager.py_connector.test.mla_variant_facts import (
    FROZEN,
    FROZEN_GROUPS,
    GOLDEN_GROUPS,
    load_golden,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from kv_cache_manager.py_connector.vllm import v1_connector
from kv_cache_manager.py_connector.vllm.vllm_common import parse_groups, spec_name

EXTRA_CONFIG: Dict[str, Any] = {
    "manager_uri": "http://127.0.0.1:1",
    "coordinator_base_port": 19999,
    "instance_group": "test_group",
    "instance_id": "test_instance",
}


def _spec(row: str) -> Any:
    kwargs = dict(FROZEN[row]["kwargs"])
    mode = kwargs.get("kv_quant_mode")
    if isinstance(mode, str):
        kwargs["kv_quant_mode"] = getattr(KVQuantMode, mode)
    kwargs.setdefault("num_kv_heads", 1)
    return MLAAttentionSpec(**kwargs)


def _worker_view(row: str = "v32_wrapper_worker") -> Tuple[Any, Any]:
    """The worker's config for one packed group: the wrapper, all layers."""
    facts = FROZEN_GROUPS[row]
    wrapper = UniformTypeKVCacheSpecs(
        block_size=facts["block_size"],
        kv_cache_specs={name: _spec(spec_row) for name, spec_row in facts["layers"]},
    )
    group = SimpleNamespace(
        layer_names=[name for name, _ in facts["layers"]],
        kv_cache_spec=wrapper,
        is_eagle_group=False,
    )
    return wrapper, group


def _scheduler_view(wrapper: Any, layer_names: List[str]) -> Any:
    """What generate_scheduler_kv_cache_config leaves: any one sub spec (the
    first, in vLLM's implementation) with the group's full layer list."""
    folded = next(iter(wrapper.kv_cache_specs.values()))
    return SimpleNamespace(
        layer_names=list(layer_names),
        kv_cache_spec=folded,
        is_eagle_group=False,
    )


def _payload(metas: List[Any], tp_size: int = 1) -> List[Dict[str, Any]]:
    return [
        {"name": spec_name(rank, meta), "size": meta.per_block_bytes}
        for rank in range(tp_size)
        for meta in metas
    ]


def _parse(group: Any, mbs: int) -> List[Any]:
    config: Any = SimpleNamespace(kv_cache_groups=[group])
    return parse_groups(config, mbs, calculate_kv_scales=False)


class TestRoleViews(unittest.TestCase):
    """P1: one config, two views -- same group geometry, different buckets."""

    def setUp(self):
        self.wrapper, self.worker_group = _worker_view()
        self.sched_group = _scheduler_view(self.wrapper, self.worker_group.layer_names)
        self.worker_metas = _parse(self.worker_group, mbs=64)
        self.sched_metas = _parse(self.sched_group, mbs=64)

    def test_worker_view_splits_into_buckets(self):
        self.assertEqual([m.spec_suffix for m in self.worker_metas], ["", "_b1"])
        self.assertEqual([m.per_block_bytes for m in self.worker_metas], [83968, 16896])

    def test_scheduler_view_is_folded(self):
        # The folded spec is the first sub spec with the whole group's layers:
        # one location whose bytes differ from either worker bucket.
        self.assertEqual(len(self.sched_metas), 1)
        self.assertEqual(self.sched_metas[0].page_bytes, 41984)
        self.assertEqual(self.sched_metas[0].per_block_bytes, 167936)
        self.assertEqual(self.sched_metas[0].layer_names, self.worker_group.layer_names)

    def test_role_invariant_fields_agree(self):
        # R3: group_idx / block_size / block-table geometry are identical in
        # both views -- everything the scheduler actually consumes.
        self.assertEqual(
            {(m.group_idx, m.block_size) for m in self.worker_metas},
            {(m.group_idx, m.block_size) for m in self.sched_metas},
        )

    def test_the_two_views_would_register_different_payloads(self):
        # Why registration is worker-only: the folded view would publish a
        # different name -> bytes map than the worker's, and the manager
        # refuses the second, mismatching registration.
        self.assertNotEqual(_payload(self.worker_metas), _payload(self.sched_metas))


class _RoleSpy:
    """Records the constructor arguments of the role object it replaces."""

    instances: List["_RoleSpy"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        _RoleSpy.instances.append(self)


class _FakeManagerClient:
    def __init__(self) -> None:
        self.register_requests: List[Dict[str, Any]] = []

    def register_instance(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.register_requests.append(request)
        return {"storage_configs": "[]"}

    def close(self) -> None:
        return None


def _vllm_config() -> Any:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            served_model_name="tiny", dtype="bfloat16", use_mla=True
        ),
        cache_config=SimpleNamespace(
            block_size=64, cache_dtype="auto", calculate_kv_scales=False
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=dict(EXTRA_CONFIG)
        ),
    )


class TestRegistrationRole(unittest.TestCase):
    """Only a worker-role connector registers the instance's locations."""

    def setUp(self):
        _RoleSpy.instances = []
        self.manager = _FakeManagerClient()
        _, self.group = _worker_view()
        self.connector = self._build(KVConnectorRole.WORKER)

    def _build(self, role: Any) -> Any:
        with mock.patch.object(
            v1_connector.KvCacheManagerClient,
            "from_connector_config",
            return_value=self.manager,
        ):
            with mock.patch.object(
                v1_connector, "TpCoordinatorClient", return_value=mock.MagicMock()
            ):
                with mock.patch.object(v1_connector, "ConnectorScheduler", _RoleSpy):
                    with mock.patch.object(v1_connector, "ConnectorWorker", _RoleSpy):
                        config: Any = SimpleNamespace(kv_cache_groups=[self.group])
                        return v1_connector.TairKvCacheConnector(
                            _vllm_config(), role, config
                        )

    def test_worker_registers_the_bucket_payload(self):
        self.assertEqual(len(self.manager.register_requests), 1)
        request = self.manager.register_requests[0]
        self.assertEqual(
            request["location_spec_infos"],
            [
                {"name": "tp0_g0", "size": 83968},
                {"name": "tp0_g0_b1", "size": 16896},
            ],
        )
        self.assertEqual(request["block_size"], 64)
        self.assertEqual(request["instance_id"], EXTRA_CONFIG["instance_id"])
        # No state groups: no location_spec_groups advertised.
        self.assertNotIn("location_spec_groups", request)
        # The role object was built with the registration response.
        self.assertEqual(len(_RoleSpy.instances), 1)
        self.assertEqual(_RoleSpy.instances[0].args[-1], {"storage_configs": "[]"})

    def test_scheduler_role_does_not_register(self):
        _RoleSpy.instances = []
        self.manager.register_requests = []
        connector = self._build(KVConnectorRole.SCHEDULER)
        self.assertEqual(self.manager.register_requests, [])
        self.assertIsNone(connector.connector_worker)
        self.assertIsNotNone(connector.connector_scheduler)
        # The scheduler role object still gets the manager client: it needs it
        # for the location queries.
        self.assertIs(_RoleSpy.instances[0].args[5], self.manager)


class TestGoldenDrift(unittest.TestCase):
    """The frozen group facts must match the local tiny-model golden.

    Generated outside this repository (see 07/08); skipped when the file is
    not on this machine, so the fast ring stays self-contained.
    """

    def setUp(self):
        self.golden: Any = load_golden()
        if self.golden is None:
            self.skipTest(
                "tiny-model golden not available (set KVCM_MLA_GOLDEN to enable)"
            )

    def test_v32_group_matches_the_golden(self):
        for row, (model, index) in GOLDEN_GROUPS.items():
            with self.subTest(group=row):
                facts = FROZEN_GROUPS[row]
                golden_group = self.golden["models"][model]["groups"][index]
                self.assertEqual(golden_group["block_size"], facts["block_size"])
                self.assertEqual(
                    golden_group["page_size_bytes_total"], facts["page_size_bytes"]
                )
                self.assertEqual(golden_group["page_sizes"], facts["page_sizes"])
                self.assertEqual(
                    [name for name, _ in facts["layers"]],
                    golden_group["layer_names"],
                )
                specs = self.golden["models"][model]["specs"]
                for name, spec_row in facts["layers"]:
                    with self.subTest(layer=name):
                        golden_spec = specs[name]
                        frozen = FROZEN[spec_row]
                        self.assertEqual(
                            golden_spec["real_page_size_bytes"],
                            frozen["real_page_size_bytes"],
                        )
                        self.assertEqual(
                            golden_spec["page_size_padded"],
                            frozen["page_size_padded"],
                        )
                        self.assertEqual(
                            golden_spec["storage_block_size"],
                            frozen["storage_block_size"],
                        )
                        self.assertEqual(
                            golden_spec["dtype"], frozen["kwargs"]["dtype"]
                        )
                        self.assertEqual(
                            golden_spec["bytes_per_storage_row"],
                            frozen["real_page_size_bytes"]
                            // frozen["storage_block_size"],
                        )

    def test_golden_block_size_matches_the_group_fixture(self):
        # The V3.2 engine block size (64) is what the manager block size
        # defaults to; a drift here would move every per-block byte count.
        model = self.golden["models"]["dsv32_tiny"]
        self.assertEqual(
            model["block_size"], FROZEN_GROUPS["v32_wrapper_worker"]["block_size"]
        )


if __name__ == "__main__":
    unittest.main()
