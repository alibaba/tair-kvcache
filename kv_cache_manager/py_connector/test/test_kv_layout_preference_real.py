"""The layout preference against the *installed* vLLM.

``test_kv_layout_preference.py`` decides on the stand-in ring; this module
re-derives the same claims from the real engine classes of this era:

* the preference that goes out matches what the era's layout name space can
  express (``BLNHC`` from 0.29.0 on, ``None`` while only NHD/HND exist);
* through the engine's own resolution (``get_kv_connector_cache_layout`` +
  ``resolve_kv_cache_layout``) the packed-block model set lands on ``BLNHC``
  while a standard model keeps the engine default ``LBNHC``, byte for byte the
  same as with no preference at all;
* the fallback list of backends that declare no layout *contains* ``BLNHC``
  (mutation-style negative control: without the model gate the same standard
  model would be moved onto a packed-block layout);
* on an era that has no candidate-set resolution, the engine adopts a
  connector's answer verbatim -- which is why the hook stays silent there;
* the register-time token-major gate of ``ConnectorWorker`` accepts pages built
  in the preferred layout and rejects the head-major one the engine would pick
  without the preference.

Run it once per dual-anchor era:

    kvtest unit <worktree>                      # default env: vllm-0.26.0
    kvtest unit <worktree> --env vllm-0.31.0

Deliberately NOT dependent on ``vllm_stubs``: the whole point is the real class
tree. Skips itself when vLLM or the connector is unavailable (open-source CI) or
when the stubs already replaced the module space in this process.
"""

import importlib
import unittest
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

try:
    import torch
    import vllm
except ImportError as exc:  # pragma: no cover - environment dependent
    torch = None  # ty: ignore[invalid-assignment]
    vllm = None  # ty: ignore[invalid-assignment]
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""

try:
    from kv_cache_manager.py_connector.vllm.v1_connector import (
        TairKvCacheConnector,
    )
except ImportError as exc:  # pragma: no cover - needs the compiled client
    TairKvCacheConnector = None  # ty: ignore[invalid-assignment]
    _CONNECTOR_ERROR = str(exc)
else:
    _CONNECTOR_ERROR = ""

#: The vLLM architectures of the packed-block family (indexer pages beside the
#: MLA latent pages, see _PACKED_BLOCK_ARCHITECTURES).
V41_ARCHITECTURES = ["DeepseekV41ForCausalLM"]
#: A model the connector serves today; the engine gives it LBNHC.
STANDARD_ARCHITECTURES = ["Qwen2ForCausalLM"]

CONNECTOR_MODULE = "kv_cache_manager.py_connector.vllm.v1_connector"


def _skip_reason() -> str:
    if _IMPORT_ERROR:
        return f"real vLLM not importable ({_IMPORT_ERROR})"
    if getattr(vllm, "_kvcm_test_stub", False) or "vllm_stubs" in str(
        getattr(vllm, "__file__", "")
    ):
        return "vLLM is shadowed by the test stubs in this process"
    if _CONNECTOR_ERROR:
        return f"the connector is not importable ({_CONNECTOR_ERROR})"
    return ""


def _module(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _layout_enum() -> Optional[Any]:
    """The engine's layout name space, or None while it has only NHD/HND."""
    module = _module("vllm.v1.kv_cache_layout")
    return getattr(module, "KVCacheLayout", None) if module else None


def _engine_utils() -> Optional[Any]:
    return _module("vllm.v1.attention.backends.utils")


def _config(
    architectures: Any,
    connector: str = "TairKvCacheConnector",
    module_path: Optional[str] = CONNECTOR_MODULE,
) -> Any:
    """A vllm_config-shaped stub for the layout hook.

    The negotiation only reads ``kv_transfer_config``'s connector name/module
    (to reach the connector class) and writes the resolved ``kv_cache_layout``
    onto ``cache_config``; ``model_config.architectures`` is what the hook
    itself reads. A real ``CacheConfig`` is used where the era has one.
    """
    from vllm.config import CacheConfig

    # ``architecture`` mirrors what a real ModelConfig carries: the
    # implementation the engine resolved from ``architectures``. The layout
    # predicate reads it (falling back to the registry's supported names), so
    # the fixture records the same pick a real config would.
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=architectures, architecture=architectures[0]
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector, kv_connector_module_path=module_path
        ),
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
    )


class RealEngineCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reason = _skip_reason()
        if reason:
            raise unittest.SkipTest(reason)


class TestEraContract(RealEngineCase):
    """The hook answers only in the eras that can carry the answer."""

    def test_packed_block_model_preference_matches_the_layout_name_space(self):
        enum = _layout_enum()
        preference = TairKvCacheConnector.get_required_kvcache_layout(
            _config(V41_ARCHITECTURES)
        )
        if enum is not None and hasattr(enum, "BLNHC"):
            self.assertEqual(preference, "BLNHC")
            return
        # vLLM <= 0.28: only NHD/HND exist, and the engine adopts a connector's
        # answer without checking it against the backends. Assert the era is
        # really in that state, so this branch cannot pass vacuously.
        self.assertIsNone(preference)
        utils = _engine_utils()
        validator = getattr(utils, "is_valid_kv_cache_layout", None)
        if validator is not None:
            self.assertFalse(validator("BLNHC"))

    def test_standard_model_gets_no_preference(self):
        self.assertIsNone(
            TairKvCacheConnector.get_required_kvcache_layout(
                _config(STANDARD_ARCHITECTURES)
            )
        )


class TestNegotiatedLayout(RealEngineCase):
    """The engine's own resolution, fed the candidates this era reports."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        utils = _engine_utils()
        if _layout_enum() is None or not hasattr(utils, "resolve_kv_cache_layout"):
            raise unittest.SkipTest("era has no backend candidate-set resolution")
        cls.utils = utils

    def resolve_layout(self, candidates, architectures, specs=None) -> str:
        config = _config(architectures)
        resolved = self.utils.resolve_kv_cache_layout(
            config, [list(candidates)], specs or None
        )
        # The engine records the resolution on the cache config it was handed.
        self.assertEqual(config.cache_config.kv_cache_layout, resolved.name)
        return resolved.name

    def candidates(self, backends) -> list:
        layouts = self.utils.get_supported_kv_cache_layouts(backends)
        return [layout.name for layout in layouts]

    def attention_backend(self, module_name: str, class_name: str):
        module = _module(module_name)
        backend = getattr(module, class_name, None) if module else None
        if backend is None:
            self.skipTest(f"{module_name}.{class_name} is not in this era")
        return backend

    def v41_candidates(self) -> list:
        """The V4.1 layer set's candidate set, from its real backends.

        Only the indexer backends declare layouts (measured on 0.31.0: the MLA
        and SWA backends return None, and an undeclared backend does not
        constrain the intersection); the pair they declare is what makes the
        engine pick the head-major BLHNC by default.
        """
        declaring = (
            ("vllm.v1.attention.backends.mla.indexer", "DeepseekV41IndexerBackend"),
            (
                "vllm.v1.attention.backends.mla.sparse_indexer",
                "DeepseekV41SparseIndexerBackend",
            ),
        )
        return self.candidates(
            [self.attention_backend(module, name) for module, name in declaring]
        )

    def test_packed_block_model_resolves_to_the_token_major_layout(self):
        candidates = self.v41_candidates()
        self.assertIn("BLNHC", candidates)
        # The trap this PR exists for: the engine's own first pick is the
        # head-major layout the transfer kernel cannot read.
        self.assertEqual(candidates[0], "BLHNC")
        self.assertEqual(self.resolve_layout(candidates, V41_ARCHITECTURES), "BLNHC")

    def test_standard_model_keeps_the_engine_default(self):
        # A connector with no preference at all, i.e. the pre-PR behaviour.
        no_preference = classmethod(lambda cls, vllm_config: None)
        for module_name, class_name in (
            ("vllm.v1.attention.backends.triton_attn", "TritonAttentionBackend"),
            ("vllm.v1.attention.backends.flash_attn", "FlashAttentionBackend"),
        ):
            backend = self.attention_backend(module_name, class_name)
            candidates = self.candidates([backend])
            with self.subTest(backend=class_name):
                self.assertEqual(
                    self.resolve_layout(candidates, STANDARD_ARCHITECTURES),
                    "LBNHC",
                )
                # Same answer as a connector without a preference: the hook is
                # transparent for models it has nothing to ask for.
                with mock.patch.object(
                    TairKvCacheConnector,
                    "get_required_kvcache_layout",
                    no_preference,
                ):
                    self.assertEqual(
                        self.resolve_layout(candidates, STANDARD_ARCHITECTURES),
                        "LBNHC",
                    )

    def test_standard_model_with_mixed_spec_shapes_keeps_the_engine_default(self):
        # The residual narrowing the engine applies to specs that mix HNC
        # shapes (blocks/heads/sizes) must not change the outcome either.
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        backend = self.attention_backend(
            "vllm.v1.attention.backends.triton_attn", "TritonAttentionBackend"
        )
        candidates = self.candidates([backend])
        specs = [
            FullAttentionSpec(
                block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
            ),
            FullAttentionSpec(
                block_size=16, num_kv_heads=4, head_size=64, dtype=torch.bfloat16
            ),
        ]
        self.assertEqual(
            self.resolve_layout(candidates, STANDARD_ARCHITECTURES, specs), "LBNHC"
        )

    def test_unguarded_preference_would_move_standard_models(self):
        """Negative control: the model gate is load-bearing, not decoration.

        vLLM's fallback preference list (used for backends that declare no
        layout -- flash_attn/triton) contains BLNHC, so an unconditional
        ``"BLNHC"`` is accepted and silently moves the model onto a
        packed-block layout.
        """
        backend = self.attention_backend(
            "vllm.v1.attention.backends.triton_attn", "TritonAttentionBackend"
        )
        candidates = self.candidates([backend])
        if "BLNHC" not in candidates:
            self.skipTest("engine no longer offers BLNHC to undeclared backends")
        unguarded = classmethod(lambda cls, vllm_config: "BLNHC")
        with mock.patch.object(
            TairKvCacheConnector, "get_required_kvcache_layout", unguarded
        ):
            self.assertEqual(
                self.resolve_layout(candidates, STANDARD_ARCHITECTURES), "BLNHC"
            )


class TestLegacyEraAdoptsTheAnswerVerbatim(RealEngineCase):
    """Why the hook must stay silent before the name space exists."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if _layout_enum() is not None:
            raise unittest.SkipTest("era resolves from backend candidate sets")
        if getattr(_engine_utils(), "get_kv_cache_layout", None) is None:
            raise unittest.SkipTest("era has no kv cache layout reader")

    def test_engine_takes_an_unknown_connector_answer_verbatim(self):
        import vllm.envs as envs
        from vllm.config import set_current_vllm_config

        utils = _engine_utils()
        reader = utils.get_kv_cache_layout
        setter = getattr(utils, "set_kv_cache_layout", None)
        if setter is not None:
            setter(None)
            self.addCleanup(setter, None)
        unguarded = classmethod(lambda cls, vllm_config: "BLNHC")
        with (
            mock.patch.object(
                TairKvCacheConnector, "get_required_kvcache_layout", unguarded
            ),
            set_current_vllm_config(_config(V41_ARCHITECTURES)),
            mock.patch.object(envs, "VLLM_KV_CACHE_LAYOUT", None),
        ):
            reader.cache_clear()
            self.assertEqual(reader(), "BLNHC")


class TestRegistrationUnderEachLayout(unittest.TestCase):
    """The register-time token-major gate, on tensors the engine built."""

    LAYERS = ("l0", "l1")
    BLOCK_SIZE = 16

    @classmethod
    def setUpClass(cls) -> None:
        reason = _skip_reason()
        if reason:
            raise unittest.SkipTest(reason)
        if _layout_enum() is None:
            raise unittest.SkipTest("era has no block-outer layouts")
        interfaces = _module("vllm.v1.kv_cache_interface")
        if not hasattr(interfaces, "create_kv_cache_views"):
            raise unittest.SkipTest("era builds no per-layer cache views")
        cls.interfaces = interfaces
        from kv_cache_manager.py_connector.vllm.connector_worker import (
            ConnectorWorker,
        )
        from kv_cache_manager.py_connector.vllm.vllm_common import (
            AttentionGroupMeta,
        )

        cls.worker_cls = ConnectorWorker
        cls.meta_cls = AttentionGroupMeta

    def layer_views(self, spec, layout):
        """The per-layer tensors the engine hands the connector, for ``layout``.

        Placement (layer stride, block stride) comes from the engine's own
        ``compute_layout_strides``, so the tensors are laid out exactly as a
        running engine would lay them out.
        """
        num_blocks, layers = 8, len(self.LAYERS)
        shape = self.interfaces.compute_layer_kv_cache_shape_bytes(
            spec, num_blocks, self.BLOCK_SIZE
        )
        strides = self.interfaces.compute_layout_strides(
            spec, num_blocks, layers, layout, self.BLOCK_SIZE
        )
        size = sum(extent * stride for extent, stride in zip((layers, *shape), strides))
        raw = torch.zeros(size, dtype=torch.int8)
        placement = self.interfaces.KVCacheTensor(
            size=size,
            layers=list(self.LAYERS),
            layer_stride=strides[0],
            block_stride=strides[1],
        )
        views = self.interfaces.create_kv_cache_views(
            raw, spec, num_blocks, layout, placement, self.BLOCK_SIZE
        )
        return dict(zip(self.LAYERS, views))

    def build(self, spec, layout):
        worker = self.worker_cls.__new__(self.worker_cls)
        worker._tp_rank = 0
        worker._device = "cpu"
        page_bytes = spec.page_size_bytes
        meta = self.meta_cls(
            group_idx=0,
            layer_names=list(self.LAYERS),
            block_size=self.BLOCK_SIZE,
            per_block_bytes=(page_bytes // self.BLOCK_SIZE)
            * self.BLOCK_SIZE
            * len(self.LAYERS),
            page_bytes=page_bytes,
        )
        return worker._build_attention_group(meta, self.layer_views(spec, layout))

    def test_token_major_layouts_register_and_the_engine_default_does_not(self):
        """BLNHC is the point of the preference: its pages are readable.

        LBNHC (the engine default for standard models) and BLNHC both register;
        BLHNC -- what the engine picks for packed-block models without the
        preference -- trips the connector's token-major gate.
        """
        spec = self.interfaces.FullAttentionSpec(
            block_size=self.BLOCK_SIZE,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )
        enum = _layout_enum()
        for layout, block_stride in (
            (enum.LBNHC, 0),
            (enum.BLNHC, None),  # None: only "strided, not flat" is asserted
        ):
            with self.subTest(layout=layout.name):
                group = self.build(spec, layout)
                self.assertEqual(group.kernel_block_size, self.BLOCK_SIZE)
                self.assertEqual(group.per_token_dim, 8 * 2 * 128)
                if block_stride == 0:
                    self.assertEqual(group.block_stride, 0)
                else:
                    self.assertNotEqual(group.block_stride, 0)
        with self.subTest(layout="BLHNC"):
            with self.assertRaises(AssertionError) as ctx:
                self.build(spec, enum.BLHNC)
            self.assertIn("not token-major", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
