"""The same spec-receipt table, re-derived from the *real* vLLM of this era.

``test_spec_gate.py`` runs the zoo on the stand-in (its BUILD target has no
vLLM dependency); this module runs the identical rows through the engine that
is actually installed, so the table cannot be a stub-only belief. Run it once
per dual-anchor era:

    kvtest unit <worktree>                  # default env: vllm-0.26.0
    kvtest unit <worktree> --env vllm-0.31.0
    # or, without bazel, from the host:
    PYTHONPATH=<worktree> envs/vllm-0.31.0/bin/python \
        -m unittest kv_cache_manager.py_connector.test.test_spec_gate_real -v

Deliberately NOT dependent on ``vllm_stubs`` (see the BUILD comment on
test_mla_spec_facts): the whole point is the real class tree. When vLLM is
absent (open-source CI) or its module space was already replaced by the stubs
(a same-process full-suite run), every test skips instead of comparing the
stand-in to the table it was built from.
"""

import inspect
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from kv_cache_manager.py_connector.test import spec_gate_facts as facts

#: Where vLLM itself is *absent* (open-source CI), the suite skips. Only the
#: engine may be missing: a failure to import a *connector* symbol must not be
#: laundered into a skip, or the dual-era job would report green while the very
#: interface it guards regressed -- see _missing_engine below.
_MISSING_ENGINE_MODULES = ("vllm", "torch")


def _missing_engine(exc: ImportError) -> bool:
    """Is this error 'vLLM is not installed here' rather than a regression?

    True only when the *top-level* vLLM or torch package is missing: a broken
    wheel that cannot load an internal module (vllm._C, torch._C, ...) is a
    regression too, and skipping the dual-era suite for it would be exactly the
    false green this guard exists to prevent. Everything else -- a connector
    symbol that vanished, any nested module, a non-ModuleNotFoundError -- must
    fail the run. Everything else -- a connector symbol that vanished, a
    broken install, a non-ModuleNotFoundError -- must fail the run.
    """
    if not isinstance(exc, ModuleNotFoundError):
        return False
    return exc.name in _MISSING_ENGINE_MODULES


def _engine_absent(exc: ImportError) -> str:
    return f"real vLLM not installed here ({exc})"


try:
    import torch
    import vllm
except ImportError as _engine_import_error:  # pragma: no cover - env dependent
    if not _missing_engine(_engine_import_error):
        raise
    torch = None  # ty: ignore[invalid-assignment]
    vllm = None  # ty: ignore[invalid-assignment]
    KVCacheSpec = None  # ty: ignore[invalid-assignment]
    _check_attention_spec_supported = None  # ty: ignore[invalid-assignment]
    parse_groups = None  # ty: ignore[invalid-assignment]
    _IMPORT_ERROR = _engine_absent(_engine_import_error)
else:
    # vLLM is here: these imports must work, so an interface regression fails
    # the suite instead of skipping it.
    from vllm.v1.kv_cache_interface import KVCacheSpec

    from kv_cache_manager.py_connector.vllm.vllm_common import (
        _check_attention_spec_supported,
        build_spec_groups,
        parse_groups,
    )

    _IMPORT_ERROR = ""


def _skip_reason() -> str:
    if _IMPORT_ERROR:
        return _IMPORT_ERROR
    if getattr(vllm, "_kvcm_test_stub", False) or "vllm_stubs" in str(
        getattr(vllm, "__file__", "")
    ):
        return "vLLM is shadowed by the test stubs in this process"
    return ""


class RealEngineCase(unittest.TestCase):
    """Shared setup: the era's spec module behind the facts.Engine adapter."""

    @classmethod
    def setUpClass(cls) -> None:
        reason = _skip_reason()
        if reason:
            raise unittest.SkipTest(reason)
        cls.version: str = vllm.__version__
        cls.module = sys.modules["vllm.v1.kv_cache_interface"]
        cls.engine = facts.Engine(
            module=cls.module,
            dtype_of=lambda name: getattr(torch, name),
            parse_groups=parse_groups,
            version=cls.version,
        )
        cls.expected = facts.expected_classes(cls.version)

    def _spec_classes(self) -> Dict[str, type]:
        """Every KVCacheSpec subclass this vLLM exposes by name."""
        return {
            name: obj
            for name, obj in vars(self.module).items()
            if isinstance(obj, type)
            and issubclass(obj, KVCacheSpec)  # ty: ignore[invalid-argument-type]
            and obj is not KVCacheSpec
        }

    def assert_unavailable_is_expected(self, row: Dict[str, Any], message: str) -> None:
        """A row this era cannot build must be unable for a *declared* reason."""
        self.assertIsNotNone(
            self.expected,
            f"{row['name']}: {message}, and vLLM {self.version} is not a known era",
        )
        absent = [name for name in facts.class_names(row) if name not in self.expected]
        if absent:
            return
        # The class tree is there, so the era must be missing a *field* the row
        # declares (the compress_ratio <-> tokens_per_state migration).
        classes = [self.engine.class_of(name) for name in facts.needed_classes(row)]
        missing = [
            field
            for cls in classes
            if cls is not None
            for field in row.get("requires", ())
            if field not in inspect.signature(cls).parameters
        ]
        self.assertTrue(
            missing, f"{row['name']}: {message}, but its classes have every field"
        )


class TestZooOnTheRealEngine(RealEngineCase):
    """Every zoo row, on the real class tree of the installed era."""

    def test_class_table_matches_this_era(self):
        # The era map is the dual-anchor claim made checkable: 0.26 has no
        # CircularBuffer/KpoolTail/HiSparse class, 0.31 has no TQFullAttention.
        if self.expected is None:
            self.skipTest(f"vLLM {self.version} is not in the era table")
        present = set(self._spec_classes()) | {"AttentionSpec", "KVCacheSpec"}
        self.assertEqual(self.expected, present)

    def test_disposition_of_every_row(self):
        built = 0
        for row in facts.ZOO:
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                if result.outcome == "unavailable":
                    self.assert_unavailable_is_expected(row, result.message)
                    continue
                built += 1
                expected = facts.expected_outcome(row, self.engine)
                self.assertEqual(expected, result.outcome, result.message)
                if result.outcome != "refuse":
                    continue
                for keyword in row.get("keywords", ()):
                    self.assertIn(keyword, result.message)
                if facts.SPEC_DISPOSITIONS[row["cls"]] == "refuse":
                    # Whole-family refusals explain why and what would change
                    # it; variant refusals are pinned by their keywords.
                    self.assertIn("TairKvCacheConnector", result.message)
        # The oldest era in the table (0.14) builds 12 rows; the floor only
        # catches a table that silently stops running.
        self.assertGreater(built, 10, "the era built too few rows to be a check")

    def test_retention_of_every_windowed_row(self):
        # The real classes carry the window fields: the retention every consumer
        # keys on must come out of the *era's own* spec objects, not just the
        # union stand-in (review_r1 N3).
        for row in facts.ZOO:
            if row["disposition"] != "accept":
                continue
            result = facts.run_row(row, self.engine)
            if result.outcome != "accept":
                continue
            with self.subTest(row=row["name"]):
                own = [meta for meta in result.metas if meta.group_idx == 0][0]
                expected = row.get("retention")
                if expected is None:
                    self.assertIsNone(own.retention)
                    continue
                self.assertEqual(expected["kind"], own.retention.kind)
                self.assertEqual(expected["window_tokens"], own.retention.window_tokens)
                self.assertEqual(
                    expected.get("extra_retained_tokens", 0),
                    own.retention.extra_retained_tokens,
                )

    def test_sizes_of_every_accepted_row(self):
        for row in facts.ZOO:
            if row["disposition"] != "accept" or "page_bytes" not in row:
                continue
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                if result.outcome == "unavailable":
                    continue
                own = [meta for meta in result.metas if meta.group_idx == 0]
                self.assertTrue(own, result.message)
                self.assertEqual(row["page_bytes"], own[0].page_bytes)
                self.assertEqual(row["per_block_bytes"], own[0].per_block_bytes)

    def test_skipped_rows_keep_the_other_groups(self):
        for row in facts.ZOO:
            if facts.expected_outcome(row, self.engine) != "skip":
                continue
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                if result.outcome == "unavailable":
                    continue
                self.assertEqual([meta.group_idx for meta in result.metas], [1])


class TestUndeclaredSubclassesFailClosed(RealEngineCase):
    """A new FullAttentionSpec / MLA subclass must be declared before it transfers.

    The reviewer's probe for the pre-fix hole: an undeclared subclass -- and in
    particular one whose window hides in a brand-new field name, the RSWASpec
    shape -- used to slip through the family accept rule. Exact-type acceptance
    refuses it at the gate and at parse_groups, while the declared classes and
    the declared HiddenStateCacheSpec marker keep working.
    """

    def _subclass(self, base_name: str, class_name: str, **kwargs):
        base = self.engine.class_of(base_name)
        self.assertIsNotNone(base, f"{base_name} missing in vLLM {self.version}")
        return type(class_name, (base,), {})(**kwargs)

    def _gate_refusal(self, spec) -> str:
        with self.assertRaises(NotImplementedError) as ctx:
            _check_attention_spec_supported("group 0", spec, calculate_kv_scales=False)
        return str(ctx.exception)

    def _assert_parse_refuses(self, spec) -> None:
        with self.assertRaises(NotImplementedError):
            parse_groups(
                SimpleNamespace(
                    kv_cache_groups=[
                        SimpleNamespace(
                            layer_names=["l0"], kv_cache_spec=spec, is_eagle_group=False
                        )
                    ]
                ),
                16,
                calculate_kv_scales=False,
            )

    def test_an_undeclared_subclass_is_refused_even_with_a_brand_new_window(self):
        # The review report's CPU probe, now an assertion: the base gate refused
        # this spec (window whitelist), the pre-fix head accepted it.
        spec = self._subclass(
            "FullAttentionSpec",
            "FooBrandNewWindowSpec",
            block_size=16,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.bfloat16,
        )
        object.__setattr__(spec, "brand_new_window", 4096)
        self.assertIn("FooBrandNewWindowSpec", self._gate_refusal(spec))
        self._assert_parse_refuses(spec)

    def test_an_undeclared_subclass_with_a_known_window_is_refused_too(self):
        spec = self._subclass(
            "FullAttentionSpec",
            "FooWindowSpec",
            block_size=16,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.bfloat16,
            sliding_window=1024,
        )
        self.assertIn("FooWindowSpec", self._gate_refusal(spec))
        self._assert_parse_refuses(spec)
        # The exact declared class with the same fields is accepted: the
        # decision is the class, not the field set (the promote escape hatch).
        full = self.engine.class_of("FullAttentionSpec")
        self.assertIsNotNone(full)
        self.assertIsNone(
            _check_attention_spec_supported(
                "group 0",
                full(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=64,
                    dtype=torch.bfloat16,
                    sliding_window=1024,
                ),
                calculate_kv_scales=False,
            )
        )

    def test_undeclared_mla_subclasses_are_refused(self):
        for base_name, class_name in (
            ("MLAAttentionSpec", "FooMLASpec"),
            ("HiddenStateCacheSpec", "FooHiddenStateSpec"),
        ):
            with self.subTest(cls=class_name):
                if self.engine.class_of(base_name) is None:
                    continue
                spec = self._subclass(
                    base_name,
                    class_name,
                    block_size=16,
                    num_kv_heads=1,
                    head_size=576,
                    dtype=torch.bfloat16,
                )
                self.assertIn(class_name, self._gate_refusal(spec))

    #: Qwen3.5-4B mamba shapes (survey c1_gate_matrix): 2195456 B/page.
    MAMBA_SHAPES = ((3, 8192), (32, 128, 128))

    def _mamba(self, **extra):
        cls = self.engine.class_of("MambaSpec")
        self.assertIsNotNone(cls, f"MambaSpec missing in vLLM {self.version}")
        return cls(
            block_size=16,
            shapes=self.MAMBA_SHAPES,
            dtypes=(torch.float32, torch.float32),
            **extra,
        )

    def _hybrid_config(self, state_spec):
        full = self.engine.class_of("FullAttentionSpec")
        self.assertIsNotNone(full)
        return SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["m0", "m1"], kv_cache_spec=state_spec, is_eagle_group=False
                ),
                SimpleNamespace(
                    layer_names=["a0"],
                    kv_cache_spec=full(
                        block_size=16,
                        num_kv_heads=4,
                        head_size=256,
                        head_size_v=256,
                        dtype=torch.bfloat16,
                    ),
                    is_eagle_group=False,
                ),
            ]
        )

    def test_an_undeclared_mamba_subclass_is_refused(self):
        # The review probe: a state spec subclass carrying its own field used to
        # be taken as an ordinary state group (the same mechanism as the RSWA
        # hole, one family over). The transfer path relies on "a block's state
        # covers the prefix ending at that block" -- a subclass may not claim
        # that by inheritance.
        base = self.engine.class_of("MambaSpec")
        self.assertIsNotNone(base)
        foo_cls = type("FooMambaSpec", (base,), {})
        spec = foo_cls(
            block_size=16,
            shapes=self.MAMBA_SHAPES,
            dtypes=(torch.float32, torch.float32),
        )
        object.__setattr__(spec, "window", 1024)
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(self._hybrid_config(spec), 16, calculate_kv_scales=False)
        self.assertIn("FooMambaSpec", str(ctx.exception))

    def test_a_plain_mamba_group_is_still_a_state_group(self):
        # The control: the declared class keeps transferring as state, hybrid
        # instance included (the path PR2/PR3 build on).
        metas = parse_groups(self._hybrid_config(self._mamba()), 16, calculate_kv_scales=False)
        self.assertEqual(
            [type(meta).__name__ for meta in metas],
            ["StateGroupMeta", "AttentionGroupMeta"],
        )
        self.assertEqual(metas[0].page_size_bytes, 2195456)
        self.assertEqual(metas[0].per_block_bytes, 2195456 * 2)

    def test_an_undeclared_uniform_wrapper_subclass_is_refused(self):
        wrapper = self.engine.class_of("UniformTypeKVCacheSpecs")
        self.assertIsNotNone(wrapper)
        inner = facts.build_spec(facts.ROW_BY_NAME["wrapper_v32_pair"], self.engine)
        foo_cls = type("FooWrapperSpec", (wrapper,), {})
        spec = foo_cls(block_size=inner.block_size, kv_cache_specs=inner.kv_cache_specs)
        config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=sorted(inner.kv_cache_specs),
                    kv_cache_spec=spec,
                    is_eagle_group=False,
                )
            ]
        )
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(config, 64, calculate_kv_scales=False)
        self.assertIn("FooWrapperSpec", str(ctx.exception))

    def test_the_group_receipt_classes_have_no_subclasses_here(self):
        # The audit behind the exact match, per era: nothing in this vLLM
        # inherits from the two group-level receipts (the AST scan over all nine
        # era trees says the same for every era).
        for base_name in ("MambaSpec", "UniformTypeKVCacheSpecs"):
            base = self.engine.class_of(base_name)
            with self.subTest(base=base_name):
                self.assertEqual(
                    [],
                    [
                        name
                        for name, obj in self._spec_classes().items()
                        if obj is not base and issubclass(obj, base)
                    ],
                )

    def test_a_windowed_hidden_state_spec_is_refused(self):
        # N1: the marker gets no window exemption -- it walks the same window
        # test and latent-layout gate as MLAAttentionSpec.
        hidden = self.engine.class_of("HiddenStateCacheSpec")
        if hidden is None:
            self.skipTest(f"vLLM {self.version} has no HiddenStateCacheSpec")
        if "sliding_window" not in inspect.signature(hidden).parameters:
            self.skipTest("this era's HiddenStateCacheSpec takes no window field")
        windowed = hidden(
            block_size=16,
            num_kv_heads=3,
            head_size=2560,
            dtype=torch.bfloat16,
            sliding_window=1024,
        )
        message = self._gate_refusal(windowed)
        self.assertIn("HiddenStateCacheSpec", message)
        self.assertIn("sliding_window=1024", message)
        # The plain marker (the only shape vLLM builds) stays accepted.
        self.assertEqual(
            "accept",
            facts.run_row(facts.ROW_BY_NAME["hidden_state"], self.engine).outcome,
        )


class TestTransferProjection(RealEngineCase):
    """vLLM 0.29+ projects the group list for connectors (enable_kv_transfer).

    Membership comes from the engine's own projection; the index we keep is the
    raw vLLM group index, because that is the space of the block-id payloads
    this connector receives (KVCacheBlocks.get_block_ids / new_block_ids).
    """

    def _group_spec(self):
        cls = getattr(self.module, "KVCacheGroupSpec", None)
        if cls is None or "enable_kv_transfer" not in inspect.signature(cls).parameters:
            self.skipTest(f"vLLM {self.version} has no enable_kv_transfer")
        return cls

    def _config(self, groups):
        cls = self._group_spec()
        spec = {
            "full": facts.build_spec(facts.ROW_BY_NAME["full"], self.engine),
            "mla_bf16": facts.build_spec(facts.ROW_BY_NAME["mla_bf16"], self.engine),
        }
        KVCacheConfig = getattr(self.module, "KVCacheConfig")
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[
                cls([layer], spec[row], **extra) for layer, row, extra in groups
            ],
        )

    def test_the_config_projection_and_the_parser_agree(self):
        config = self._config(
            [
                ("a0", "full", {}),
                ("hidden", "mla_bf16", {"enable_kv_transfer": False}),
                ("a2", "full", {}),
            ]
        )
        # The engine says which groups are transferable...
        self.assertEqual((0, 2), config.transfer_group_ids)
        # ... and the parser must publish exactly those, keeping their indices:
        # the payload stays indexed by the vLLM group id (0 and 2 here).
        metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual([0, 2], [meta.group_idx for meta in metas])
        from kv_cache_manager.py_connector.vllm.vllm_common import spec_name

        self.assertEqual(["tp7_g0", "tp7_g2"], [spec_name(7, meta) for meta in metas])

    def test_a_cacheable_class_is_still_dropped_when_the_engine_says_so(self):
        # MLAAttentionSpec is accepted on its own; only the projection can say
        # this instance's group is engine-owned (the HiSparse shape).
        config = self._config(
            [
                ("hidden", "mla_bf16", {"enable_kv_transfer": False}),
                ("a1", "full", {}),
            ]
        )
        with self.assertLogs(
            "kv_cache_manager.py_connector.common.logger", level="WARNING"
        ) as captured:
            metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual([1], [meta.group_idx for meta in metas])
        self.assertTrue(
            any("does not transfer it" in msg for msg in captured.output),
            captured.output,
        )


class TestTransferableScratch(RealEngineCase):
    """A transferable scratch group is refused -- unless the engine rebuilds it.

    The compression ring (V4.1's CircularBufferSpec) is rebuilt per request, so
    it is skipped even though the engine exposes it; an unexplained scratch
    class stays refused (fail-closed).
    """

    def _config(self, spec_row, enable_kv_transfer):
        group_cls = getattr(self.module, "KVCacheGroupSpec", None)
        if group_cls is None or "enable_kv_transfer" not in inspect.signature(
            group_cls
        ).parameters:
            self.skipTest(f"vLLM {self.version} has no enable_kv_transfer")
        config_cls = getattr(self.module, "KVCacheConfig", None)
        if config_cls is None or not hasattr(config_cls, "transfer_group_ids"):
            self.skipTest(f"vLLM {self.version} has no transfer projection")
        full = facts.build_spec(facts.ROW_BY_NAME["full"], self.engine)
        return config_cls(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[
                group_cls(
                    ["ring"],
                    facts.build_spec(facts.ROW_BY_NAME[spec_row], self.engine),
                    enable_kv_transfer=enable_kv_transfer,
                ),
                group_cls(["attn"], full),
            ],
        )

    def _needs_scratch_contract(self, spec_row):
        row = facts.ROW_BY_NAME[spec_row]
        if self.version not in facts.skip_eras(row):
            self.skipTest(f"vLLM {self.version} does not declare {row['cls']} scratch")

    def test_a_transferable_compression_ring_is_skipped(self):
        # enable_kv_transfer=True (the default) exposes the group to
        # connectors, but the ring is rebuilt per request by the compression
        # machinery ("one block per request holding the raw keys of the token
        # group that is still being compressed"), so dropping it is not a hole
        # -- an external hit restores the main MLA rows and the ring refills.
        self._needs_scratch_contract("circular_buffer")
        config = self._config("circular_buffer", True)
        self.assertEqual((0, 1), config.transfer_group_ids)
        with self.assertLogs(
            "kv_cache_manager.py_connector.common.logger", level="WARNING"
        ) as captured:
            metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual([1], [meta.group_idx for meta in metas])
        self.assertTrue(
            any("compression ring" in msg for msg in captured.output), captured.output
        )

    def test_a_transferable_ring_pack_is_skipped(self):
        # The wrapper form: V4.1 packs the compressor rings into one
        # UniformTypeKVCacheSpecs, and every member being a ring is what makes
        # the pack engine-rebuilt scratch.
        self._needs_scratch_contract("circular_buffer")
        wrapper = self.engine.class_of("UniformTypeKVCacheSpecs")
        self.assertIsNotNone(wrapper)
        rings = facts.build_spec(facts.ROW_BY_NAME["circular_buffer"], self.engine)
        group = wrapper.from_specs({"l.cb0": rings, "l.cb1": rings})
        self.assertIsNotNone(group)
        group_cls = getattr(self.module, "KVCacheGroupSpec", None)
        config = getattr(self.module, "KVCacheConfig")(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[
                group_cls(["l.cb0", "l.cb1"], group),
                group_cls(
                    ["attn"], facts.build_spec(facts.ROW_BY_NAME["full"], self.engine)
                ),
            ],
        )
        self.assertEqual((0, 1), config.transfer_group_ids)
        with self.assertLogs(
            "kv_cache_manager.py_connector.common.logger", level="WARNING"
        ) as captured:
            metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual([1], [meta.group_idx for meta in metas])
        self.assertTrue(
            any(
                "UniformTypeKVCacheSpecs" in msg and "compression ring" in msg
                for msg in captured.output
            ),
            captured.output,
        )

    def test_an_undeclared_ring_subclass_is_refused(self):
        # A custom spec subclassing the ring inherits no exemption: the
        # engine-rebuilt exception matches exact types, so the group lands on
        # the transferable-scratch refusal (fail-closed) -- for the spec itself
        # and for a pack that contains it.
        self._needs_scratch_contract("circular_buffer")
        ring_cls = self.engine.class_of("CircularBufferSpec")
        self.assertIsNotNone(ring_cls)
        foo = type("FooRingSpec", (ring_cls,), {})
        declared = facts.build_spec(facts.ROW_BY_NAME["circular_buffer"], self.engine)
        subclass = object.__new__(foo)
        subclass.__dict__.update(declared.__dict__)
        group_cls = getattr(self.module, "KVCacheGroupSpec", None)
        config_cls = getattr(self.module, "KVCacheConfig")
        full = facts.build_spec(facts.ROW_BY_NAME["full"], self.engine)
        config = config_cls(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[
                group_cls(["ring"], subclass),
                group_cls(["attn"], full),
            ],
        )
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(config, 16, calculate_kv_scales=False)
        self.assertIn("FooRingSpec", str(ctx.exception))

        wrapper = self.engine.class_of("UniformTypeKVCacheSpecs")
        pack = wrapper.from_specs({"l.cb0": subclass, "l.cb1": subclass})
        self.assertIsNotNone(pack)
        config = config_cls(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[
                group_cls(["l.cb0", "l.cb1"], pack),
                group_cls(["attn"], full),
            ],
        )
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(config, 16, calculate_kv_scales=False)
        self.assertIn("UniformTypeKVCacheSpecs", str(ctx.exception))

    def test_an_unknown_transferable_scratch_class_is_refused(self):
        # KpoolTailSpec: same manager family, but nothing shows its data path is
        # rebuilt per request, so a transferable group is refused (fail-closed).
        self._needs_scratch_contract("kpool_tail")
        config = self._config("kpool_tail", True)
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(config, 16, calculate_kv_scales=False)
        message = str(ctx.exception)
        self.assertIn("KpoolTailSpec", message)
        self.assertIn("enable_kv_transfer=True", message)

    def test_a_scratch_group_the_engine_owns_is_still_skipped(self):
        self._needs_scratch_contract("circular_buffer")
        config = self._config("circular_buffer", False)
        self.assertEqual((1,), config.transfer_group_ids)
        with self.assertLogs(
            "kv_cache_manager.py_connector.common.logger", level="WARNING"
        ) as captured:
            metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual([1], [meta.group_idx for meta in metas])
        self.assertTrue(
            any("does not transfer it" in msg for msg in captured.output),
            captured.output,
        )


class TestPooledFullAttention(RealEngineCase):
    """One row per token is the transfer model: pooled rows are refused."""

    def _has_tokens_per_state(self) -> bool:
        cls = self.engine.class_of("FullAttentionSpec")
        self.assertIsNotNone(cls)
        return "tokens_per_state" in inspect.signature(cls).parameters

    def _full(self, **extra):
        cls = self.engine.class_of("FullAttentionSpec")
        self.assertIsNotNone(cls)
        return cls(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            **extra,
        )

    def _pooled(self, tokens_per_state=None, **extra):
        from fractions import Fraction

        if not self._has_tokens_per_state():
            self.skipTest(f"vLLM {self.version} has no tokens_per_state")
        if tokens_per_state is None:
            tokens_per_state = Fraction(1, 8)
        return self._full(tokens_per_state=tokens_per_state, **extra)

    def test_pooled_and_compressed_full_attention_are_refused(self):
        # Whisper pools (Fraction(1, n) = several states per token row) and the
        # promote path can keep the field on a FullAttentionSpec; both are
        # refused by field name instead of failing at registration.
        cases = (
            ("bare", self._pooled(), "tokens_per_state=1/8"),
            (
                "windowed (the promote shape)",
                self._pooled(sliding_window=1024),
                "tokens_per_state=1/8",
            ),
            ("compressed", self._pooled(tokens_per_state=2), "tokens_per_state=2"),
        )
        for label, spec, keyword in cases:
            with self.subTest(shape=label):
                with self.assertRaises(NotImplementedError) as ctx:
                    _check_attention_spec_supported(
                        "group 0", spec, calculate_kv_scales=False
                    )
                message = str(ctx.exception)
                self.assertIn(keyword, message)
                self.assertIn("one state per token", message)

    def test_the_one_state_per_token_shapes_still_pass(self):
        # Runs on every era: the plain and promote shapes need no new field,
        # and the explicit tokens_per_state=1 case joins them where the field
        # exists (it is the default, never a way in).
        cases = [{}, dict(sliding_window=1024)]
        if self._has_tokens_per_state():
            cases.append(dict(tokens_per_state=1))
        for extra in cases:
            with self.subTest(extra=extra):
                self.assertIsNone(
                    _check_attention_spec_supported(
                        "group 0", self._full(**extra), calculate_kv_scales=False
                    )
                )


class TestBucketDtype(RealEngineCase):
    """AttentionGroupMeta.dtype is the bucket's element type (spec.dtype)."""

    def test_accepted_buckets_report_their_spec_dtype(self):
        for row_name, expected in (
            ("full", torch.bfloat16),
            ("mla_bf16", torch.bfloat16),
            ("mla_fp8_ds", torch.uint8),
            ("mla_indexer", torch.uint8),
        ):
            with self.subTest(row=row_name):
                row = facts.ROW_BY_NAME[row_name]
                spec = facts.build_spec(row, self.engine)
                metas = facts.run_row(row, self.engine).metas
                self.assertEqual([expected], [meta.dtype for meta in metas])
                self.assertEqual([spec.dtype], [meta.dtype for meta in metas])

    def test_wrapper_buckets_keep_their_own_dtype(self):
        # A packed group is one vLLM group but several buckets: V3.2 packs an
        # fp8_ds_mla main with an fp8 indexer (both uint8 element-wise, distinct
        # page sizes), and block_size 64 also admits a bf16 main next to the
        # uint8 indexer -- one instance, two element sizes, and the staging
        # pool keys by dtype, so the bucket must carry it.
        wrapper = self.engine.class_of("UniformTypeKVCacheSpecs")
        self.assertIsNotNone(wrapper)
        packed = facts.build_spec(facts.ROW_BY_NAME["wrapper_v32_pair"], self.engine)
        from dataclasses import replace

        main = replace(
            facts.build_spec(facts.ROW_BY_NAME["mla_bf16"], self.engine), block_size=64
        )
        indexer = facts.build_spec(facts.ROW_BY_NAME["mla_indexer"], self.engine)
        cases = {
            "v32_pair": packed.kv_cache_specs,
            "mixed_element_sizes": {"l.main": main, "l.indexer": indexer},
        }
        for label, specs in cases.items():
            with self.subTest(shape=label):
                group = wrapper.from_specs(dict(specs))
                self.assertIsNotNone(group, label)
                metas = parse_groups(
                    SimpleNamespace(
                        kv_cache_groups=[
                            SimpleNamespace(
                                layer_names=sorted(specs),
                                kv_cache_spec=group,
                                is_eagle_group=False,
                            )
                        ]
                    ),
                    64,
                    calculate_kv_scales=False,
                )
                self.assertEqual(
                    sorted(str(spec.dtype) for spec in specs.values()),
                    sorted(str(meta.dtype) for meta in metas),
                )
                self.assertEqual(
                    {spec.dtype for spec in specs.values()},
                    {meta.dtype for meta in metas},
                )
        self.assertEqual(
            {torch.uint8, torch.bfloat16},
            {spec.dtype for specs in cases.values() for spec in specs.values()},
        )

    def test_the_spec_dtype_is_the_dtype_a_tensor_gets_for_that_cache(self):
        # Both sides call kv_cache_dtype_str_to_dtype on the same cache_dtype
        # (gpu_model_runner.py for the KV tensors, the attention layer for the
        # spec), so the bucket dtype the connector records is the staging tensor
        # dtype the worker will see. Assert it on the fp8 shapes, where a wrong
        # guess is silent (uint8 vs a float8 view share the byte count).
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        for row_name, cache_dtype in (
            ("mla_fp8_ds", "fp8_ds_mla"),
            ("mla_indexer", "fp8"),
        ):
            with self.subTest(row=row_name):
                spec = facts.build_spec(facts.ROW_BY_NAME[row_name], self.engine)
                self.assertEqual(torch.uint8, spec.dtype)
                self.assertEqual(
                    torch.uint8, kv_cache_dtype_str_to_dtype(cache_dtype, None)
                )


class TestImportGuard(RealEngineCase):
    """A missing engine skips; a broken connector import must fail the run."""

    def test_only_a_missing_engine_counts_as_absent(self):
        self.assertTrue(
            _missing_engine(ModuleNotFoundError("No module named 'vllm'", name="vllm"))
        )
        self.assertTrue(
            _missing_engine(
                ModuleNotFoundError("No module named 'torch'", name="torch")
            )
        )
        # A connector symbol that vanished, or any other import failure, is a
        # regression: it must not be reported as "vLLM is not installed here".
        self.assertFalse(
            _missing_engine(ImportError("cannot import name 'parse_groups'"))
        )
        self.assertFalse(
            _missing_engine(
                ModuleNotFoundError("No module named 'x'", name="some_new_dependency")
            )
        )
        # Installed but broken: a missing internal module is a regression, not
        # an absent engine (a false skip here would hide a broken wheel).
        for nested in ("vllm._C", "torch._C", "vllm.v1.kv_cache_interface"):
            with self.subTest(module=nested):
                self.assertFalse(
                    _missing_engine(
                        ModuleNotFoundError("No module named x", name=nested)
                    )
                )

    def _probe(self, blocked: str):
        root = str(Path(__file__).resolve().parents[3])
        script = textwrap.dedent(
            """
            import importlib.abc, sys

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name == {blocked!r} or name.startswith({blocked!r} + "."):
                        raise ModuleNotFoundError(
                            f"No module named {{name!r}}", name=name
                        )
                    return None

            sys.meta_path.insert(0, Blocker())
            import kv_cache_manager.py_connector.test.test_spec_gate_real as m
            print("IMPORTED", m._IMPORT_ERROR)
            """
        ).format(blocked=blocked)
        env = dict(os.environ, PYTHONPATH=root)
        return subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )

    def test_a_missing_vllm_is_still_a_skip(self):
        proc = self._probe("vllm")
        self.assertEqual(0, proc.returncode, proc.stderr[-800:])
        self.assertIn("real vLLM not installed here", proc.stdout)
        self.assertIn("IMPORTED", proc.stdout)

    def test_a_broken_connector_import_fails_instead_of_skipping(self):
        proc = self._probe("kv_cache_manager.py_connector.vllm.vllm_common")
        self.assertNotEqual(0, proc.returncode, proc.stdout)
        self.assertIn("kv_cache_manager.py_connector.vllm.vllm_common", proc.stderr)
        self.assertNotIn("IMPORTED", proc.stdout)


class TestEraShapes(RealEngineCase):
    """What this era spells and declares, through the same code path."""

    def test_the_eras_spelling_is_the_one_that_runs(self):
        # At most one of the two compressed rows is buildable here; the other
        # era's shape is reported as such instead of being silently built from
        # the wrong field. vLLM 0.14 has neither spelling (and no compressed
        # MLA at all). Since PR6 the buildable one is *accepted* as a
        # row-granular bucket, with the ratio the era's own field carries.
        compress = facts.run_row(facts.ROW_BY_NAME["mla_compressed_0_26"], self.engine)
        tokens = facts.run_row(facts.ROW_BY_NAME["mla_compressed_0_31"], self.engine)
        for result in (compress, tokens):
            self.assertIn(result.outcome, ("accept", "unavailable"), result.message)
        for result, ratio in ((compress, 4), (tokens, 2)):
            if result.outcome != "accept":
                continue
            own = [meta for meta in result.metas if meta.group_idx == 0][0]
            self.assertEqual(ratio, own.row_granular.tokens_per_row)
        if "accept" in (compress.outcome, tokens.outcome):
            self.assertEqual(
                1,
                (compress.outcome == "accept") + (tokens.outcome == "accept"),
                "both compressed spellings were buildable at once",
            )

    def test_scratch_flag_shapes_are_the_measured_ones(self):
        # The data behind the era-dependent skip (facts.SCRATCH_FLAGS): the
        # group is dropped only where the era's spec opts out of *both* halves,
        # which is why 0.29 and 0.30 refuse what 0.31 skips.
        measured_era = facts.SCRATCH_FLAGS.get(self.version)
        if not measured_era:
            self.skipTest(f"no measured scratch flags for vLLM {self.version}")
        for name, flags in measured_era.items():
            row = next(row for row in facts.ZOO if row["cls"] == name)
            with self.subTest(cls=name):
                spec = facts.build_spec(row, self.engine)
                measured = tuple(
                    getattr(spec, attribute, None)
                    for attribute in (
                        "prefix_cacheable",
                        "uses_slot_mapping",
                        "has_layer_views",
                    )
                )
                self.assertEqual(flags, measured)

    def test_skip_eras_agree_with_the_measured_flags(self):
        # Two independent data sets (the measured flags and the era list) must
        # say the same thing about where the skip applies.
        for row in facts.ZOO:
            if row["disposition"] != "skip":
                continue
            with self.subTest(row=row["cls"]):
                flags_data = facts.SCRATCH_FLAGS.get(self.version, {}).get(
                    facts.needed_classes(row)[0]
                )
                if flags_data is None:
                    self.assertNotIn(self.version, facts.skip_eras(row))
                    continue
                declared = flags_data[0] is False and (
                    flags_data[1] is False or flags_data[2] is False
                )
                self.assertEqual(declared, self.version in facts.skip_eras(row))


class TestCoverageGuard(RealEngineCase):
    """A spec class this vLLM ships without a declared disposition fails here.

    That is the RSWA lesson turned into a test: the hole existed because the
    gate tested *fields*, so a class with a differently-named window slipped
    through. The gate now accepts exact declared classes only and refuses
    anything else, so this guard is its other half -- it tells us when a new
    era ships a class (or a subclass) that should be declared instead of being
    refused in the field.
    """

    def test_every_spec_class_has_a_disposition(self):
        present = set(self._spec_classes())
        undeclared = present - set(facts.SPEC_DISPOSITIONS)
        self.assertEqual(set(), undeclared, "add these classes to SPEC_DISPOSITIONS")
        if self.expected is not None:
            unknown = present - self.expected
            self.assertEqual(set(), unknown, f"unexpected in vLLM {self.version}")


class TestDifferentialOnTheRealZoo(RealEngineCase):
    """Today's accepted surface stays accepted (the two holes excepted)."""

    def test_legacy_surface_is_contained(self):
        checked = 0
        for row in facts.ZOO:
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                if result.outcome == "unavailable":
                    continue
                legacy = facts.legacy_accepts_row(row, self.engine)
                self.assertEqual(row["legacy"] == "accept", legacy)
                if legacy:
                    checked += 1
                    if result.outcome == "refuse":
                        self.assertIn(row.get("hole"), ("rswa", "tokens_per_state"))
        self.assertGreater(checked, 5, "too few rows accepted before to be a check")


class TestV41CompressedOnTheRealEngine(RealEngineCase):
    """The field-migration hole, read from the real 0.31 spec.

    PR1 closed the hole by refusing what the old gate had accepted silently
    (it read compress_ratio, which this era no longer has); PR6 closes it from
    the other side: the era's own field is read, and the layout is transferred
    as rows. The size the old gate produced by accident (a per-token formula on
    a two-token row) is the same number here -- the hole was never the size.
    """

    def test_compressed_main_is_a_row_granular_bucket(self):
        row = facts.ROW_BY_NAME["mla_compressed_0_31"]
        result = facts.run_row(row, self.engine)
        if result.outcome == "unavailable":
            self.assert_unavailable_is_expected(row, result.message)
            return
        self.assertEqual("accept", result.outcome, result.message)
        own = [meta for meta in result.metas if meta.group_idx == 0][0]
        self.assertEqual(2, own.row_granular.tokens_per_row)
        self.assertEqual(128, own.row_granular.rows_per_page)
        self.assertEqual(584, own.row_granular.bytes_per_row)
        self.assertEqual(74752, own.per_block_bytes)  # 584 B x 128 rows
        # What the pre-PR gate sized from the same spec, and what the real
        # spec's page is: the same 74752 B/block, from a per-token formula.
        spec = facts.build_spec(row, self.engine)
        self.assertTrue(facts.legacy_accepts(spec, self.engine))
        self.assertEqual(74752, spec.real_page_size_bytes)


class TestFoldedGroupCoverageOnTheRealEngine(RealEngineCase):
    """The reachable way to make one group's buckets disagree (PR6 review r1).

    ``UniformTypeKVCacheSpecs.from_specs`` merges an MLA layer with an R-SWA one
    -- their uniform base is the same FullAttentionSpec -- but the two carry
    different coverage (one is a windowed contract, the other full prefix). The
    engine hands the scheduler one member of such a group
    (generate_scheduler_kv_cache_config keeps the first), whose masks would then
    miss the other bucket's dimension: stale or zero-page bytes would be
    published as cacheable. The guard must refuse it, on the real class tree of
    the installed era.
    """

    def _mixed_wrapper(self):
        module = self.module
        members = {
            "mla_l": module.MLAAttentionSpec(
                block_size=64,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.uint8,
            ),
            "rswa_l": module.RSWASpec(
                block_size=64,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.float16,
                rswa_window=128,
            ),
        }
        merged = module.UniformTypeKVCacheSpecs.from_specs(members)
        return module, members, merged

    def test_the_era_really_folds_these_two_specs(self):
        module, _members, merged = self._mixed_wrapper()
        self.assertIsNotNone(
            merged,
            "the era no longer merges MLA with R-SWA: this test would be dead; "
            "pick the spec pair that folds on this version",
        )
        del module

    def test_a_group_with_two_coverages_is_refused(self):
        module, members, merged = self._mixed_wrapper()
        self.assertIsNotNone(merged)
        config = module.KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                module.KVCacheGroupSpec(layer_names=list(members), kv_cache_spec=merged)
            ],
        )
        metas = parse_groups(config, 64, calculate_kv_scales=False)
        self.assertEqual(
            [None, 128],
            [
                meta.retention.window_tokens if meta.retention else None
                for meta in metas
            ],
            "the two coverages must really be in the bucket set",
        )
        # The guard runs where the connector announces spec coverage: the
        # worker's registration (build_spec_groups), before any request. The
        # parse-time bucket split serves *agreeing* packed windowed groups;
        # a disagreement survives parsing and must be refused here.
        with self.assertRaises(NotImplementedError) as ctx:
            build_spec_groups(metas, tp_size=1)
        message = str(ctx.exception)
        self.assertIn("different coverage", message)
        self.assertIn("group 0", message)


if __name__ == "__main__":
    unittest.main()
