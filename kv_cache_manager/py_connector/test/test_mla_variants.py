"""MLA variant semantics on the fast ring (FakeTensor / stub vLLM, no GPU).

Covers the connector's variant-aware parse: bucket splitting of a
``UniformTypeKVCacheSpecs`` group (V3.2 packs a 656 B/token main spec and a
132 B/indexer spec into one block table), the per-bucket page/byte sizing, the
(cache_dtype_str, kv_quant_mode, compress_ratio, calculate_kv_scales) acceptance
gate, and the wire-name invariant. The expectations all come from
``mla_variant_facts.FROZEN`` (measured on vLLM 0.26.0) -- see that module.

Runs without torch/CUDA: the specs come from the ``vllm_stubs`` stand-in.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest import mock

from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)
from kv_cache_manager.py_connector.test.mla_variant_facts import (
    ACCEPTED,
    FROZEN,
    FROZEN_GROUPS,
)
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from kv_cache_manager.py_connector.vllm import vllm_common
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta,
    parse_groups,
    spec_name,
)


def _spec(row: str) -> Any:
    """The stub MLAAttentionSpec of one FROZEN row."""
    return _spec_from_kwargs(dict(FROZEN[row]["kwargs"]))


def _spec_from_kwargs(kwargs: Dict[str, Any]) -> Any:
    mode = kwargs.get("kv_quant_mode", "NONE")
    if isinstance(mode, str):
        kwargs["kv_quant_mode"] = getattr(KVQuantMode, mode)
    kwargs.setdefault("num_kv_heads", 1)
    return MLAAttentionSpec(**kwargs)


def _group(spec: Any, layers: Tuple[str, ...] = ("l0",)) -> Any:
    return SimpleNamespace(
        layer_names=list(layers), kv_cache_spec=spec, is_eagle_group=False
    )


def _wrapper(layers: List[Tuple[str, Any]], block_size: int) -> Any:
    return UniformTypeKVCacheSpecs(
        block_size=block_size, kv_cache_specs={name: spec for name, spec in layers}
    )


def _parse(groups: List[Any], mbs: int, calculate_kv_scales: bool = False) -> List[Any]:
    config: Any = SimpleNamespace(kv_cache_groups=groups)
    return parse_groups(config, mbs, calculate_kv_scales=calculate_kv_scales)


def _wrapper_group(row: str) -> Tuple[Any, Any]:
    """(wrapper spec, kv_cache_group) for one FROZEN_GROUPS worker-view row."""
    facts = FROZEN_GROUPS[row]
    layers = [(name, _spec(spec_row)) for name, spec_row in facts["layers"]]
    wrapper = _wrapper(layers, facts["block_size"])
    return wrapper, _group(wrapper, tuple(name for name, _ in facts["layers"]))


class TestMLASizeTranslation(unittest.TestCase):
    """spec -> AttentionGroupMeta: the per-bucket page and block byte sizes."""

    def test_plain_mla_single_group(self):
        # A1: GLM bf16 baseline, 16 x 576 x 2 B = 18432 B/page, mbs=64.
        metas = _parse([_group(_spec("m0"))], mbs=64)
        self.assertEqual(len(metas), 1)
        meta = metas[0]
        self.assertIsInstance(meta, AttentionGroupMeta)
        self.assertEqual(meta.group_idx, 0)
        self.assertEqual(meta.spec_suffix, "")
        self.assertEqual(meta.block_size, 16)
        self.assertEqual(meta.page_bytes, 18432)
        self.assertEqual(meta.per_block_bytes, 73728)

    def test_layer_count_multiplies_the_block_bytes(self):
        # A2: two layers -> twice the bytes per manager block.
        metas = _parse([_group(_spec("m0"), layers=("l0", "l1"))], mbs=64)
        self.assertEqual(metas[0].per_block_bytes, 147456)

    def test_fp8_ds_mla_packed_spec_is_accepted(self):
        # A3/D1/D17: the V3.2 main layer (656 B/token, row-internal scale,
        # kv_quant_mode=FP8_PER_TENSOR) is sized from its own page algebra.
        metas = _parse([_group(_spec("m1"))], mbs=64)
        self.assertEqual(metas[0].page_bytes, 41984)
        self.assertEqual(metas[0].per_block_bytes, 41984)

    def test_manager_block_bigger_than_the_spec_block(self):
        # A4/B3: M > B scan -- 128 tokens at 656 B each, page unchanged.
        metas = _parse([_group(_spec("m1"))], mbs=128)
        self.assertEqual(metas[0].page_bytes, 41984)
        self.assertEqual(metas[0].per_block_bytes, 656 * 128)

    def test_v32_indexer_spec_is_accepted(self):
        # A9/D3: the real V3.2 indexer has no alignment: page == compact 8448.
        metas = _parse([_group(_spec("m2c2"))], mbs=64)
        self.assertEqual(metas[0].page_bytes, 8448)
        self.assertEqual(metas[0].per_block_bytes, 8448)

    def test_per_tensor_fp8_spec_is_accepted(self):
        # M3a: the layer-level scales stay outside the cached bytes.
        metas = _parse([_group(_spec("m3b"))], mbs=64)
        self.assertEqual(metas[0].page_bytes, 18432)
        self.assertEqual(metas[0].per_block_bytes, 73728)

    def test_size_invariant_over_accepted_variants(self):
        # A13: per_block_bytes == rows-per-block * row-bytes * layers, and the
        # compact page stays an exact multiple of the row size.
        for row in ACCEPTED:
            with self.subTest(variant=row):
                facts = FROZEN[row]
                mbs = facts["mbs"]
                block_size = facts["kwargs"]["block_size"]
                row_bytes = facts["real_page_size_bytes"] // block_size
                metas = _parse([_group(_spec(row), layers=("l0", "l1"))], mbs=mbs)
                self.assertEqual(metas[0].per_block_bytes, row_bytes * mbs * 2)
                self.assertEqual(metas[0].per_block_bytes % (row_bytes * 2), 0)


class TestUniformGroupBuckets(unittest.TestCase):
    """A UniformTypeKVCacheSpecs group maps to one bucket per page layout."""

    def test_v32_pair_splits_into_two_buckets(self):
        # A10: main + indexer share the group (and its block table) but get
        # their own wire names and their own per-block byte sizes.
        wrapper, group = _wrapper_group("v32_wrapper_pair")
        metas = _parse([group], mbs=64)
        self.assertEqual([m.spec_suffix for m in metas], ["", "_b1"])
        self.assertEqual([m.group_idx for m in metas], [0, 0])
        self.assertEqual([m.page_bytes for m in metas], [41984, 8448])
        self.assertEqual([m.per_block_bytes for m in metas], [41984, 8448])
        self.assertEqual([m.block_size for m in metas], [64, 64])
        self.assertEqual([spec_name(0, m) for m in metas], ["tp0_g0", "tp0_g0_b1"])
        self.assertEqual(
            [m.layer_names for m in metas],
            [
                ["model.layers.0.self_attn.attn"],
                ["model.layers.0.self_attn.indexer.k_cache"],
            ],
        )
        # The wrapper itself still reports the vLLM-side sum (50432).
        self.assertEqual(wrapper.page_size_bytes, 50432)

    def test_real_tiny_v32_group_buckets(self):
        # The 2-layer tiny model (golden line): two layers per bucket, the
        # per-block bytes scale with the layer count.
        _, group = _wrapper_group("v32_wrapper_worker")
        metas = _parse([group], mbs=64)
        self.assertEqual([m.page_bytes for m in metas], [41984, 8448])
        self.assertEqual([m.per_block_bytes for m in metas], [83968, 16896])
        self.assertEqual([len(m.layer_names) for m in metas], [2, 2])

    def test_bucket_suffixes_ignore_the_layer_order(self):
        # The suffix is a pure function of the bucket *set*: another rank (or
        # a future vLLM) may hand the layers in another order.
        forward, group = _wrapper_group("v32_wrapper_pair")
        reversed_group = _group(
            _wrapper(list(reversed(forward.kv_cache_specs.items())), 64),
            tuple(reversed(list(forward.kv_cache_specs.keys()))),
        )
        names = [spec_name(0, m) for m in _parse([group], mbs=64)]
        self.assertEqual(
            [spec_name(0, m) for m in _parse([reversed_group], mbs=64)], names
        )

    def test_bf16_group_keeps_mixed_element_sizes_apart(self):
        # V3.2 + auto KV: a bf16 main spec next to a uint8 indexer in the same
        # group. Both buckets size independently (2 B vs 1 B elements).
        _, group = _wrapper_group("v32_bf16_wrapper")
        metas = _parse([group], mbs=64)
        self.assertEqual([m.page_bytes for m in metas], [73728, 8448])
        self.assertEqual([m.per_block_bytes for m in metas], [73728, 8448])


class TestMLAVariantGate(unittest.TestCase):
    """The (cache_dtype_str, kv_quant_mode, compress_ratio) acceptance table."""

    def _refusal(
        self, spec: Any, mbs: int = 64, calculate_kv_scales: bool = False
    ) -> str:
        with self.assertRaises(NotImplementedError) as ctx:
            _parse([_group(spec)], mbs=mbs, calculate_kv_scales=calculate_kv_scales)
        return str(ctx.exception)

    def test_fp8_ds_mla_row_scale_is_not_refused_as_quantized(self):
        # D17 (regression): a real V3.2 main spec carries
        # kv_quant_mode=FP8_PER_TENSOR. Any truthiness-based quant gate would
        # refuse it and make every real V3.2 engine unreachable.
        metas = _parse([_group(_spec("m1"))], mbs=64)
        self.assertEqual(metas[0].per_block_bytes, 41984)

    def test_fp8_ds_mla_bytes_ignore_the_quant_mode(self):
        # D17: the mode drives the gate, never the bytes.
        quantized = _spec_from_kwargs(dict(FROZEN["m1"]["kwargs"]))
        plain = _spec_from_kwargs(
            dict(FROZEN["m1"]["kwargs"], kv_quant_mode=KVQuantMode.NONE)
        )
        self.assertEqual(quantized.real_page_size_bytes, plain.real_page_size_bytes)
        self.assertEqual(
            quantized.unpadded_page_size_bytes, plain.unpadded_page_size_bytes
        )
        self.assertEqual(quantized.page_size_padded, plain.page_size_padded)
        self.assertEqual(quantized.storage_block_size, plain.storage_block_size)

    def test_fp8_ds_mla_with_foreign_quant_mode_is_refused(self):
        # Defence in depth: vLLM derives the mode from the dtype string, so
        # this pair cannot occur -- guessing a scale layout would be worse.
        spec = _spec_from_kwargs(
            dict(FROZEN["m1"]["kwargs"], kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD)
        )
        message = self._refusal(spec)
        self.assertIn("fp8_ds_mla", message)
        self.assertIn("INT4_PER_TOKEN_HEAD", message)

    def test_plain_cache_dtypes_accept_only_mode_none(self):
        # D18 (zero regression): float16/bfloat16 are legal --kv-cache-dtype
        # values today and must stay accepted; only an inconsistent mode is
        # refused.
        for cache_dtype in ("float16", "bfloat16", None, "auto"):
            with self.subTest(cache_dtype=cache_dtype):
                spec = _spec_from_kwargs(
                    dict(
                        block_size=16,
                        head_size=576,
                        dtype="bfloat16",
                        cache_dtype_str=cache_dtype,
                    )
                )
                metas = _parse([_group(spec)], mbs=64)
                self.assertEqual(metas[0].per_block_bytes, 73728)
        inconsistent = _spec_from_kwargs(
            dict(
                block_size=16,
                head_size=576,
                dtype="float16",
                cache_dtype_str="float16",
                kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
            )
        )
        message = self._refusal(inconsistent)
        self.assertIn("kv_quant_mode", message)
        self.assertIn("FP8_PER_TENSOR", message)

    def test_per_token_head_and_nvfp4_modes_are_refused(self):
        # D4-D6 (04 keeps the refusal; 06 pins the wording): the mode name is
        # printed, not the raw integer.
        for mode in (
            "INT8_PER_TOKEN_HEAD",
            "FP8_PER_TOKEN_HEAD",
            "INT4_PER_TOKEN_HEAD",
            "NVFP4",
        ):
            with self.subTest(mode=mode):
                spec = _spec_from_kwargs(
                    dict(
                        block_size=16,
                        head_size=576,
                        dtype="bfloat16",
                        cache_dtype_str="auto",
                        kv_quant_mode=mode,
                    )
                )
                message = self._refusal(spec)
                self.assertIn(mode, message)
                self.assertIn("kv_quant_mode", message)

    def test_unknown_cache_dtype_is_refused(self):
        spec = _spec_from_kwargs(
            dict(
                block_size=16,
                head_size=576,
                dtype="bfloat16",
                cache_dtype_str="fp8_e5m2",
            )
        )
        message = self._refusal(spec)
        self.assertIn("fp8_e5m2", message)
        self.assertIn("fp8_ds_mla", message)  # the list of known layouts

    def test_runtime_calibrated_scales_are_refused(self):
        # The layer-level fp8 scales are calibrated from the request when
        # calculate_kv_scales is on: the cached bytes then stop being
        # reproducible, so the connector refuses instead of storing them.
        spec = _spec("m3b")
        message = self._refusal(spec, calculate_kv_scales=True)
        self.assertIn("calculate_kv_scales", message)
        # ...while the same spec stays accepted without calibration.
        metas = _parse([_group(spec)], mbs=64)
        self.assertEqual(metas[0].per_block_bytes, 73728)

    def test_compressed_mla_is_refused(self):
        # M2 (the wording is completed in the next wave): one row per several
        # tokens has no per-token slot to gather.
        message = self._refusal(_spec("m2b"), mbs=256)
        self.assertIn("compress_ratio", message)

    def test_refusals_are_independent_of_calculate_kv_scales(self):
        # The calibration gate only guards the per-tensor path: a compressed
        # spec must fail for its own reason either way.
        for flag in (False, True):
            with self.subTest(calculate_kv_scales=flag):
                message = self._refusal(_spec("m2b"), mbs=256, calculate_kv_scales=flag)
                self.assertIn("compress_ratio", message)


class TestMLAGateOriginContext(unittest.TestCase):
    """A refusal inside a packed group must name the offending layer."""

    def test_wrapper_refusal_points_at_the_layer(self):
        layers = [
            ("model.layers.0.self_attn.attn", _spec("m1")),
            ("model.layers.0.self_attn.indexer.k_cache", _spec("m3")),
        ]
        wrapper = _wrapper(layers, 64)
        with self.assertRaises(NotImplementedError) as ctx:
            _parse([_group(wrapper)], mbs=64)
        message = str(ctx.exception)
        self.assertIn("group 0", message)
        self.assertIn("UniformTypeKVCacheSpecs", message)
        self.assertIn("model.layers.0.self_attn.indexer.k_cache", message)
        self.assertIn("kv_quant_mode", message)

    def test_unknown_spec_class_is_refused_with_its_name(self):
        class HiddenStateCacheSpec:
            block_size = 64

        with self.assertRaises(NotImplementedError) as ctx:
            _parse([_group(HiddenStateCacheSpec())], mbs=64)
        self.assertIn("HiddenStateCacheSpec", str(ctx.exception))

    def test_windowed_full_attention_is_still_refused(self):
        spec = _spec("m0")
        spec.sliding_window = 1024
        with self.assertRaises(NotImplementedError) as ctx:
            _parse([_group(spec)], mbs=64)
        self.assertIn("sliding_window", str(ctx.exception))


class TestWireNameInvariant(unittest.TestCase):
    """I1: the manager keys locations by name, so names must be unique."""

    @staticmethod
    def _meta(group_idx: int, suffix: str = "") -> AttentionGroupMeta:
        return AttentionGroupMeta(
            group_idx=group_idx,
            layer_names=[f"l{group_idx}{suffix}"],
            block_size=16,
            per_block_bytes=1024,
            page_bytes=2048,
            spec_suffix=suffix,
        )

    def test_duplicate_names_are_refused(self):
        # A forgotten suffix (two buckets of group 0 claiming "g0") is the
        # failure mode this guard exists for: it must fail at parse time with
        # the colliding names, not silently overwrite a manager location.
        with self.assertRaises(AssertionError) as ctx:
            vllm_common._check_unique_names([self._meta(0), self._meta(0)])
        message = str(ctx.exception)
        self.assertIn("collision", message)
        self.assertEqual(message.count("'g0'"), 2)

    def test_distinct_suffixes_are_accepted(self):
        vllm_common._check_unique_names([self._meta(0), self._meta(0, "_b1")])

    def test_rank_prefix_does_not_collide_across_two_digit_ranks(self):
        # tp1 vs tp10 must stay distinguishable: the guard compares names
        # without the rank prefix, and spec_name only ever prefixes.
        metas = [self._meta(0), self._meta(0, "_b1")]
        names = [spec_name(rank, meta) for rank in (1, 10) for meta in metas]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, ["tp1_g0", "tp1_g0_b1", "tp10_g0", "tp10_g0_b1"])


class TestEraCompatibility(unittest.TestCase):
    """vllm 0.14 has no UniformTypeKVCacheSpecs / SlidingWindowMLASpec /
    KVQuantMode: the sentinels must not break the plain path."""

    def test_missing_wrapper_types_do_not_break_plain_mla(self):
        with mock.patch.object(vllm_common, "UniformTypeKVCacheSpecs", None):
            with mock.patch.object(vllm_common, "SlidingWindowMLASpec", None):
                with mock.patch.object(vllm_common, "KVQuantMode", None):
                    metas = _parse([_group(_spec("m0"))], mbs=64)
        self.assertEqual(metas[0].per_block_bytes, 73728)


class TestFrozenFactsAgainstStub(unittest.TestCase):
    """F2: the vllm stubs must reproduce the measured vLLM 0.26.0 algebra."""

    def test_specs_reproduce_frozen(self):
        for row, facts in FROZEN.items():
            with self.subTest(variant=row):
                spec = _spec(row)
                expected_page = (
                    facts["page_size_padded"]
                    if facts["page_size_padded"] is not None
                    else facts["unpadded_page_size_bytes"]
                )
                self.assertEqual(spec.storage_block_size, facts["storage_block_size"])
                self.assertEqual(
                    spec.real_page_size_bytes, facts["real_page_size_bytes"]
                )
                self.assertEqual(
                    spec.unpadded_page_size_bytes, facts["unpadded_page_size_bytes"]
                )
                self.assertEqual(spec.page_size_padded, facts["page_size_padded"])
                self.assertEqual(spec.page_size_bytes, expected_page)
                self.assertEqual(str(spec.dtype), facts["kwargs"]["dtype"])

    def test_groups_reproduce_frozen(self):
        for row, facts in FROZEN_GROUPS.items():
            if "folded_row" in facts:
                continue
            with self.subTest(group=row):
                wrapper, _ = _wrapper_group(row)
                self.assertEqual(wrapper.block_size, facts["block_size"])
                self.assertEqual(wrapper.page_size_bytes, facts["page_size_bytes"])
                self.assertEqual(sorted(wrapper.get_page_sizes()), facts["page_sizes"])
                metas = _parse([_wrapper_group(row)[1]], mbs=facts["mbs"])
                self.assertEqual(
                    [m.spec_suffix for m in metas],
                    [b["suffix"] for b in facts["buckets"]],
                )
                self.assertEqual(
                    [m.per_block_bytes for m in metas],
                    [b["per_block_bytes"] for b in facts["buckets"]],
                )


class TestCompressedRowsAreStructuralFacts(unittest.TestCase):
    """The compressed rows are frozen too: they document what vLLM 0.26.0
    computes for V4, and the next wave pins the refusals on top of them."""

    def test_frozen_rows_have_positive_pages(self):
        for row, facts in FROZEN.items():
            with self.subTest(variant=row):
                block_size = facts["kwargs"]["block_size"]
                self.assertGreater(facts["real_page_size_bytes"], 0)
                # One row per storage block: the compact page divides exactly
                # by the storage block size (block_size for c == 1).
                self.assertEqual(
                    facts["real_page_size_bytes"] % facts["storage_block_size"], 0
                )
                self.assertEqual(
                    facts["storage_block_size"],
                    block_size // facts["compress_ratio"],
                )


if __name__ == "__main__":
    unittest.main()
