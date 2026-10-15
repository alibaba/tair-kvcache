"""MLA variant semantics on the fast ring (FakeTensor / stub vLLM, no GPU).

Covers the connector's variant-aware parse: bucket splitting of a
``UniformTypeKVCacheSpecs`` group (V3.2 packs a 656 B/token main spec and a
132 B/indexer spec into one block table), the per-bucket page/byte sizing, the
(cache_dtype_str, kv_quant_mode, compress_ratio, calculate_kv_scales) acceptance
gate, the refusals of the DeepSeek V4 family (compressed rows, sliding-window
and compressor-state caches, malformed ratios / model_version) and the
wire-name invariant. The expectations all come from
``mla_variant_facts.FROZEN`` / ``FROZEN_GROUPS`` (measured on vLLM 0.26.0 and
reconciled with the tiny-model golden) -- see that module.

Runs without torch/CUDA: the specs come from the ``vllm_stubs`` stand-in.
"""

import importlib
import sys
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest import mock

from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)
from kv_cache_manager.py_connector.test.mla_variant_facts import (
    ACCEPTED,
    FROZEN,
    FROZEN_GROUPS,
    REJECTED_COMPRESSED,
    REJECTED_INVALID,
    REJECTED_QUANTIZED,
    REJECTED_WINDOW,
    V4_TINY_GROUPS,
)
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from kv_cache_manager.py_connector.vllm import vllm_common
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta,
    parse_groups,
    spec_name,
)


def _spec(row: str) -> Any:
    """The stub spec of one FROZEN row (``kind`` selects the spec class)."""
    facts = FROZEN[row]
    return _spec_from_kwargs(dict(facts["kwargs"]), facts.get("kind", "mla"))


def _spec_from_kwargs(kwargs: Dict[str, Any], kind: str = "mla") -> Any:
    mode = kwargs.get("kv_quant_mode", "NONE")
    if isinstance(mode, str):
        kwargs["kv_quant_mode"] = getattr(KVQuantMode, mode)
    kwargs.setdefault("num_kv_heads", 1)
    if kind == "swa":
        return SlidingWindowMLASpec(**kwargs)
    if kind == "full":
        return FullAttentionSpec(**kwargs)
    return MLAAttentionSpec(**kwargs)


def _spec_without_quant_mode(kwargs: Dict[str, Any]) -> Any:
    """A vLLM 0.14-shaped spec: every 0.26 field except kv_quant_mode."""
    base = _spec_from_kwargs(dict(kwargs))
    cls = type("LegacyMLASpec", (type(base),), {})
    spec = object.__new__(cls)
    spec.__dict__.update(
        {k: v for k, v in base.__dict__.items() if k != "kv_quant_mode"}
    )
    return spec


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
        # A18: the non-packed fp8 cache dtypes keep layer-level scales outside
        # the cached bytes, and the engine stores that cache as uint8: 9216
        # B/page, 576 B/token at block 16 (01 section 2.3; the 18432/bf16 row
        # 02 section 3.2 probed was a template dtype, never an engine spec).
        # fp8_e5m2 maps to FP8_PER_TENSOR and uint8 exactly like the e4m3
        # variant (anchored in test_mla_spec_facts).
        for cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            with self.subTest(cache_dtype=cache_dtype):
                spec = _spec_from_kwargs(
                    dict(FROZEN["m3b"]["kwargs"], cache_dtype_str=cache_dtype)
                )
                self.assertEqual(str(spec.dtype), "uint8")
                self.assertIsNone(spec.page_size_padded)
                self.assertEqual(spec.unpadded_page_size_bytes, 9216)
                metas = _parse([_group(spec)], mbs=64)
                self.assertEqual(metas[0].page_bytes, 9216)
                self.assertEqual(metas[0].per_block_bytes, 576 * 64)

    def test_legacy_era_packed_fp8_without_quant_mode_is_accepted(self):
        # vLLM 0.14's MLAAttentionSpec has cache_dtype_str but no
        # kv_quant_mode field: the dtype string alone defines the layout
        # there, so the (cds, mode) consistency rule must not refuse the
        # whole compatibility era. The 656 B packed layout (10496 B/page at
        # block 16) and the plain fp8 layout (9216 B/page) both transfer.
        for cache_dtype, page in (("fp8_ds_mla", 10496), ("fp8", 9216)):
            with self.subTest(cache_dtype=cache_dtype):
                spec = _spec_without_quant_mode(
                    dict(
                        block_size=16,
                        head_size=576,
                        dtype="uint8",
                        cache_dtype_str=cache_dtype,
                        kv_quant_mode="FP8_PER_TENSOR",
                    )
                )
                self.assertFalse(hasattr(spec, "kv_quant_mode"))
                metas = _parse([_group(spec)], mbs=64)
                self.assertEqual(metas[0].page_bytes, page)
                self.assertEqual(metas[0].per_block_bytes, page // 16 * 64)

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

    def test_same_page_different_geometry_split_into_buckets(self):
        # A UniformTypeKVCacheSpecs group only guarantees spec type and block
        # size, not head geometry: 2 heads x 64 and 1 head x 128 share a page
        # (8192 B here) but not a tensor shape, so they must not share a
        # transfer bucket.
        wide = _spec_from_kwargs(
            dict(
                block_size=16,
                num_kv_heads=2,
                head_size=64,
                dtype="bfloat16",
            ),
            kind="full",
        )
        tall = _spec_from_kwargs(
            dict(
                block_size=16,
                num_kv_heads=1,
                head_size=128,
                dtype="bfloat16",
            ),
            kind="full",
        )
        self.assertEqual(wide.real_page_size_bytes, tall.real_page_size_bytes)
        wrapper = _wrapper([("l0", wide), ("l1", tall)], block_size=16)
        metas = _parse([_group(wrapper, ("l0", "l1"))], mbs=64)
        self.assertEqual([m.spec_suffix for m in metas], ["", "_b1"])
        self.assertEqual([m.group_idx for m in metas], [0, 0])
        self.assertEqual([m.page_bytes for m in metas], [8192, 8192])
        self.assertEqual([m.per_block_bytes for m in metas], [8192 // 16 * 64] * 2)
        self.assertEqual([len(m.layer_names) for m in metas], [1, 1])
        # Geometry breaks the page tie: fewer heads first, deterministically.
        self.assertEqual([m.layer_names for m in metas], [["l1"], ["l0"]])
        self.assertEqual([spec_name(0, m) for m in metas], ["tp0_g0", "tp0_g0_b1"])

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

    def test_plain_fp8_requires_the_per_tensor_mode(self):
        # #6b, second half (defence in depth, symmetric with D18's counter-
        # example): vLLM maps the plain fp8 cache dtypes to FP8_PER_TENSOR,
        # so a plain fp8 cache without it cannot occur -- refuse instead of
        # guessing.
        for cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            with self.subTest(cache_dtype=cache_dtype):
                spec = _spec_from_kwargs(
                    dict(
                        FROZEN["m3b"]["kwargs"],
                        cache_dtype_str=cache_dtype,
                        kv_quant_mode=KVQuantMode.NONE,
                    )
                )
                self.assertEqual(spec.kv_quant_mode, KVQuantMode.NONE)
                self.assertEqual(vllm_common._quant_mode(spec), 0)
                message = self._refusal(spec)
                self.assertIn(f"cache_dtype_str='{cache_dtype}'", message)
                self.assertIn("kv_quant_mode=NONE", message)
                self.assertIn("is an inconsistent spec", message)

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

    def test_int4_per_token_head_is_refused_with_the_mode_name(self):
        # D4: the frozen INT4 row (kv_quant_mode value 4, truthy) must be
        # refused by the mode *name*, never by a bare "if kv_quant_mode".
        spec = _spec_from_kwargs(dict(FROZEN["m3"]["kwargs"]))
        self.assertEqual(
            vllm_common._quant_mode(spec), int(KVQuantMode.INT4_PER_TOKEN_HEAD)
        )
        message = self._refusal(spec)
        self.assertIn("kv_quant_mode=INT4_PER_TOKEN_HEAD", message)
        self.assertNotIn("kv_quant_mode=4", message)
        self.assertIn("per-token-head", message)
        self.assertIn("outside the transferable rows", message)
        self.assertIn("TairKvCacheConnector", message)
        self.assertNotIn("one latent row per", message)

    def test_per_token_head_modes_are_refused_with_the_mode_name(self):
        # D5: INT8/FP8 per-token-head scales are quant modes, not dtypes.
        for mode in ("INT8_PER_TOKEN_HEAD", "FP8_PER_TOKEN_HEAD"):
            with self.subTest(mode=mode):
                spec = self._spec_with_quant_mode(mode)
                self.assertEqual(
                    vllm_common._quant_mode(spec), int(getattr(KVQuantMode, mode))
                )
                message = self._refusal(spec)
                self.assertIn(f"kv_quant_mode={mode}", message)
                self.assertNotIn(
                    f"kv_quant_mode={int(getattr(KVQuantMode, mode))}", message
                )
                self.assertIn("per-token-head scales", message)

    def test_nvfp4_mode_is_refused_with_the_mode_name(self):
        # D6: NVFP4 is not per-token-head but shares the refusal (its scales
        # sit in the page budget, outside the transferable rows).
        spec = self._spec_with_quant_mode("NVFP4")
        message = self._refusal(spec)
        self.assertIn("kv_quant_mode=NVFP4", message)
        self.assertIn("NVFP4 quantized MLA KV is not supported", message)

    def _spec_with_quant_mode(self, mode: str) -> Any:
        return _spec_from_kwargs(
            dict(
                block_size=16,
                head_size=576,
                dtype="bfloat16",
                cache_dtype_str="auto",
                kv_quant_mode=mode,
            )
        )

    def test_unknown_cache_dtype_is_refused(self):
        spec = _spec_from_kwargs(
            dict(
                block_size=16,
                head_size=576,
                dtype="bfloat16",
                cache_dtype_str="turboquant_k8v4",
            )
        )
        message = self._refusal(spec)
        self.assertIn("turboquant_k8v4", message)
        self.assertIn("fp8_ds_mla", message)  # the list of known layouts
        self.assertIn("fp8_e5m2", message)  # a per-tensor fp8 value, not unknown

    def test_runtime_calibrated_scales_are_refused(self):
        # D7b: with calculate_kv_scales on, vLLM calibrates the layer-level fp8
        # scales from the request, so the cached bytes stop being
        # reproducible; the refusal says why and what to do instead.
        spec = _spec("m3b")
        message = self._refusal(spec, calculate_kv_scales=True)
        self.assertIn("cache_dtype_str=fp8", message)
        self.assertIn("calculate_kv_scales=True", message)
        self.assertIn("wrong scale", message)
        self.assertIn("--calculate-kv-scales off", message)
        # ...while the same spec stays accepted without calibration.
        metas = _parse([_group(spec)], mbs=64)
        self.assertEqual(metas[0].per_block_bytes, 576 * 64)

    def test_compressed_mla_is_refused_with_the_v4_coupling_reason(self):
        # D2a: the ratio is named, and the reason is V4's multi-group coupling
        # -- the compressed rows are read with the SWA / compressor-state
        # caches, so a request's KV groups go as a whole.
        for row in REJECTED_COMPRESSED:
            with self.subTest(variant=row):
                facts = FROZEN[row]
                ratio = facts["compress_ratio"]
                message = self._refusal(_spec(row), mbs=facts["mbs"])
                self.assertIn("MLAAttentionSpec", message)
                self.assertIn(f"compress_ratio={ratio}", message)
                self.assertIn(f"one latent row per {ratio} tokens", message)
                self.assertIn("sliding-window and", message)
                self.assertIn("compressor-state", message)
                self.assertIn("separate vLLM groups", message)
                self.assertIn("as a whole", message)
                self.assertIn("rows-per-block", message)

    def test_negative_compress_ratio_is_refused_as_invalid(self):
        # A15/D9: c < 1 is not a compression at all -- vLLM would floor it into
        # garbage (c=0 even divides by zero) -- so it is refused as an invalid
        # ratio instead of being described as a row-per-N-tokens layout.
        for compress_ratio in (0, -4):
            with self.subTest(compress_ratio=compress_ratio):
                spec = _spec_from_kwargs(
                    dict(
                        block_size=16,
                        head_size=512,
                        dtype="bfloat16",
                        cache_dtype_str="auto",
                        compress_ratio=compress_ratio,
                    )
                )
                message = self._refusal(spec, mbs=16)
                self.assertIn("compress_ratio", message)
                self.assertIn("must be >= 1", message)
                self.assertIn(f"compress_ratio={compress_ratio} is invalid", message)

    def test_non_divisible_compress_ratio_is_refused(self):
        # A16/D10: X2 floors 16 // 3 to 5 rows per block in vLLM's algebra, so
        # the row bytes would silently describe a layout nobody allocated.
        message = self._refusal(_spec("X2"), mbs=16)
        self.assertIn("block_size=16", message)
        self.assertIn("compress_ratio=3", message)
        self.assertIn("not divisible", message)
        # The structural branch, not the compression description.
        self.assertNotIn("one latent row per", message)

    def test_compress_ratio_above_the_block_size_is_refused(self):
        # A17/D11: 64 // 128 == 0 storage rows -- refused by the same
        # divisibility check, before any storage_block_size read.
        spec = _spec_from_kwargs(
            dict(
                block_size=64,
                head_size=512,
                dtype="bfloat16",
                cache_dtype_str="auto",
                compress_ratio=128,
            )
        )
        message = self._refusal(spec, mbs=64)
        self.assertIn("block_size=64", message)
        self.assertIn("compress_ratio=128", message)
        self.assertIn("not divisible", message)
        self.assertNotIn("one latent row per", message)

    def test_unknown_model_version_is_refused(self):
        # D8: vLLM itself does not validate model_version (its own
        # real_page_size_bytes only branches on "deepseek_v4"), so an unknown
        # value has no size algebra -- refuse by name.
        spec = _spec_from_kwargs(
            dict(
                block_size=16,
                head_size=512,
                dtype="bfloat16",
                cache_dtype_str="auto",
                model_version="bogus",
            )
        )
        message = self._refusal(spec, mbs=16)
        self.assertIn("model_version='bogus'", message)
        self.assertIn("not a layout this connector knows", message)
        self.assertIn('"deepseek_v4"', message)
        self.assertNotIn("one latent row per", message)

    def test_structure_is_checked_before_the_compression_refusal(self):
        # Ordering (03 section 4.1): 1.1/1.2/1.3 run before 1.4, so a c>1 spec
        # with an unknown model_version reports the model_version, not the
        # compression.
        spec = _spec_from_kwargs(dict(FROZEN["m2b"]["kwargs"], model_version="bogus"))
        message = self._refusal(spec, mbs=256)
        self.assertIn("model_version='bogus'", message)
        self.assertNotIn("one latent row per", message)

    def test_refusals_are_independent_of_calculate_kv_scales(self):
        # The calibration gate only guards the per-tensor path: a compressed
        # spec must fail for its own reason either way.
        for flag in (False, True):
            with self.subTest(calculate_kv_scales=flag):
                message = self._refusal(_spec("m2b"), mbs=256, calculate_kv_scales=flag)
                self.assertIn("compress_ratio", message)


class TestWindowedMLARefusals(unittest.TestCase):
    """M2/D15/D16: SlidingWindowMLASpec (V4 attention SWA + compressor state).

    The class is not a FullAttentionSpec (its MRO is
    SlidingWindowMLASpec -> SlidingWindowSpec -> AttentionSpec), so it must be
    recognised before that branch, and its refusal carries the window reason
    plus the V4 coupling -- both the bare and the wrapped (real V4) shapes.
    """

    def _refusal(self, spec: Any, *, layers=("l0",), mbs: int = 64) -> str:
        with self.assertRaises(NotImplementedError) as ctx:
            _parse([_group(spec, tuple(layers))], mbs=mbs)
        return str(ctx.exception)

    def test_swa_spec_names_window_prefix_and_coupling(self):
        # D15: a V4 SWA spec (compress_ratio=4 in the fixture) is refused by
        # class + window + coupling, never by the compressed-row wording.
        spec = _spec_from_kwargs(
            dict(
                block_size=256,
                sliding_window=2048,
                head_size=512,
                dtype="uint8",
                cache_dtype_str="fp8_ds_mla",
                compress_ratio=4,
                model_version="deepseek_v4",
                kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
            ),
            kind="swa",
        )
        message = self._refusal(spec)
        self.assertIn("SlidingWindowMLASpec", message)
        self.assertIn("sliding_window=2048", message)
        self.assertIn("not the full prefix", message)
        self.assertIn("compressor-state", message)
        self.assertIn("TairKvCacheConnector", message)
        self.assertNotIn("one latent row per", message)

    def test_swa_group_refusal_names_the_layer(self):
        # D2b/D16: V4 wraps each SWA cache in its own UniformTypeKVCacheSpecs
        # group; the refusal points at the wrapper and the layer.
        _, group = _wrapper_group("v4_swa_wrapper_l1")
        facts = FROZEN_GROUPS["v4_swa_wrapper_l1"]
        with self.assertRaises(NotImplementedError) as ctx:
            _parse([group], mbs=facts["mbs"])
        message = str(ctx.exception)
        self.assertIn(
            "UniformTypeKVCacheSpecs, layer model.layers.1.attn.swa_cache", message
        )
        self.assertIn("sliding_window=128", message)
        self.assertIn("not the full prefix", message)

    def test_compressor_state_groups_are_refused_as_windowed_state(self):
        # D2c/D16: the compressor-state caches are SlidingWindowMLASpec too
        # (fp32, one row per window step): windowed *partial state*, and the
        # state of one group feeds the compressed rows of another. The layer
        # named is the first *registered* one -- for the l1 group that is the
        # indexer's state, not the attention's (engine registration order).
        for row, layer in (
            (
                "v4_state_wrapper_l1",
                "model.layers.1.attn.indexer.compressor.state_cache",
            ),
            ("v4_state_wrapper_l2", "model.layers.2.attn.compressor.state_cache"),
        ):
            with self.subTest(group=row):
                _, group = _wrapper_group(row)
                self.assertEqual(layer, FROZEN_GROUPS[row]["layers"][0][0])
                with self.assertRaises(NotImplementedError) as ctx:
                    _parse([group], mbs=FROZEN_GROUPS[row]["mbs"])
                message = str(ctx.exception)
                self.assertIn(layer, message)
                self.assertIn("partial state", message)
                self.assertIn("compressor-state MLA KV is not supported", message)


class TestV4GroupFacts(unittest.TestCase):
    """F3: the six V4 tiny groups as frozen from the golden, refused as a whole.

    vLLM builds the V4 tiny model into six packed groups (one full-MLA group,
    one SWA group per layer, two compressor-state groups) and folds each of
    them to its *first* sub spec for the scheduler. Both role views must be
    refused -- the connector serves a request's KV groups as a whole, so a
    partial transfer (some groups stored, some skipped) is never an option.
    """

    def test_worker_view_reproduces_the_frozen_group(self):
        for row in V4_TINY_GROUPS:
            with self.subTest(group=row):
                facts = FROZEN_GROUPS[row]
                wrapper, _ = _wrapper_group(row)
                self.assertEqual(wrapper.block_size, facts["block_size"])
                self.assertEqual(sorted(wrapper.get_page_sizes()), facts["page_sizes"])
                self.assertEqual(wrapper.page_size_bytes, facts["page_size_bytes"])

    def test_every_group_has_a_refused_sub_spec(self):
        # The option-A predicate: one refused sub spec per group refuses the
        # instance. For the real V4 model *every* sub spec is refused, which is
        # why the A and B cases have the same acceptance surface.
        for row in V4_TINY_GROUPS:
            with self.subTest(group=row):
                facts = FROZEN_GROUPS[row]
                refused = [
                    name
                    for name, spec_row in facts["layers"]
                    if self._refused(spec_row, name)
                ]
                self.assertTrue(refused, f"{row}: no sub spec refused")
                self.assertEqual(refused, [name for name, _ in facts["layers"]])

    @staticmethod
    def _refused(spec_row: str, layer_name: str) -> bool:
        origin = f"group 0 (UniformTypeKVCacheSpecs, layer {layer_name})"
        try:
            vllm_common._check_attention_spec_supported(
                origin, _spec(spec_row), calculate_kv_scales=False
            )
        except NotImplementedError:
            return True
        return False

    def test_worker_view_refusal_names_the_first_offending_layer(self):
        # A11/D2a-c: the wrapper is gated layer by layer, so the message points
        # at the first refused sub spec -- never at the wrapper class only.
        for row in V4_TINY_GROUPS:
            with self.subTest(group=row):
                facts = FROZEN_GROUPS[row]
                _, group = _wrapper_group(row)
                with self.assertRaises(NotImplementedError) as ctx:
                    _parse([group], mbs=facts["mbs"])
                first_layer = facts["layers"][0][0]
                self.assertIn(
                    f"group 0 (UniformTypeKVCacheSpecs, layer {first_layer})",
                    str(ctx.exception),
                )

    def test_scheduler_folded_view_is_refused_too(self):
        # P1/F3: the scheduler's folded spec (the first sub spec, all layer
        # names) refuses too, but its message has no layer context -- that is
        # the worker view's; the PR body states the wording difference.
        for row in V4_TINY_GROUPS:
            with self.subTest(group=row):
                facts = FROZEN_GROUPS[row]
                wrapper, _ = _wrapper_group(row)
                # Generate the fold the way the engine does: the first entry of
                # the wrapper dict, which is registration order (S1). The frozen
                # sched_row must name exactly that sub spec.
                self.assertEqual(facts["sched_row"], facts["layers"][0][1])
                folded_spec = next(iter(wrapper.kv_cache_specs.values()))
                self.assertEqual(
                    folded_spec.real_page_size_bytes,
                    FROZEN[facts["sched_row"]]["real_page_size_bytes"],
                )
                layer_names = [name for name, _ in facts["layers"]]
                folded = _group(folded_spec, tuple(layer_names))
                with self.assertRaises(NotImplementedError) as ctx:
                    _parse([folded], mbs=facts["mbs"])
                message = str(ctx.exception)
                if FROZEN[facts["sched_row"]].get("kind") == "swa":
                    self.assertIn("SlidingWindowMLASpec", message)
                else:
                    self.assertIn("compress_ratio=", message)
                self.assertNotIn(layer_names[0], message)


class TestRefusedRowsStayRefused(unittest.TestCase):
    """Table-driven: every row the facts mark as refused must fail the gate
    with its own reason -- and never with a ZeroDivisionError/AssertionError
    (the structural checks read raw fields before any derived page size)."""

    CASES = (
        (REJECTED_COMPRESSED, "one latent row per"),
        (REJECTED_QUANTIZED, "per-token-head"),
        (REJECTED_WINDOW, "SlidingWindowMLASpec"),
        (REJECTED_INVALID, "not divisible"),
    )

    def test_refused_rows_fail_with_their_reason(self):
        for rows, keyword in self.CASES:
            for row in rows:
                with self.subTest(variant=row):
                    with self.assertRaises(NotImplementedError) as ctx:
                        _parse([_group(_spec(row))], mbs=FROZEN[row]["mbs"])
                    self.assertIn(keyword, str(ctx.exception))


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

    def test_partial_symbol_failure_keeps_the_wrapper_path(self):
        # vLLM 0.14 ships UniformTypeKVCacheSpecs but not KVQuantMode /
        # SlidingWindowMLASpec: each optional import must fail on its own,
        # or one missing symbol nulls the wrapper type too and every packed
        # group dies as an unsupported spec.
        interface = sys.modules["vllm.v1.kv_cache_interface"]
        saved_symbols = (
            getattr(interface, "KVQuantMode"),
            getattr(interface, "SlidingWindowMLASpec"),
        )
        saved_globals = dict(vllm_common.__dict__)
        delattr(interface, "KVQuantMode")
        delattr(interface, "SlidingWindowMLASpec")
        try:
            importlib.reload(vllm_common)
            self.assertIsNone(vllm_common.KVQuantMode)
            self.assertIsNone(vllm_common.SlidingWindowMLASpec)
            self.assertIsNotNone(vllm_common.UniformTypeKVCacheSpecs)
            # The wrapper path must still split a packed group.
            _, group = _wrapper_group("v32_wrapper_pair")
            metas = _parse([group], mbs=64)
            self.assertEqual([m.page_bytes for m in metas], [41984, 8448])
        finally:
            setattr(interface, "KVQuantMode", saved_symbols[0])
            setattr(interface, "SlidingWindowMLASpec", saved_symbols[1])
            # Restore the exact pre-reload objects: a plain reload would
            # rebind every class and break later isinstance checks.
            vllm_common.__dict__.clear()
            vllm_common.__dict__.update(saved_globals)


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
            if "buckets" not in facts:
                # The folded views and the V4 groups are covered elsewhere
                # (test_role_views / TestV4GroupFacts): they have no buckets.
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


class TestFoldRowsAreRegistrationOrdered(unittest.TestCase):
    """S1: the scheduler fold keeps the wrapper dict's first entry, which is
    engine registration order (DeepSeek registers the indexer before the
    attention module) -- never an alphabetically or numerically first one.
    Every frozen fold row must name exactly that sub spec."""

    def test_sched_and_folded_rows_name_layers_zero(self):
        for row, facts in FROZEN_GROUPS.items():
            layers = facts.get("layers")
            if not isinstance(layers, list):
                continue  # count-only rows (v32_wrapper_sched): checked below
            with self.subTest(group=row):
                for key in ("sched_row", "folded_row"):
                    if key in facts:
                        self.assertEqual(facts[key], layers[0][1], key)

    def test_v32_sched_row_names_the_worker_groups_first_layer(self):
        worker = FROZEN_GROUPS["v32_wrapper_worker"]
        self.assertEqual(
            FROZEN_GROUPS["v32_wrapper_sched"]["folded_row"], worker["layers"][0][1]
        )
        self.assertEqual(worker["sched_row"], "m2c2")


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
