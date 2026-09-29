"""F1: the real vLLM must reproduce ``mla_variant_facts.FROZEN``.

A deliberate mirror of ``test_mla_variants.TestFrozenFactsAgainstStub``: the
stub and the engine cannot both be wrong in the same direction, so a vLLM
formula change (or a spec field rename) turns one of them red instead of
silently skewing every size/layout assertion.

This module must NOT import ``vllm_stubs`` (its BUILD target has no such
dependency): it needs the real ``vllm.v1.kv_cache_interface``. When vLLM is
absent (open-source CI) or its module space was already replaced by the stubs
(a same-process full-suite run), the test skips instead of comparing the stub
against the table it was built from.
"""

import unittest
from typing import Any, Dict

from kv_cache_manager.py_connector.test.mla_variant_facts import (
    FROZEN,
    FROZEN_GROUPS,
)

try:
    import torch
    import vllm
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVQuantMode,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )

    _IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - environment dependent
    torch = None  # ty: ignore[invalid-assignment]
    vllm = None  # ty: ignore[invalid-assignment]
    generate_scheduler_kv_cache_config = None  # ty: ignore[invalid-assignment]
    FullAttentionSpec = None  # ty: ignore[invalid-assignment]
    KVCacheConfig = None  # ty: ignore[invalid-assignment]
    KVCacheGroupSpec = None  # ty: ignore[invalid-assignment]
    KVQuantMode = None  # ty: ignore[invalid-assignment]
    MLAAttentionSpec = None  # ty: ignore[invalid-assignment]
    SlidingWindowMLASpec = None  # ty: ignore[invalid-assignment]
    UniformTypeKVCacheSpecs = None  # ty: ignore[invalid-assignment]
    _IMPORT_ERROR = str(exc)


def _skip_reason() -> str:
    if _IMPORT_ERROR:
        return f"real vLLM not importable ({_IMPORT_ERROR})"
    if getattr(vllm, "_kvcm_test_stub", False) or "vllm_stubs" in str(
        getattr(vllm, "__file__", "")
    ):
        return "vLLM is shadowed by the test stubs in this process"
    return ""


def _spec(row: str) -> Any:
    """The real spec of one FROZEN row (``kind`` selects the spec class)."""
    kwargs = _spec_kwargs(row)
    if FROZEN[row].get("kind") == "swa":
        return SlidingWindowMLASpec(**kwargs)  # ty: ignore[call-non-callable]
    return MLAAttentionSpec(**kwargs)  # ty: ignore[call-non-callable]


def _spec_kwargs(row: str) -> Dict[str, Any]:
    """FROZEN kwargs with the names mapped back to real objects."""
    kwargs = dict(FROZEN[row]["kwargs"])
    kwargs["dtype"] = getattr(torch, kwargs["dtype"])
    mode = kwargs.get("kv_quant_mode")
    kwargs["kv_quant_mode"] = (
        getattr(KVQuantMode, mode) if mode is not None else KVQuantMode.NONE
    )
    kwargs.setdefault("num_kv_heads", 1)
    return kwargs


class TestRealVLLMFacts(unittest.TestCase):
    """Every frozen row must hold on the installed vLLM 0.26.0."""

    @classmethod
    def setUpClass(cls) -> None:
        reason = _skip_reason()
        if reason:
            raise unittest.SkipTest(reason)

    def test_real_spec_reproduces_frozen(self):
        for row, facts in FROZEN.items():
            with self.subTest(variant=row):
                spec = _spec(row)
                expected_page = (
                    facts["page_size_padded"]
                    if facts["page_size_padded"] is not None
                    else facts["unpadded_page_size_bytes"]
                )
                self.assertEqual(spec.block_size, facts["kwargs"]["block_size"])
                self.assertEqual(spec.storage_block_size, facts["storage_block_size"])
                self.assertEqual(
                    spec.real_page_size_bytes, facts["real_page_size_bytes"]
                )
                self.assertEqual(
                    spec.unpadded_page_size_bytes, facts["unpadded_page_size_bytes"]
                )
                self.assertEqual(spec.page_size_padded, facts["page_size_padded"])
                self.assertEqual(spec.page_size_bytes, expected_page)

    def test_real_fp8_ds_mla_carries_the_per_tensor_mode(self):
        # F-n: every real fp8_ds_mla spec is built with kv_quant_mode set (the
        # connector must not read it as "quantized, refuse").
        spec = _spec("m1")
        self.assertEqual(spec.kv_quant_mode, KVQuantMode.FP8_PER_TENSOR)
        self.assertNotEqual(spec.kv_quant_mode, KVQuantMode.NONE)

    def test_real_plain_fp8_cache_is_uint8_storage(self):
        # M3a: the engine derives the spec dtype from the cache dtype string
        # (`MLAAttention.get_kv_cache_spec` -> kv_cache_dtype_str_to_dtype), and
        # "fp8" maps to uint8 (1 B/element), so the page is block x 576 x 1 B.
        # The bf16 18432 row 02 section 3.2 probed was a template dtype the
        # engine never produces (01 section 2.3: 576 B/token).
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        # "fp8" is not "auto", so the model_config argument is never read.
        dtype = kv_cache_dtype_str_to_dtype("fp8", None)  # ty: ignore[invalid-argument-type]
        self.assertEqual(dtype, torch.uint8)
        spec = MLAAttentionSpec(  # ty: ignore[call-non-callable]
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=dtype,
            cache_dtype_str="fp8",
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        )
        self.assertEqual(spec.real_page_size_bytes, 9216)
        self.assertEqual(spec.unpadded_page_size_bytes, 9216)
        self.assertEqual(spec.page_size_bytes, 9216)

    def test_real_sliding_window_specs_are_not_full_attention(self):
        # The step-0 refusal in _check_attention_spec_supported exists because
        # SlidingWindowMLASpec is NOT a FullAttentionSpec subclass; if vLLM ever
        # makes it one, the gate (and this guard) must be revisited.
        for row in ("v4_swa", "v4_state_l1"):
            with self.subTest(variant=row):
                spec = _spec(row)
                self.assertIsInstance(spec, SlidingWindowMLASpec)  # ty: ignore[invalid-argument-type]
                self.assertNotIsInstance(spec, FullAttentionSpec)  # ty: ignore[invalid-argument-type]

    def test_scheduler_config_folds_a_packed_group(self):
        """The scheduler's view of a packed group is a *folded* one.

        This is the fact the worker-only registration rests on
        (v1_connector): the scheduler gets one sub spec while the worker's
        wrapper still carries every page layout, so the two roles derive
        different location specs from the same model. The fold keeps the
        wrapper dict's *first* entry, which is engine registration order --
        for DeepSeek the indexer registers before the attention module, so the
        scheduler sees the indexer spec (S1). If vLLM ever stops folding, the
        payloads converge and the registration split (plus this expectation)
        must be revisited."""
        for row in ("v32_wrapper_pair", "v4_full_c4a", "v4_state_wrapper_l1"):
            with self.subTest(group=row):
                facts = FROZEN_GROUPS[row]
                # FROZEN_GROUPS.layers is registration order, so layers[0] is
                # the sub spec the engine folds to; sched_row must name it.
                self.assertEqual(facts["sched_row"], facts["layers"][0][1])
                layer_names = [name for name, _ in facts["layers"]]
                wrapper = UniformTypeKVCacheSpecs(  # ty: ignore[call-non-callable]
                    block_size=facts["block_size"],
                    kv_cache_specs={
                        name: _spec(spec_row) for name, spec_row in facts["layers"]
                    },
                )
                self.assertEqual(sorted(wrapper.get_page_sizes()), facts["page_sizes"])
                config = KVCacheConfig(  # ty: ignore[call-non-callable]
                    num_blocks=128,
                    kv_cache_tensors=[],
                    kv_cache_groups=[
                        KVCacheGroupSpec(  # ty: ignore[call-non-callable]
                            layer_names=layer_names,
                            kv_cache_spec=wrapper,
                            is_eagle_group=False,
                        )
                    ],
                )
                folded = generate_scheduler_kv_cache_config([config])  # ty: ignore[call-non-callable]
                group = folded.kv_cache_groups[0]
                self.assertNotIsInstance(
                    group.kv_cache_spec,
                    UniformTypeKVCacheSpecs,  # ty: ignore[invalid-argument-type]
                )
                # The fold deep-copies the config, so compare structurally: the
                # folded spec equals the first entry (registration order), not
                # the last one.
                self.assertEqual(
                    group.kv_cache_spec, wrapper.kv_cache_specs[layer_names[0]]
                )
                self.assertNotEqual(
                    group.kv_cache_spec, wrapper.kv_cache_specs[layer_names[-1]]
                )
                self.assertEqual(group.layer_names, layer_names)
                # The folded spec is the indexer sub spec, not the group total.
                frozen = FROZEN[facts["sched_row"]]
                expected_page = (
                    frozen["page_size_padded"]
                    if frozen["page_size_padded"] is not None
                    else frozen["unpadded_page_size_bytes"]
                )
                self.assertEqual(group.kv_cache_spec.page_size_bytes, expected_page)
                # The worker's own config keeps the wrapper (nothing is folded
                # there).
                self.assertIs(config.kv_cache_groups[0].kv_cache_spec, wrapper)


if __name__ == "__main__":
    unittest.main()
