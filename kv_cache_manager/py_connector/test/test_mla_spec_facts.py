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

from kv_cache_manager.py_connector.test.mla_variant_facts import FROZEN

try:
    import torch
    import vllm
    from vllm.v1.kv_cache_interface import KVQuantMode, MLAAttentionSpec

    _IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - environment dependent
    torch = None  # ty: ignore[invalid-assignment]
    vllm = None  # ty: ignore[invalid-assignment]
    KVQuantMode = None  # ty: ignore[invalid-assignment]
    MLAAttentionSpec = None  # ty: ignore[invalid-assignment]
    _IMPORT_ERROR = str(exc)


def _skip_reason() -> str:
    if _IMPORT_ERROR:
        return f"real vLLM not importable ({_IMPORT_ERROR})"
    if getattr(vllm, "_kvcm_test_stub", False) or "vllm_stubs" in str(
        getattr(vllm, "__file__", "")
    ):
        return "vLLM is shadowed by the test stubs in this process"
    return ""


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
                spec = MLAAttentionSpec(**_spec_kwargs(row))  # ty: ignore[call-non-callable]
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
        spec = MLAAttentionSpec(**_spec_kwargs("m1"))  # ty: ignore[call-non-callable]
        self.assertEqual(spec.kv_quant_mode, KVQuantMode.FP8_PER_TENSOR)
        self.assertNotEqual(spec.kv_quant_mode, KVQuantMode.NONE)


if __name__ == "__main__":
    unittest.main()
