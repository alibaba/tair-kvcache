"""The layout preference gates: which model, on which era, gets BLNHC.

``TairKvCacheConnector.get_required_kvcache_layout`` asks the engine for a
token-major page layout (``BLNHC``) only where asking can help:

* only when the installed era's layout name space has ``BLNHC`` at all. vLLM
  <= 0.28 knows only ``NHD``/``HND`` and adopts a connector's answer verbatim
  (no candidate-set check), so any other name there would land as-is.
* only for models whose KV blocks pack several per-layer pages: their backends
  support block-outer layouts only, and the engine's own pick among those is
  the head-major ``BLHNC``, which the transfer kernel cannot read. Every other
  model already gets a token-major layout (``LBNHC``) from the engine, and
  ``BLNHC`` is part of the engine's *fallback* preference list -- declaring it
  unconditionally would silently move flash-attn/triton models onto a
  packed-block layout (measured in test_kv_layout_preference_real.py).

Runs on the shared stand-in ring (no vLLM); the same decisions are re-derived
against the installed engine in test_kv_layout_preference_real.py.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from kv_cache_manager.py_connector.test.vllm_stubs import STUBBED  # noqa: F401
import kv_cache_manager.py_connector.vllm.v1_connector as connector_module
from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector


class _LayoutNames:
    """Stand-in for the engine's ``KVCacheLayout``: membership is all the hook
    asks about it."""

    def __init__(self, *names: str) -> None:
        for name in names:
            setattr(self, name, name)


#: The 0.29.0+ name space (``vllm/v1/kv_cache_layout.py``).
_ERA_WITH_BLNHC = _LayoutNames("LBNHC", "LBHNC", "LHBNC", "BLHNC", "BLNHC", "BHLNC")
#: A name space that never heard of the block-outer family.
_ERA_WITHOUT_BLNHC = _LayoutNames("NHD", "HND")

#: Architectures the connector serves today: the engine picks a token-major
#: layout for all of them, so the hook must stay out of the way.
_SERVED_ARCHITECTURES = (
    ("Qwen2ForCausalLM",),  # full attention
    ("Qwen3_5ForCausalLM",),  # hybrid mamba
    ("Gemma3ForCausalLM",),  # sliding window / mm
    ("Glm4MoeForCausalLM",),
    ("DeepseekV32ForCausalLM",),  # MLA, but a layer-compact candidate set
    ("Qwen3_5ForCausalLM", "WrongExtraArchitecture"),
)


class LayoutPreferenceCase(unittest.TestCase):
    def preference(self, architectures, layout_namespace):
        """The hook's answer for a config shaped like the engine's."""
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architectures=list(architectures))
        )
        with mock.patch.object(connector_module, "_KVCacheLayout", layout_namespace):
            return TairKvCacheConnector.get_required_kvcache_layout(vllm_config)


class TestPackedBlockModels(LayoutPreferenceCase):
    """Models that pack per-layer pages inside one block: ask for BLNHC."""

    def test_packed_block_model_prefers_blnhc(self):
        for arch in ("DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"):
            with self.subTest(arch=arch):
                self.assertEqual(self.preference([arch], _ERA_WITH_BLNHC), "BLNHC")

    def test_unknown_layout_name_space_stays_silent(self):
        # vLLM <= 0.28 has no such name at all; the answer would be adopted
        # verbatim, so the hook must not answer.
        self.assertIsNone(self.preference(["DeepseekV41ForCausalLM"], None))

    def test_name_space_without_the_name_stays_silent(self):
        self.assertIsNone(
            self.preference(["DeepseekV41ForCausalLM"], _ERA_WITHOUT_BLNHC)
        )


class TestOtherModelsStayOnTheEngineDefault(LayoutPreferenceCase):
    """Everything else: no preference (the engine already picks token-major).

    BLNHC belongs to the engine's fallback preference list, so answering here
    would be accepted and would move the model onto a packed-block layout.
    """

    def test_served_architectures_get_no_preference(self):
        for architectures in _SERVED_ARCHITECTURES:
            with self.subTest(architectures=architectures):
                self.assertIsNone(self.preference(architectures, _ERA_WITH_BLNHC))

    def test_prefix_match_is_not_a_substring_match(self):
        # DeepSeek-V3.2 must not ride on the V4 prefix.
        self.assertIsNone(self.preference(["DeepseekV32ForCausalLM"], _ERA_WITH_BLNHC))
        self.assertIsNone(self.preference(["NotDeepseekV4Either"], _ERA_WITH_BLNHC))

    def test_missing_or_empty_model_config_stays_silent(self):
        for model_config in (
            None,
            SimpleNamespace(),
            SimpleNamespace(architectures=None),
            SimpleNamespace(architectures=[]),
        ):
            with self.subTest(model_config=model_config):
                vllm_config = SimpleNamespace(model_config=model_config)
                with mock.patch.object(
                    connector_module, "_KVCacheLayout", _ERA_WITH_BLNHC
                ):
                    self.assertIsNone(
                        TairKvCacheConnector.get_required_kvcache_layout(vllm_config)
                    )
        with mock.patch.object(connector_module, "_KVCacheLayout", _ERA_WITH_BLNHC):
            # A config without model_config at all (standalone tools build
            # such things) must not raise either.
            self.assertIsNone(
                TairKvCacheConnector.get_required_kvcache_layout(SimpleNamespace())
            )


if __name__ == "__main__":
    unittest.main()
