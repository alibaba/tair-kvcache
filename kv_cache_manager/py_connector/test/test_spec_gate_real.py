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
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from kv_cache_manager.py_connector.test import spec_gate_facts as facts

try:
    import torch
    import vllm

    from vllm.v1.kv_cache_interface import KVCacheSpec

    from kv_cache_manager.py_connector.vllm.vllm_common import parse_groups

    _IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - environment dependent
    torch = None  # ty: ignore[invalid-assignment]
    vllm = None  # ty: ignore[invalid-assignment]
    KVCacheSpec = None  # ty: ignore[invalid-assignment]
    parse_groups = None  # ty: ignore[invalid-assignment]
    _IMPORT_ERROR = str(exc)


def _skip_reason() -> str:
    if _IMPORT_ERROR:
        return f"real vLLM not importable ({_IMPORT_ERROR})"
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
        import sys

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


class TestEraShapes(RealEngineCase):
    """The 0.26 -> 0.31 field migration, per era, through the same code path."""

    def test_the_eras_spelling_is_the_one_that_runs(self):
        # At most one of the two compressed rows is buildable here; the other
        # era's shape is reported as such instead of being silently built from
        # the wrong field. vLLM 0.14 has neither spelling (and no compressed
        # MLA at all).
        compress = facts.run_row(facts.ROW_BY_NAME["mla_compressed_0_26"], self.engine)
        tokens = facts.run_row(facts.ROW_BY_NAME["mla_compressed_0_31"], self.engine)
        outcomes = [compress.outcome, tokens.outcome]
        self.assertLessEqual(outcomes.count("refuse"), 1, "both spellings refused")
        if "refuse" not in outcomes:
            self.assertEqual(["unavailable", "unavailable"], outcomes)
        elif compress.outcome == "refuse":
            self.assertIn("compress_ratio=4", compress.message)
        else:
            self.assertIn("tokens_per_state=2", tokens.message)

    def test_scratch_flag_shapes_are_the_measured_ones(self):
        # The data behind the era-dependent skip (facts.SCRATCH_FLAGS): the
        # group is dropped only where the era's spec opts out of *both* halves,
        # which is why 0.29 and 0.30 refuse what 0.31 skips.
        for version, classes in facts.SCRATCH_FLAGS.items():
            if version != self.version:
                continue
            for name, flags in classes.items():
                row = next(row for row in facts.ZOO if row["cls"] == name)
                with self.subTest(cls=name):
                    spec = facts.build_spec(row, self.engine)
                    measured = tuple(
                        getattr(spec, attribute, None)
                        for attribute in ("prefix_cacheable", "uses_slot_mapping", "has_layer_views")
                    )
                    self.assertEqual(flags, measured)
            return

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
    through. The table is now the gate's contract, and a new class (or a new
    era) must be added to it deliberately.
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


class TestV41HoleOnTheRealEngine(RealEngineCase):
    """The field-migration hole, at the size the old gate actually produced."""

    def test_compressed_main_is_refused_at_the_legacy_size(self):
        row = facts.ROW_BY_NAME["mla_compressed_0_31"]
        result = facts.run_row(row, self.engine)
        if result.outcome == "unavailable":
            self.assert_unavailable_is_expected(row, result.message)
            return
        self.assertEqual("refuse", result.outcome)
        self.assertIn("tokens_per_state=2", result.message)
        # What the pre-PR gate sized from the same spec (74752 B/block at
        # mbs 256, i.e. a page that described a layout the tensor never had).
        spec = facts.build_spec(row, self.engine)
        self.assertTrue(facts.legacy_accepts(spec, self.engine))
        self.assertEqual(74752, spec.real_page_size_bytes)


if __name__ == "__main__":
    unittest.main()
