"""The vLLM spec-receipt gate, class by class (stand-in ring, no GPU/vLLM).

PR1 turned ``_check_attention_spec_supported`` from a *field* whitelist into an
explicit per-class dispatch, added the parse-time skip for per-request scratch
groups, and made the MLAAttentionSpec tokens-per-state read era-safe. This
module pins that table on the ``vllm_stubs`` stand-in -- which mirrors the union
of the vLLM eras (the 0.31 class tree and capability flags plus the 0.26
``compress_ratio`` spelling), so both spellings are exercised with no vLLM
installed at all. ``test_spec_gate_real.py`` re-derives the same table from the
real engine: the two cannot drift in the same direction unnoticed.

Three things are pinned here that a plain table check would miss:

* the **reversed** assertions: RSWASpec and a compressed MLA spec were
  *accepted* by the pre-PR gate (``legacy_accepts``, a transcription of that
  predicate) and are refused now;
* the **differential**: today's accepted surface stays accepted, except for the
  two documented holes;
* the **wire hole** left by a skipped group (vLLM group numbers are kept, so a
  skipped group is simply absent from the name set).
"""

import sys
import unittest

from kv_cache_manager.py_connector.test import spec_gate_facts as facts
from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)

from kv_cache_manager.py_connector.vllm.vllm_common import (
    parse_groups,
    spec_name,
)

LOGGER_NAME = "kv_cache_manager.py_connector.common.logger"


def engine() -> facts.Engine:
    """The stand-in engine: dtype names stay names, the stub parses them.

    Its version is the era the stand-in mirrors (0.31), which is what makes the
    scratch rows' era-dependent skip expectations apply here.
    """
    return facts.Engine(
        module=sys.modules["vllm.v1.kv_cache_interface"],
        dtype_of=lambda name: name,
        parse_groups=parse_groups,
        version="0.31.0",
    )


class TestZooDispositions(unittest.TestCase):
    """Every row of the zoo (test/spec_gate_facts.ZOO), through parse_groups."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def test_every_row_is_buildable_on_the_union_stand_in(self):
        # The stand-in carries every era's fields, so no row may be skipped
        # here: a gap in the mirror must not be able to hide a gap in the gate.
        for row in facts.ZOO:
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                self.assertNotEqual("unavailable", result.outcome, result.message)

    def test_disposition_of_every_row(self):
        for row in facts.ZOO:
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                expected = facts.expected_outcome(row, self.engine)
                self.assertEqual(expected, result.outcome, result.message)
                if result.outcome != "refuse":
                    continue
                for keyword in row.get("keywords", ()):
                    self.assertIn(keyword, result.message)
                if facts.SPEC_DISPOSITIONS[row["cls"]] == "refuse":
                    # A whole-family refusal explains why and what would change
                    # it; a variant refusal (compressed / windowed MLA) is
                    # pinned by its keywords instead.
                    self.assertIn("TairKvCacheConnector", result.message)

    def test_sizes_of_every_accepted_row(self):
        # Sizing is re-pinned on the real engine too; here it proves the
        # stand-in and the gate agree on *what* is being transferred.
        for row in facts.ZOO:
            if row["disposition"] != "accept" or "page_bytes" not in row:
                continue
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                own = [meta for meta in result.metas if meta.group_idx == 0]
                self.assertTrue(own, result.message)
                self.assertEqual(row["page_bytes"], own[0].page_bytes)
                self.assertEqual(row["per_block_bytes"], own[0].per_block_bytes)

    def test_skipped_rows_leave_the_companion_alone(self):
        for row in facts.ZOO:
            if facts.expected_outcome(row, self.engine) != "skip":
                continue
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                self.assertEqual([meta.group_idx for meta in result.metas], [1])


class TestScratchGroupSkip(unittest.TestCase):
    """A scratch group is dropped, not refused -- and the wire shows a hole."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def _parse(self, groups, mbs=16):
        from types import SimpleNamespace

        return parse_groups(
            SimpleNamespace(kv_cache_groups=groups), mbs, calculate_kv_scales=False
        )

    def _group(self, row_name, layers):
        from types import SimpleNamespace

        return SimpleNamespace(
            layer_names=layers,
            kv_cache_spec=facts.build_spec(facts.ROW_BY_NAME[row_name], self.engine),
            is_eagle_group=False,
        )

    def test_group_numbers_survive_the_skip(self):
        # [scratch, full]: the scratch group keeps index 0, so the surviving
        # group keeps the name tp{r}_g1 -- exactly the EAGLE hole shape.
        metas = self._parse(
            [
                self._group("circular_buffer", ["ring"]),
                self._group("full", ["attn"]),
            ]
        )
        self.assertEqual([meta.group_idx for meta in metas], [1])
        self.assertEqual([spec_name(0, meta) for meta in metas], ["tp0_g1"])

    def test_skip_between_two_full_groups(self):
        metas = self._parse(
            [
                self._group("full", ["a0"]),
                self._group("hisparse_resident", ["hot"]),
                self._group("full", ["a1"]),
            ]
        )
        self.assertEqual([meta.group_idx for meta in metas], [0, 2])
        self.assertEqual(
            [spec_name(0, meta) for meta in metas], ["tp0_g0", "tp0_g2"]
        )

    def test_skip_logs_one_line_with_the_class(self):
        # The operator must be able to see *what* was left out of the manager.
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured:
            self._parse(
                [
                    self._group("kpool_tail", ["tail"]),
                    self._group("full", ["attn"]),
                ]
            )
        skipped = [msg for msg in captured.output if "skip" in msg]
        self.assertEqual(len(skipped), 1)
        self.assertIn("KpoolTailSpec", skipped[0])
        self.assertIn("group 0", skipped[0])
        self.assertIn("1 layers", skipped[0])

    def test_a_scratch_only_instance_is_refused(self):
        with self.assertRaisesRegex(
            NotImplementedError, "no usable kv cache groups"
        ):
            self._parse([self._group("circular_buffer", ["ring"])])

    def test_the_skip_predicate_is_capability_based(self):
        # The predicate is "not prefix cacheable AND not transferable"; both
        # halves are needed. SlidingWindowMLASpec with bounded replay is the
        # counter-example that keeps a plain "prefix_cacheable is False" rule
        # from silently dropping a group the table still refuses.
        spec = facts.build_spec(
            facts.ROW_BY_NAME["swa_mla_bounded_replay"], self.engine
        )
        self.assertFalse(spec.prefix_cacheable)
        self.assertTrue(spec.uses_slot_mapping)
        from kv_cache_manager.py_connector.vllm.vllm_common import _is_scratch_spec

        self.assertFalse(_is_scratch_spec(spec))
        # ... and the 0.26 shapes answer nothing at all: never skip on a flag
        # the era cannot answer.
        class NoFlags:
            block_size = 16

        self.assertFalse(_is_scratch_spec(NoFlags()))


class TestHolesAreReversed(unittest.TestCase):
    """The two silent holes: accepted before this change, refused now."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def _row(self, name):
        row = facts.ROW_BY_NAME[name]
        spec = facts.build_spec(row, self.engine)
        return row, spec, facts.legacy_accepts(spec, self.engine)

    def test_rswa_was_accepted_and_is_refused_now(self):
        # Pre-PR: RSWASpec is a FullAttentionSpec with no window field in the
        # whitelist (rswa_window) -> ACCEPT, and the freed gap blocks were
        # published as a token prefix. Now the class itself is the decision.
        row, spec, legacy = self._row("rswa")
        self.assertTrue(legacy, "the transcription must reproduce the old hole")
        result = facts.run_row(row, self.engine)
        self.assertEqual("refuse", result.outcome)
        self.assertIn("rswa_window=128", result.message)
        # Same geometry, plain FullAttentionSpec: accepted. The difference is
        # the class (the window semantics), not the fields.
        accepted_row = {
            "name": row["name"] + "_as_full",
            "cls": "FullAttentionSpec",
            "kwargs": {k: v for k, v in row["kwargs"].items() if k != "rswa_window"},
            "disposition": "accept",
            "mbs": row["mbs"],
        }
        plain = facts.build_spec(accepted_row, self.engine)
        self.assertEqual(spec.real_page_size_bytes, plain.real_page_size_bytes)
        self.assertEqual("accept", facts.run_row(accepted_row, self.engine).outcome)

    def test_compressed_mla_was_accepted_and_is_refused_now(self):
        # Pre-PR: the gate read compress_ratio, which the 0.29+ spec no longer
        # has (getattr -> 1) -> ACCEPT, while the real layout is one row per 2
        # tokens. Now the era-safe read sees tokens_per_state.
        row, _, legacy = self._row("mla_compressed_0_31")
        self.assertTrue(legacy)
        result = facts.run_row(row, self.engine)
        self.assertEqual("refuse", result.outcome)
        self.assertIn("tokens_per_state=2", result.message)

    def test_pooled_states_are_refused_too(self):
        # The same field family: Fraction(1, n) means several states per token.
        row, _, legacy = self._row("mla_pooled_fraction")
        self.assertTrue(legacy)
        result = facts.run_row(row, self.engine)
        self.assertEqual("refuse", result.outcome)
        self.assertIn("tokens_per_state=1/8", result.message)


class TestEraSafeTokensPerState(unittest.TestCase):
    """One code path, two spellings: whichever field carries a value wins."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def _message(self, **kwargs) -> str:
        from kv_cache_manager.py_connector.vllm.vllm_common import (
            _check_attention_spec_supported,
        )

        spec_cls = self.engine.class_of("MLAAttentionSpec")
        assert spec_cls is not None
        spec = spec_cls(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype="bfloat16",
            cache_dtype_str="auto",
            **kwargs,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            _check_attention_spec_supported(
                "group 0", spec, calculate_kv_scales=False
            )
        return str(ctx.exception)

    def test_the_026_spelling_is_named(self):
        self.assertIn("compress_ratio=4", self._message(compress_ratio=4))

    def test_the_031_spelling_is_named(self):
        self.assertIn("tokens_per_state=2", self._message(tokens_per_state=2))

    def test_the_newer_spelling_wins_when_both_carry_a_value(self):
        message = self._message(compress_ratio=4, tokens_per_state=2)
        self.assertIn("tokens_per_state=2", message)
        self.assertNotIn("compress_ratio=4", message)

    def test_both_defaults_mean_one_row_per_token(self):
        from kv_cache_manager.py_connector.vllm.vllm_common import (
            _tokens_per_state,
        )

        spec_cls = self.engine.class_of("MLAAttentionSpec")
        assert spec_cls is not None
        spec = spec_cls(
            block_size=16, num_kv_heads=1, head_size=576, dtype="bfloat16"
        )
        self.assertEqual((1, "tokens_per_state"), _tokens_per_state(spec))
        plain = engine().class_of("FullAttentionSpec")
        assert plain is not None
        self.assertEqual(
            (1, "tokens_per_state"),
            _tokens_per_state(plain(block_size=16, num_kv_heads=1, head_size=64)),
        )


class TestAcceptanceSurfaceDifferential(unittest.TestCase):
    """Every class accepted today is still accepted (or properly skipped)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def test_declared_legacy_matches_the_transcription(self):
        # The `legacy` column is a claim about the pre-PR gate; the transcription
        # is its executable form. Drift in either turns red.
        for row in facts.ZOO:
            with self.subTest(row=row["name"]):
                self.assertEqual(
                    row["legacy"] == "accept",
                    facts.legacy_accepts_row(row, self.engine),
                )

    def test_legacy_surface_is_contained(self):
        # accepted before => accepted or skipped now. The only rows allowed to
        # shrink are the two documented holes.
        for row in facts.ZOO:
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                if not facts.legacy_accepts_row(row, self.engine):
                    continue
                if result.outcome == "refuse":
                    self.assertIn(
                        row.get("hole"),
                        ("rswa", "tokens_per_state"),
                        f"{row['name']} was accepted before and is refused now",
                    )
                else:
                    self.assertIn(result.outcome, ("accept", "skip"))

    def test_the_widened_rows_are_the_promote_shapes(self):
        # Widening is the other direction: only the promoted FullAttentionSpec
        # window shapes (the --disable-hybrid-kv-cache-manager escape hatch).
        widened = [
            row["name"]
            for row in facts.ZOO
            if not facts.legacy_accepts_row(row, self.engine)
            and facts.run_row(row, self.engine).outcome == "accept"
        ]
        self.assertEqual(["full_window", "full_chunked"], widened)


class TestTableInvariants(unittest.TestCase):
    """The table itself must stay complete and internally consistent."""

    def test_class_table_and_rows_agree(self):
        # The class table is the family-level decision; a row may be narrower
        # (a compressed or windowed MLA variant is refused inside an accepted
        # family). The other direction is absolute: a refused or skipped
        # family has no accepted row.
        for row in facts.ZOO:
            family = facts.SPEC_DISPOSITIONS[row["cls"]]
            with self.subTest(row=row["name"]):
                if family in ("refuse", "skip"):
                    self.assertEqual(family, row["disposition"])
                else:
                    self.assertIn(family, ("accept", "split"))
                    self.assertIn(row["disposition"], ("accept", "refuse", "skip"))

    def test_every_accepted_family_has_an_accepted_row(self):
        for name, disposition in facts.SPEC_DISPOSITIONS.items():
            if disposition != "accept":
                continue
            with self.subTest(cls=name):
                self.assertTrue(
                    any(
                        row["cls"] == name and row["disposition"] == "accept"
                        for row in facts.ZOO
                    ),
                    f"{name} is declared accepted but no row proves it",
                )

    def test_every_known_class_is_covered(self):
        covered = {row["cls"] for row in facts.ZOO} | set(facts.NOT_IN_ZOO)
        self.assertEqual(set(facts.SPEC_DISPOSITIONS), covered)

    def test_windowed_and_scratch_classes_keep_their_side(self):
        # The red line of this PR, spelled out: windowed classes stay refused
        # (their data-plane mechanism arrives in the follow-up work), scratch
        # classes are skipped, and nothing else moved.
        self.assertEqual(
            {
                "SlidingWindowSpec": "refuse",
                "ChunkedLocalAttentionSpec": "refuse",
                "SlidingWindowMLASpec": "refuse",
                "RSWASpec": "refuse",
                "CircularBufferSpec": "skip",
                "KpoolTailSpec": "skip",
                "HiSparseHotSpec": "skip",
                "HiSparseResidentSpec": "skip",
            },
            {
                name: facts.SPEC_DISPOSITIONS[name]
                for name in (
                    "SlidingWindowSpec",
                    "ChunkedLocalAttentionSpec",
                    "SlidingWindowMLASpec",
                    "RSWASpec",
                    "CircularBufferSpec",
                    "KpoolTailSpec",
                    "HiSparseHotSpec",
                    "HiSparseResidentSpec",
                )
            },
        )

    def test_shrinking_rows_declare_a_hole(self):
        for row in facts.ZOO:
            if row["legacy"] == "accept" and row["disposition"] == "refuse":
                self.assertIn(row.get("hole"), ("rswa", "tokens_per_state"))


if __name__ == "__main__":
    unittest.main()
