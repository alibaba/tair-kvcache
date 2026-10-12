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
  documented holes (the RSWA window, the tokens-per-state field);
  the third shape -- pooled / compressed *full attention* -- carries the same
  field, and undeclared subclasses are pinned by their own tests;
* the **wire hole** left by a skipped group (vLLM group numbers are kept, so a
  skipped group is simply absent from the name set).
"""

import sys
import unittest
from types import SimpleNamespace

from kv_cache_manager.py_connector.test import spec_gate_facts as facts
from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)

from kv_cache_manager.py_connector.vllm.vllm_common import (
    _check_attention_spec_supported,
    _is_scratch_spec,
    _tokens_per_state,
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

    def test_retention_of_every_windowed_row(self):
        # The retention metadata is the contract the data plane consumes:
        # one kind, one window, one trailing-edge retention per bucket. Every
        # accepted *non*-windowed row must carry None (full-prefix semantics
        # unchanged), which is what keeps the full-attention path untouched.
        for row in facts.ZOO:
            if row["disposition"] != "accept":
                continue
            with self.subTest(row=row["name"]):
                result = facts.run_row(row, self.engine)
                own = [meta for meta in result.metas if meta.group_idx == 0][0]
                expected = row.get("retention")
                if expected is None:
                    self.assertIsNone(own.retention)
                    return
                self.assertEqual(expected["kind"], own.retention.kind)
                self.assertEqual(expected["window_tokens"], own.retention.window_tokens)
                self.assertEqual(
                    expected.get("extra_retained_tokens", 0),
                    own.retention.extra_retained_tokens,
                )

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


class TestUndeclaredSubclassesFailClosed(unittest.TestCase):
    """Acceptance is per exact class: a new subclass must be declared first.

    This is the RSWA lesson applied to the mechanism itself: RSWASpec is a
    FullAttentionSpec subclass whose window lives in a field the gate did not
    know, and the field whitelist let it pass. Naming RSWA fixes that one
    class; refusing *undeclared* subclasses fixes the class of hole.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def _subclass(self, base_name: str, class_name: str, **kwargs):
        base = self.engine.class_of(base_name)
        assert base is not None
        cls = type(class_name, (base,), {})
        return cls(**kwargs)

    def test_undeclared_subclasses_are_refused_whatever_the_fields_say(self):
        # Three shapes: a bare subclass of each accepted full-prefix class, and
        # one that keeps a *known* window field (the promote escape hatch is
        # about the declared class, not about the field set).
        cases = (
            ("FullAttentionSpec", "FooFullSpec", {}),
            ("FullAttentionSpec", "FooWindowSpec", {"sliding_window": 1024}),
            ("SinkFullAttentionSpec", "FooSinkSpec", {}),
        )
        for base, class_name, extra in cases:
            kwargs = dict(
                block_size=16, num_kv_heads=4, head_size=64, dtype="bfloat16", **extra
            )
            with self.subTest(cls=class_name):
                spec = self._subclass(base, class_name, **kwargs)
                with self.assertRaises(NotImplementedError) as ctx:
                    _check_attention_spec_supported(
                        "group 0", spec, calculate_kv_scales=False
                    )
                message = str(ctx.exception)
                self.assertIn(class_name, message)
                self.assertIn("Unsupported kv cache spec", message)

    def test_the_rswa_shaped_subclass_is_refused_too(self):
        # A brand-new window field on a FullAttentionSpec subclass: accepted by
        # neither the gate nor parse_groups (the pre-B1 hole).
        spec = self._subclass(
            "FullAttentionSpec", "FooBrandNewWindowSpec",
            block_size=16, num_kv_heads=4, head_size=64, dtype="bfloat16",
        )
        spec.brand_new_window = 4096
        with self.assertRaises(NotImplementedError) as ctx:
            _check_attention_spec_supported("group 0", spec, calculate_kv_scales=False)
        self.assertIn("FooBrandNewWindowSpec", str(ctx.exception))
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

    def test_an_undeclared_mla_subclass_is_refused(self):
        for base, name in (
            ("MLAAttentionSpec", "FooMLASpec"),
            ("HiddenStateCacheSpec", "FooHiddenStateSpec"),
        ):
            with self.subTest(cls=name):
                spec = self._subclass(
                    base, name, block_size=16, num_kv_heads=1, head_size=576,
                    dtype="bfloat16",
                )
                with self.assertRaises(NotImplementedError) as ctx:
                    _check_attention_spec_supported(
                        "group 0", spec, calculate_kv_scales=False
                    )
                self.assertIn(name, str(ctx.exception))

    def test_declared_classes_are_unaffected(self):
        # The contrast that keeps the fix honest: the declared classes (and the
        # declared marker) still transfer.
        for row_name in ("full", "full_window", "sink", "hidden_state", "tq_full"):
            with self.subTest(row=row_name):
                result = facts.run_row(facts.ROW_BY_NAME[row_name], self.engine)
                self.assertEqual("accept", result.outcome, result.message)

    def test_an_undeclared_mamba_subclass_is_refused(self):
        # The group-level receipt is exact-typed too: a state spec subclass is a
        # new claim about what a block's state covers, so it must not be waved
        # through as an ordinary state group just because of its base class.
        state_cls = self.engine.class_of("MambaSpec")
        assert state_cls is not None
        foo = type("FooMambaSpec", (state_cls,), {})
        spec = foo(block_size=16, page_size_bytes=16384)
        spec.window = 1024  # a brand-new field, like the RSWA shape
        config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=["m0"], kv_cache_spec=spec, is_eagle_group=False),
                SimpleNamespace(
                    layer_names=["a0"],
                    kv_cache_spec=facts.build_spec(facts.ROW_BY_NAME["full"], self.engine),
                    is_eagle_group=False,
                ),
            ]
        )
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(config, 16, calculate_kv_scales=False)
        self.assertIn("FooMambaSpec", str(ctx.exception))

    def test_a_plain_mamba_group_is_still_a_state_group(self):
        # The control: the declared class keeps transferring as state (the
        # hybrid path is what PR2/PR3 build on).
        state_cls = self.engine.class_of("MambaSpec")
        assert state_cls is not None
        config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["m0", "m1"],
                    kv_cache_spec=state_cls(block_size=16, page_size_bytes=2048),
                    is_eagle_group=False,
                ),
                SimpleNamespace(
                    layer_names=["a0"],
                    kv_cache_spec=facts.build_spec(facts.ROW_BY_NAME["full"], self.engine),
                    is_eagle_group=False,
                ),
            ]
        )
        metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual(
            [type(meta).__name__ for meta in metas],
            ["StateGroupMeta", "AttentionGroupMeta"],
        )
        self.assertEqual(metas[0].per_block_bytes, 2048 * 2)
        self.assertEqual(metas[0].page_size_bytes, 2048)

    def test_an_undeclared_uniform_wrapper_subclass_is_refused(self):
        wrapper_cls = self.engine.class_of("UniformTypeKVCacheSpecs")
        assert wrapper_cls is not None
        inner = facts.build_spec(facts.ROW_BY_NAME["wrapper_v32_pair"], self.engine)
        foo = type("FooWrapperSpec", (wrapper_cls,), {})
        spec = foo(block_size=inner.block_size, kv_cache_specs=inner.kv_cache_specs)
        config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=sorted(inner.kv_cache_specs), kv_cache_spec=spec,
                    is_eagle_group=False,
                )
            ]
        )
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(config, 64, calculate_kv_scales=False)
        self.assertIn("FooWrapperSpec", str(ctx.exception))

    def test_a_declared_mamba_subclass_of_an_era_is_absent(self):
        # The audit behind the exact match: vLLM 0.14 - 0.31 ships no MambaSpec
        # or UniformTypeKVCacheSpecs subclass (AST scan over every era tree), so
        # tightening the group receipts refuses no real shape.
        module = self.engine.module
        for base_name in ("MambaSpec", "UniformTypeKVCacheSpecs"):
            base = self.engine.class_of(base_name)
            subclasses = [
                obj.__name__
                for obj in vars(module).values()
                if isinstance(obj, type) and issubclass(obj, base) and obj is not base
            ]
            with self.subTest(base=base_name):
                self.assertEqual([], subclasses)

    def test_a_windowed_subclass_names_itself_in_the_refusal(self):
        # N4: the refusal must name the concrete class (vLLM 0.30's
        # KpoolTailSpec is refused through this branch, and calling it a
        # SlidingWindowSpec would send the operator looking for the wrong spec).
        spec = self._subclass(
            "SlidingWindowSpec", "MyWindowedSpec",
            block_size=16, num_kv_heads=4, head_size=64, dtype="bfloat16",
            sliding_window=512,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            _check_attention_spec_supported("group 0", spec, calculate_kv_scales=False)
        message = str(ctx.exception)
        self.assertIn("MyWindowedSpec", message)
        self.assertIn("sliding_window=512", message)


class TestHiddenStateWindows(unittest.TestCase):
    """N1: the extract_hidden_states marker gets no window exemption."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def test_a_windowed_hidden_state_spec_is_refused(self):
        spec_cls = self.engine.class_of("HiddenStateCacheSpec")
        assert spec_cls is not None
        spec = spec_cls(
            block_size=16,
            num_kv_heads=3,
            head_size=2560,
            dtype="bfloat16",
        )
        # The real class inherits the window fields from FullAttentionSpec (only
        # this stand-in's constructor drops them), so set the one vLLM's promote
        # path would leave behind.
        spec.sliding_window = 1024
        with self.assertRaises(NotImplementedError) as ctx:
            _check_attention_spec_supported("group 0", spec, calculate_kv_scales=False)
        message = str(ctx.exception)
        self.assertIn("HiddenStateCacheSpec", message)
        self.assertIn("sliding_window=1024", message)
        # ... while the plain marker (the only shape vLLM builds) is accepted.
        self.assertEqual(
            "accept",
            facts.run_row(facts.ROW_BY_NAME["hidden_state"], self.engine).outcome,
        )


class TestTransferProjection(unittest.TestCase):
    """vLLM (0.30+) projects the group list for connectors; honor it.

    enable_kv_transfer=False marks a group the engine rebuilds itself. Such a
    group must not be registered even when its spec looks cacheable -- and the
    remaining groups keep their raw vLLM indices, because that is the index
    space of the block-id payloads this connector receives.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def _group(self, row_name, layers, **attrs):
        spec = facts.build_spec(facts.ROW_BY_NAME[row_name], self.engine)
        return SimpleNamespace(
            layer_names=layers, kv_cache_spec=spec, is_eagle_group=False, **attrs
        )

    def _config(self, groups, transfer_ids=None):
        config = SimpleNamespace(kv_cache_groups=groups)
        if transfer_ids is not None:
            config.transfer_group_ids = transfer_ids
        return config

    def test_a_non_transferable_group_stays_out_of_the_wire(self):
        # [full(transfer), full(no transfer), full(transfer)]: the middle group
        # is engine-owned; the wire keeps the raw indices 0 and 2.
        config = self._config(
            [
                self._group("full", ["a0"]),
                self._group("mla_bf16", ["x1"]),
                self._group("full", ["a2"]),
            ],
            transfer_ids=(0, 2),
        )
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured:
            metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual([meta.group_idx for meta in metas], [0, 2])
        self.assertEqual([spec_name(0, m) for m in metas], ["tp0_g0", "tp0_g2"])
        self.assertTrue(
            any("does not transfer it" in msg for msg in captured.output),
            captured.output,
        )

    def test_the_engine_projection_decides_membership_not_the_class_flags(self):
        # mla_bf16 is a perfectly cacheable class: only the engine's projection
        # can say it is rebuilt engine-side (that is the HiSparse shape).
        config = self._config(
            [self._group("full", ["a0"]), self._group("mla_bf16", ["hidden"])],
            transfer_ids=(0,),
        )
        metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual([meta.group_idx for meta in metas], [0])

    def test_an_era_without_the_projection_keeps_every_group(self):
        # vLLM <= 0.29 hands no transfer_group_ids attribute: all groups.
        config = self._config(
            [self._group("full", ["a0"]), self._group("mla_bf16", ["x1"])]
        )
        metas = parse_groups(config, 16, calculate_kv_scales=False)
        self.assertEqual([meta.group_idx for meta in metas], [0, 1])

    def test_all_groups_filtered_out_is_a_clean_refusal(self):
        config = self._config([self._group("full", ["a0"])], transfer_ids=())
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(config, 16, calculate_kv_scales=False)
        self.assertIn("no usable kv cache groups", str(ctx.exception))


class TestPooledFullAttentionRefusal(unittest.TestCase):
    """One row per token is the transfer model: pooled rows are refused."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def _gate(self, row_name):
        spec = facts.build_spec(facts.ROW_BY_NAME[row_name], self.engine)
        return _check_attention_spec_supported(
            "group 0", spec, calculate_kv_scales=False
        )

    def test_a_pooled_full_attention_spec_is_refused(self):
        # Whisper pools: tokens_per_state = Fraction(1, n) means several states
        # per token row, which _attention_meta would size as one per token.
        with self.assertRaises(NotImplementedError) as ctx:
            self._gate("full_pooled")
        message = str(ctx.exception)
        self.assertIn("tokens_per_state=1/8", message)
        self.assertIn("one state per token", message)

    def test_a_compressed_full_attention_spec_is_refused(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self._gate("full_compressed")
        self.assertIn("tokens_per_state=2", str(ctx.exception))

    def test_the_plain_and_windowed_promote_specs_still_pass(self):
        for row_name in ("full", "full_window", "full_chunked", "sink", "tq_full"):
            with self.subTest(row=row_name):
                self.assertIsNone(self._gate(row_name))


class TestBucketDtype(unittest.TestCase):
    """AttentionBucketMeta.dtype is the bucket's element type (spec.dtype)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def test_attention_metas_carry_the_spec_dtype(self):
        # One instance may mix element sizes (V3.2 bf16 main + uint8 indexer),
        # so the bucket -- not the instance -- is the dtype granularity.
        for row_name in ("full", "mla_bf16", "mla_fp8_ds"):
            with self.subTest(row=row_name):
                row = facts.ROW_BY_NAME[row_name]
                spec = facts.build_spec(row, self.engine)
                metas = facts.run_row(row, self.engine).metas
                self.assertEqual([meta.dtype for meta in metas], [spec.dtype])

    def test_wrapper_buckets_each_carry_their_member_dtype(self):
        row = facts.ROW_BY_NAME["wrapper_v32_pair"]
        members = facts.member_specs(row, self.engine)
        metas = facts.run_row(row, self.engine).metas
        self.assertGreater(len(metas), 1)
        self.assertEqual(
            sorted(str(meta.dtype) for meta in metas),
            sorted(str(spec.dtype) for spec in members.values()),
        )


class TestScratchGroupSkip(unittest.TestCase):
    """A scratch group is dropped, not refused -- and the wire shows a hole."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def _parse(self, groups, mbs=16):
        return parse_groups(
            SimpleNamespace(kv_cache_groups=groups), mbs, calculate_kv_scales=False
        )

    def _group(self, row_name, layers):
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
        # The line says why the group is gone *and* where a class that claims
        # the opt-out flags but is meant to be cached has to be named.
        self.assertIn("prefix_cacheable=False", skipped[0])
        self.assertIn("vllm_common.py", skipped[0])

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
        self.assertFalse(_is_scratch_spec(spec))
        # ... and the 0.26 shapes answer nothing at all: never skip on a flag
        # the era cannot answer.
        class NoFlags:
            block_size = 16

        self.assertFalse(_is_scratch_spec(NoFlags()))


class TestHolesAreReversed(unittest.TestCase):
    """The silent holes: accepted before this change, refused now."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine()

    def _row(self, name):
        row = facts.ROW_BY_NAME[name]
        spec = facts.build_spec(row, self.engine)
        return row, spec, facts.legacy_accepts(spec, self.engine)

    def test_rswa_was_silently_accepted_and_now_declares_its_window(self):
        # Pre-PR: RSWASpec is a FullAttentionSpec with no window field in the
        # whitelist (rswa_window) -> ACCEPT, and the freed gap blocks were
        # published as a token prefix. Now the class itself is the decision: it
        # is accepted *with* the retention metadata that drives the eager
        # capture, the null disposition and the hit clamp.
        row, spec, legacy = self._row("rswa")
        self.assertTrue(legacy, "the transcription must reproduce the old hole")
        result = facts.run_row(row, self.engine)
        self.assertEqual("accept", result.outcome)
        own = [meta for meta in result.metas if meta.group_idx == 0][0]
        self.assertEqual("rswa_gap", own.retention.kind)
        self.assertEqual(128, own.retention.window_tokens)
        # Same geometry, plain FullAttentionSpec: accepted without retention.
        # The difference is the class (the window semantics), not the fields.
        accepted_row = {
            "name": row["name"] + "_as_full",
            "cls": "FullAttentionSpec",
            "kwargs": {k: v for k, v in row["kwargs"].items() if k != "rswa_window"},
            "disposition": "accept",
            "mbs": row["mbs"],
        }
        plain = facts.build_spec(accepted_row, self.engine)
        self.assertEqual(spec.real_page_size_bytes, plain.real_page_size_bytes)
        plain_result = facts.run_row(accepted_row, self.engine)
        self.assertEqual("accept", plain_result.outcome)
        self.assertIsNone(plain_result.metas[0].retention)

    def test_a_pooled_full_attention_spec_was_accepted_and_is_refused_now(self):
        # Pre-PR: FullAttentionSpec isinstance -> ACCEPT with the page sized as
        # one row per token, and registration failed much later (Whisper's
        # block pooling, and the promote path can keep the field). Now the
        # accepted branch reads the field itself.
        for row_name, keyword in (
            ("full_pooled", "tokens_per_state=1/8"),
            ("full_compressed", "tokens_per_state=2"),
        ):
            with self.subTest(row=row_name):
                row, _, legacy = self._row(row_name)
                self.assertTrue(legacy, "the transcription must reproduce the hole")
                result = facts.run_row(row, self.engine)
                self.assertEqual("refuse", result.outcome)
                self.assertIn(keyword, result.message)

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
        # shrink are the documented holes (rswa / tokens_per_state).
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

    def test_the_widened_rows_are_the_declared_mechanism_shapes(self):
        # Widening is the other direction, and it is a *declared* list: the two
        # promoted FullAttentionSpec window shapes (the
        # --disable-hybrid-kv-cache-manager escape hatch, which needs no
        # windowed machinery -- those blocks are the full prefix) plus the three
        # windowed classes this PR accepts with their retention.
        widened = [
            row["name"]
            for row in facts.ZOO
            if not facts.legacy_accepts_row(row, self.engine)
            and facts.run_row(row, self.engine).outcome == "accept"
        ]
        # (RSWASpec is not in the list: the old gate already accepted it, so it
        # is a hole closed with a mechanism rather than a widened class -- see
        # TestHolesAreReversed.)
        self.assertEqual(
            ["full_window", "full_chunked", "sliding_window",
             "sliding_window_retained", "chunked_local"],
            widened,
        )


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
        # The red line of this PR, spelled out: the three windowed classes this
        # PR implements are accepted (with retention), the windowed MLA variant
        # stays refused (its correct disposition is the DSA skip), the scratch
        # classes are skipped, and nothing else moved.
        self.assertEqual(
            {
                "SlidingWindowSpec": "accept",
                "ChunkedLocalAttentionSpec": "accept",
                "SlidingWindowMLASpec": "refuse",
                "RSWASpec": "accept",
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
