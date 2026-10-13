"""DSA V4.1 (PR6): the row-granular data plane, pinned against the measurements.

Three independent facts are checked here, and each of them has a negative
control -- the shape that the pre-PR6 code got wrong:

* **address math** (``_attn_slots``): vLLM addresses a compressed cache by
  (storage block, row within it), one row per ``tokens_per_state`` tokens, and
  a block-table entry is one kernel page (SM90: 64 tokens, SM100 family: 128).
  The reference is the survey's measured table
  (``data/pin/sa-b/out_b1_index_table.txt``): with mbs 256, block 64, 32 rows
  per page and tps 2, the connector's old formula multiplied the page number by
  ``gbs // kbs`` and was wrong for *every* token but the first. The table below
  is that measurement, transcribed.
* **the arithmetic preconditions** (``mbs % tps == 0``, ``block_size % tps ==
  0``): a manager block and a block-table entry must stay whole numbers of
  rows. The table uses mbs 528 (vLLM's merge block size) so the ratios that pass
  and the ones that cannot are both covered.
* **the bucket structure** of the real V4.1 shape (9 groups at SM90: seven
  SWA-MLA groups skipped, one packed group of eight layers -> four buckets, one
  CircularBuffer group skipped), with the per-block bytes the survey measured on
  a real vLLM registration (``out_b1_register_sm90_blnhc.txt``).

The stub ring is enough for all three: ``vllm_stubs`` mirrors the 0.31 field
set (``tokens_per_state``, ``state_content_bytes``, ``bounded_replay``) and the
page algebra (page = (block_size // tps) rows x state row), which
``test_mla_spec_facts``/``test_role_views`` reconcile against the real classes
and the measured golden.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)

from kv_cache_manager.py_connector.vllm.vllm_common import (
    ALL_SPEC_GROUP,
    AttentionGroupMeta,
    GroupMeta,
    StateGroupMeta,
    RowGranularMeta,
    WindowRetention,
    build_spec_groups,
    parse_groups,
    spec_name,
)

#: The survey's measured geometry (out_b1_index_table.txt): the V4.1 main layer
#: at SM90 -- block 64 tokens per block-table entry, tps 2, so 32 rows of 584 B
#: per page.
TABLE_MBS = 256
TABLE_BLOCK = 64
TABLE_TPS = 2
TABLE_ROWS_PER_PAGE = TABLE_BLOCK // TABLE_TPS
TABLE_BLOCK_TABLE = [7, 9, 11, 13, 15, 17]
#: (token, vLLM's page, vLLM's row) -- transcribed from the measured table.
VLLM_INDEX_TABLE: List[Tuple[int, int, int]] = [
    (0, 7, 0),
    (1, 7, 0),
    (2, 7, 1),
    (3, 7, 1),
    (4, 7, 2),
    (5, 7, 2),
    (6, 7, 3),
    (7, 7, 3),
    (30, 7, 15),
    (31, 7, 15),
    (32, 7, 16),
    (33, 7, 16),
    (62, 7, 31),
    (63, 7, 31),
    (64, 9, 0),
    (65, 9, 0),
    (126, 9, 31),
    (127, 9, 31),
    (128, 11, 0),
    (190, 11, 31),
    (255, 13, 31),
]


def _quant(name: str) -> Any:
    """The stub's quant mode (an enum: the stub casts it with int())."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    return getattr(KVQuantMode, name)


def _row_granular_spec(**kwargs: Any) -> Any:
    """The stub MLAAttentionSpec of one V4.1-like compressed row."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    fields: Dict[str, Any] = dict(
        block_size=TABLE_BLOCK,
        num_kv_heads=1,
        head_size=512,
        dtype="uint8",
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        kv_quant_mode=_quant("FP8_PER_TENSOR"),
        tokens_per_state=TABLE_TPS,
        state_content_bytes=584,
    )
    fields.update(kwargs)
    return MLAAttentionSpec(**fields)


def _group(spec: Any, layers: Tuple[str, ...] = ("l0",)) -> Any:
    return SimpleNamespace(
        layer_names=list(layers), kv_cache_spec=spec, is_eagle_group=False
    )


def _parse(groups: List[Any], mbs: int) -> List[GroupMeta]:
    return parse_groups(
        SimpleNamespace(kv_cache_groups=groups), mbs, calculate_kv_scales=False
    )


class TestRowGranularAddressMath(unittest.TestCase):
    """The token -> (page, row) mapping, against the survey's measured table."""

    @classmethod
    def setUpClass(cls) -> None:
        from kv_cache_manager.py_connector.test.vllm_stubs import make_connector
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            AttentionTransferGroup,
        )

        cls.meta = _parse([_group(_row_granular_spec())], TABLE_MBS)[0]
        assert isinstance(cls.meta, AttentionGroupMeta)
        group = AttentionTransferGroup(
            group_idx=0,
            spec_name="tp0_g0",
            layer_names=["l0"],
            block_size=cls.meta.block_size,
            per_block_bytes=cls.meta.per_block_bytes,
            layer_num=1,
            kv_layout=None,  # type: ignore[arg-type]
            kvcache_ptr_tensor_gpu=None,  # type: ignore[arg-type]
            num_kv_ptrs=1,
            per_token_dim=584,
            # The tensor's page: rows, not tokens (the whole point of PR6).
            kernel_block_size=TABLE_ROWS_PER_PAGE,
            block_stride=0,
            dtype="uint8",  # type: ignore[arg-type]
            tokens_per_row=TABLE_TPS,
        )
        cls.conn = make_connector(manager_block_size=TABLE_MBS)
        cls.slots = cls.conn._attn_slots(group, [0], TABLE_BLOCK_TABLE)[0][0]

    def _connector_slot(self, token: int) -> Tuple[int, int]:
        # _attn_slots stages one entry per row of the manager block; the row of
        # a token is its index inside that block.
        flat = self.slots[token // self.meta.row_granular.tokens_per_row]
        return flat // TABLE_ROWS_PER_PAGE, flat % TABLE_ROWS_PER_PAGE

    def test_every_measured_token_maps_to_vllms_own_slot(self):
        for token, page, row in VLLM_INDEX_TABLE:
            with self.subTest(token=token):
                self.assertEqual((page, row), self._connector_slot(token))

    def test_the_mapping_is_one_row_per_tps_tokens(self):
        # Two tokens share a row, and the row is what the kernel copies: the
        # manager block stages `mbs // tps` entries, not `mbs`.
        self.assertEqual(len(self.slots), TABLE_MBS // TABLE_TPS)
        self.assertEqual(self._connector_slot(0), self._connector_slot(1))
        self.assertNotEqual(self._connector_slot(0), self._connector_slot(2))

    def test_the_pre_pr6_formula_was_wrong_for_every_token_but_the_first(self):
        # The negative control that documents *why* this PR exists: the old code
        # took the tensor's row count as a token count, so `ratio = gbs // kbs`
        # doubled the page number and shifted every row by one page.
        def legacy(token: int) -> Tuple[int, int]:
            logical = token // TABLE_BLOCK
            off = token % TABLE_BLOCK
            ratio = TABLE_BLOCK // TABLE_ROWS_PER_PAGE
            phys = TABLE_BLOCK_TABLE[logical] * ratio + off // TABLE_ROWS_PER_PAGE
            return phys, off % TABLE_ROWS_PER_PAGE

        wrong = [
            token
            for token, _, _ in VLLM_INDEX_TABLE
            if legacy(token) != self._connector_slot(token)
        ]
        self.assertEqual(wrong, [token for token, _, _ in VLLM_INDEX_TABLE])
        # The old code looked at page 7 x 2 for the very first entry: the page
        # number itself was doubled, so even token 0 (whose *row* still agrees)
        # reads the wrong page.
        self.assertEqual((14, 0), legacy(0))
        self.assertEqual(0, self._connector_slot(0)[1])

    def test_a_page_smaller_than_the_entry_covers_several_of_them(self):
        # The mapping must not assume one page per entry. For a token-granular
        # bucket that is production (the kernel-split layouts: block 64 with a
        # 32-token kernel page); for a *row-granular* one registration pins
        # `kernel_block_size == rows_per_page`, so the geometry below is only
        # reachable through a bypassed registration -- kept as the defensive pin
        # for the formula's `pages_per_entry` term.
        from kv_cache_manager.py_connector.test.vllm_stubs import make_connector
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            AttentionTransferGroup,
        )

        meta = _parse([_group(_row_granular_spec())], 256)[0]
        group = AttentionTransferGroup(
            group_idx=0,
            spec_name="tp0_g0",
            layer_names=["l0"],
            block_size=meta.block_size,
            per_block_bytes=meta.per_block_bytes,
            layer_num=1,
            kv_layout=None,  # type: ignore[arg-type]
            kvcache_ptr_tensor_gpu=None,  # type: ignore[arg-type]
            num_kv_ptrs=1,
            per_token_dim=584,
            # A page of 8 rows: one 64-token entry (32 rows) spans four pages.
            kernel_block_size=8,
            block_stride=0,
            dtype="uint8",  # type: ignore[arg-type]
            tokens_per_row=TABLE_TPS,
        )
        conn = make_connector(manager_block_size=256)
        slots = conn._attn_slots(group, [0], TABLE_BLOCK_TABLE)[0][0]
        # Token 32 -> row 16 -> page 7 * 4 + 2 = 30, row 0.
        self.assertEqual((30, 0), (slots[16] // 8, slots[16] % 8))
        # Token 64 -> row 32 -> the next entry's first page (9 * 4 = 36).
        self.assertEqual((36, 0), (slots[32] // 8, slots[32] % 8))

    def test_the_manager_block_stays_whole_rows(self):
        # tokens_per_row 2 and mbs 256: 128 rows per block, the block table's
        # entries stay aligned with the manager's blocks (the two preconditions
        # asserted while parsing).
        self.assertEqual(self.meta.row_granular, RowGranularMeta(2, 32, 584))
        self.assertEqual(self.meta.per_block_bytes, 584 * 128)


class TestRowGranularPreconditions(unittest.TestCase):
    """mbs % tps == 0 and block_size % tps == 0, over the whole ratio range."""

    MBS = 528  # vLLM's merge block size: 528 = 16 x 33, so 32 and 64 do not divide

    def _verdict(self, *, block_size: int, tps: int, mbs: int = MBS) -> str:
        spec = _row_granular_spec(block_size=block_size, tokens_per_state=tps)
        try:
            _parse([_group(spec)], mbs)
        except NotImplementedError:
            return "refuse"
        return "accept"

    def test_every_ratio_that_divides_both_passes(self):
        for tps in (2, 4, 8, 16):
            with self.subTest(tps=tps):
                self.assertEqual("accept", self._verdict(block_size=64, tps=tps))

    def test_a_ratio_the_manager_block_cannot_hold_is_refused(self):
        # 32 and 64 divide a 64-token block but not a 528-token manager block.
        for tps in (32, 64):
            with self.subTest(tps=tps):
                with self.assertRaises(NotImplementedError) as ctx:
                    spec = _row_granular_spec(block_size=64, tokens_per_state=tps)
                    _parse([_group(spec)], self.MBS)
                self.assertIn("manager block size 528", str(ctx.exception))
                self.assertIn(f"tokens_per_state={tps}", str(ctx.exception))

    def test_a_ratio_the_page_cannot_hold_is_refused(self):
        # 128 divides neither the 64-token block nor the 528-token manager
        # block; the block check fires first (the gate runs before the sizing).
        with self.assertRaises(NotImplementedError) as ctx:
            spec = _row_granular_spec(block_size=64, tokens_per_state=128)
            _parse([_group(spec)], self.MBS)
        self.assertIn("block_size=64", str(ctx.exception))
        self.assertIn("tokens_per_state=128", str(ctx.exception))

    def test_a_ratio_that_does_not_divide_the_block_is_refused(self):
        with self.assertRaises(NotImplementedError) as ctx:
            spec = _row_granular_spec(block_size=80, tokens_per_state=5)
            _parse([_group(spec)], self.MBS)
        self.assertIn("tokens_per_state=5", str(ctx.exception))
        self.assertIn("not divisible", str(ctx.exception))

    def test_a_large_ratio_needs_a_manager_block_that_holds_it(self):
        # 128 divides a 256-token block but not a 528-token manager block; the
        # same ratio is fine once the manager block is a multiple of it.
        self.assertEqual("refuse", self._verdict(block_size=256, tps=128))
        self.assertEqual("accept", self._verdict(block_size=256, tps=128, mbs=512))

    def test_a_fractional_ratio_is_refused_by_name(self):
        from fractions import Fraction

        spec = _row_granular_spec(tokens_per_state=Fraction(1, 8))
        with self.assertRaises(NotImplementedError) as ctx:
            _parse([_group(spec)], self.MBS)
        message = str(ctx.exception)
        self.assertIn("tokens_per_state=1/8", message)
        self.assertIn("whole number", message)
        self.assertIn("cache_dtype_str=", message)


class TestReplaySkipIsExactType(unittest.TestCase):
    """The replay skip must not swallow an undeclared subclass.

    A subclass may add layout or retention state the base replay does not
    regenerate, so it must reach the attention gate and be refused by name
    (the same fail-closed policy the receipt branches follow) instead of
    being skipped wholesale as a replayed window.
    """

    def test_a_subclass_is_refused_not_skipped(self):
        from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

        class ReplayedSWASub(SlidingWindowMLASpec):
            pass

        spec = ReplayedSWASub(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            sliding_window=128,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            bounded_replay=True,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            _parse([_group(spec)], 64)
        self.assertIn("ReplayedSWASub", str(ctx.exception))
        self.assertIn("bounded_replay", str(ctx.exception))

    def test_the_base_class_is_still_skipped(self):
        from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

        spec = SlidingWindowMLASpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            sliding_window=128,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            bounded_replay=True,
        )
        # A lone skipped group trips the aggregate guard ("all groups
        # skipped?"), so a plain attention group rides along and only the
        # replayed one disappears.
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        plain = FullAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            sliding_window=None,
        )
        metas = _parse([_group(spec), _group(plain, ("l1",))], 64)
        self.assertEqual([["l1"]], [meta.layer_names for meta in metas])


class TestV41TinyGolden(unittest.TestCase):
    """The real V4.1-Flash group structure: what survives parse_groups.

    Config mother: ``data/pin/sa-b/v41_flash_config.json`` (40 layers, head 512,
    rope 64, index 128/topk 512, sw 128, compress_ratios [0,0, 2x18, 1x20,
    0,0,0], kv_source_layer_ids [2, 8, 14, 20]); measured on 0.31/SM90 in
    ``out_b1_groups_sm90.txt`` (9 groups) and ``out_b1_register_sm90_blhnc.txt``
    (bucket sizes). This module rebuilds the *tiny* shape of it: the seven
    SWA-MLA groups, the packed 8-layer group and the compressor rings, with the
    same field values, and pins the connector's view.
    """

    BLOCK = 64  # SM90 kernel block for the compressed groups
    SWA_BLOCK = 32  # the SWA-MLA page/block size
    MBS = 256
    #: gemma-style stride split of the 40 SWA layers over seven groups.
    SWA_GROUPS: Tuple[int, ...] = (6, 6, 6, 6, 6, 5, 5)
    #: The measured bucket sizes at mbs 256 (out_b1_register_sm90_blhnc.txt).
    EXPECTED_BUCKETS = (
        ("tp0_g7", 1, 37376, 149504),
        ("tp0_g7_b1", 3, 18688, 224256),
        ("tp0_g7_b2", 1, 8448, 33792),
        ("tp0_g7_b3", 3, 4224, 50688),
    )

    def _specs(self) -> Any:
        from vllm.v1.kv_cache_interface import (
            CircularBufferSpec,
            MLAAttentionSpec,
            SlidingWindowMLASpec,
            UniformTypeKVCacheSpecs,
        )

        def swa(layer: int) -> Any:
            return SlidingWindowMLASpec(
                block_size=self.SWA_BLOCK,
                num_kv_heads=1,
                head_size=512,
                dtype="uint8",
                sliding_window=128,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                bounded_replay=True,
            )

        def mla(*, head: int, tps: int, block: int) -> Any:
            return MLAAttentionSpec(
                block_size=block,
                num_kv_heads=1,
                head_size=head,
                dtype="uint8",
                cache_dtype_str="fp8_ds_mla" if head == 512 else None,
                model_version="deepseek_v4" if head == 512 else None,
                kv_quant_mode=_quant("FP8_PER_TENSOR" if head == 512 else "NONE"),
                tokens_per_state=tps,
                state_content_bytes=584 if head == 512 else None,
                alignment=576 if head == 512 else None,
            )

        def ring() -> Any:
            return CircularBufferSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=1024,
                head_size_v=0,
                dtype="bfloat16",
            )

        groups = []
        index = 0
        for size in self.SWA_GROUPS:
            members = {f"swa_{index + i}": swa(index + i) for i in range(size)}
            groups.append(
                SimpleNamespace(
                    layer_names=list(members),
                    kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(members),
                    is_eagle_group=False,
                )
            )
            index += size
        # The four kv-source layers: 2, 8, 14 compressed (tps 2, one page per
        # 64-token entry) and 20 uncompressed (tps 1, 64 rows), each with its
        # indexer K cache.
        packed = {}
        for layer in (2, 8, 14):
            packed[f"model.layers.{layer}.attn"] = mla(
                head=512, tps=2, block=self.BLOCK
            )
            packed[f"model.layers.{layer}.attn.indexer.k_cache"] = mla(
                head=132, tps=2, block=self.BLOCK
            )
        packed["model.layers.20.attn"] = mla(head=512, tps=1, block=self.BLOCK)
        packed["model.layers.20.attn.indexer.k_cache"] = mla(
            head=132, tps=1, block=self.BLOCK
        )
        groups.append(
            SimpleNamespace(
                layer_names=list(packed),
                kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(packed),
                is_eagle_group=False,
            )
        )
        rings = {
            f"model.layers.{layer}.compressor.state_cache": ring()
            for layer in (2, 8, 14)
        }
        groups.append(
            SimpleNamespace(
                layer_names=list(rings),
                kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(rings),
                is_eagle_group=False,
            )
        )
        return groups

    def setUp(self) -> None:
        self.groups = self._specs()
        self.metas = _parse(self.groups, self.MBS)

    def test_the_nine_groups_fold_into_four_buckets(self):
        self.assertEqual(9, len(self.groups))
        self.assertEqual(
            [self.EXPECTED_BUCKETS[i][0] for i in range(len(self.EXPECTED_BUCKETS))],
            [spec_name(0, meta) for meta in self.metas],
        )

    def test_the_skipped_groups_keep_their_numbers(self):
        # Groups 0..6 (SWA-MLA, replayed) and 8 (compressor rings) hold no
        # prefix data: they are absent from the wire names, and the surviving
        # group keeps its vLLM index (7).
        self.assertEqual([7, 7, 7, 7], [meta.group_idx for meta in self.metas])
        self.assertEqual(
            ["tp0_g7", "tp0_g7_b1", "tp0_g7_b2", "tp0_g7_b3"],
            [spec_name(0, meta) for meta in self.metas],
        )

    def test_every_bucket_matches_the_measured_registration(self):
        for (name, layers, page_bytes, per_block), meta in zip(
            self.EXPECTED_BUCKETS, self.metas
        ):
            with self.subTest(bucket=name):
                self.assertEqual(layers, len(meta.layer_names))
                self.assertEqual(page_bytes, meta.page_bytes)
                self.assertEqual(per_block, meta.per_block_bytes)
                # 4 buckets of one block table: all share the group's block size.
                self.assertEqual(self.BLOCK, meta.block_size)

    def test_the_row_granular_facts_of_each_bucket(self):
        by_name = {spec_name(0, meta): meta for meta in self.metas}
        # tps=1 buckets: one row per token, so no row metadata at all.
        self.assertIsNone(by_name["tp0_g7"].row_granular)
        self.assertIsNone(by_name["tp0_g7_b2"].row_granular)
        # tps=2 buckets: 32 rows of 584 B (main) / 132 B (indexer) per page.
        self.assertEqual(RowGranularMeta(2, 32, 584), by_name["tp0_g7_b1"].row_granular)
        self.assertEqual(RowGranularMeta(2, 32, 132), by_name["tp0_g7_b3"].row_granular)
        # One manager block is 128 rows for them, 256 for the tps=1 pair --
        # which is why the per-block sizes differ by exactly 2x for one layer.
        self.assertEqual(584 * 128 * 3, by_name["tp0_g7_b1"].per_block_bytes)
        self.assertEqual(584 * 256 * 1, by_name["tp0_g7"].per_block_bytes)

    def test_the_bucket_set_is_a_pure_function_of_the_specs(self):
        # Registration order cannot move the names: the buckets are ordered by
        # page size (and geometry), not by layer order.
        shuffled = list(reversed(self.groups))
        for group in shuffled:
            group.layer_names = list(reversed(group.layer_names))
        metas = _parse(shuffled, self.MBS)
        self.assertEqual(
            [meta.spec_suffix for meta in metas],
            [meta.spec_suffix for meta in self.metas],
        )
        self.assertEqual(
            [meta.per_block_bytes for meta in metas],
            [meta.per_block_bytes for meta in self.metas],
        )


class TestBucketKeyKeepsRowGranularityApart(unittest.TestCase):
    """Identical bytes, different address math: the key must keep them apart.

    The M6 coincidence in its purest form: a 32768-byte page, 512-wide head,
    uint8, same class and block size -- and a different row granularity (64
    rows of 512 B vs 32 rows of 1024 B). Everything the pre-PR6 key looked at
    agrees, so without the tokens_per_state dimension the two layers would
    silently share one bucket (and one of them would be transferred with the
    other's address math).
    """

    def _pair(self) -> Tuple[Any, Any]:
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        token_granular = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            kv_quant_mode=_quant("NONE"),
            tokens_per_state=1,
        )
        compressed = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            kv_quant_mode=_quant("FP8_PER_TENSOR"),
            tokens_per_state=2,
            state_content_bytes=1024,
        )
        self.assertEqual(
            token_granular.real_page_size_bytes, compressed.real_page_size_bytes
        )
        return token_granular, compressed

    def test_the_pair_shares_every_key_component_but_the_row_size(self):
        from kv_cache_manager.py_connector.vllm.vllm_common import _bucket_key

        slow, fast = self._pair()
        slow_key = _bucket_key(slow, "group 0")
        fast_key = _bucket_key(fast, "group 0")
        self.assertNotEqual(slow_key, fast_key)
        # Everything but the row granularity agrees -- including
        # cache_dtype_str, which is deliberately *not* part of the key (the
        # page and the head geometry already describe the bytes; a
        # dtype-string-only difference would be one layout under two names).
        self.assertEqual(
            slow_key[:8],
            fast_key[:8],
            "the pre-PR6 key components must all agree for this pair",
        )
        self.assertEqual((1, 2), (slow_key.tokens_per_state, fast_key.tokens_per_state))

    def test_the_pair_becomes_two_buckets_of_one_group(self):
        # The data-plane consequence: one wrapper holding both layouts yields
        # two buckets, each sized and addressed by its own row granularity.
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

        slow, fast = self._pair()
        wrapper = UniformTypeKVCacheSpecs.from_specs({"l_slow": slow, "l_fast": fast})
        metas = _parse([_group(wrapper, ("l_slow", "l_fast"))], 256)
        self.assertEqual([meta.spec_suffix for meta in metas], ["", "_b1"])
        by_ratio = {
            (1 if m.row_granular is None else m.row_granular.tokens_per_row): m
            for m in metas
        }
        self.assertIsNone(by_ratio[1].row_granular)
        self.assertEqual(by_ratio[2].row_granular, RowGranularMeta(2, 32, 1024))
        # 64 rows per manager block vs 32: the same page, half the rows.
        self.assertEqual(by_ratio[1].per_block_bytes, 512 * 256)
        self.assertEqual(by_ratio[2].per_block_bytes, 1024 * 128)


class TestCoverageSurvivesFolding(unittest.TestCase):
    """The folding interaction: one vLLM group, one coverage.

    The engine hands the scheduler one member per packed group
    (generate_scheduler_kv_cache_config), so the scheduler's window / state masks
    come from that member. Buckets that agree fold harmlessly -- the extra
    buckets keep their registration *and* their announcement name (the group's
    member list is the worker's registration) -- while buckets that disagree
    would mask the wrong blocks, which is what this pins.
    """

    def _metas(self, *, packed: bool, windowed: bool) -> List[GroupMeta]:
        metas: List[GroupMeta] = []
        if packed:
            metas.extend(
                [
                    AttentionGroupMeta(
                        group_idx=0,
                        layer_names=["main"],
                        block_size=64,
                        per_block_bytes=16384,
                        page_bytes=16384,
                        row_granular=RowGranularMeta(2, 32, 584),
                        spec_suffix="",
                    ),
                    AttentionGroupMeta(
                        group_idx=0,
                        layer_names=["indexer"],
                        block_size=64,
                        per_block_bytes=4096,
                        page_bytes=4096,
                        row_granular=RowGranularMeta(2, 32, 132),
                        spec_suffix="_b1",
                    ),
                ]
            )
        else:
            metas.append(
                AttentionGroupMeta(
                    group_idx=0,
                    layer_names=["main"],
                    block_size=64,
                    per_block_bytes=16384,
                    page_bytes=16384,
                    row_granular=RowGranularMeta(2, 32, 584),
                    spec_suffix="",
                )
            )
        if windowed:
            metas.append(
                AttentionGroupMeta(
                    group_idx=1,
                    layer_names=["swa"],
                    block_size=16,
                    per_block_bytes=65536,
                    page_bytes=65536,
                    retention=WindowRetention(kind="sliding", window_tokens=1024),
                    spec_suffix="",
                )
            )
        return metas

    def test_a_packed_group_with_an_independent_windowed_group_is_announceable(
        self,
    ):
        # The usable combination (measured end to end: both buckets of the
        # packed group are registered *and* named, the windowed group keeps its
        # own name, and every block round-trips bit-exact). The declaration
        # names both packed buckets for the coverage that has the window.
        groups = build_spec_groups(self._metas(packed=True, windowed=True), tp_size=1)
        names = {group["name"]: group["spec_names"] for group in groups}
        self.assertEqual({"tp0_g0", "tp0_g0_b1", "tp0_g1"}, set(names[ALL_SPEC_GROUP]))

    def test_two_buckets_of_one_group_may_not_disagree_about_the_window(self):
        # The combination the guard exists for: a group whose buckets fold to
        # one member would let the scheduler mask blocks by a window the *other*
        # bucket does not have (or miss a window it cannot see), publishing
        # stale / zero-page bytes as cacheable.
        metas = self._metas(packed=True, windowed=False)
        metas[-1] = AttentionGroupMeta(
            group_idx=metas[-1].group_idx,
            layer_names=["rswa"],
            block_size=metas[-1].block_size,
            per_block_bytes=metas[-1].per_block_bytes,
            page_bytes=metas[-1].page_bytes,
            row_granular=metas[-1].row_granular,
            spec_suffix="_b2",
            retention=WindowRetention(kind="sliding", window_tokens=128),
        )
        with self.assertRaises(NotImplementedError) as ctx:
            build_spec_groups(metas, tp_size=1)
        message = str(ctx.exception)
        self.assertIn("group 0", message)
        self.assertIn("different coverage", message)
        self.assertIn("'rswa'", message)

    def test_two_buckets_of_one_group_may_not_carry_different_windows(self):
        # Same dimension, different scope: the folded member would supply the
        # *narrower* window to the mask, so blocks whose wider window is gone
        # would still be published.
        metas = self._metas(packed=True, windowed=False)
        for suffix, window in (("_b1", 1024), ("_b2", 4096)):
            metas.append(
                AttentionGroupMeta(
                    group_idx=0,
                    layer_names=[f"swa{window}"],
                    block_size=16,
                    per_block_bytes=65536,
                    page_bytes=65536,
                    retention=WindowRetention(kind="sliding", window_tokens=window),
                    spec_suffix=suffix,
                )
            )
        with self.assertRaises(NotImplementedError) as ctx:
            build_spec_groups(metas, tp_size=1)
        self.assertIn("different coverage", str(ctx.exception))

    def test_a_group_cannot_mix_a_state_bucket_with_an_attention_bucket(self):
        # The defensive half of the predicate: vLLM's uniform bases keep mamba
        # specs in a group of their own today, so this cannot be built from a
        # real KVCacheConfig -- but the fold would lose the state dimension all
        # the same, and the guard is total on purpose.
        metas: List[GroupMeta] = self._metas(packed=False, windowed=False)
        metas.append(
            StateGroupMeta(
                group_idx=0,
                layer_names=["mamba"],
                block_size=16,
                per_block_bytes=2048,
                page_size_bytes=2048,
            )
        )
        with self.assertRaises(NotImplementedError) as ctx:
            build_spec_groups(metas, tp_size=1)
        self.assertIn("different coverage", str(ctx.exception))

    def test_a_packed_group_without_sparse_coverage_declares_nothing(self):
        # No declared groups means the announcement carries no names and the
        # manager stores every registered spec -- which is exactly what a
        # coverage-free packed group does (V4.1 runs this path).
        self.assertEqual(
            [], build_spec_groups(self._metas(packed=True, windowed=False), tp_size=1)
        )

    def test_a_single_bucket_group_with_sparse_coverage_is_announceable(self):
        # With one bucket per group there is nothing to fold away: every
        # windowed / hybrid model that works today takes this path.
        groups = build_spec_groups(self._metas(packed=False, windowed=True), tp_size=1)
        names = {group["name"]: group["spec_names"] for group in groups}
        self.assertIn(ALL_SPEC_GROUP, names)
        self.assertEqual({"tp0_g0", "tp0_g1"}, set(names[ALL_SPEC_GROUP]))


if __name__ == "__main__":
    unittest.main()
