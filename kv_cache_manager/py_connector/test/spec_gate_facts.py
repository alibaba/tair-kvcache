"""Spec gate facts: one row per vLLM KV cache spec class, plus the era map.

This is the shared table behind ``test_spec_gate.py`` (the vllm_stubs stand-in,
runs everywhere) and ``test_spec_gate_real.py`` (the real vLLM of whatever era
the environment carries). Each row is one *construction point* -- the field
values a real model produces, taken from the survey experiments
(``~/.wing/tasks/swa-dsa-support/data/pin/sa-c/c1_gate_matrix.py``,
``data/exp_gate_v41.py``), never invented ones -- together with the disposition
the connector must give it:

``accept``  the group transfers (full prefix / latent rows);
``refuse``  parse_groups raises NotImplementedError (message keywords pinned);
``skip``    parse_groups drops the group (per-request scratch) and the rest of
            the instance still parses.

``legacy`` is what the pre-PR gate (7b55b853) did with the same spec, so the
differential ("today's accepted surface must stay accepted") is data, and
``legacy_accepts`` below is the executable transcription of that predicate. The
only rows allowed to go from ``legacy=accept`` to ``refuse`` are the documented
holes, named by ``hole``: the RSWA window (``rswa``), the tokens-per-state field
migration (``tokens_per_state``) -- which also covers the pooled/compressed
full-attention rows below -- and, for undeclared subclasses, the tests that
build them.

``requires`` lists constructor fields a row *needs* to be meaningful: a row
whose fields the running era lacks is skipped (the era map below says why), so
the same table serves vLLM 0.26 (``compress_ratio``) and 0.31
(``tokens_per_state``) without pretending one shape is the other.
"""

import inspect
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

MIB = 2**20

#: The 19 KVCacheSpec classes of vLLM 0.31 and what the parser does with each.
#: ``test_spec_gate_real`` fails when the installed vLLM exposes a class that is
#: missing here -- the mechanism against the next RSWA-style surprise.
#: SlidingWindowMLASpec is the one class whose disposition depends on a policy
#: field rather than the class alone: the replayed form is skipped, the
#: non-replayed one refused -- never transferred either way (PR6).
SPEC_DISPOSITIONS: Dict[str, str] = {
    "FullAttentionSpec": "accept",
    "TQFullAttentionSpec": "accept",
    "SinkFullAttentionSpec": "accept",
    "MLAAttentionSpec": "accept",
    "HiddenStateCacheSpec": "accept",
    "RSWASpec": "accept",
    "SlidingWindowSpec": "accept",
    "ChunkedLocalAttentionSpec": "accept",
    "SlidingWindowMLASpec": "skip_or_refuse",
    "CircularBufferSpec": "skip",
    "KpoolTailSpec": "skip",
    "HiSparseHotSpec": "skip",
    "HiSparseResidentSpec": "skip",
    "CrossAttentionSpec": "refuse",
    "EncoderOnlyAttentionSpec": "refuse",
    "MambaSpec": "state",
    "UniformTypeKVCacheSpecs": "split",
    "AttentionSpec": "family",  # never a group's spec on its own
    "KVCacheSpec": "family",  # the root, likewise
}

#: Classes no ZOO row covers, with where their parse-level disposition is
#: pinned instead: MambaSpec becomes a StateGroupMeta (parse_groups, covered by
#: test_scheduler_state's hybrid cases) and the two roots never appear as a
#: group spec at all.
NOT_IN_ZOO: Tuple[str, ...] = ("MambaSpec", "AttentionSpec", "KVCacheSpec")

#: Which vLLM eras ship which spec class (read off
#: ``vllm/v1/kv_cache_interface.py`` in every era under ``envs/``). Only the
#: listed versions are asserted; an era outside the table skips the check
#: instead of guessing.
_ALL_ERAS = (
    "0.14.0",
    "0.22.1",
    "0.23.0",
    "0.26.0",
    "0.27.0",
    "0.28.0",
    "0.29.0",
    "0.30.0",
    "0.31.0",
)
_FROM_0_22 = _ALL_ERAS[1:]
_FROM_0_26 = _ALL_ERAS[3:]
_FROM_0_29 = _ALL_ERAS[6:]
SPEC_CLASS_ERAS: Dict[str, Tuple[str, ...]] = {
    # The full-attention family and the always-present zoo.
    "FullAttentionSpec": _ALL_ERAS,
    "SinkFullAttentionSpec": _ALL_ERAS,
    "MLAAttentionSpec": _ALL_ERAS,
    "SlidingWindowSpec": _ALL_ERAS,
    "ChunkedLocalAttentionSpec": _ALL_ERAS,
    "CrossAttentionSpec": _ALL_ERAS,
    "EncoderOnlyAttentionSpec": _ALL_ERAS,
    "MambaSpec": _ALL_ERAS,
    "UniformTypeKVCacheSpecs": _ALL_ERAS,
    "AttentionSpec": _ALL_ERAS,
    "KVCacheSpec": _ALL_ERAS,
    # Arrived (and, for TQ, left again) along the way.
    "TQFullAttentionSpec": ("0.22.1", "0.23.0", "0.26.0", "0.27.0"),
    "HiddenStateCacheSpec": _FROM_0_22,
    "SlidingWindowMLASpec": _FROM_0_22,
    "RSWASpec": _FROM_0_26,
    "CircularBufferSpec": _FROM_0_29,
    "KpoolTailSpec": ("0.30.0", "0.31.0"),
    "HiSparseHotSpec": ("0.30.0", "0.31.0"),
    "HiSparseResidentSpec": ("0.30.0", "0.31.0"),
}


def expected_classes(version: str) -> Optional[set]:
    """The spec classes a *known* vLLM version ships (None = unknown era)."""
    if not any(version in eras for eras in SPEC_CLASS_ERAS.values()):
        return None
    return {name for name, eras in SPEC_CLASS_ERAS.items() if version in eras}


#: Measured capability flags of the scratch families per era: (prefix_cacheable,
#: uses_slot_mapping, has_layer_views), None = the era has no such attribute.
#: The skip rests on the era *declaring* both halves (see _is_scratch_spec), so
#: vLLM 0.29 (ring specs: no slot-mapping flag) and 0.30 (KpoolTailSpec keeps
#: the slot mapping) refuse those groups -- their pre-PR behaviour, fail-closed.
SCRATCH_FLAGS: Dict[str, Dict[str, Tuple[Any, Any, Any]]] = {
    "0.29.0": {
        "CircularBufferSpec": (False, None, None),
    },
    "0.30.0": {
        "CircularBufferSpec": (False, False, True),
        "KpoolTailSpec": (False, True, True),
        "HiSparseHotSpec": (False, False, False),
        "HiSparseResidentSpec": (False, True, False),
    },
    "0.31.0": {
        "CircularBufferSpec": (False, False, True),
        "KpoolTailSpec": (False, False, True),
        "HiSparseHotSpec": (False, False, False),
        "HiSparseResidentSpec": (False, True, False),
    },
}

#: PR6's declared relaxations of the PR1 table, as ``row -> (PR1's verdict,
#: this PR's verdict)``. The compressed main/indexer rows are *accepted* (one
#: row per 2 or 4 tokens, addressed in rows -- see vllm_common.RowGranularMeta),
#: the packed V4.1 group with them, and the engine-replayed SWA-MLA window is
#: *skipped* (vLLM rebuilds those rows on a hit, so the group holds no prefix
#: data to serve). Everything else in the tps/replay family stays refused: the
#: fractional pooling ratio (its row unit is smaller than a token) and the
#: non-replayed windowed MLA (it needs the windowed retention contract,
#: deferred). ``test_spec_gate`` asserts that these four are exactly the rows
#: that left the refuse set.
PR6_DECISIONS: Dict[str, Tuple[str, str]] = {
    "mla_compressed_0_26": ("refuse", "accept"),
    "mla_compressed_0_31": ("refuse", "accept"),
    "wrapper_v41_packed": ("refuse", "accept"),
    "swa_mla_bounded_replay": ("refuse", "skip"),
}

#: The ZOO fields that put a row in the tps / replay family (see
#: PR6_DECISIONS): used to assert that *only* the declared rows left the refuse
#: set, so a future row cannot be relaxed by accident.
TPS_OR_REPLAY_FIELDS = ("tokens_per_state", "compress_ratio", "bounded_replay")


def in_tps_or_replay_family(row: Dict[str, Any]) -> bool:
    """Is this row (or any of its members, for a wrapper row) in the family
    whose rows PR6 relaxes? See PR6_DECISIONS."""
    if set(row.get("kwargs", {})) & set(TPS_OR_REPLAY_FIELDS):
        return True
    return any(
        in_tps_or_replay_family(ROW_BY_NAME[member])
        for member in row.get("members", {}).values()
    )


#: Eras whose KVCacheConfig declares the transfer projection (KVCacheGroupSpec.
#: enable_kv_transfer / transfer_group_ids / prefix_cacheable_group_ids). The
#: enable_kv_transfer / transfer_group_ids arrived in vLLM 0.29; the
#: prefix_cacheable_group_ids subset (the codex comment's second contract) in
#: 0.30. Measured per tree: 0.14-0.28 carry neither, 0.29-0.31 both here.
TRANSFER_PROJECTION_ERAS: Tuple[str, ...] = ("0.29.0", "0.30.0", "0.31.0")


def declares_transfer_projection(engine: "Engine") -> bool:
    """Does this era tell connectors which groups are theirs to transfer?"""
    return engine.version in TRANSFER_PROJECTION_ERAS


def _is_compression_ring_row(row: Dict[str, Any]) -> bool:
    """Is every spec this row builds a compression ring (CircularBufferSpec)?"""
    return set(needed_classes(row)) == {"CircularBufferSpec"}


def skip_eras(row: Dict[str, Any]) -> Tuple[str, ...]:
    """The eras whose spec declares the whole scratch contract for this row."""
    if row["cls"] == "UniformTypeKVCacheSpecs":
        member = next(iter(_members(row).values()))
        return skip_eras(member)
    eras = SCRATCH_FLAGS.keys()
    return tuple(
        version
        for version in eras
        if (flags := SCRATCH_FLAGS[version].get(row["cls"])) is not None
        and flags[0] is False
        and (flags[1] is False or flags[2] is False)
    )


def expected_outcome(row: Dict[str, Any], engine: "Engine") -> str:
    """The row's disposition *in this era*.

    Two families answer differently. The scratch rows are era-dependent, for
    two reasons: the era's own spec must declare the scratch contract (see
    skip_eras), and the group must not be exposed to connectors -- an era
    that still claims it, or a group the engine hands to connectors
    (transferable, the vLLM default: parse_groups refuses it instead of
    silently dropping it), is refused rather than dropped. Two exceptions
    keep skipping: the engine-replayed window (``skip_kind`` ``replay``, a
    policy field -- wherever the flag exists, vLLM rebuilds the rows), and
    the compression ring, which the engine rebuilds per request.
    """
    if row["disposition"] != "skip":
        return row["disposition"]
    if row.get("skip_kind") == "replay":
        return "skip"
    if engine.version not in skip_eras(row):
        return "refuse"
    if declares_transfer_projection(engine) and row.get("transferable", True):
        # A transferable scratch group is refused -- unless the engine itself
        # rebuilds it per request. The compression ring is the proven case
        # (CircularBufferSpec: "one block per request holding the raw keys of
        # the token group that is still being compressed", refilled by the
        # compression machinery), including V4.1's UniformType pack of rings.
        if not _is_compression_ring_row(row):
            return "refuse"
    return "skip"



# --------------------------------------------------------------------------- #
# The zoo: one row per spec class, at a real construction point.
#
# ``dtype`` and ``kv_quant_mode`` values are *names*: the engine adapter maps
# them (the real vLLM needs torch dtypes / KVQuantMode, the stand-in also takes
# names). ``mbs`` is the manager block size the row is parsed with;
# ``page_bytes`` / ``per_block_bytes`` are the connector's sizing of that
# one-layer group (None = not pinned). ``companion`` parses the row next to a
# plain full-attention group, which is how a skipped group really occurs.
# --------------------------------------------------------------------------- #
ZOO: List[Dict[str, Any]] = [
    dict(
        name="full",
        cls="FullAttentionSpec",
        # gemma-3 attention group geometry (4 kv heads x 256).
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            head_size_v=256,
            dtype="bfloat16",
            kv_quant_mode="NONE",
        ),
        disposition="accept",
        legacy="accept",
        mbs=16,
        page_bytes=65536,
        per_block_bytes=65536,
        source="gemma-3-4b attention group; the connector's baseline",
    ),
    dict(
        name="full_window",
        cls="FullAttentionSpec",
        # kv_cache_utils._promote_local_kv_cache_specs output
        # (--disable-hybrid-kv-cache-manager): the allocation covers all tokens
        # and the window only narrows the compute -> full prefix.
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            head_size_v=256,
            dtype="bfloat16",
            kv_quant_mode="NONE",
            sliding_window=1024,
        ),
        disposition="accept",
        legacy="refuse",
        mbs=16,
        page_bytes=65536,
        per_block_bytes=65536,
        source="promoted gemma-3 SWA layer (kv_cache_utils.py:1887)",
    ),
    dict(
        name="full_chunked",
        cls="FullAttentionSpec",
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            head_size_v=256,
            dtype="bfloat16",
            kv_quant_mode="NONE",
            attention_chunk_size=8192,
        ),
        disposition="accept",
        legacy="refuse",
        mbs=16,
        page_bytes=65536,
        per_block_bytes=65536,
        source="promoted llama4 chunked-local layer (same promote path)",
    ),
    dict(
        name="full_pooled",
        cls="FullAttentionSpec",
        # Whisper block pooling: every token row holds several states, so the
        # one-row-per-token transfer model would mis-size the page (the gate
        # refuses it by field name; registration used to fail much later).
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            dtype="bfloat16",
            tokens_per_state=Fraction(1, 8),
        ),
        disposition="refuse",
        legacy="accept",
        hole="tokens_per_state",
        requires=("tokens_per_state",),
        keywords=("tokens_per_state=1/8", "one state per token"),
        mbs=16,
        source="whisper_causal block pooling (block_pool_size)",
    ),
    dict(
        name="full_compressed",
        cls="FullAttentionSpec",
        # The same field on a plain full-attention spec (a state spanning
        # several tokens), reachable through the hybrid-manager fallbacks.
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            dtype="bfloat16",
            tokens_per_state=2,
        ),
        disposition="refuse",
        legacy="accept",
        hole="tokens_per_state",
        requires=("tokens_per_state",),
        keywords=("tokens_per_state=2", "one state per token"),
        mbs=16,
        source="the field's other direction on an accepted class",
    ),
    dict(
        name="mla_bf16",
        cls="MLAAttentionSpec",
        # GLM-4.7-Flash: one bf16 latent row per token.
        kwargs=dict(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype="bfloat16",
            cache_dtype_str="auto",
        ),
        disposition="accept",
        legacy="accept",
        mbs=64,
        page_bytes=18432,
        per_block_bytes=73728,
        source="GLM-4.7-Flash / FROZEN m0",
    ),
    dict(
        name="mla_fp8_ds",
        cls="MLAAttentionSpec",
        # DeepSeek V3.2 main: 656 B/token packed (NoPE + RoPE + scale). 0.26
        # derives that from its fp8_ds_mla branch; 0.31 states it as
        # state_content_bytes (mla_attention.py:1361) -- same page either way.
        kwargs=dict(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            dtype="uint8",
            cache_dtype_str="fp8_ds_mla",
            kv_quant_mode="FP8_PER_TENSOR",
            state_content_bytes=656,
        ),
        disposition="accept",
        legacy="accept",
        mbs=64,
        page_bytes=41984,
        per_block_bytes=41984,
        source="V3.2 main MLA / FROZEN m1 + vllm 0.31 get_kv_cache_spec",
    ),
    dict(
        name="mla_indexer",
        cls="MLAAttentionSpec",
        # The V3.2 indexer: 132 B/row, no alignment, plain uint8 elements.
        kwargs=dict(
            block_size=64,
            num_kv_heads=1,
            head_size=132,
            dtype="uint8",
        ),
        disposition="accept",
        legacy="accept",
        mbs=64,
        page_bytes=8448,
        per_block_bytes=8448,
        source="V3.2 indexer / FROZEN m2c2",
    ),
    dict(
        name="mla_compressed_0_26",
        cls="MLAAttentionSpec",
        # V4 c4a bf16 (the vllm <= 0.28 spelling: compress_ratio). One row per
        # 4 tokens, 1024 B/row (512 bf16 x 2 = 1024 B, K and V halves in one
        # row); 64 rows in the 256-token page.
        kwargs=dict(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype="bfloat16",
            cache_dtype_str="auto",
            compress_ratio=4,
            alignment=512,
            model_version="deepseek_v4",
        ),
        disposition="accept",
        legacy="refuse",
        requires=("compress_ratio",),
        row_granular=dict(tokens_per_row=4, rows_per_page=64, bytes_per_row=1024),
        mbs=256,
        page_bytes=65536,
        per_block_bytes=65536,  # 1024 B/row x (256 // 4) rows
        source="FROZEN m2a",
    ),
    dict(
        name="mla_compressed_0_31",
        cls="MLAAttentionSpec",
        # DeepSeek-V4.1-Flash compressed main row (vllm >= 0.29 spelling:
        # tokens_per_state): 1 row = 2 tokens, 584 B/row, 128 rows/page. Before
        # PR1 the gate read compress_ratio (absent here -> 1) and accepted it,
        # sizing 74752 B/block at mbs 256; PR1 refused it (the silent-acceptance
        # hole), PR6 accepts it *with* the row-granular mapping -- the block
        # size was never wrong, the address math was (see test_dsa_v41).
        kwargs=dict(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            tokens_per_state=2,
            cache_dtype_str="fp8_ds_mla",
            alignment=576,
            state_content_bytes=584,
            model_version="deepseek_v4",
            kv_quant_mode="FP8_PER_TENSOR",
        ),
        disposition="accept",
        legacy="accept",
        requires=("tokens_per_state",),
        row_granular=dict(tokens_per_row=2, rows_per_page=128, bytes_per_row=584),
        mbs=256,
        page_bytes=74752,
        per_block_bytes=74752,  # 584 B/row x (256 // 2) rows
        source="V4.1 run_gate experiment (swa-dsa-support data/exp_gate_v41.py)",
    ),
    dict(
        name="mla_pooled_fraction",
        cls="MLAAttentionSpec",
        # Whisper block pooling spells the same field as a Fraction < 1: more
        # states than tokens, so the row unit is smaller than a token -- the
        # row-granular model cannot address it either.
        kwargs=dict(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype="bfloat16",
            cache_dtype_str="auto",
            tokens_per_state=Fraction(1, 8),
        ),
        disposition="refuse",
        legacy="accept",
        hole="tokens_per_state",
        requires=("tokens_per_state",),
        keywords=("tokens_per_state=1/8", "whole number", "cache_dtype_str"),
        mbs=64,
        source="whisper_causal.py:408 (Fraction(1, block_pool_size))",
    ),
    dict(
        name="mla_tps_not_dividing_mbs",
        cls="MLAAttentionSpec",
        # The other arithmetic precondition (mbs % tokens_per_state == 0): a
        # manager block must stay a whole number of rows. Synthetic ratio at
        # the shape of the PR6 table test (mbs 528 is vLLM's default merge
        # block size; 528 % 32 != 0 while the block itself divides).
        kwargs=dict(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            tokens_per_state=32,
            cache_dtype_str="fp8_ds_mla",
            state_content_bytes=584,
            model_version="deepseek_v4",
            kv_quant_mode="FP8_PER_TENSOR",
        ),
        disposition="refuse",
        legacy="accept",
        hole="tokens_per_state",
        requires=("tokens_per_state",),
        keywords=("manager block size 528", "tokens_per_state=32"),
        mbs=528,
        source="mbs divisibility precondition (vllm_common._attention_meta)",
    ),
    dict(
        name="mla_nvfp4",
        cls="MLAAttentionSpec",
        # The SM100-only NVFP4 record: an in-row scale plus a packed cell, 288 B
        # per row for V4.1. Named refusal (the pre-PR6 wording fell through to
        # "unknown cache_dtype_str", which reads as a typo rather than a
        # layout the connector deliberately does not size).
        kwargs=dict(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            tokens_per_state=2,
            cache_dtype_str="nvfp4_ds_mla",
            state_content_bytes=288,
            model_version="deepseek_v4",
        ),
        disposition="refuse",
        legacy="refuse",
        requires=("tokens_per_state",),
        keywords=(
            "nvfp4_ds_mla",
            "288",
            "tokens_per_state=2",
            "TairKvCacheConnector",
        ),
        mbs=64,
        source="deepseek_v41/attention.py:505-512 (SM100 NVFP4 record)",
    ),
    dict(
        name="hidden_state",
        cls="HiddenStateCacheSpec",
        # extract_hidden_states: an MLA-shaped cache of hidden states, 3 eagle
        # aux layers of a 2560-wide target model.
        kwargs=dict(
            block_size=16,
            num_kv_heads=3,
            head_size=2560,
            dtype="bfloat16",
        ),
        disposition="accept",
        legacy="accept",
        mbs=64,
        page_bytes=245760,
        per_block_bytes=983040,
        source="extract_hidden_states.py:316 (c1_gate_matrix row 2)",
    ),
    dict(
        name="sink",
        cls="SinkFullAttentionSpec",
        # OpenPangu-2.0-Flash geometry: head 128 NoPE + 64 RoPE, v head 128.
        kwargs=dict(
            block_size=16,
            num_kv_heads=8,
            head_size=192,
            head_size_v=128,
            dtype="bfloat16",
            sink_len=128,
            kv_quant_mode="NONE",
        ),
        disposition="accept",
        legacy="accept",
        mbs=16,
        page_bytes=81920,
        per_block_bytes=81920,
        source="static_sink_attention.py:217 (c1_gate_matrix row 3)",
    ),
    dict(
        name="tq_full",
        cls="TQFullAttentionSpec",
        # vllm 0.22 - 0.27: full attention whose page counts TQ slots; slot 0
        # keeps the ordinary head x dtype page.
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            dtype="bfloat16",
            kv_quant_mode="NONE",
            tq_slot_size=0,
        ),
        disposition="accept",
        legacy="accept",
        mbs=16,
        page_bytes=65536,
        per_block_bytes=65536,
        source="0.26 TQFullAttentionSpec (turboquant full attention)",
    ),
    dict(
        name="rswa",
        cls="RSWASpec",
        # DeepSeek-OCR-2 geometry (hidden 1280 / 10 heads) with the UnlimitedOCR
        # default window. The pre-PR gate accepted it as full attention:
        # rswa_window is not in the window-field whitelist.
        kwargs=dict(
            block_size=16,
            num_kv_heads=10,
            head_size=128,
            head_size_v=128,
            dtype="bfloat16",
            kv_quant_mode="NONE",
            rswa_window=128,
        ),
        disposition="accept",
        legacy="accept",
        # The pre-PR gate read the full-attention family and let the window
        # through; now the class itself decides and names the retention.
        retention=dict(kind="rswa_gap", window_tokens=128),
        mbs=16,
        page_bytes=81920,
        per_block_bytes=81920,
        source="deepseek_v2.py:208-221 + config.py:195-206 (c1 row 1)",
    ),
    dict(
        name="sliding_window",
        cls="SlidingWindowSpec",
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            dtype="bfloat16",
            kv_quant_mode="NONE",
            sliding_window=1024,
        ),
        disposition="accept",
        legacy="refuse",
        retention=dict(kind="sliding", window_tokens=1024),
        mbs=16,
        page_bytes=65536,
        per_block_bytes=65536,
        source="gemma-3 SWA layer (SlidingWindowSpec, win 1024)",
    ),
    dict(
        name="sliding_window_retained",
        cls="SlidingWindowSpec",
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            dtype="bfloat16",
            kv_quant_mode="NONE",
            sliding_window=1024,
            extra_retained_tokens=8,
        ),
        disposition="accept",
        legacy="refuse",
        requires=("extra_retained_tokens",),
        retention=dict(kind="sliding", window_tokens=1024, extra_retained_tokens=8),
        mbs=16,
        page_bytes=65536,
        per_block_bytes=65536,
        source="multi-module spec decoding trailing edge (vllm >= 0.29)",
    ),
    dict(
        name="chunked_local",
        cls="ChunkedLocalAttentionSpec",
        # Llama-4-Scout: chunk 8192, 8 kv heads, head 128.
        kwargs=dict(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype="bfloat16",
            kv_quant_mode="NONE",
            attention_chunk_size=8192,
        ),
        disposition="accept",
        legacy="refuse",
        retention=dict(kind="chunked_local", window_tokens=8192),
        mbs=16,
        page_bytes=65536,
        per_block_bytes=65536,
        source="llama4.py:251 (c1_gate_matrix row 10)",
    ),
    dict(
        name="swa_mla_bounded_replay",
        cls="SlidingWindowMLASpec",
        # V4/V4.1 SWA cache: 584 B rows with bounded replay ON (the vLLM 0.31
        # config default), so vLLM replays the window itself and rebuilds these
        # rows on a hit. Skipped while parsing (the group holds no reusable
        # prefix data), with the wire name absent -- same shape as an EAGLE or
        # scratch group.
        kwargs=dict(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            sliding_window=128,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            tokens_per_state=1,
            bounded_replay=True,
        ),
        disposition="skip",
        legacy="refuse",
        requires=("bounded_replay",),
        # One predicate (the replay policy), not the era-dependent scratch
        # contract: wherever the flag exists, the group is dropped.
        skip_kind="replay",
        companion=True,
        mbs=64,
        page_bytes=37376,
        source="V4.1 SWA-MLA group (c1_gate_matrix row 11)",
    ),
    dict(
        name="swa_mla_prefix_replay",
        cls="SlidingWindowMLASpec",
        # bounded_replay=False is prefix_cacheable -- still refused: vLLM does
        # not rebuild those rows, so they would need the windowed retention
        # contract, which the connector implements for the plain windowed
        # classes only (deferred, see the gate's message).
        kwargs=dict(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype="uint8",
            sliding_window=128,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            tokens_per_state=1,
            bounded_replay=False,
        ),
        disposition="refuse",
        legacy="refuse",
        requires=("bounded_replay",),
        keywords=("SlidingWindowMLASpec", "sliding_window=128", "bounded_replay=False"),
        mbs=64,
        page_bytes=37376,
        source="same spec with the replay policy off (M5: default is on)",
    ),
    dict(
        name="circular_buffer",
        cls="CircularBufferSpec",
        # The V4.1 compressor ring: one block per request, block_size is the
        # ring capacity (QSA shape: compress_ratio 128 with no speculation).
        kwargs=dict(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype="bfloat16",
        ),
        disposition="skip",
        legacy="refuse",
        companion=True,
        mbs=16,
        page_bytes=32768,
        source="qwen4_exp QSA / V4.1 compressor (c1_gate_matrix row 4)",
    ),
    dict(
        name="kpool_tail",
        cls="KpoolTailSpec",
        kwargs=dict(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            head_size_v=0,
            dtype="bfloat16",
            sliding_window=16,
        ),
        disposition="skip",
        legacy="refuse",
        companion=True,
        mbs=16,
        page_bytes=8192,
        source="glm5next index_kpool tail (c1_gate_matrix row 5)",
    ),
    dict(
        name="hisparse_hot",
        cls="HiSparseHotSpec",
        kwargs=dict(block_size=16, page_size=2 * MIB, blocks_per_request=64),
        disposition="skip",
        legacy="refuse",
        companion=True,
        # v1/hisparse/layout.py builds this group with enable_kv_transfer=False:
        # the engine rebuilds it, so it is not part of the connector contract.
        transferable=False,
        source="v1/hisparse/layout.py:185 (c1_gate_matrix row 8)",
    ),
    dict(
        name="hisparse_resident",
        cls="HiSparseResidentSpec",
        # uses_slot_mapping is True here (inherited): the measured reason the
        # scratch predicate also checks has_layer_views.
        kwargs=dict(block_size=16, page_size=2 * MIB),
        disposition="skip",
        legacy="refuse",
        companion=True,
        # Same layout, same flag: the resident tier is engine-owned too.
        transferable=False,
        source="v1/hisparse/layout.py (c1_gate_matrix row 8b)",
    ),
    dict(
        name="cross_attention",
        cls="CrossAttentionSpec",
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            dtype="bfloat16",
        ),
        disposition="refuse",
        legacy="refuse",
        keywords=("Unsupported kv cache spec CrossAttentionSpec",),
        mbs=16,
        source="encoder-decoder cross attention (whisper, glm4_1v)",
    ),
    dict(
        name="encoder_only",
        cls="EncoderOnlyAttentionSpec",
        kwargs=dict(
            block_size=16,
            num_kv_heads=4,
            head_size=256,
            dtype="bfloat16",
        ),
        disposition="refuse",
        legacy="refuse",
        keywords=("Unsupported kv cache spec EncoderOnlyAttentionSpec",),
        mbs=16,
        source="encoder-only layers (vLLM returns no spec for them)",
    ),
    dict(
        name="wrapper_v32_pair",
        cls="UniformTypeKVCacheSpecs",
        # V3.2 main + indexer packed into one block table (the accepted wrapper
        # shape).
        members={"l.main": "mla_fp8_ds", "l.indexer": "mla_indexer"},
        disposition="accept",
        legacy="accept",
        mbs=64,
        source="FROZEN_GROUPS.v32_wrapper_pair",
    ),
    dict(
        name="wrapper_v41_packed",
        cls="UniformTypeKVCacheSpecs",
        # V4.1 packs its compressed main and indexer into one group; both carry
        # tokens_per_state=2 (the second member is declared below). The buckets
        # are the two page layouts (74752 B main first, then the 16896 B
        # indexer), each sized in rows: 584 B x 32 rows, 132 B x 32 rows at
        # mbs 64.
        members={"l.main": "mla_compressed_0_31", "l.indexer": "mla_indexer_tps2"},
        disposition="accept",
        legacy="accept",
        requires=("tokens_per_state",),
        # The first bucket (largest page) is the compressed main: the wrapper
        # row's own expectation is that bucket's, as in every other row.
        row_granular=dict(tokens_per_row=2, rows_per_page=128, bytes_per_row=584),
        mbs=64,
        page_bytes=74752,
        per_block_bytes=18688,
        source="exp_gate_v41.py section 4 (V4.1 UniformType main+indexer)",
    ),
    dict(
        name="wrapper_circular",
        cls="UniformTypeKVCacheSpecs",
        # vLLM packs the compressor rings the same way; the wrapper's
        # prefix_cacheable is all(members), so the whole group is scratch.
        members={"l.cb": "circular_buffer"},
        disposition="skip",
        legacy="refuse",
        companion=True,
        mbs=16,
        source="V4.1 CircularBuffer groups (M6: 3 rings, own block size)",
    ),
]

#: Sub-spec rows that exist only to be packed into a wrapper: they are built
#: through their wrapper, never asserted standalone.
MEMBER_ROWS: Dict[str, Dict[str, Any]] = {
    "mla_indexer_tps2": dict(
        cls="MLAAttentionSpec",
        kwargs=dict(
            block_size=256,
            num_kv_heads=1,
            head_size=132,
            dtype="uint8",
            tokens_per_state=2,
            alignment=576,
            block_stride_alignment=576,
        ),
        requires=("tokens_per_state",),
        source="exp_gate_v41.py section 4 indexer",
    ),
}

ROW_BY_NAME: Dict[str, Dict[str, Any]] = {row["name"]: row for row in ZOO}
ROW_BY_NAME.update(MEMBER_ROWS)


@dataclass(frozen=True)
class Engine:
    """One vLLM era under test: its spec module, dtype mapper and parse entry.

    ``module`` is ``vllm.v1.kv_cache_interface`` (real or stand-in);
    ``dtype_of`` maps a dtype *name* to whatever that engine wants;
    ``parse_groups`` is ``vllm_common.parse_groups``, imported by the test
    module *after* its own engine setup (so the stand-in wins in the stub ring).
    """

    module: Any
    dtype_of: Callable[[str], Any]
    parse_groups: Callable[..., Sequence[Any]]
    #: The vLLM era this engine speaks (the stand-in passes the era it mirrors).
    version: str = "0.31.0"

    def class_of(self, name: str) -> Optional[type]:
        cls = getattr(self.module, name, None)
        return cls if isinstance(cls, type) else None

    def quant_of(self, name: str) -> Any:
        return getattr(self.module.KVQuantMode, name)


class RowUnavailable(Exception):
    """The running era cannot build this row (missing class or field)."""


def _kwargs_for(cls: type, kwargs: Dict[str, Any], engine: Engine) -> Dict[str, Any]:
    """Row kwargs the class accepts, with names mapped to engine values.

    A class with ``**kwargs`` (the stand-in's thin subclasses forward to their
    base) accepts every name, so nothing is filtered out there.
    """
    params = inspect.signature(cls).parameters
    takes_anything = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    resolved = {}
    for key, value in kwargs.items():
        if not takes_anything and key not in params:
            continue
        if key == "dtype":
            value = engine.dtype_of(value)
        elif key == "kv_quant_mode":
            value = engine.quant_of(value)
        resolved[key] = value
    return resolved


def _missing_fields(cls: type, requires: Sequence[str]) -> List[str]:
    params = inspect.signature(cls).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return []
    return [field for field in requires if field not in params]


def _members(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    members = {}
    for layer_name, member_name in row.get("members", {}).items():
        member = ROW_BY_NAME.get(member_name)
        if member is None:  # pragma: no cover - table typo
            raise AssertionError(f"{row['name']}: unknown member row {member_name}")
        members[layer_name] = member
    return members


def member_specs(row: Dict[str, Any], engine: Engine) -> Dict[str, Any]:
    """The row's member spec objects (a wrapper row's buckets), by layer name."""
    return {
        layer_name: build_spec(member, engine)
        for layer_name, member in _members(row).items()
    }


def needed_classes(row: Dict[str, Any]) -> List[str]:
    """The classes a row's *specs* need: its members' for a wrapper, else its own."""
    members = _members(row)
    if members:
        return [member["cls"] for member in members.values()]
    return [row["cls"]]


def class_names(row: Dict[str, Any]) -> List[str]:
    """The spec classes a row needs: its own plus its wrapper members'."""
    return [row["cls"]] + [member["cls"] for member in _members(row).values()]


def _classes_of(row: Dict[str, Any], engine: Engine) -> List[type]:
    """Every class the row needs, or RowUnavailable for this era."""
    classes = []
    for cls_name in class_names(row):
        cls = engine.class_of(cls_name)
        if cls is None:
            raise RowUnavailable(f"{cls_name} is not part of this era")
        classes.append(cls)
    requires = tuple(row.get("requires", ()))
    missing: List[str] = []
    # A wrapper row's requirements are about the specs it packs, not about the
    # wrapper itself (whose signature is just block_size + the spec dict).
    for cls_name in needed_classes(row):
        cls = engine.class_of(cls_name)
        assert cls is not None  # checked above
        missing += _missing_fields(cls, requires)
    if missing:
        raise RowUnavailable(f"era lacks {'/'.join(sorted(set(missing)))}")
    return classes


def build_spec(row: Dict[str, Any], engine: Engine) -> Any:
    """The row's spec object (a UniformTypeKVCacheSpecs for wrapper rows)."""
    _classes_of(row, engine)
    members = _members(row)
    if members:
        specs = {}
        for layer_name, member in members.items():
            member_cls = engine.class_of(member["cls"])
            assert member_cls is not None  # checked above
            specs[layer_name] = member_cls(
                **_kwargs_for(member_cls, member["kwargs"], engine)
            )
        wrapper = engine.class_of("UniformTypeKVCacheSpecs")
        assert wrapper is not None
        return wrapper.from_specs(specs)
    cls = engine.class_of(row["cls"])
    assert cls is not None  # checked above
    return cls(**_kwargs_for(cls, row.get("kwargs", {}), engine))


@dataclass(frozen=True)
class RowResult:
    """Outcome of one zoo row: accept / refuse / skip / unavailable."""

    outcome: str
    message: str
    metas: Sequence[Any]


def run_row(row: Dict[str, Any], engine: Engine) -> RowResult:
    """Run one row through the real parse path.

    ``skip`` rows are parsed next to a plain full-attention companion group
    (index 1), so the instance still parses and the row's group index simply
    has no meta.

    Where the era declares the transfer projection, the config carries the
    realistic one: the row's group (unless the row is ``transferable=False``,
    as the HiSparse layout builds them) and the companion are exposed. A
    transferable scratch group is then *refused* -- see expected_outcome.
    """
    try:
        spec = build_spec(row, engine)
    except RowUnavailable as exc:
        return RowResult("unavailable", str(exc), ())
    layer_names = sorted(row["members"]) if row.get("members") else ["l0"]
    groups = [
        SimpleNamespace(
            layer_names=layer_names, kv_cache_spec=spec, is_eagle_group=False
        )
    ]
    if row.get("companion"):
        groups.append(
            SimpleNamespace(
                layer_names=["companion"],
                kv_cache_spec=build_spec(ROW_BY_NAME["full"], engine),
                is_eagle_group=False,
            )
        )
    config = SimpleNamespace(kv_cache_groups=groups)
    if declares_transfer_projection(engine):
        ids = [1] if row.get("companion") else []
        if row.get("transferable", True):
            ids.insert(0, 0)
        config.transfer_group_ids = tuple(sorted(ids))
    try:
        metas = engine.parse_groups(
            config, row.get("mbs", 16), calculate_kv_scales=False
        )
    except NotImplementedError as exc:
        return RowResult("refuse", str(exc), ())
    if any(meta.group_idx == 0 for meta in metas):
        return RowResult("accept", "", metas)
    assert len(metas) == len(groups) - 1, f"{row['name']}: unexpected parse result"
    return RowResult("skip", "", metas)


# --------------------------------------------------------------------------- #
# The pre-PR gate, transcribed: the differential's reference.
# --------------------------------------------------------------------------- #
_QUANT_FP8_PER_TENSOR = 1
_QUANT_PER_TOKEN_HEAD = (2, 3, 4)
_QUANT_NVFP4 = 5
_PLAIN_CACHE_DTYPES = (None, "auto", "float16", "bfloat16")
_PLAIN_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3", "fp8_e5m2")


def legacy_accepts_row(row: Dict[str, Any], engine: Engine) -> bool:
    """The pre-PR gate's verdict for a whole row.

    Wrapper rows descend into their members, exactly as the old
    ``_uniform_group_metas`` did (a member the old gate refused refused the
    group).
    """
    members = _members(row)
    if members:
        return all(
            legacy_accepts(build_spec(member, engine), engine)
            for member in members.values()
        )
    return legacy_accepts(build_spec(row, engine), engine)


def legacy_accepts(spec: Any, engine: Engine) -> bool:
    """Did the pre-PR gate (7b55b853) accept this spec?

    Compact transcription of the old ``_check_attention_spec_supported`` +
    ``_check_mla_structure`` + ``_check_mla_variant``: the FullAttentionSpec
    isinstance test, the two-field window whitelist and -- the hole -- the MLA
    read of ``getattr(spec, "compress_ratio", 1)``. It exists only as the
    differential reference; the zoo rows exercise exactly these paths (all of
    them with ``calculate_kv_scales=False``, the only mode the zoo runs).
    """
    swa_mla = engine.class_of("SlidingWindowMLASpec")
    full_attention = engine.class_of("FullAttentionSpec")
    mla = engine.class_of("MLAAttentionSpec")
    if swa_mla is not None and isinstance(spec, swa_mla):
        return False
    if full_attention is None or not isinstance(spec, full_attention):
        return False
    if mla is not None and isinstance(spec, mla):
        ratio = getattr(spec, "compress_ratio", 1)
        if ratio < 1 or (ratio > 1 and spec.block_size % ratio):
            return False
        if ratio > 1:
            return False
        if getattr(spec, "model_version", None) not in (None, "deepseek_v4"):
            return False
        cache_dtype_str = getattr(spec, "cache_dtype_str", None)
        mode = int(getattr(spec, "kv_quant_mode", 0) or 0)
        if cache_dtype_str == "fp8_ds_mla":
            if hasattr(spec, "kv_quant_mode") and mode != _QUANT_FP8_PER_TENSOR:
                return False
        elif mode in _QUANT_PER_TOKEN_HEAD or mode == _QUANT_NVFP4:
            return False
        elif cache_dtype_str in _PLAIN_CACHE_DTYPES:
            if mode != 0:
                return False
        elif cache_dtype_str in _PLAIN_FP8_CACHE_DTYPES:
            if hasattr(spec, "kv_quant_mode") and mode != _QUANT_FP8_PER_TENSOR:
                return False
        else:
            return False
    return not any(
        getattr(spec, field, None) is not None
        for field in ("sliding_window", "attention_chunk_size")
    )
