"""Frozen MLA variant facts: the shared ring of truth for the MLA tests.

Every row is one attention spec as vLLM 0.26.0 materializes it for a real
model (DeepSeek V3.2 fp8/bfloat16, DeepSeek V4 c4/c128, GLM-4.7-Flash, plus the
boundary rows the 02/03 designs fixed), together with the page numbers the
connector sizes its transfers from. ``kind`` defaults to ``"mla"``
(``MLAAttentionSpec``); ``kind="swa"`` rows are ``SlidingWindowMLASpec``, the
V4 windowed / compressor-state caches. Two independent tests re-derive the same
table:

* ``test_mla_variants.py`` through the ``vllm_stubs`` stand-in (CI, no vllm);
* ``test_mla_spec_facts.py`` through the real vLLM (skipped when unavailable).

A vLLM formula change and a stub drift therefore both turn red, instead of
silently skewing every layout/size assertion downstream.

``dtype`` and ``kv_quant_mode`` are stored as *names*: this module stays free
of torch (the CI has none) and of vLLM's enum. Tests map the names back.

``page_size_padded`` is the page size the connector actually sees -- i.e. after
vLLM's *grouping* pass; for the V4 state rows that is larger than what the
spec's own ``alignment`` produces (``page_size_padded_before_grouping``), so the
row pins the grouped value and records the other one for the golden check.

``FROZEN_GROUPS`` adds the group-level facts: what vLLM packs into one
``UniformTypeKVCacheSpecs`` group, what the transfer buckets of that group are,
and what the scheduler's *folded* view of the same group looks like (the two
role views differ -- see vllm_common.parse_groups). ``sched_row`` names the sub
spec ``generate_scheduler_kv_cache_config`` keeps, i.e. the folded view of that
group: the *first* entry of the wrapper dict, which is engine registration order
(indexer before attention), not the golden's sorted ``layer_names``.
``V4_TINY_GROUPS`` lists the six V4 groups in golden order.

Source of truth: the 01 variant research (vLLM 0.26.0 measurements) and the
tiny-model golden under ``e2e/mla-cq/golden/tiny_model_specs.json`` (generated
outside this repository; ``golden_drift`` checks below reconcile the two when
the file is readable).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

#: One attention spec per variant: the constructor kwargs plus the derived
#: numbers of a single layer. ``per_block_bytes`` assumes manager_block_size
#: == ``mbs`` and is None for the rows the gate refuses (compressed, windowed),
#: so the connector never sizes them.
FROZEN: Dict[str, Dict[str, Any]] = {
    # --- accepted variants: one latent row per token ---------------------- #
    "m0": dict(  # GLM-4.7-Flash bf16 (the regression baseline)
        kwargs=dict(
            block_size=16,
            head_size=576,
            dtype="bfloat16",
            cache_dtype_str="auto",
            num_kv_heads=1,
        ),
        itemsize=2,
        compress_ratio=1,
        mbs=64,
        storage_block_size=16,
        real_page_size_bytes=18432,
        unpadded_page_size_bytes=18432,
        page_size_padded=None,
        per_block_bytes=73728,
    ),
    "m0_b64": dict(  # the bf16 main layer of a V3.2 group (block size 64)
        kwargs=dict(
            block_size=64,
            head_size=576,
            dtype="bfloat16",
            cache_dtype_str="auto",
            num_kv_heads=1,
        ),
        itemsize=2,
        compress_ratio=1,
        mbs=64,
        storage_block_size=64,
        real_page_size_bytes=73728,
        unpadded_page_size_bytes=73728,
        page_size_padded=None,
        per_block_bytes=73728,
    ),
    "m1": dict(  # DeepSeek V3.2 fp8_ds_mla main: 656 B/token, scale in-row
        kwargs=dict(
            block_size=64,
            head_size=576,
            dtype="uint8",
            cache_dtype_str="fp8_ds_mla",
            kv_quant_mode="FP8_PER_TENSOR",
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=1,
        mbs=64,
        storage_block_size=64,
        real_page_size_bytes=41984,
        unpadded_page_size_bytes=41984,
        page_size_padded=None,
        per_block_bytes=41984,
    ),
    "m2c2": dict(  # the real V3.2 indexer: 132 B/row, no alignment padding
        kwargs=dict(
            block_size=64,
            head_size=132,
            dtype="uint8",
            cache_dtype_str=None,
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=1,
        mbs=64,
        storage_block_size=64,
        real_page_size_bytes=8448,
        unpadded_page_size_bytes=8448,
        page_size_padded=None,
        per_block_bytes=8448,
    ),
    "m3b": dict(  # non-packed per-tensor fp8 (layer-level scales in the model)
        # The engine stores this cache as uint8 (1 B/element): the dtype map
        # (`kv_cache_dtype_str_to_dtype("fp8") == torch.uint8`) feeds
        # `MLAAttention.get_kv_cache_spec`, so the page is block x 576 x 1 B
        # (9216 at block 16), not the bf16 18432 02 section 3.2 probed with a
        # template dtype (01 section 2.3: 576 B/token).
        kwargs=dict(
            block_size=16,
            head_size=576,
            dtype="uint8",
            cache_dtype_str="fp8",
            kv_quant_mode="FP8_PER_TENSOR",
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=1,
        mbs=64,
        storage_block_size=16,
        real_page_size_bytes=9216,
        unpadded_page_size_bytes=9216,
        page_size_padded=None,
        per_block_bytes=36864,  # 576 B/token x 64
    ),
    # --- V4 compressed family (refused: one row per several tokens) ------- #
    "m2a": dict(  # V4 c4a bf16, block 256: alignment holds exactly
        kwargs=dict(
            block_size=256,
            head_size=512,
            dtype="bfloat16",
            cache_dtype_str="auto",
            compress_ratio=4,
            alignment=512,
            model_version="deepseek_v4",
            num_kv_heads=1,
        ),
        itemsize=2,
        compress_ratio=4,
        mbs=256,
        storage_block_size=64,
        real_page_size_bytes=65536,
        unpadded_page_size_bytes=65536,
        page_size_padded=None,
        per_block_bytes=None,
    ),
    "m2a_s": dict(  # the same spec on a small page (block 64)
        kwargs=dict(
            block_size=64,
            head_size=512,
            dtype="bfloat16",
            cache_dtype_str="auto",
            compress_ratio=4,
            alignment=512,
            model_version="deepseek_v4",
            num_kv_heads=1,
        ),
        itemsize=2,
        compress_ratio=4,
        mbs=64,
        storage_block_size=16,
        real_page_size_bytes=16384,
        unpadded_page_size_bytes=16384,
        page_size_padded=None,
        per_block_bytes=None,
    ),
    "m2b": dict(  # V4 c4a fp8_ds_mla (584 B/row) with a 64 B alignment tail
        kwargs=dict(
            block_size=256,
            head_size=512,
            dtype="uint8",
            cache_dtype_str="fp8_ds_mla",
            compress_ratio=4,
            alignment=576,
            model_version="deepseek_v4",
            kv_quant_mode="FP8_PER_TENSOR",
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=4,
        mbs=256,
        storage_block_size=64,
        real_page_size_bytes=37376,
        unpadded_page_size_bytes=37376,
        page_size_padded=37440,
        per_block_bytes=None,
    ),
    "m2b_s": dict(  # the same spec on a small page (block 64)
        kwargs=dict(
            block_size=64,
            head_size=512,
            dtype="uint8",
            cache_dtype_str="fp8_ds_mla",
            compress_ratio=4,
            alignment=576,
            model_version="deepseek_v4",
            kv_quant_mode="FP8_PER_TENSOR",
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=4,
        mbs=64,
        storage_block_size=16,
        real_page_size_bytes=9344,
        unpadded_page_size_bytes=9344,
        page_size_padded=9792,
        per_block_bytes=None,
    ),
    "m2b2": dict(  # V4 c128a: two rows per page, 560 B of alignment tail
        kwargs=dict(
            block_size=256,
            head_size=512,
            dtype="uint8",
            cache_dtype_str="fp8_ds_mla",
            compress_ratio=128,
            alignment=576,
            model_version="deepseek_v4",
            kv_quant_mode="FP8_PER_TENSOR",
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=128,
        mbs=256,
        storage_block_size=2,
        real_page_size_bytes=1168,
        unpadded_page_size_bytes=1168,
        page_size_padded=1728,
        per_block_bytes=None,
    ),
    "m2c": dict(  # V4 indexer (132 B/row) with a 192 B alignment tail
        kwargs=dict(
            block_size=256,
            head_size=132,
            dtype="uint8",
            cache_dtype_str=None,
            compress_ratio=4,
            alignment=576,
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=4,
        mbs=256,
        storage_block_size=64,
        real_page_size_bytes=8448,
        unpadded_page_size_bytes=8448,
        page_size_padded=8640,
        per_block_bytes=None,
    ),
    # --- V4 windowed / compressor-state rows (refused: not the full prefix) #
    # They pin the *grouped* spec: page_size_padded is a constructor input
    # (grouping applied it) and the pre-grouping value is kept for the golden.
    "v4_swa": dict(  # the V4 attention sliding-window cache (584 B/row)
        kind="swa",
        kwargs=dict(
            block_size=64,
            sliding_window=128,
            head_size=512,
            dtype="uint8",
            cache_dtype_str="fp8_ds_mla",
            compress_ratio=1,
            model_version="deepseek_v4",
            kv_quant_mode="FP8_PER_TENSOR",
            page_size_padded=37440,
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=1,
        mbs=64,
        storage_block_size=64,
        real_page_size_bytes=37376,
        unpadded_page_size_bytes=37376,
        page_size_padded=37440,
        page_size_padded_before_grouping=37440,
        per_block_bytes=None,
    ),
    "v4_state_l1": dict(  # layer-1 compressor state: fp32, one row per step
        kind="swa",
        kwargs=dict(
            block_size=4,
            sliding_window=8,
            head_size=2048,
            dtype="float32",
            compress_ratio=1,
            page_size_padded=37440,
            num_kv_heads=1,
        ),
        itemsize=4,
        compress_ratio=1,
        mbs=4,
        storage_block_size=4,
        real_page_size_bytes=32768,
        unpadded_page_size_bytes=32768,
        page_size_padded=37440,
        page_size_padded_before_grouping=32832,
        per_block_bytes=None,
    ),
    "v4_state_indexer": dict(  # the indexer's compressor state (head 512)
        kind="swa",
        kwargs=dict(
            block_size=4,
            sliding_window=8,
            head_size=512,
            dtype="float32",
            compress_ratio=1,
            page_size_padded=8640,
            num_kv_heads=1,
        ),
        itemsize=4,
        compress_ratio=1,
        mbs=4,
        storage_block_size=4,
        real_page_size_bytes=8192,
        unpadded_page_size_bytes=8192,
        page_size_padded=8640,
        page_size_padded_before_grouping=8640,
        per_block_bytes=None,
    ),
    "v4_state_l2": dict(  # layer-2 compressor state (block 8, head 1024)
        kind="swa",
        kwargs=dict(
            block_size=8,
            sliding_window=128,
            head_size=1024,
            dtype="float32",
            compress_ratio=1,
            page_size_padded=37440,
            num_kv_heads=1,
        ),
        itemsize=4,
        compress_ratio=1,
        mbs=8,
        storage_block_size=8,
        real_page_size_bytes=32768,
        unpadded_page_size_bytes=32768,
        page_size_padded=37440,
        page_size_padded_before_grouping=32832,
        per_block_bytes=None,
    ),
    # --- boundary / refused rows ------------------------------------------ #
    "m3": dict(  # INT4 per-token-head: the scale budget sits inside the page
        kwargs=dict(
            block_size=16,
            head_size=576,
            dtype="bfloat16",
            cache_dtype_str="auto",
            kv_quant_mode="INT4_PER_TOKEN_HEAD",
            num_kv_heads=1,
        ),
        itemsize=2,
        compress_ratio=1,
        mbs=64,
        storage_block_size=16,
        real_page_size_bytes=9216,
        unpadded_page_size_bytes=9344,
        page_size_padded=None,
        per_block_bytes=None,
    ),
    "X1": dict(  # synthetic: fp8_ds_mla + c>1 + no model_version (vLLM's
        # non-V4 branch sizes real from block_size, not storage_block_size)
        kwargs=dict(
            block_size=256,
            head_size=576,
            dtype="uint8",
            cache_dtype_str="fp8_ds_mla",
            compress_ratio=4,
            kv_quant_mode="FP8_PER_TENSOR",
            num_kv_heads=1,
        ),
        itemsize=1,
        compress_ratio=4,
        mbs=64,
        storage_block_size=64,
        real_page_size_bytes=167936,
        unpadded_page_size_bytes=167936,
        page_size_padded=None,
        per_block_bytes=None,
    ),
    "X2": dict(  # synthetic: block_size 16 with compress_ratio 3 floors to 5
        kwargs=dict(
            block_size=16,
            head_size=512,
            dtype="bfloat16",
            cache_dtype_str="auto",
            compress_ratio=3,
            num_kv_heads=1,
        ),
        itemsize=2,
        compress_ratio=3,
        mbs=16,
        storage_block_size=5,
        real_page_size_bytes=5120,
        unpadded_page_size_bytes=5120,
        page_size_padded=None,
        per_block_bytes=None,
    ),
}

#: Variants the gate accepts today (c == 1, self-contained rows).
ACCEPTED: List[str] = ["m0", "m0_b64", "m1", "m2c2", "m3b"]
#: Variants the gate refuses, grouped by the rule that refuses them.
REJECTED_COMPRESSED: List[str] = ["m2a", "m2a_s", "m2b", "m2b_s", "m2b2", "m2c", "X1"]
REJECTED_QUANTIZED: List[str] = ["m3"]
#: SlidingWindowMLASpec rows (V4 SWA / compressor state: not the full prefix).
REJECTED_WINDOW: List[str] = [
    "v4_swa",
    "v4_state_l1",
    "v4_state_indexer",
    "v4_state_l2",
]
#: Structurally invalid rows (block_size not divisible by compress_ratio).
REJECTED_INVALID: List[str] = ["X2"]

#: The V3.2 wrapper group as vLLM builds it: one layer pair per model layer,
#: all sharing one block table. ``layers`` is (vLLM layer name, FROZEN row) in
#: *engine registration order* (see FROZEN_GROUPS): the indexer registers
#: before the attention module, so it comes first.
_V32_LAYERS = [
    ("model.layers.0.self_attn.indexer.k_cache", "m2c2"),
    ("model.layers.0.self_attn.attn", "m1"),
    ("model.layers.1.self_attn.indexer.k_cache", "m2c2"),
    ("model.layers.1.self_attn.attn", "m1"),
]

#: Group-level facts. ``layers`` is in engine registration order, which is the
#: order vLLM iterates ``static_forward_context`` in: it is the wrapper's dict
#: order, the group's ``layer_names`` order, and therefore the order
#: ``generate_scheduler_kv_cache_config`` folds (it keeps the *first* entry --
#: vllm 0.26.0 kv_cache_utils.py:1813 -- not an alphabetically first one). For
#: the DeepSeek families that means the indexer sub spec comes first. The local
#: golden generator dumps ``layer_names`` sorted, so the drift check sorts.
#: ``buckets`` = the worker view of a packed group (buckets in wire-name
#: order); ``folded_row`` marks a row that *is* the folded view (v32),
#: ``sched_row`` the sub spec a worker-view row folds to.
FROZEN_GROUPS: Dict[str, Dict[str, Any]] = {
    "v32_wrapper_worker": dict(
        block_size=64,
        mbs=64,  # the FLASHMLA_SPARSE preferred block size
        layers=_V32_LAYERS,
        page_size_bytes=100864,  # 2 x 41984 + 2 x 8448
        page_sizes=[8448, 41984],
        sched_row="m2c2",  # the indexer registers first -> folded entry
        buckets=[
            dict(suffix="", row="m1", layers=2, per_block_bytes=83968),
            dict(suffix="_b1", row="m2c2", layers=2, per_block_bytes=16896),
        ],
    ),
    "v32_wrapper_pair": dict(  # one main + one indexer layer (the 02 A10 row)
        block_size=64,
        mbs=64,
        layers=_V32_LAYERS[:2],
        page_size_bytes=50432,  # 41984 + 8448
        page_sizes=[8448, 41984],
        sched_row="m2c2",
        buckets=[
            dict(suffix="", row="m1", layers=1, per_block_bytes=41984),
            dict(suffix="_b1", row="m2c2", layers=1, per_block_bytes=8448),
        ],
    ),
    "v32_wrapper_sched": dict(  # the folded view: the indexer spec x all layers
        block_size=64,
        mbs=64,
        folded_row="m2c2",
        layers=4,
        page_bytes=8448,
        per_block_bytes=33792,  # 132 B/token x 64 x 4 layers
    ),
    "v32_bf16_wrapper": dict(  # bf16 main + uint8 indexer: mixed itemsizes
        # Same fold fact as the pair row (the indexer registers first, so the
        # fold keeps m2c2); this row only pins the mixed element sizes.
        block_size=64,
        mbs=64,
        layers=[
            ("model.layers.0.self_attn.indexer.k_cache", "m2c2"),
            ("model.layers.0.self_attn.attn", "m0_b64"),
        ],
        page_size_bytes=82176,  # 73728 + 8448
        page_sizes=[8448, 73728],
        buckets=[
            dict(suffix="", row="m0_b64", layers=1, per_block_bytes=73728),
            dict(suffix="_b1", row="m2c2", layers=1, per_block_bytes=8448),
        ],
    ),
    # --- DeepSeek V4 tiny: six packed groups, all refused (M2/A case) ------ #
    # The full-MLA group packs two compressed mains (c=4/c=128) and the
    # compressed indexer; every sub spec of every group is refused. The
    # indexer registers first, so it is the group's first layer and the folded
    # row the scheduler sees (see the FROZEN_GROUPS comment).
    "v4_full_c4a": dict(
        block_size=256,
        mbs=256,
        layers=[
            ("model.layers.1.attn.indexer.k_cache", "m2c"),
            ("model.layers.1.attn", "m2b"),
            ("model.layers.2.attn", "m2b2"),
        ],
        page_sizes=[1728, 8640, 37440],
        page_size_bytes=47808,  # 37440 + 8640 + 1728
        sched_row="m2c",
    ),
    "v4_swa_wrapper_l0": dict(  # one SWA group per layer
        block_size=64,
        mbs=64,
        layers=[("model.layers.0.attn.swa_cache", "v4_swa")],
        page_sizes=[37440],
        page_size_bytes=37440,
        sched_row="v4_swa",
    ),
    "v4_swa_wrapper_l1": dict(
        block_size=64,
        mbs=64,
        layers=[("model.layers.1.attn.swa_cache", "v4_swa")],
        page_sizes=[37440],
        page_size_bytes=37440,
        sched_row="v4_swa",
    ),
    "v4_swa_wrapper_l2": dict(
        block_size=64,
        mbs=64,
        layers=[("model.layers.2.attn.swa_cache", "v4_swa")],
        page_sizes=[37440],
        page_size_bytes=37440,
        sched_row="v4_swa",
    ),
    "v4_state_wrapper_l1": dict(  # two compressor states share one table
        block_size=4,
        mbs=4,
        layers=[
            ("model.layers.1.attn.indexer.compressor.state_cache", "v4_state_indexer"),
            ("model.layers.1.attn.compressor.state_cache", "v4_state_l1"),
        ],
        page_sizes=[8640, 37440],
        page_size_bytes=46080,  # 37440 + 8640
        sched_row="v4_state_indexer",
    ),
    "v4_state_wrapper_l2": dict(  # layer-2 compressor state (block 8)
        block_size=8,
        mbs=8,
        layers=[("model.layers.2.attn.compressor.state_cache", "v4_state_l2")],
        page_sizes=[37440],
        page_size_bytes=37440,
        sched_row="v4_state_l2",
    ),
}

#: The six V4 groups in golden order (index = golden group index).
V4_TINY_GROUPS: List[str] = [
    "v4_state_wrapper_l1",
    "v4_state_wrapper_l2",
    "v4_swa_wrapper_l0",
    "v4_swa_wrapper_l1",
    "v4_swa_wrapper_l2",
    "v4_full_c4a",
]

#: Local tiny-model golden (generated outside this repository; see 07/08) --
#: only used to reconcile the frozen numbers when the file is readable.
DEFAULT_GOLDEN_PATH = "/root/ws/kv/e2e/mla-cq/golden/tiny_model_specs.json"
#: Key order: variant name -> (golden model, golden group index).
GOLDEN_GROUPS: Dict[str, Tuple[str, int]] = {
    "v32_wrapper_worker": ("dsv32_tiny", 0),
    **{row: ("dsv4_tiny", index) for index, row in enumerate(V4_TINY_GROUPS)},
}


def load_golden(path: Optional[str] = None) -> Optional[dict]:
    """The tiny-model golden, or None when unavailable (CI / other machines)."""
    path = path or os.environ.get("KVCM_MLA_GOLDEN", DEFAULT_GOLDEN_PATH)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
