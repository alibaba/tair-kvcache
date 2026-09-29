"""Frozen MLA variant facts: the shared ring of truth for the MLA tests.

Every row is one ``MLAAttentionSpec`` as vLLM 0.26.0 materializes it for a real
model (DeepSeek V3.2 fp8/bfloat16, DeepSeek V4 c4/c128, GLM-4.7-Flash, plus the
boundary rows the 02/03 designs fixed), together with the page numbers the
connector sizes its transfers from. Two independent tests re-derive the same
table:

* ``test_mla_variants.py`` through the ``vllm_stubs`` stand-in (CI, no vllm);
* ``test_mla_spec_facts.py`` through the real vLLM (skipped when unavailable).

A vLLM formula change and a stub drift therefore both turn red, instead of
silently skewing every layout/size assertion downstream.

``dtype`` and ``kv_quant_mode`` are stored as *names*: this module stays free
of torch (the CI has none) and of vLLM's enum. Tests map the names back.

``FROZEN_GROUPS`` adds the group-level facts: what vLLM packs into one
``UniformTypeKVCacheSpecs`` group, what the transfer buckets of that group are,
and what the scheduler's *folded* view of the same group looks like (the two
role views differ -- see vllm_common.parse_groups).

Source of truth: the 01 variant research (vLLM 0.26.0 measurements) and the
tiny-model golden under ``e2e/mla-cq/golden/tiny_model_specs.json`` (generated
outside this repository; ``golden_drift`` checks below reconcile the two when
the file is readable).
"""

import json
import os
from typing import Any, Dict, List, Optional

#: One MLAAttentionSpec per variant: the constructor kwargs plus the derived
#: numbers of a single layer. ``per_block_bytes`` assumes manager_block_size
#: == ``mbs`` and is None for the compressed rows (c > 1 is refused, so the
#: connector never sizes them).
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
    "m3b": dict(  # non-packed per-tensor fp8 (layer-level scales)
        kwargs=dict(
            block_size=16,
            head_size=576,
            dtype="bfloat16",
            cache_dtype_str="fp8",
            kv_quant_mode="FP8_PER_TENSOR",
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

#: The V3.2 wrapper group as vLLM builds it: one layer pair per model layer,
#: all sharing one block table. ``layers`` is (vLLM layer name, FROZEN row).
_V32_LAYERS = [
    ("model.layers.0.self_attn.attn", "m1"),
    ("model.layers.0.self_attn.indexer.k_cache", "m2c2"),
    ("model.layers.1.self_attn.attn", "m1"),
    ("model.layers.1.self_attn.indexer.k_cache", "m2c2"),
]

#: Group-level facts. ``buckets`` = the transfer buckets of the worker view, in
#: wire-name order (suffix, FROZEN row, layers, per_block_bytes at ``mbs``);
#: ``folded`` = what generate_scheduler_kv_cache_config leaves for the
#: scheduler (one sub spec, all layer names).
FROZEN_GROUPS: Dict[str, Dict[str, Any]] = {
    "v32_wrapper_worker": dict(
        block_size=64,
        mbs=64,  # the FLASHMLA_SPARSE preferred block size
        layers=_V32_LAYERS,
        page_size_bytes=100864,  # 2 x 41984 + 2 x 8448
        page_sizes=[8448, 41984],
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
        buckets=[
            dict(suffix="", row="m1", layers=1, per_block_bytes=41984),
            dict(suffix="_b1", row="m2c2", layers=1, per_block_bytes=8448),
        ],
    ),
    "v32_wrapper_sched": dict(  # the folded view: main fields x all layers
        block_size=64,
        mbs=64,
        folded_row="m1",
        layers=4,
        page_bytes=41984,
        per_block_bytes=167936,  # 41984 / 64 * 64 * 4
    ),
    "v32_bf16_wrapper": dict(  # bf16 main + uint8 indexer: mixed itemsizes
        block_size=64,
        mbs=64,
        layers=[
            ("model.layers.0.self_attn.attn", "m0_b64"),
            ("model.layers.0.self_attn.indexer.k_cache", "m2c2"),
        ],
        page_size_bytes=82176,  # 73728 + 8448
        page_sizes=[8448, 73728],
        buckets=[
            dict(suffix="", row="m0_b64", layers=1, per_block_bytes=73728),
            dict(suffix="_b1", row="m2c2", layers=1, per_block_bytes=8448),
        ],
    ),
}

#: Local tiny-model golden (generated outside this repository; see 07/08) --
#: only used to reconcile the frozen numbers when the file is readable.
DEFAULT_GOLDEN_PATH = "/root/ws/kv/e2e/mla-cq/golden/tiny_model_specs.json"
#: Key order: variant name -> (golden model, golden group index).
GOLDEN_GROUPS = {
    "v32_wrapper_worker": ("dsv32_tiny", 0),
}


def load_golden(path: Optional[str] = None) -> Optional[dict]:
    """The tiny-model golden, or None when unavailable (CI / other machines)."""
    path = path or os.environ.get("KVCM_MLA_GOLDEN", DEFAULT_GOLDEN_PATH)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
