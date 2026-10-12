"""Pure helpers for the vLLM e2e harness: model detection, paths,
tokenization, manager queries, prompt/answer traffic and capture
comparison. No process state lives here."""


import glob
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import uuid
from typing import Optional

import requests

logger = logging.getLogger("vllm_e2e")
# This module only runs inside test drivers; make the orchestration evidence
# (block counts, verification report, log-scan results) visible in test.log.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Required: path to a local HF model directory (config.json + weights).
# Full-attention coverage needs a plain attention model (e.g.
# Qwen2.5-7B-Instruct); hybrid coverage needs a mamba/linear + attention model
# (e.g. Qwen3.5-4B). Hybrid models are auto-detected from config.json.
MODEL_PATH = os.environ.get("KVCM_E2E_MODEL")
if not MODEL_PATH:
    raise RuntimeError(
        "KVCM_E2E_MODEL is not set. Point it at a local model directory, e.g. "
        "--test_env=KVCM_E2E_MODEL=/path/to/Qwen2.5-7B-Instruct (full "
        "attention) or /path/to/Qwen3.5-4B (hybrid). See "
        "integration_test/vllm_e2e/README.md.")

# Model-agnostic alias the driver uses in OpenAI API requests, so the tests do
# not depend on the model directory name.
SERVED_MODEL_NAME = "e2e-model"



def _model_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


#: Layer types that mean "this layer attends a bounded window", i.e. the layer
#: becomes a windowed kv_cache_group (vLLM's gemma3 / gpt_oss / llama4 configs).
_LOCAL_LAYER_TYPES = ("sliding_attention", "sliding_window", "chunked_local",
                      "local_attention")
#: Model families whose configs interleave local and full attention layers
#: without a per-config marker (gemma-3 declares sliding_window + an arch, not
#: a layer_types list).
_SWA_ARCH_MARKERS = ("Gemma2", "Gemma3", "GptOss", "Llama4")


def is_swa_model(model_path: str) -> bool:
    """Detect a model with sliding-window / chunked-local attention layers.

    A windowed layer is not a full-prefix layer: vLLM frees its blocks outside
    the window while the request lives, so the connector has to copy them in
    the step that produces them and a hit must be clamped to the servable
    window (see WindowRetention in the connector). Such models need the same
    orchestration as hybrid ones: prefix caching on, and a server restart
    between the save and load phases so the load goes through KVCM.

    The signals are the ones vLLM itself keys on, ORed:
    * ``layer_types`` naming a local layer (gemma-3-1b/gpt-oss style);
    * ``sliding_window_pattern`` (gemma-2 style interleave);
    * a known interleaved family architecture *and* a declared window.
    The architecture alone is not enough (a plain window value is not either:
    Qwen2 declares ``sliding_window=131072`` while never using it -- its
    layers are full attention, and treating it as windowed would change the
    orchestration of an existing model).
    """
    try:
        cfg = _model_config(model_path)
    except Exception:
        return False
    text_cfg = cfg.get("text_config", cfg)
    window = window_tokens(model_path)
    layer_types = text_cfg.get("layer_types") or []
    if any(t in _LOCAL_LAYER_TYPES for t in layer_types):
        return True
    pattern = text_cfg.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 0:
        return True
    archs = ", ".join(cfg.get("architectures", []) or [])
    return window is not None and any(m in archs for m in _SWA_ARCH_MARKERS)


def window_kind(model_path: str) -> Optional[str]:
    """Which retention rule the model's window follows (None = no window).

    vLLM keeps the two windowed shapes in separate managers with different
    skip rules, so the harness must not apply one's arithmetic to the other:

    * ``"sliding"`` -- ``SlidingWindowManager``: only the last ``window`` tokens
      are attended, so the skipped prefix grows with every step;
    * ``"chunked"`` -- ``ChunkedLocalAttentionManager``: attention is confined
      to the *current* chunk, so the skipped prefix is everything before the
      chunk the next token falls in (``H // chunk * chunk``).

    Sliding wins when a config declares both (that is also the key order
    ``window_tokens`` has always used).
    """
    try:
        cfg = _model_config(model_path)
    except Exception:
        return None
    text_cfg = cfg.get("text_config", cfg)
    for key, kind in (("sliding_window", "sliding"),
                      ("attention_chunk_size", "chunked")):
        value = text_cfg.get(key)
        if isinstance(value, int) and value > 0:
            return kind
    return None


def window_tokens(model_path: str) -> Optional[int]:
    """The attention window of a windowed model, in tokens (None = full).

    Reads the same config field the connector turns into
    ``WindowRetention.window_tokens`` (SlidingWindowSpec.sliding_window, or
    ChunkedLocalAttentionSpec.attention_chunk_size for chunked-local);
    ``window_kind`` says which of the two it is.
    """
    try:
        cfg = _model_config(model_path)
    except Exception:
        return None
    text_cfg = cfg.get("text_config", cfg)
    for key in ("sliding_window", "attention_chunk_size"):
        value = text_cfg.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def skipped_window_tokens(matched_tokens: int, window: Optional[int],
                          extra_retained: int = 0,
                          kind: str = "sliding") -> int:
    """Tokens of a hit that vLLM parks on the null block (its own rule).

    ``SingleTypeKVCacheManager.get_num_skipped_tokens`` in vLLM 0.31, per
    manager, with H the hit length:

    * sliding window: ``max(0, H - window + 1 - extra_retained)``;
    * chunked local: ``H // chunk_size * chunk_size`` -- the whole chunks before
      the one the next token falls in, *not* an H-(window-1) tail (H=13,
      chunk=8 -> 8 skipped tokens, not 6). It ignores ``extra_retained``: that
      term only exists in the sliding manager.

    Full attention (window None) never skips.
    """
    if window is None:
        return 0
    if kind == "chunked":
        return (matched_tokens // window) * window
    return max(0, matched_tokens - (window - 1) - extra_retained)


def window_tail_blocks(matched_tokens: int, window: Optional[int],
                       manager_block_size: int, extra_retained: int = 0,
                       kind: str = "sliding") -> int:
    """Manager blocks vLLM gives real slots on an external hit of that size.

    vLLM's own allocation, for a hit of H tokens (vLLM 0.31
    SingleTypeKVCacheManager.allocate_external_computed_blocks): the leading
    ``skipped_window_tokens(H) // bs`` blocks are replaced by the null block and
    ``cdiv(H, bs)`` blocks are allocated, so the blocks with a real block table
    entry are the tail of the window (``kind`` decides the rule). Full-attention
    models (window None) keep every block.

    This is *one* half of the load expectation; the other is
    :func:`loaded_block_count`, which adds the full-attention group's whole
    prefix when the model has one.
    """
    blocks = -(-matched_tokens // manager_block_size)
    skipped = skipped_window_tokens(matched_tokens, window, extra_retained, kind)
    return blocks - skipped // manager_block_size


def has_full_attention_group(model_path: str) -> bool:
    """Does the model keep at least one non-windowed attention layer?

    vLLM groups attention layers by spec; when a config mixes local and full
    layers (gemma-3: six sliding + one full, gpt-oss: one of each) the full
    group keeps *every* matched block. A hit therefore loads the whole matched
    prefix, not just the tail window -- the loaded-capture expectation follows
    the same split.

    Unknown or undeclared layouts answer True: expecting the larger set fails
    loudly instead of quietly shrinking the assertion surface.
    """
    try:
        cfg = _model_config(model_path)
    except Exception:
        return True
    text_cfg = cfg.get("text_config", cfg)
    layer_types = text_cfg.get("layer_types") or []
    if layer_types:
        return any(t not in _LOCAL_LAYER_TYPES for t in layer_types)
    pattern = text_cfg.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 0:
        # "every pattern-th layer is full attention" (gemma-2 style): a full
        # layer exists only if the model is at least that deep.
        return int(text_cfg.get("num_hidden_layers") or 0) >= pattern
    return True


def loaded_block_count(matched_tokens: int, window: Optional[int],
                       manager_block_size: int,
                       full_attention_group: bool = True,
                       kind: str = "sliding") -> int:
    """Manager blocks whose KV a hit of that size really loads.

    vLLM hands out slots per group, so the loaded set is a union: every group
    gets the tail window of the match (``window_tail_blocks``), and a
    full-attention group -- one without a window -- additionally keeps the whole
    matched prefix. For a model whose attention groups are *all* windowed the
    load stops at the window; for one with a full group it covers everything
    (a windowed model such as gemma-3 loads its full group's out-of-window
    blocks, which the harness must capture and verify like any other block).

    ``window`` None (no windowed layers) means the whole matched prefix either
    way.
    """
    blocks = -(-matched_tokens // manager_block_size)
    if window is None or full_attention_group:
        return blocks
    return window_tail_blocks(matched_tokens, window, manager_block_size,
                              kind=kind)


def is_hybrid_model(model_path: str) -> bool:
    """Detect a hybrid (mamba/linear + full attention) model from its config."""
    try:
        cfg = _model_config(model_path)
    except Exception:
        return False
    text_cfg = cfg.get("text_config", cfg)
    # Any known hybrid marker wins: architecture names cover the families we
    # know (e.g. Qwen3NextForCausalLM), the layer knobs cover configs that
    # interleave linear/mamba layers with full attention without naming a
    # known family (e.g. Qwen3_5ForConditionalGeneration declares its own
    # architecture). The signals are ORed -- a non-matching architecture
    # string must never shadow the knobs.
    archs = ", ".join(cfg.get("architectures", []) or [])
    arch_markers = ("Qwen3Next", "Zamba", "FalconH1", "Samba", "Jamba")
    return (
        any(m in archs for m in arch_markers)
        or "full_attention_interval" in text_cfg
        or "linear_conv_kernel_dim" in text_cfg
        or str(cfg.get("model_type", "")).startswith("qwen3_")
    )


def is_multigroup_model(model_path: str) -> bool:
    """Does this model need the multi-group orchestration?

    prefix caching on (per-group block tables / windowed blocks survive) and a
    vLLM restart between the save and the load phase, so phase 2 loads from
    KVCM instead of hitting vLLM's own prefix cache. True for hybrid
    (mamba/linear + attention) and for windowed (SWA / chunked-local) models;
    single-group full-attention models keep the original orchestration.
    """
    return is_hybrid_model(model_path) or is_swa_model(model_path)


def model_fingerprint(model_path: str) -> tuple:
    """``(architectures[0], declared window)``: a path-independent identity.

    Used to key per-model budgets (e.g. the pinned-host gate) so that a copied,
    renamed or symlinked checkpoint keeps the gate it had under its original
    directory name. The window component separates same-architecture siblings
    (gemma-3-4b 1024 vs gemma-3-1b / 270m 512).
    """
    try:
        cfg = _model_config(model_path)
    except Exception:
        cfg = {}
    archs = cfg.get("architectures") or []
    return (archs[0] if archs else None, window_tokens(model_path))


def scenario_tp_size(default: int = 1) -> int:
    """TP size for a scenario: KVCM_E2E_TP overrides the per-scenario default.

    Models too large for a single GPU (e.g. GLM-4.7-Flash on 24 GB cards, even
    fp8-quantized) run every scenario at tp=2 via this knob instead of editing
    each scenario file."""
    return int(os.environ.get("KVCM_E2E_TP", "") or default)


def is_mla_model(model_path: str) -> bool:
    """Detect an MLA model from its config (kv_lora_rank is the defining
    knob -- vLLM's own is_deepseek_mla check keys on it for every MLA
    family). Used to stop the harness from forcing the FLASH_ATTN backend,
    which has no MLA implementation and would abort engine init."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        return False
    text_cfg = cfg.get("text_config", cfg)
    return "kv_lora_rank" in text_cfg


# --------------------------------------------------------------------------- #
# Paths / binaries
# --------------------------------------------------------------------------- #

def _runfiles_root() -> Optional[str]:
    return os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")


def find_repo_root() -> str:
    """Locate the KVCM repository root (works under Bazel runfiles and plain)."""
    here = os.path.dirname(os.path.abspath(__file__))
    # integration_test/vllm_e2e/e2e_lib.py -> repo root is two levels up.
    candidate = os.path.abspath(os.path.join(here, "..", ".."))
    if os.path.exists(os.path.join(candidate, "WORKSPACE")):
        return candidate
    runfiles = _runfiles_root()
    if runfiles:
        cand = os.path.join(runfiles, "kv_cache_manager")
        if os.path.exists(os.path.join(cand, "WORKSPACE")):
            return cand
    return candidate


def find_manager_binary(repo_root: str) -> str:
    candidates = [
        os.path.join(repo_root, "bazel-bin/kv_cache_manager/kv_cache_manager_bin"),
        os.path.join(repo_root, "bazel-out/k8-opt/bin/kv_cache_manager/kv_cache_manager_bin"),
    ]
    runfiles = _runfiles_root()
    if runfiles:
        candidates.append(
            os.path.join(runfiles, "kv_cache_manager", "kv_cache_manager",
                         "kv_cache_manager_bin")
        )
    for c in candidates:
        if os.path.exists(c):
            return c
    raise RuntimeError(
        "kv_cache_manager_bin not found; build it with: "
        "bazelisk build //kv_cache_manager:kv_cache_manager_bin"
    )


def find_python() -> str:
    # Required: python interpreter of a venv with vLLM >= 0.26.0 and both KVCM
    # wheels (kvcm_py_client, kvcm_vllm_connector) installed; see README.md.
    python = os.environ.get("KVCM_E2E_PYTHON")
    if not python:
        raise RuntimeError(
            "KVCM_E2E_PYTHON is not set. Point it at the python of a vLLM "
            "venv with the KVCM wheels installed, e.g. "
            "--test_env=KVCM_E2E_PYTHON=/path/to/venv/bin/python. See "
            "integration_test/vllm_e2e/README.md.")
    return python


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Loopback only: everything in this harness is single-machine.
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float, post_body: Optional[dict] = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if post_body is not None:
                r = requests.post(url, json=post_body, timeout=3)
            else:
                r = requests.get(url, timeout=3)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


# --------------------------------------------------------------------------- #
# KVCM manager

# --------------------------------------------------------------------------- #
def tokenize(base_url: str, prompt: str) -> list[int]:
    r = requests.post(f"{base_url}/tokenize",
                      json={"model": SERVED_MODEL_NAME, "prompt": prompt}, timeout=60)
    r.raise_for_status()
    return r.json()["tokens"]


def get_manager_block_size(manager_uri: str, instance_id: str) -> int:
    """Ask the manager for the registered instance's manager block size."""
    r = requests.post(f"{manager_uri}/api/getInstanceInfo",
                      json={"trace_id": "e2e_bs", "instance_id": instance_id},
                      timeout=10)
    r.raise_for_status()
    block_size = r.json()["instance_info"]["block_size"]
    assert block_size > 0, f"bad manager block size: {block_size}"
    return block_size


def block_token_hash(token_ids: list[int]) -> str:
    """Token-content hash used in capture file names (mirrors test_connector)."""
    import hashlib
    import torch
    return hashlib.sha256(
        torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes()
    ).hexdigest()[:16]


def full_block_hashes(token_ids: list[int], manager_block_size: int) -> list[str]:
    """Per-manager-block capture hashes for the full blocks of a token stream."""
    n = len(token_ids) // manager_block_size
    return [
        block_token_hash(token_ids[i * manager_block_size:(i + 1) * manager_block_size])
        for i in range(n)
    ]


def wait_for_prefix_cached(manager_uri: str, instance_id: str,
                           token_ids: list[int], min_blocks: int,
                           timeout: float = 120.0) -> bool:
    """Poll the manager until at least min_blocks of the prefix are committed.

    Mirrors what the connector's get_num_new_matched_tokens queries
    (query_type=QT_PREFIX_MATCH, block_mask offset=0 for a fresh request), so it
    guarantees phase 2 will actually hit the external cache for all min_blocks.
    """
    deadline = time.time() + timeout
    payload = {
        "trace_id": "e2e_probe",
        "token_ids": token_ids,
        "instance_id": instance_id,
        "query_type": "QT_PREFIX_MATCH",
        "block_mask": {"offset": 0},
    }
    while time.time() < deadline:
        try:
            r = requests.post(f"{manager_uri}/api/getCacheLocation",
                              json=payload, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("header", {}).get("status", {}).get("code") == "OK":
                    locs = data.get("locations", [])
                    if len(locs) >= min_blocks:
                        logger.info("prefix cached: %d location(s)", len(locs))
                        return True
        except Exception:
            pass
        time.sleep(1.0)
    logger.warning("timed out waiting for prefix to be cached")
    return False


def send_completions(base_url: str, prompts: list, max_tokens: int = 4,
                     temperature: float = 0.0, **extra_payload) -> list[dict]:
    """Send prompts concurrently and return the OpenAI responses.

    Each prompt may be a string or a list of token ids (the completions API
    accepts both). extra_payload is merged into the request body (e.g.
    return_token_ids=True)."""
    from concurrent.futures import ThreadPoolExecutor

    client_url = f"{base_url}/v1/completions"

    def _one(prompt) -> dict:
        payload = {
            "model": SERVED_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **extra_payload,
        }
        r = requests.post(client_url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as ex:
        return list(ex.map(_one, prompts))


# --------------------------------------------------------------------------- #
# Capture comparison

# --------------------------------------------------------------------------- #
def count_captures(capture_dir: str, kind: str) -> int:
    return len(glob.glob(os.path.join(capture_dir, f"{kind}_*.pt")))


def wait_for_captures(capture_dir: str, kind: str, expected: int,
                      timeout: float = 120.0) -> int:
    """Wait until at least ``expected`` captures of ``kind`` exist.

    Raises AssertionError on timeout: a missing capture means the connector
    never exercised the code path under test, so the scenario must fail rather
    than silently verify fewer blocks.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        n = count_captures(capture_dir, kind)
        if n >= expected:
            logger.info("saw %d/%d %s captures", n, expected, kind)
            return n
        time.sleep(1.0)
    n = count_captures(capture_dir, kind)
    raise AssertionError(
        f"timed out waiting for {kind} captures: got {n}, want {expected}")


def compare_captures(capture_dir: str) -> dict:
    """Compare loaded captures against reference captures.

    Capture file names carry the tp rank, so no tp_size is needed.

    Every block that was *loaded* from KVCM must correspond to a *reference*
    capture (same tp rank + token content) with matching KV data. The direction
    matters: saves are incremental, so some saved blocks may legitimately not be
    reloaded (e.g. the tokenization boundary block) -- but every loaded block
    must match something that was saved.

    Returns a report dict; the caller asserts on it.
    """
    import torch

    refs = {}
    loaded = {}
    for path in glob.glob(os.path.join(capture_dir, "*.pt")):
        name = os.path.basename(path)[:-3]  # strip .pt
        parts = name.split("_")
        kind, tp, token_hash = parts[0], parts[1], "_".join(parts[2:])
        key = (tp, token_hash)
        (refs if kind == "ref" else loaded)[key] = path

    report = {
        "num_refs": len(refs),
        "num_loaded": len(loaded),
        "matched": 0,
        "bit_exact": 0,
        "failures": [],
        "loaded_without_ref": [],
        "matched_keys": [],
    }

    for key, loaded_path in sorted(loaded.items()):
        if key not in refs:
            report["loaded_without_ref"].append(key)
            continue
        ref = torch.load(refs[key], map_location="cpu", weights_only=True)
        got = torch.load(loaded_path, map_location="cpu", weights_only=True)

        assert ref["token_ids"] == got["token_ids"], f"token id mismatch for {key}"

        all_bit_exact = True
        # Compare every layer present in the *loaded* capture: each one was
        # actually written by the connector and must match its reference.
        # Mamba "align" state layers can legitimately be absent on either side
        # (vLLM materializes states only at segment boundaries; interior blocks
        # get the null block, which is neither saved nor loaded) -- but a
        # loaded layer without a reference is a hard error.
        for layer_name, got_kv in got["kv"].items():
            assert layer_name in ref["kv"], (
                f"loaded layer {layer_name} of {key} has no reference capture")
            ref_kv = ref["kv"][layer_name]
            # Attention groups are a single Tensor; mamba/linear/gdn groups are a
            # list[Tensor] (e.g. [conv_state, ssm_state]). Compare uniformly.
            if isinstance(ref_kv, (list, tuple)):
                ref_parts = list(ref_kv)
                got_parts = list(got_kv)
                assert len(ref_parts) == len(got_parts), (
                    f"state count mismatch {layer_name}: "
                    f"{len(ref_parts)} vs {len(got_parts)}"
                )
            else:
                ref_parts = [ref_kv]
                got_parts = [got_kv]

            for si, (ref_t, got_t) in enumerate(zip(ref_parts, got_parts)):
                assert ref_t.shape == got_t.shape, (
                    f"shape mismatch {layer_name}[{si}]: {ref_t.shape} vs {got_t.shape}"
                )
                # The transfer is a verbatim byte round trip, so the loaded data
                # must be bit-identical to what was saved -- no tolerance.
                if not torch.equal(ref_t, got_t):
                    all_bit_exact = False
                    report["failures"].append({
                        "key": key,
                        "layer": f"{layer_name}[{si}]",
                        "mismatched_elems": int((ref_t != got_t).sum()),
                        "num_elems": ref_t.numel(),
                    })

        report["matched"] += 1
        report["matched_keys"].append(key)
        if all_bit_exact:
            report["bit_exact"] += 1

    return report


def loaded_capture_keys(capture_dir: str) -> set:
    """The (tp rank, token-content hash) keys of the loaded captures."""
    keys = set()
    for path in glob.glob(os.path.join(capture_dir, "loaded_*.pt")):
        name = os.path.basename(path)[:-3]  # strip .pt
        parts = name.split("_")
        keys.add((parts[1], "_".join(parts[2:])))
    return keys


def assert_loaded_blocks(capture_dir: str, expected_keys: set, tp_size: int):
    """The loaded captures must be *exactly* the expected blocks, per tp rank.

    The load expectation is a *set*, not a count, and for a windowed model it
    is the union of every group's slots (``loaded_block_count``): the hit's
    tail window plus, when the model has a full-attention group, the whole
    matched prefix. Asserting the set catches both directions -- a load that
    stops at the window and a model whose windowed groups were served their
    out-of-window (null-page) positions -- even though the blocks that are
    loaded are bit-exact on their own.
    """
    actual = loaded_capture_keys(capture_dir)
    missing = expected_keys - actual
    extra = actual - expected_keys
    if missing or extra:
        raise AssertionError(
            f"loaded captures are not the expected region: missing {sorted(missing)}, "
            f"unexpected {sorted(extra)} (expected {len(expected_keys)} blocks "
            f"across {tp_size} rank(s), got {len(actual)})")


def assert_loaded_nonempty(capture_dir: str, tp_size: int):
    """Every loaded capture must carry real bytes in every layer.

    Used where the reference comparison is skipped on purpose (see
    test_cross_request_prefix): the refs there come from an *independent*
    recomputation of the same tokens, whose low bits legitimately differ, so
    the bit-exact comparison only applies to scenarios whose refs and loads
    come from the same run. An all-zero layer, though, would mean the load
    target was the shared null page -- a published block that is in fact
    empty -- and must fail everywhere.
    """
    import torch

    paths = sorted(glob.glob(os.path.join(capture_dir, "loaded_*.pt")))
    if not paths:
        raise AssertionError(f"no loaded captures in {capture_dir}")
    for path in paths:
        rec = torch.load(path, map_location="cpu", weights_only=True)
        for layer_name, kv in rec["kv"].items():
            parts = kv if isinstance(kv, (list, tuple)) else [kv]
            for part in parts:
                if part.numel() and float(part.float().abs().sum()) == 0.0:
                    raise AssertionError(
                        f"{os.path.basename(path)}: layer {layer_name} of a "
                        f"loaded block is all zero (null page, not loaded KV)")
    logger.info("loaded captures non-empty: %d file(s) over %d rank(s)",
                len(paths), tp_size)


def block_tables_from_logs(lines: list) -> int:
    """Block-table count from the connector's registration line(s).

    The connector's load-failure reporting mode depends on how many vLLM block
    tables the request carries (``report_failures = len(all_block_ids) == 1``),
    so the harness reads the same number out of its own registration log
    instead of guessing from the model family. Disagreeing registrations (e.g.
    a restart that resolved a different layout) are a hard error.
    """
    seen = {int(g[0]) if isinstance(g, tuple) else int(g) for g in lines}
    assert seen, ("the connector registration line carries no block_tables "
                  "count; the load-failure branch cannot be decided")
    assert len(seen) == 1, f"inconsistent block-table counts: {sorted(seen)}"
    return seen.pop()


def assert_pinned_ok(lines: list, limit_gib: float, model_path: str,
                     expected_ranks: int = 1):
    """Assert every `total pinned = X GiB` line of the run stays under budget.

    PR2 made the pinned total machine-parseable exactly so this gate can read
    it; the connector prints one line per pool-planning instance, i.e. one per
    tp rank (plus one per restarted server). Both failure directions are hard:
    a line above the budget, and fewer lines than ranks -- losing one worker's
    report must not pass on the surviving worker's value, otherwise the gate
    can be bypassed exactly when per-rank pool planning regresses.
    """
    values = [float(g[0]) if isinstance(g, tuple) else float(g) for g in lines]
    assert len(values) >= expected_ranks, (
        f"pinned report incomplete: {len(values)} 'total pinned' line(s) in "
        f"the connector logs of {model_path}, expected at least "
        f"{expected_ranks} (one per tp rank); the pinned gate cannot cover "
        f"every rank")
    if not values:
        raise AssertionError(
            f"no 'total pinned' line found in the connector logs of "
            f"{model_path}; the pinned gate cannot be evaluated")
    worst = max(values)
    if worst > limit_gib:
        raise AssertionError(
            f"pinned budget breached: {worst:.3f} GiB > {limit_gib} GiB "
            f"(all observations: {values})")
    logger.info("pinned gate OK: max %.3f GiB <= %.3f GiB (%d observation(s))",
                worst, limit_gib, len(values))


def assert_report_ok(report: dict, min_matched: int = 1):
    """Assert the comparison succeeded.

    min_matched is the exact lower bound of (ref, loaded) capture pairs computed
    from the prompts' tokenization (num full manager blocks x tp ranks); a lower
    count means some blocks were silently never saved or never loaded.
    """
    problems = []
    if report["loaded_without_ref"]:
        problems.append(
            f"loaded captures with no matching reference: {report['loaded_without_ref']}"
        )
    if report["failures"]:
        problems.append(f"bit-exact failures: {report['failures']}")
    if report["matched"] < min_matched:
        problems.append(
            f"matched {report['matched']} loaded captures, expected >= {min_matched}"
        )
    if problems:
        raise AssertionError("KV verification failed: " + "; ".join(problems))
    logger.info(
        "KV verification OK: matched=%d bit_exact=%d (refs=%d loaded=%d)",
        report["matched"], report["bit_exact"],
        report["num_refs"], report["num_loaded"],
    )


# --------------------------------------------------------------------------- #
# Scenario runner

def make_base_prompts(num_prompts: int, hybrid: bool,
                      window: Optional[int] = None) -> list[str]:
    """Distinct, deterministic prompts. Each sentence carries a unique counter
    so every manager block has unique token content -- this avoids hash
    collisions between blocks with identical text but different KV (RoPE is
    position-dependent).

    Hybrid models pin the manager block size to the scheduler block size (528),
    so their prompts must be much longer to span multiple manager blocks
    (empirically 140 sentences ~ 2100 tokens > 3 x 528). Full-attention models
    use manager blocks of 16/32 tokens, where 40 sentences (~580 tokens) already
    span dozens of blocks.

    Windowed models additionally need the prompt to cross the attention window
    repeatedly (the save/load block-count difference and the tail-window load
    are only meaningful past the first window): ask for ~2.5 windows, assuming
    a conservative 12 tokens per sentence (measured: 14.5 for gemma-3's
    tokenizer, 12.1 for gpt-oss's). run_e2e asserts the result really crosses
    the window twice, so a shorter tokenizer fails loudly instead of silently
    weakening the assertions.
    """
    num_sentences = 140 if hybrid else 40
    if window:
        num_sentences = max(num_sentences, window * 5 // 2 // 12 + 10)
    return [
        f"Prompt number {i}. " + " ".join(
            f"Sentence {j} of prompt {i} has value {j * 7 + i * 131}."
            for j in range(num_sentences)
        )
        for i in range(num_prompts)
    ]


def shared_token_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
