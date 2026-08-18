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



def is_hybrid_model(model_path: str) -> bool:
    """Detect a hybrid (mamba/linear + full attention) model from its config."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
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


def compare_captures(capture_dir: str, tp_size: int) -> dict:
    """Compare loaded captures against reference captures.

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

def make_base_prompts(num_prompts: int, hybrid: bool) -> list[str]:
    """Distinct, deterministic prompts. Each sentence carries a unique counter
    so every manager block has unique token content -- this avoids hash
    collisions between blocks with identical text but different KV (RoPE is
    position-dependent).

    Hybrid models pin the manager block size to the scheduler block size (528),
    so their prompts must be much longer to span multiple manager blocks
    (empirically 140 sentences ~ 2100 tokens > 3 x 528). Full-attention models
    use manager blocks of 16/32 tokens, where 40 sentences (~580 tokens) already
    span dozens of blocks.
    """
    num_sentences = 140 if hybrid else 40
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
