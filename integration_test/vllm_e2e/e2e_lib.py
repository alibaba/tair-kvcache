"""Orchestration for the KVCM <-> vLLM end-to-end KV cache verification test.

This module is imported by the Bazel ``py_test`` targets. It:

1. Starts a KVCM manager (``kv_cache_manager_bin``) with a local-file storage
   backend.
2. Starts a vLLM OpenAI server configured with the ``VerifyingConnector``
   (injected via ``kv_connector_module_path`` -- no vLLM files are modified).
3. Drives prompts through the OpenAI API in two phases and compares the
   independently-captured KV data (reference from the save path vs loaded from
   the load path).

The comparison is done on KV data captured from vLLM's paged cache using vLLM's
own block-table mapping (independent of the connector's per-group translation),
which is what makes the test able to detect symmetric save/load translation
bugs. See ``test_connector.py`` for the capture-side details.

Full-attention vs hybrid models
-------------------------------
The same test targets run against either model, selected by ``$KVCM_E2E_MODEL``:

* Full-attention (e.g. Qwen2.5): a single ``FullAttentionSpec`` group. Prefix
  caching is disabled so every phase-2 request is served through the connector
  (no local prefix hit); the two phases share one server.
* Hybrid (e.g. Qwen3.5): several ``MambaSpec`` groups plus a ``FullAttentionSpec``
  group. Prefix caching must be enabled for vLLM to produce per-group block
  tables (``mamba_cache_mode="align"``). Because that also populates the local
  prefix cache, the vLLM server is restarted between the two phases so phase 2
  genuinely loads from KVCM instead of hitting the local cache.
"""

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
    # Hybrid models interleave linear/mamba layers with full attention and
    # expose a full_attention_interval / linear_* knob.
    return (
        "full_attention_interval" in text_cfg
        or "linear_conv_kernel_dim" in text_cfg
        or cfg.get("model_type", "").startswith("qwen3_5")
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
class ManagerProcess:
    def __init__(self, workdir: str, storage_root: str, key_count_per_file: int = 8):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.rpc_port = free_port()
        self.http_port = free_port()
        self.admin_rpc_port = free_port()
        self.admin_http_port = free_port()
        self.storage_root = storage_root
        self.key_count_per_file = key_count_per_file
        self.proc: Optional[subprocess.Popen] = None
        self.config_path = os.path.join(workdir, "startup_config.json")

    def manager_uri(self) -> str:
        return f"http://127.0.0.1:{self.http_port}"

    def _write_config(self):
        cfg = {
            "storage_config": {
                "type": "file",
                "global_unique_name": "nfs_01",
                "storage_spec": {
                    # The backend concatenates root_path + key with no
                    # separator; the trailing slash keeps files inside the dir.
                    "root_path": self.storage_root.rstrip("/") + "/",
                    "key_count_per_file": self.key_count_per_file,
                },
            },
            "instance_group": {
                "name": "default",
                "storage_candidates": ["nfs_01"],
                "global_quota_group_name": "default_quota_group",
                "max_instance_count": 100,
                "quota": {
                    "capacity": 30000000000,
                    "quota_config": [
                        {"storage_type": "file", "capacity": 10000000000},
                        {"storage_type": "hf3fs", "capacity": 10000000000},
                        {"storage_type": "pace", "capacity": 10000000000},
                    ],
                },
                "cache_config": {
                    "reclaim_strategy": {
                        "reclaim_policy": 1,
                        "trigger_strategy": {"used_percentage": 0.8},
                        "delay_before_delete_ms": 1000,
                    },
                    "cache_prefer_strategy": 2,
                    "meta_indexer_config": {
                        "max_key_count": 1000000,
                        "mutex_shard_num": 16,
                        "batch_key_size": 16,
                        "meta_storage_backend_config": {
                            "storage_type": "local",
                            "storage_uri": "",
                        },
                        "meta_cache_policy_config": {
                            "type": "LRU",
                            "capacity": 10000,
                            "cache_shard_bits": 0,
                            "high_pri_pool_ratio": 0.0,
                        },
                    },
                },
                "user_data": '{"description": "vllm e2e test instance group"}',
                "version": 1,
            },
        }
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def start(self, repo_root: str):
        self._write_config()
        binary = find_manager_binary(repo_root)
        cmd = [
            binary,
            "--env", f"kvcm.service.rpc_port={self.rpc_port}",
            "--env", f"kvcm.service.http_port={self.http_port}",
            "--env", f"kvcm.service.admin_rpc_port={self.admin_rpc_port}",
            "--env", f"kvcm.service.admin_http_port={self.admin_http_port}",
            "--env", f"kvcm.startup_config={self.config_path}",
            "--env", "kvcm.logger.log_level=5",
        ]
        logger.info("starting manager: %s (cwd=%s)", " ".join(cmd), self.workdir)
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            stdout=open(os.path.join(self.workdir, "manager.stdout"), "w"),
            stderr=open(os.path.join(self.workdir, "manager.stderr"), "w"),
        )
        if not wait_http(
            f"{self.manager_uri()}/api/getClusterInfo",
            timeout=60,
            post_body={"trace_id": "probe", "instance_id": "probe"},
        ):
            raise RuntimeError("manager did not become ready; see manager.stderr")
        logger.info("manager ready at %s", self.manager_uri())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# vLLM server
# --------------------------------------------------------------------------- #
class VllmServer:
    def __init__(self, workdir: str, capture_dir: str, manager_uri: str,
                 tp_size: int, coordinator_base_port: int,
                 instance_id: str, preferred_block_size: int,
                 enable_prefix_caching: bool,
                 connector_name: str = "VerifyingConnector",
                 log_level: str = "INFO",
                 extra_config_overrides: Optional[dict] = None,
                 kv_load_failure_policy: Optional[str] = None):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.capture_dir = capture_dir
        os.makedirs(capture_dir, exist_ok=True)
        self.port = free_port()
        self.manager_uri = manager_uri
        self.tp_size = tp_size
        self.coordinator_base_port = coordinator_base_port
        self.instance_id = instance_id
        self.preferred_block_size = preferred_block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.connector_name = connector_name
        self.log_level = log_level
        self.extra_config_overrides = extra_config_overrides or {}
        self.kv_load_failure_policy = kv_load_failure_policy
        self.proc: Optional[subprocess.Popen] = None

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, repo_root: str, log_suffix: str = ""):
        extra_config = {
            "manager_uri": self.manager_uri,
            "coordinator_base_port": self.coordinator_base_port,
            "instance_group": "default",
            "instance_id": self.instance_id,
            "preferred_block_size": self.preferred_block_size,
            "log_level": self.log_level,
        }
        extra_config.update(self.extra_config_overrides)
        kv_transfer_config = {
            "kv_connector": self.connector_name,
            "kv_role": "kv_both",
            "kv_connector_module_path": "test_connector",
            "kv_connector_extra_config": extra_config,
        }
        if self.kv_load_failure_policy:
            kv_transfer_config["kv_load_failure_policy"] = self.kv_load_failure_policy
        cmd = [
            find_python(), "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_PATH,
            "--served-model-name", SERVED_MODEL_NAME,
            "--host", "127.0.0.1",  # loopback only: single-machine harness
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp_size),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.85",
            "--enforce-eager",
            "--max-num-seqs", "16",
            "--kv-transfer-config", json.dumps(kv_transfer_config),
        ]
        if self.enable_prefix_caching:
            # Hybrid models need prefix caching to expose per-group block tables
            # (mamba_cache_mode="align"); align mode requires chunked prefill.
            cmd += ["--enable-prefix-caching", "--enable-chunked-prefill"]
        else:
            cmd += ["--no-enable-prefix-caching"]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
        env["KVCM_E2E_CAPTURE_DIR"] = self.capture_dir
        # Keep the connector's KV cache layout matching its expected
        # [2, num_blocks, block_size, num_kv_heads, head_size] shape.
        env.setdefault("VLLM_KV_CACHE_LAYOUT", "NHD")
        # Force FlashAttention for the full-attention layers: it produces the
        # [2, num_blocks, block_size, num_kv_heads, head_size] layout the
        # connector expects, and avoids the flashinfer backend entirely.
        env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        # Use the PyTorch-native sampler; the flashinfer sampler JIT-compiles
        # with ninja, which is not available in the test environment.
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # The test venv ships mismatched flashinfer / flashinfer-cubin wheels;
        # skip the version check so importing vLLM's attention registry does not
        # crash before the FlashAttention backend is selected.
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        logger.info("starting vllm: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            env=env,
            stdout=open(os.path.join(self.workdir, f"vllm{log_suffix}.stdout"), "w"),
            stderr=open(os.path.join(self.workdir, f"vllm{log_suffix}.stderr"), "w"),
        )
        if not wait_http(f"{self.base_url()}/health", timeout=600):
            raise RuntimeError("vllm did not become ready; see vllm.stderr")
        logger.info("vllm ready at %s", self.base_url())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# Request driver
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
        # get the null block and the connector transfers them vacuously) -- but
        # a loaded layer without a reference is a hard error.
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
# --------------------------------------------------------------------------- #
class ScenarioEnv:
    """Owns one scenario's manager + vLLM server lifecycle and scratch dirs.

    Custom scenarios (partial-hit, full-hit, load-failure, multi-turn) share
    this; run_e2e keeps its own two-phase flow on top of the same pieces.
    """

    def __init__(self, scenario: str, tp_size: int = 1,
                 preferred_block_size: int = 0,
                 enable_prefix_caching: Optional[bool] = None,
                 connector_name: str = "VerifyingConnector",
                 log_level: str = "INFO",
                 extra_config_overrides: Optional[dict] = None,
                 key_count_per_file: int = 8,
                 kv_load_failure_policy: Optional[str] = None):
        self.scenario = scenario
        self.tp_size = tp_size
        self.hybrid = is_hybrid_model(MODEL_PATH)
        self.preferred_block_size = 0 if self.hybrid else preferred_block_size
        self.enable_prefix_caching = (self.hybrid if enable_prefix_caching is None
                                      else enable_prefix_caching)
        self.connector_name = connector_name
        self.log_level = log_level
        self.extra_config_overrides = extra_config_overrides
        self.kv_load_failure_policy = kv_load_failure_policy

        self.repo_root = find_repo_root()
        scratch_root = (os.environ.get("TEST_TMPDIR")
                        or os.environ.get("TMPDIR") or "/tmp")
        self.base_workdir = os.path.join(scratch_root, "kvcm_vllm_e2e", scenario)
        if os.path.exists(self.base_workdir):
            shutil.rmtree(self.base_workdir)
        self.storage_root = os.path.join(self.base_workdir, "nfs")
        self.capture_dir = os.path.join(self.base_workdir, "captures")
        self.vllm_dir = os.path.join(self.base_workdir, "vllm")
        os.makedirs(self.storage_root, exist_ok=True)

        self.instance_id = f"e2e-{scenario}-{uuid.uuid4().hex[:8]}"
        self.manager = ManagerProcess(
            os.path.join(self.base_workdir, "manager"), self.storage_root,
            key_count_per_file=key_count_per_file)
        self.vllm: Optional[VllmServer] = None

    def start_manager(self):
        self.manager.start(self.repo_root)

    def start_vllm(self, log_suffix: str = "") -> VllmServer:
        self.vllm = VllmServer(
            self.vllm_dir, self.capture_dir, self.manager.manager_uri(),
            self.tp_size, coordinator_base_port=free_port(),
            instance_id=self.instance_id,
            preferred_block_size=self.preferred_block_size,
            enable_prefix_caching=self.enable_prefix_caching,
            connector_name=self.connector_name,
            log_level=self.log_level,
            extra_config_overrides=self.extra_config_overrides,
            kv_load_failure_policy=self.kv_load_failure_policy,
        )
        self.vllm.start(self.repo_root, log_suffix=log_suffix)
        return self.vllm

    def restart_vllm(self, log_suffix: str = "") -> VllmServer:
        """Restart vLLM to drop its local prefix cache (KVCM state persists)."""
        if self.vllm:
            self.vllm.stop()
        return self.start_vllm(log_suffix)

    def manager_block_size(self) -> int:
        return get_manager_block_size(self.manager.manager_uri(), self.instance_id)

    def scan_connector_logs(self, pattern: str) -> list:
        """Regex-scan all vLLM std streams; returns list of match groups."""
        import re
        out = []
        for path in glob.glob(os.path.join(self.vllm_dir, "vllm*.std*")):
            with open(path, errors="replace") as f:
                for line in f:
                    m = re.search(pattern, line)
                    if m:
                        out.append(m.groups() if m.groups() else m.group(0))
        return out

    def stop(self):
        if self.vllm:
            self.vllm.stop()
        self.manager.stop()


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


def run_e2e(scenario: str, tp_size: int, num_prompts: int,
            preferred_block_size: int, connector_name: str = "VerifyingConnector",
            expect_verification_failure: bool = False):
    """Run one full save-then-load verification scenario.

    Full-attention models: prefix caching off, one server across both phases.
    Hybrid models: prefix caching on (align mode -> per-group block tables), the
    vLLM server is restarted between phases so phase 2 loads from KVCM instead of
    hitting the local prefix cache.

    connector_name selects the connector class inside test_connector.py; the
    mutation meta-test passes "MutatedConnector" and sets
    expect_verification_failure=True to prove the harness detects an injected
    off-by-one in the token translation.
    """
    import torch  # noqa: F401  (ensure torch importable early for clear errors)

    env = ScenarioEnv(scenario, tp_size=tp_size,
                      preferred_block_size=preferred_block_size,
                      connector_name=connector_name)
    hybrid = env.hybrid
    capture_dir = env.capture_dir
    logger.info("scenario=%s model=%s hybrid=%s tp=%d prompts=%d preferred_bs=%d",
                scenario, MODEL_PATH, hybrid, tp_size, num_prompts,
                env.preferred_block_size)

    try:
        env.start_manager()
        vllm = env.start_vllm(log_suffix="" if not hybrid else "_p1")

        base_prompts = make_base_prompts(num_prompts, hybrid)
        suffixes = [f" Now answer question {i}: what is 2+2?" for i in range(num_prompts)]
        phase2_prompts = [p + s for p, s in zip(base_prompts, suffixes)]

        # Compute per-prompt expected block counts from the actual tokenization
        # so a silently dropped prompt (or block) fails the run.
        mbs = env.manager_block_size()
        base_tokens = [tokenize(vllm.base_url(), p) for p in base_prompts]
        phase2_tokens = [tokenize(vllm.base_url(), p) for p in phase2_prompts]
        expected_save_blocks = [len(t) // mbs for t in base_tokens]
        # Loads cover the shared token prefix (the base/suffix boundary token may
        # re-merge under tokenization, shortening the shared prefix by one).
        expected_load_blocks = [
            min(shared_token_prefix_len(b, p2) // mbs, s)
            for b, p2, s in zip(base_tokens, phase2_tokens, expected_save_blocks)
        ]
        assert all(n >= 1 for n in expected_load_blocks), (
            f"prompts too short to span a manager block (mbs={mbs}): "
            f"{expected_load_blocks}")
        logger.info("mbs=%d expected save blocks=%s load blocks=%s",
                    mbs, expected_save_blocks, expected_load_blocks)

        # ---- Phase 1: fresh prefill -> connector saves -> reference capture.
        logger.info("phase 1: sending %d fresh prompts", num_prompts)
        send_completions(vllm.base_url(), base_prompts)
        wait_for_captures(capture_dir, "ref",
                          expected=tp_size * sum(expected_save_blocks), timeout=180)

        # The save is committed to the manager asynchronously after the ref
        # capture (which fires when the save is submitted). Wait until the manager
        # actually has the prefix, otherwise phase 2 would find no match.
        for toks, blocks in zip(base_tokens, expected_save_blocks):
            if not wait_for_prefix_cached(env.manager.manager_uri(), env.instance_id,
                                          toks, min_blocks=blocks):
                raise AssertionError("save was not committed to the manager in time")

        # Hybrid models keep prefix caching on, which also populates the local
        # prefix cache; restart vLLM so phase 2 loads from KVCM, not locally.
        if hybrid:
            logger.info("restarting vLLM before phase 2 (clear local prefix cache)")
            vllm = env.restart_vllm(log_suffix="_p2")

        # ---- Phase 2: same prefix + suffix -> connector loads -> loaded capture.
        logger.info("phase 2: sending %d prefix+suffix prompts", num_prompts)
        send_completions(vllm.base_url(), phase2_prompts)
        wait_for_captures(capture_dir, "loaded",
                          expected=tp_size * sum(expected_load_blocks), timeout=180)

        report = compare_captures(capture_dir, tp_size)
        min_matched = tp_size * sum(expected_load_blocks)
        if expect_verification_failure:
            try:
                assert_report_ok(report, min_matched=min_matched)
            except AssertionError as e:
                logger.info("verification failed as expected: %s", e)
                return
            raise AssertionError(
                "mutated connector passed KV verification; the harness is blind")
        assert_report_ok(report, min_matched=min_matched)
        logger.info("scenario %s PASSED: %s", scenario, json.dumps(
            {k: v for k, v in report.items() if k not in ("failures", "matched_keys")},
            default=str))
    finally:
        env.stop()
