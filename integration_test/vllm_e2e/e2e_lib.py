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

MODEL_PATH = os.environ.get("KVCM_E2E_MODEL", "/root/ws/resources/models/Qwen2.5-7B-Instruct")
COSINE_THRESHOLD = 0.9999


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
    return os.environ.get("KVCM_E2E_PYTHON", "/root/ws/env/global_vllm/.venv/bin/python")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
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
    def __init__(self, workdir: str, storage_root: str):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.rpc_port = free_port()
        self.http_port = free_port()
        self.admin_rpc_port = free_port()
        self.admin_http_port = free_port()
        self.storage_root = storage_root
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
                    "root_path": self.storage_root,
                    "key_count_per_file": 8,
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
                 enable_prefix_caching: bool):
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
            "log_level": "INFO",
        }
        kv_transfer_config = {
            "kv_connector": "VerifyingConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "test_connector",
            "kv_connector_extra_config": extra_config,
        }
        cmd = [
            find_python(), "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_PATH,
            "--served-model-name", "qwen",
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
                      json={"model": "qwen", "prompt": prompt}, timeout=60)
    r.raise_for_status()
    return r.json()["tokens"]


def wait_for_prefix_cached(manager_uri: str, instance_id: str,
                           token_ids: list[int], timeout: float = 120.0) -> bool:
    """Poll the manager until the prefix for token_ids is cached (committed).

    Mirrors what the connector's get_num_new_matched_tokens queries
    (query_type=QT_PREFIX_MATCH, block_mask offset=0 for a fresh request), so it
    guarantees phase 2 will actually hit the external cache.
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
                    if locs:
                        logger.info("prefix cached: %d location(s)", len(locs))
                        return True
        except Exception:
            pass
        time.sleep(1.0)
    logger.warning("timed out waiting for prefix to be cached")
    return False


def send_completions(base_url: str, prompts: list[str], max_tokens: int = 4,
                     temperature: float = 0.0) -> list[dict]:
    """Send prompts concurrently and return the OpenAI responses."""
    from concurrent.futures import ThreadPoolExecutor

    client_url = f"{base_url}/v1/completions"

    def _one(prompt: str) -> dict:
        payload = {
            "model": "qwen",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
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
                      timeout: float = 120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        n = count_captures(capture_dir, kind)
        if n >= expected:
            logger.info("saw %d/%d %s captures", n, expected, kind)
            return n
        time.sleep(1.0)
    n = count_captures(capture_dir, kind)
    logger.warning("timed out waiting for %s captures: got %d, want %d",
                   kind, n, expected)
    return n


def _cosine(a, b) -> float:
    import torch
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


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
        "cosine_pass": 0,
        "failures": [],
        "loaded_without_ref": [],
    }

    for key, loaded_path in sorted(loaded.items()):
        if key not in refs:
            report["loaded_without_ref"].append(key)
            continue
        ref = torch.load(refs[key], map_location="cpu", weights_only=True)
        got = torch.load(loaded_path, map_location="cpu", weights_only=True)

        assert ref["token_ids"] == got["token_ids"], f"token id mismatch for {key}"

        all_bit_exact = True
        worst_cosine = 1.0
        for layer_name, ref_kv in ref["kv"].items():
            got_kv = got["kv"][layer_name]
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
                if not torch.equal(ref_t, got_t):
                    all_bit_exact = False
                    cos = _cosine(ref_t, got_t)
                    worst_cosine = min(worst_cosine, cos)
                    if cos < COSINE_THRESHOLD:
                        report["failures"].append({
                            "key": key,
                            "layer": f"{layer_name}[{si}]",
                            "cosine": cos,
                        })

        report["matched"] += 1
        if all_bit_exact:
            report["bit_exact"] += 1
        else:
            report["cosine_pass"] += 1
            logger.warning("capture %s not bit-exact (worst cosine=%.6f)",
                           key, worst_cosine)

    return report


def assert_report_ok(report: dict):
    problems = []
    if report["loaded_without_ref"]:
        problems.append(
            f"loaded captures with no matching reference: {report['loaded_without_ref']}"
        )
    if report["failures"]:
        problems.append(f"cosine failures: {report['failures']}")
    if report["matched"] == 0:
        problems.append("no loaded captures were matched against references")
    if problems:
        raise AssertionError("KV verification failed: " + "; ".join(problems))
    logger.info(
        "KV verification OK: matched=%d bit_exact=%d cosine_pass=%d (refs=%d loaded=%d)",
        report["matched"], report["bit_exact"], report["cosine_pass"],
        report["num_refs"], report["num_loaded"],
    )


# --------------------------------------------------------------------------- #
# Scenario runner
# --------------------------------------------------------------------------- #
def run_e2e(scenario: str, tp_size: int, num_prompts: int,
            preferred_block_size: int):
    """Run one full save-then-load verification scenario.

    Full-attention models: prefix caching off, one server across both phases.
    Hybrid models: prefix caching on (align mode -> per-group block tables), the
    vLLM server is restarted between phases so phase 2 loads from KVCM instead of
    hitting the local prefix cache.
    """
    import torch  # noqa: F401  (ensure torch importable early for clear errors)

    hybrid = is_hybrid_model(MODEL_PATH)
    # Hybrid mamba state is per scheduler block, so preferred_block_size can only
    # differ from the vLLM block size for pure-attention models.
    if hybrid:
        preferred_block_size = 0

    repo_root = find_repo_root()
    scratch_root = os.environ.get("TEST_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    base_workdir = os.path.join(scratch_root, "kvcm_vllm_e2e", scenario)
    if os.path.exists(base_workdir):
        shutil.rmtree(base_workdir)
    storage_root = os.path.join(base_workdir, "nfs")
    manager_dir = os.path.join(base_workdir, "manager")
    vllm_dir = os.path.join(base_workdir, "vllm")
    capture_dir = os.path.join(base_workdir, "captures")
    os.makedirs(storage_root, exist_ok=True)

    instance_id = f"e2e-{scenario}-{uuid.uuid4().hex[:8]}"
    logger.info("scenario=%s model=%s hybrid=%s tp=%d prompts=%d preferred_bs=%d",
                scenario, MODEL_PATH, hybrid, tp_size, num_prompts, preferred_block_size)

    manager = ManagerProcess(manager_dir, storage_root)

    def make_server(coordinator_port):
        return VllmServer(
            vllm_dir, capture_dir, manager.manager_uri(), tp_size,
            coordinator_base_port=coordinator_port,
            instance_id=instance_id,
            preferred_block_size=preferred_block_size,
            enable_prefix_caching=hybrid,
        )

    vllm = make_server(free_port())

    try:
        manager.start(repo_root)
        vllm.start(repo_root, log_suffix="" if not hybrid else "_p1")

        # Distinct, deterministic prompts. Each sentence carries a unique counter
        # so every manager block has unique token content -- this avoids hash
        # collisions between blocks with identical text but different KV (RoPE is
        # position-dependent). Long enough to span several manager blocks so the
        # translation layer is actually exercised.
        base_prompts = [
            f"Prompt number {i}. " + " ".join(
                f"Sentence {j} of prompt {i} has value {j * 7 + i * 131}."
                for j in range(40)
            )
            for i in range(num_prompts)
        ]
        suffixes = [f" Now answer question {i}: what is 2+2?" for i in range(num_prompts)]

        # ---- Phase 1: fresh prefill -> connector saves -> reference capture.
        logger.info("phase 1: sending %d fresh prompts", num_prompts)
        send_completions(vllm.base_url(), base_prompts)
        wait_for_captures(capture_dir, "ref", expected=num_prompts, timeout=180)

        # The save is committed to the manager asynchronously after the ref
        # capture (which fires when the save is submitted). Wait until the manager
        # actually has the prefix, otherwise phase 2 would find no match.
        phase2_prompts = [p + s for p, s in zip(base_prompts, suffixes)]
        for p in base_prompts:
            toks = tokenize(vllm.base_url(), p)
            if not wait_for_prefix_cached(manager.manager_uri(), instance_id, toks):
                raise AssertionError("save was not committed to the manager in time")

        # Hybrid models keep prefix caching on, which also populates the local
        # prefix cache; restart vLLM so phase 2 loads from KVCM, not locally.
        if hybrid:
            logger.info("restarting vLLM before phase 2 (clear local prefix cache)")
            vllm.stop()
            vllm = make_server(free_port())
            vllm.start(repo_root, log_suffix="_p2")

        # ---- Phase 2: same prefix + suffix -> connector loads -> loaded capture.
        logger.info("phase 2: sending %d prefix+suffix prompts", num_prompts)
        send_completions(vllm.base_url(), phase2_prompts)
        wait_for_captures(capture_dir, "loaded", expected=num_prompts, timeout=180)

        report = compare_captures(capture_dir, tp_size)
        assert_report_ok(report)
        logger.info("scenario %s PASSED: %s", scenario, json.dumps(
            {k: v for k, v in report.items() if k != "failures"}, default=str))
    finally:
        vllm.stop()
        manager.stop()
