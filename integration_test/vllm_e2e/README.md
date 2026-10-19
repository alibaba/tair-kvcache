# vLLM <-> KVCM End-to-End KV Cache Verification

End-to-end integration tests for the KVCM vLLM connector
(`kv_cache_manager/py_connector/vllm`). Each test starts a real KVCM manager
(local-file storage backend) and a real vLLM OpenAI server, drives prompts
through the OpenAI API and verifies that the KV cache data saved to / loaded
from KVCM is correct.

Requires 1-2 GPUs and vLLM (0.22.1, 0.23.0, 0.26.0 and 0.31.0 are
e2e-verified; the connector detects the KV cache layout of each era from the
tensor shape).

## What is verified

The connector translates between three block spaces per `kv_cache_group`:

```
KVCM manager block idx  ->  global token idx  ->  group logical block
    (step 1, connector-only)     (step 2/3, shared with vLLM)
```

A bug in step 1 is *symmetric*: save gathers from the wrong slots and load
scatters back to the same wrong slots, so a transport round trip alone cannot
detect it. The test breaks the symmetry with `VerifyingConnector`
(`test_connector.py`), a subclass of the production connector that
independently captures KV data from vLLM's paged cache using only vLLM's own
block-table mapping:

1. **Phase 1** — fresh prompts: prefill -> connector saves to KVCM. The saved
   token ranges are captured from the paged cache (**reference** captures).
2. **Phase 2** — same prompts + suffix: connector reports an external match and
   loads from KVCM. The loaded blocks are captured (**loaded** captures).
3. The driver (`e2e_lib.py`) matches loaded captures against references by
   token content and compares per layer, requiring bit-exact equality (the
   transfer is a verbatim byte round trip; all scenarios achieve it).

## Model coverage

The same test targets run against either model kind, selected by
`KVCM_E2E_MODEL`:

| Kind | Example | Groups | Orchestration |
|---|---|---|---|
| Full attention | Qwen2.5-7B-Instruct | 1 `FullAttentionSpec` | prefix caching off, one server for both phases |
| MLA | GLM-4.7-Flash | 1 `MLAAttentionSpec` | same as full attention (the latent cache is token-granular: one vector per token instead of K+V; auto-selected MLA attention backend instead of the forced FLASH_ATTN) |
| Hybrid | Qwen3.5-4B | 3 `MambaSpec` + 1 `FullAttentionSpec` | prefix caching on (`mamba_cache_mode="align"`), server restarted between phases so phase 2 loads from KVCM instead of the local prefix cache |
| Windowed (SWA) | gemma-3-4b-it / gemma-3-1b-it / gpt-oss-20b | 6 `SlidingWindowSpec` + 1 `FullAttentionSpec` (5:1 / 1:1 interleaves). Two more retention shapes are recognised: a chunked-local config (`attention_chunk_size` + a `chunked_local` layer type) and R-SWA (`rswa_window`, or the architecture whose vLLM config class injects it -- `RSWASpec` keeps the whole prefill visible and evicts only the decode gap) | same as hybrid (multigroup): prefix caching on, restart between phases, and a bounded prefill chunk (`--max-num-batched-tokens 256`) so the window slides *inside* the prefill while the eager captures still fit the pinned staging arena |

Multi-group specifics verified:

* Per-group location specs (`tp{rank}_g{group}`) and per-group block tables.
* Attention groups: token-granular gather/scatter through the Triton kernel.
* Mamba/linear groups: per-block opaque state copy, where a manager block's
  *last* token selects the state block (`_state_block_ids`).
* Windowed groups: vLLM frees the blocks outside the window while the request
  lives and parks their table entries on the null block, so
  - the harness *reference* capture is driven by the connector's capture
    requests (the same scheduler iteration that produced the bytes) -- a
    save-request-driven capture would read the shared zero page;
  - the retention rule follows the window's kind (`window_kind`, named after
    the connector's `WindowRetention.kind`): a sliding window keeps the last
    `sliding_window` tokens (plus `extra_retained_tokens`, read from the
    connector's own registration), a chunked-local one keeps the *current
    chunk* (`H // chunk_size * chunk_size` skipped), and an R-SWA one keeps
    everything (its manager never parks a block on the null page), so the
    expected region is computed with the same rule vLLM allocates with. The
    registered retention is also cross-checked against the config-derived one:
    a windowed group the detection missed fails loudly instead of verifying the
    wrong region;
  - every scenario that builds a window-scaled prompt gates it on the
    *effective* `--max-model-len` (the started server's args, so a
    `KVCM_E2E_VLLM_ARGS` override counts) and skips with the numbers when the
    prompt cannot fit, instead of letting the server answer 400;
  - the connector saves the whole prefix (historian), while a *hit* gets slots
    per group: every group keeps the tail window of the match
    (`cdiv(H, mbs) - max(0, H - (window-1)) // mbs`) and a full-attention group
    (gemma-3 and gpt-oss both have one) keeps the whole matched prefix, so the
    harness asserts the loaded blocks are exactly that union -- a load that
    stops at the window, or one with a *windowed* group's out-of-window block
    in it, fails;
  - the loaded captures narrow per (group, block) for the same reason: the
    full group's out-of-window blocks are really loaded and stay under the
    byte-exact comparison, while the windowed groups' null-page positions are
    not captured (they hold no KV of this request);
  - the loaded captures additionally require vLLM's shared null block to stay
    all-zero at the positions they filter out: the filter exists to skip slots
    vLLM never allocated, and without that read it would also hide a connector
    that wrote into them;
  - the harness checks the pinned-host budget from the engine log
    (`total pinned = X GiB`), configured per model *fingerprint*
    (`architectures[0]` + declared window, so a copied or renamed checkpoint
    keeps its gate), and requires one report per TP rank. A model without a
    budget logs `pinned gate SKIPPED: ...` at WARNING level instead of skipping
    silently.

  The chunk size is a deployment constraint, not a tuning knob: a capture holds
  its arena slots until its write session completes, so the pinned arena must
  hold the captures of the previous *and* the current chunk of every
  concurrently prefilling request (~2 x prefill_concurrency x windowed_groups x
  chunk_blocks). Deployments that raise the chunk size past roughly 512 tokens
  with several concurrent prefills need a larger
  `staging_pool_max_bytes_per_group`; a capture that cannot get slots is
  withheld (fail-closed: blocks are failed, never published empty).

## Scenarios

| Test | TP | Prompts | Notes |
|---|---|---|---|
| `test_basic` | 1 | 1 | Minimal save -> load round trip |
| `test_concurrent` | 1 | 4 | Concurrent requests: ReqState tracking, per-request block attribution |
| `test_tp` | 2 | 2 | TP coordination; for full-attention models also `preferred_block_size=32` != vLLM block size (16), forcing real cross-block translation |
| `test_partial_hit` | 1 | 1 | Phase 2 extends the prompt mid-block: partial external hit. Windowed: phase 3's load covers exactly the slots vLLM handed out for the extended prefix (the sliding groups' tail window plus the full-attention group's whole match) |
| `test_partial_hit::test_mid_prefix_hit` | 1 | 1 | Windowed only: the hit ends in the middle of a saved request, so the sliding groups' loaded window is a *middle* window of A (asserted disjoint from A's tail window) |
| `test_full_hit` | 1 | 1 | Phase 2 resends the identical prompt: full-prompt hit is capped so >= 1 token is recomputed |
| `test_multi_turn` | 1 | 1 | Growing conversation: each turn loads the previous turns' blocks and saves new ones |
| `test_cross_request_prefix` | 1 | 2 | Request B is a strict token prefix of saved request A, ending inside one of A's blocks. Hybrid: B's match must be truncated to the last block whose recurrent state was really materialized, and B's output must be token-identical to a no-cache reference |
| `test_load_failure` | 1 | 1 | Storage files deleted between phases: load fails, retry loop must not spin, request still completes. The branch follows the *registered* block-table count: a single table reports the invalid blocks to vLLM, several tables cannot and must log the swallowed failure. Single-table runs verify the surviving loads over the geometry's region intersected with the undeleted head (the whole hit for full attention and R-SWA, the tail window for a purely windowed model), and the reporting-mode return still reads the capture marker |
| `test_mutation` | 1 | 1 | Meta-test: injected off-by-one in the slot translation must make verification FAIL (proves the harness is not vacuous). Only a *verification assertion* counts as the expected catch -- a capture implementation error (an index or device fault) fails the meta-test too, so a broken capture cannot masquerade as a detected corruption |

## Running

These targets are tagged `manual`: they need a GPU machine with a prepared
vLLM venv and a local model, so `bazelisk test //integration_test/...` skips
them and they must be requested explicitly (see below).

Build prerequisites (from the repo root):

```bash
bazelisk build //kv_cache_manager:kv_cache_manager_bin \
  //kv_cache_manager/client/pybind:kvcm_py_client_lib_wheel \
  //kv_cache_manager/py_connector/vllm:kvcm_vllm_connector_wheel \
  --per_file_copt='external/jsoncpp_git/.*@-Wno-error'
```

Install both wheels into the vLLM venv (rename them first: the Bazel output
name contains unstamped `{STABLE_*}` template variables; read the real version
from the wheel's `METADATA`).

Run (tagged `exclusive`, so they execute serially):

```bash
bazelisk test //integration_test/vllm_e2e:e2e_tests \
  --cache_test_results=no --test_output=errors \
  --test_env=KVCM_E2E_PYTHON=/path/to/vllm-venv/bin/python \
  --test_env=KVCM_E2E_MODEL=/path/to/model \
  --per_file_copt='external/jsoncpp_git/.*@-Wno-error'
```

## Environment variables

All environment variables used by the e2e harness:

| Variable | Required | Meaning |
|---|---|---|
| `KVCM_E2E_MODEL` | yes | Path to a local HF model directory (`config.json` + weights). Full-attention coverage needs a plain attention model (e.g. Qwen2.5-7B-Instruct); MLA coverage needs an MLA model (e.g. GLM-4.7-Flash); hybrid coverage needs a mamba/linear + attention model (e.g. Qwen3.5-4B). Both kinds are auto-detected from `config.json`. |
| `KVCM_E2E_PYTHON` | yes | Python interpreter of a venv with vLLM (any supported version, see above) and both KVCM wheels (`kvcm_py_client`, `kvcm_vllm_connector`) installed. |
| `KVCM_E2E_TP` | no | Overrides every scenario's tensor-parallel size (default 1). For models whose (quantized) weights do not fit a single GPU, e.g. GLM-4.7-Flash fp8 on 24 GB cards, set this to 2; `test_tp` then adds no new topology and can be skipped. |
| `KVCM_E2E_VLLM_ARGS` | no | JSON dict of extra vLLM CLI flags for the spawned server (keys without `--`), e.g. `{"quantization": "fp8"}`. Wins over per-scenario defaults. |
| `KVCM_E2E_PINNED_GIB` | no | Pinned-host budget (GiB) for the engine-log gate, overriding the per-model table. Unset for a model with no table entry = gate skipped. |
| `KVCM_E2E_CAPTURE_DIR` | internal | Set by the driver for the vLLM subprocess; tells `VerifyingConnector` where to write `.pt` captures. Do not set manually. |

The driver also prepends the venv's `nvidia/{cu13,cublas,cudnn}/lib` directories to
`LD_LIBRARY_PATH` for the spawned server (existing entries are preserved, a
directory already present is skipped, absent directories are skipped):
multimodal checkpoints such as gemma-3 lazily dlopen cuBLAS/cuDNN from the
vision tower and core-dump without those paths. For a library that exists in
both places the venv's copy wins -- it is the one this vLLM was built against.
cuBLAS sits in `cu13` on CUDA-13 wheels and in `cublas` on the CUDA 11/12 ones,
so both are probed (absent ones are skipped).

The driver also sets vLLM knobs for the spawned server (`VLLM_KV_CACHE_LAYOUT=NHD`,
`VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
`FLASHINFER_DISABLE_VERSION_CHECK=1`) via `env.setdefault`, so a value you
export yourself wins. MLA models are exempt from the `VLLM_ATTENTION_BACKEND`
default (FLASH_ATTN has no MLA implementation; vLLM auto-selects an MLA
backend -- every one of them registers the same 3-D latent cache layout).

## Debugging

Bazel's `test.log` only shows the driver's view (e.g. HTTP 500). The real
tracebacks live in the scenario workdir under `$TEST_TMPDIR`:

```
<TEST_TMPDIR>/kvcm_vllm_e2e/<scenario>/
  manager/manager.stdout|stderr     # KVCM manager
  vllm/vllm*.stdout|stderr          # vLLM (EngineCore tracebacks are here)
  captures/{ref|loaded}_tp{rank}_{token_hash}.pt
```
