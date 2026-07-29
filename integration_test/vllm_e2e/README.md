# vLLM <-> KVCM End-to-End KV Cache Verification

End-to-end integration tests for the KVCM vLLM connector
(`kv_cache_manager/py_connector/vllm`). Each test starts a real KVCM manager
(local-file storage backend) and a real vLLM OpenAI server, drives prompts
through the OpenAI API and verifies that the KV cache data saved to / loaded
from KVCM is correct.

Requires 1-2 GPUs and vLLM >= 0.26.0.

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
| Hybrid | Qwen3.5-4B | 3 `MambaSpec` + 1 `FullAttentionSpec` | prefix caching on (`mamba_cache_mode="align"`), server restarted between phases so phase 2 loads from KVCM instead of the local prefix cache |

Hybrid specifics verified:

* Per-group location specs (`tp{rank}_g{group}`) and per-group block tables.
* Attention groups: token-granular gather/scatter through the Triton kernel.
* Mamba/linear groups: per-block opaque state copy, where a manager block's
  *last* token selects the state block (`_state_block_ids`).

## Scenarios

| Test | TP | Prompts | Notes |
|---|---|---|---|
| `test_basic` | 1 | 1 | Minimal save -> load round trip |
| `test_concurrent` | 1 | 4 | Concurrent requests: ReqState tracking, per-request block attribution |
| `test_tp` | 2 | 2 | TP coordination; for full-attention models also `preferred_block_size=32` != vLLM block size (16), forcing real cross-block translation |
| `test_partial_hit` | 1 | 1 | Phase 2 extends the prompt mid-block: partial external hit |
| `test_full_hit` | 1 | 1 | Phase 2 resends the identical prompt: full-prompt hit is capped so >= 1 token is recomputed |
| `test_multi_turn` | 1 | 1 | Growing conversation: each turn loads the previous turns' blocks and saves new ones |
| `test_load_failure` | 1 | 1 | Storage files deleted between phases: load fails, retry loop must not spin, request still completes |
| `test_mutation` | 1 | 1 | Meta-test: injected off-by-one in the slot translation must make verification FAIL (proves the harness is not vacuous) |

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
| `KVCM_E2E_MODEL` | yes | Path to a local HF model directory (`config.json` + weights). Full-attention coverage needs a plain attention model (e.g. Qwen2.5-7B-Instruct); hybrid coverage needs a mamba/linear + attention model (e.g. Qwen3.5-4B). Hybrid models are auto-detected from `config.json`. |
| `KVCM_E2E_PYTHON` | yes | Python interpreter of a venv with vLLM >= 0.26.0 and both KVCM wheels (`kvcm_py_client`, `kvcm_vllm_connector`) installed. |
| `KVCM_E2E_CAPTURE_DIR` | internal | Set by the driver for the vLLM subprocess; tells `VerifyingConnector` where to write `.pt` captures. Do not set manually. |

The driver also sets vLLM knobs for the spawned server (`VLLM_KV_CACHE_LAYOUT=NHD`,
`VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
`FLASHINFER_DISABLE_VERSION_CHECK=1`) via `env.setdefault`, so a value you
export yourself wins.

## Debugging

Bazel's `test.log` only shows the driver's view (e.g. HTTP 500). The real
tracebacks live in the scenario workdir under `$TEST_TMPDIR`:

```
<TEST_TMPDIR>/kvcm_vllm_e2e/<scenario>/
  manager/manager.stdout|stderr     # KVCM manager
  vllm/vllm*.stdout|stderr          # vLLM (EngineCore tracebacks are here)
  captures/{ref|loaded}_tp{rank}_{token_hash}.pt
```
