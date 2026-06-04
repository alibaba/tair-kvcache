# Agentic Trace Replay: KV Cache Hit Limitation & Fix

## Background

The `codex_swebenchpro_traces` dataset mode unfolds multi-turn conversations into
one request per assistant turn:

```
Conversation:
  Turn 0: human    → H1
  Turn 1: assistant → A1
  Turn 2: human    → H2
  Turn 3: assistant → A2

Unfolded requests:
  Request 1: input = [SYS][H1],              output_len = len(A1)
  Request 2: input = [SYS][H1][A1][H2],     output_len = len(A2)
```

## The Problem

hisim's `C_ModelRunnerHook` replaces the real model with a mock that always
outputs token id `1` regardless of input (`wrapped_sample` in `sglang_hook.py`):

```python
def wrapped_sample(self, *args, **kwargs):
    logits = args[0]
    ids = torch.ones(size=(logits.next_token_logits.shape[0],), dtype=torch.int64)
    return ids  # always token id = 1
```

So after Request 1 completes, the KV cache stores:

```
[SYS][H1] → [1, 1, 1, ...]   # fake A1 tokens
```

Request 2's input contains `A1_real` (actual token ids from the trace), which does
not match the cached `[1, 1, 1, ...]`. Prefix matching breaks at the start of A1:

```
Request 2 prefix match:
  [SYS][H1]    ✅ hit   (human turn, identical)
  [A1_real]    ❌ miss  (cache has [1,1,1,...], not real A1 token ids)
  [H2]         ❌ miss
```

**Result**: the simulation underestimates KV cache reuse for multi-turn agentic
workloads. Only the human-turn prefixes benefit from caching.

## Root Cause

All requests are pre-built and enqueued at simulation start. The mock never
generates real token ids, so the KV cache never holds the actual content that
subsequent turns depend on.

## Proposed Fix

### Step 1 — Carry output token ids through the DatasetRow

Extend `DatasetRow` with an optional field:

```python
@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    timestamp: float = 0.0
    simulation: dict = field(default_factory=dict)
    output_token_ids: Optional[List[int]] = None  # NEW
```

In `sample_agentic_trace_requests`, store the real assistant token ids:

```python
input_requests.append(DatasetRow(
    prompt=prompt,
    prompt_len=input_len,
    output_len=output_len,
    output_token_ids=output_ids,   # real A1 token ids from trace
))
```

### Step 2 — Inject real token ids into the mock sampler

`wrapped_sample` needs access to the per-request expected output ids. One approach
is to attach them to the batch object via `custom_params` so the hook can read them:

```python
def wrapped_sample(self, batch, *args, **kwargs):
    real_ids = batch.get_custom_param("output_token_ids", default=None)
    if real_ids is not None:
        return real_ids  # use trace token ids
    # fallback
    logits = args[0]
    return torch.ones(size=(logits.next_token_logits.shape[0],), dtype=torch.int64)
```

### Step 3 — Wire custom_params through the request pipeline

Ensure `output_token_ids` is packed into `custom_params.simulation` in
`get_request()` alongside `created_time` / `total_request`, and that
`C_SchedulerHook` / `C_ModelRunnerHook` pass it down to the batch.

## Expected Outcome

After the fix, Request 2's prefix match becomes:

```
[SYS][H1][A1_real]    ✅ hit   (cache now holds real A1 token ids)
[H2]                  ❌ miss  (new human input, not yet cached)
```

This accurately reflects the KV cache reuse that a real inference system would
achieve for sequential multi-turn agentic conversations.
