# LiteHit: A Single Replay Produces Capacity-Independent Facts; Arbitrary LRU Capacities Are Projected After the Fact

> [中文](README_zh.md) | English

LiteHit is a lightweight, exact LRU hit-rate analyzer for full-attention KVCache blocks. The core replays a trace only once, producing **capacity-independent facts** for each request (`RequestFact`, a hit curve encoded as arithmetic-segment RLE); the hit count for any capacity is derived after the fact from the facts by the stateless projector `HitCurveProjector`. The core never receives a capacity list, nor does it accumulate any per-capacity results.

The problem LiteHit solves is:

```text
Given a set of block traces with request boundaries,
how to replay the trace only once and precisely answer, for "any" LRU capacity:

1. the number of prefix hit blocks for each request;
2. the cumulative prefix hit rate over the entire trace.

Capacity no longer needs to be given before the analysis starts.
```

The first phase only supports the following model:

- full attention;
- every complete block has the same KVCache charge (equal charge, see 6.4);
- exact LRU, with in-request reverse-order submission;
- does not handle linear attention / Mamba, blocks of different sizes, admission, prefetch, or multi-level cache policies; TTL support is "one fixed TTL per group" layered on top of LRU (see §7 TTL Replay), and does not support query-time scanning of arbitrary TTLs.

## 0. Architecture Overview

```text
                 ┌────────────────────────────────────────────┐
  Raw request    │ Shared preprocessing request_preprocess     │
 (keys,len) ───> │  NormalizeRequest: length check/derivation +│
                 │  optional ApplyPrefixHash (prefix chained   │
                 │  hash)                                      │
                 └───────────────┬────────────────────────────┘
                                 │ NormalizedRequest
                 ┌───────────────▼────────────────────────────┐
                 │ LiteHit core (capacity-independent)         │
                 │  ProcessRequest(block_keys) → RequestFact   │
                 │  state: Fenwick + last_positions            │
                 └───────┬───────────────────────┬────────────┘
                         │ RequestFact           │ RequestFact
            Online path  │                       │  Offline path
                 ┌───────▼────────┐      ┌───────▼─────────────┐
                 │ HitCurveProjector│    │ facts CSV            │
                 │ per-slot project +│   │ litehit_facts.csv    │
                 │ cumulative ints  │    │ (atomic publish)     │
                 └────────────────┘      └───────┬─────────────┘
                                                 │ any time, any capacity
                                         ┌───────▼─────────────┐
                                         │ facts query tool     │
                                         │ HitCurveProjector    │
                                         └─────────────────────┘
```

Three inviolable layering constraints:

1. **Core is capacity-independent**: `LiteHit::ProcessRequest` only receives block keys and returns a `RequestFact`; there is no capacity parameter.
2. **Projection is the sole entry point**: all "capacity → hit block count" conversions must go through `HitCurveProjector`; Online and facts query share the same implementation, and no component is allowed to implement boundary logic on its own.
3. **Byte conversion happens only at the projection boundary**: the core and facts are all in block units; `ProjectBytes` performs a single floor division using `block_bytes` at projection time.

---

## 1. Input Model and Shared Preprocessing

### 1.1 Per Request

Each request provides to the preprocessing layer:

```text
block_keys[]            complete block keys ordered from front to back of the request (or raw per-block hashes)
input_token_len         the number of input tokens of the original request
```

`NormalizeRequest(block_keys, input_token_len, block_size_tokens, enable_prefix_hash, trace_block_size_tokens = 0)`:

- `trace_block_size_tokens` is the trace's native block granularity (0 = same as `block_size_tokens`, no re-blocking); length validation happens at trace granularity:
  `block_keys.size() == floor(input_token_len / trace_block_size_tokens)`;
- when `input_token_len > 0` it is the authoritative denominator; `input_token_len <= 0` is treated as missing and derived as `block_keys.size() * trace_block_size_tokens`;
- violating the constraints (including `block_size_tokens` not being an integer multiple of `trace_block_size_tokens` — only coarsening is allowed) throws `std::invalid_argument` (Offline fails fast on this, see Section 7).

Trailing tokens that do not fill a complete block do not enter the LRU, but are retained in the hit-rate denominator.

### 1.1a Re-blocking: Granularity Coarsening with Zero Re-hashing

When the analysis granularity is coarser than the trace granularity (`block_size_tokens = k * trace_block_size_tokens`, k > 1), no hash needs to be recomputed: the `j*k`-th key of the prefix chained keys happens to encode all tokens of the first j coarse blocks, so **first build the prefix chain (if the input is per-block hashes), then take every k-th (the k-th) key** to obtain valid prefix chained keys at the coarse granularity; the tail that cannot fill k fine blocks is discarded (its tokens remain in the denominator). Only coarsening is allowed — refining requires token information inside a block, which does not exist in the trace.

### 1.2 Block Key Contract: Prefix Chained Hash

Under full attention, the KVCache of block j depends on all tokens of the first j blocks of the request, so the key must encode the entire prefix:

```text
key_j = hash(all tokens of the first j complete blocks of the request)
```

Keys are equal if and only if the entire token prefix is exactly identical. The valid trace shape is "shared prefix + fork", e.g. `[A, B, C]` and `[A, B, D]` share the first two blocks; a reordered sequence like `[B, A, C]` cannot occur under the contract.

When the input is per-block independent hashes, set `enable_prefix_hash = true` on the **Instance Group** (the key shape follows model deployment, and group is exactly that granularity; the online group-creation RPC and offline config share the same field), and preprocessing converts it to prefix chained keys using a rolling hash:

```text
PrefixHashNext(prev, raw): Jenkins 64-bit variant, explicit uint64 arithmetic (logical right shift),
bit-for-bit identical to the Python producer prefix_hash.py::hash_int64_func.
Note: intentionally does not reuse HashUtil::HashIntFunc (signed right shift, results diverge for negative hashes).
```

Two corollaries of the contract (used in Section 5):

```text
1. Keys within the same request are necessarily distinct (each key encodes a strictly growing prefix).
2. After the first cold block within a request, the prefixes of subsequent keys all contain the divergence point, so they are necessarily cold too.
```

---

## 2. Hit Semantics: prefix hit + reverse-order submission

### 2.1 prefix hit

Full-attention requests adopt prefix hit: for a given capacity, the hit block count of a request is the number of consecutively hit complete blocks starting from the first block of the request, i.e. the position of the first miss. The first miss only truncates the hits of the current request; it does not stop the LRU state update — all complete blocks are still submitted in full, otherwise the cache state seen by subsequent requests would be wrong.

The physical basis for submission not distinguishing hit / miss is: in a real prefix cache, a missed block is recomputed and **written back** to the cache, and the write itself is a touch — after the request ends, regardless of hit or miss, the block is at the MRU end. Thus "LRU update" and "hit determination" are naturally decoupled (Mattson stack algorithm property): hit determination depends on capacity and is deferred to the projection layer; LRU update does not depend on capacity, and the core unconditionally submits all blocks to the end of the position. This also means that if the trace carries output (new blocks generated by decode), the same mechanism can be extended with **read/write separation**: output blocks do not participate in phase-one hit evaluation (newly generated blocks cannot be called hits), but participate as usual in phase-two submission into the LRU, to be hit by subsequent requests.

### 2.2 State Submission in Reverse Request Order (tail first, head last)

Forward-order touching would make the head of the chain the oldest and the first to be evicted — for prefix semantics this is **value inversion**: without the head, none of the other resident blocks on the chain can contribute a single prefix hit, yet they still occupy capacity. Production prefix caches all choose tail-first eviction: vLLM puts blocks back into the free queue in reverse order, and the SGLang radix cache only evicts LRU leaves.

Reverse-order submission, combined with the prefix hash contract of 1.2, yields the invariant:

```text
A parent key is always newer than any of its resident descendants
    ⇒ the eviction victim of global LRU is always a leaf
    ⇒ reverse-order submission LRU is equivalent to "evict the least recently used leaf"
```

And the global order is still uniquely determined by the access sequence, independent of capacity — the stack inclusion property is preserved (Section 3).

The implementation has two phases: **Phase one** computes the hit curve based on the read-only LRU snapshot before the request arrives (before the first miss there are only hits; hits change order but not the member set, so snapshot evaluation is exact); **Phase two** batch-submits "by the first-occurrence position of each key, from the request tail toward the head", which is equivalent to reverse-order per-block touching even for inputs with duplicate keys (under the contract there are no duplicate keys within a request; this deduplication is defensive behavior).

---

## 3. Why a Single LRU State Can Answer All Capacities

LRU has the stack inclusion property. Suppose the current global most-recent-access order is:

```text
MRU → [X, A, Y, B] → LRU
```

Then the cache contents for capacities 1/2/3/4 are respectively the first 1/2/3/4 elements of this order. Block `B` is at position 4, so it misses at capacities 1, 2, 3 and hits at capacity ≥ 4. Therefore a single access only needs to know the depth of the block in the global LRU order to simultaneously answer all capacities — this is exactly the root of why "capacity-independent facts" are feasible.

---

## 4. Reuse Distance and Fenwick

For a repeated access:

```text
reuse distance d = number of distinct blocks appearing after the last access and before this access
required_capacity = d + 1        (capacity C hits ⇔ C >= d + 1)
```

A block appearing for the first time is a cold miss, with no finite required_capacity; even infinite capacity cannot hit a cold access.

LiteHit retains only the latest access position `last[key]` for each block; the Fenwick's logical array is 1 at "the latest position of some block" and 0 at historical stale positions:

```text
d = Fenwick.sum(i - 1) - Fenwick.sum(prev)     // number of positions still 1 within (prev, i)
then Fenwick.add(prev, -1), Fenwick.add(i, +1), last[key] = i
```

The Fenwick is not a cache; it is merely an order-statistics representation of the global LRU order. When historical abandoned positions exceed twice the number of active keys plus a fixed slack, the implementation rebuilds the position space (compaction), so that the Fenwick space is dominated by the current number of active keys rather than the total number of historical accesses — note this only reclaims **positions**, it does not delete any key (see Section 9).

---

## 5. RequestFact: Hit Curve Encoded as Arithmetic-Segment RLE

### 5.1 From Per-Block Thresholds to Hit Curve

Suppose the minimum hit capacity of each block of a request based on the snapshot is `required = [r1, ..., rm]` (cold truncates at the first infinity). To make the first j blocks hit consecutively, the capacity must satisfy all preceding blocks:

```text
prefix_required[j] = max(r1, ..., rj)
```

The sequence `prefix_required[1..h]` (h being the length before cold truncation) completely determines the hit block count of that request at **all capacities**:

```text
hit_blocks(C) = |{ j : prefix_required[j] <= C }|
```

This monotonic step function is the **hit curve** of this request — it is the entire fact of this request.

### 5.2 Under the Contract, Thresholds Are Strictly Increasing ⇒ Arithmetic-Segment RLE Is Lossless

Under reverse-order submission, the minimum hit capacity of a later block on the chain is **strictly greater** than that of an earlier block (each block deeper adds at least its parent key within the snapshot interval), so for contract inputs `prefix_required` is strictly increasing. A stronger structural property is: adjacent blocks on the chain occupy **consecutive** positions in the global LRU, and the threshold only jumps at "queue-jumping" points (positions where sibling branches interleave). So consecutive `+1` thresholds are compressed into one arithmetic segment:

```text
HitCurveSegment { start_required_blocks, run_length }
The j-th block (0-based) within a segment becomes a prefix hit at capacity >= start + j.
Number of segments = 1 + number of queue-jumps, independent of request length.
```

For example, thresholds `[1, 2, 3]` (a whole-chain replay with no queue-jumping) encode as one segment `{1, 3}`; thresholds `[1, 2, 4]` (a queue jumped in before the third block) encode as `{1, 2}, {4, 1}`. Between adjacent segments there is necessarily a threshold gap of at least 1, so they cannot be merged further.

### 5.3 Monotonic Defense for Non-Contract Inputs

For inputs that do not satisfy the contract (duplicate keys within a request, etc.), thresholds may not be strictly increasing. A monotonic defense is applied during encoding:

```text
encoded_threshold = max(prefix_required[j], last_encoded + 1)
```

For contract inputs this defense is always a no-op; for non-contract inputs it only **raises** thresholds (pessimistic, never optimistic), and the projection result is a lower bound of the true hits.

### 5.4 HitCurveProjector

```cpp
ProjectBlocks(fact, capacity_blocks)   // linear scan along segments:
                                       // hits += min(run_length, C - start + 1), until start > C
ProjectBytes(fact, capacity_bytes, block_bytes)
                                       // = ProjectBlocks(fact, floor(bytes / block_bytes))
ProjectInfinite(fact)                  // = Σ run_length (no capacity miss, only cold miss)
```

An empty curve means the request head is cold, and hits are 0 at any capacity.

---

## 6. Units and Conversions

### 6.1 Two Sizes, Different Purposes

| Name | Unit | Purpose |
|---|---:|---|
| `block_size_tokens` | token/block | convert hit block count to hit token count |
| `block_bytes` | byte/block | convert byte capacity to block capacity (projection boundary only) |

### 6.2 Token Hit Rate

```text
hit_tokens = hit_blocks * block_size_tokens
trace_hit_rate = hit_tokens / input_token_len
```

Trailing incomplete tokens are in the denominator, so even a full hit may not be 100%. The cumulative hit rate must first accumulate the integer numerator and denominator (`Σ hit_tokens / Σ input_token_len`); it must not be the arithmetic mean of per-request hit rates.

### 6.3 Byte Capacity Conversion

```text
capacity_blocks = floor(capacity_bytes / block_bytes)
capacity_gb uses binary conversion: capacity_bytes = capacity_gb * 1024^3
```

`block_bytes` comes from the sum of spec.size of each spec in the full location spec group of instance registration (`size_full`). Each row of the facts CSV records `block_bytes`, making facts self-describing: even if the charge estimate is corrected later, historical facts can be re-projected.

### 6.4 Equal-Charge Invariant (Premise of Block-Unit RLE)

The hit curve is in block units and `ProjectBytes` performs a single floor division, which is exact **if and only if** the charges of all participating blocks are exactly equal — full-only instances satisfy this (each block's charge is constantly `size_full`, an exact value not an average). For linear/Mamba mixed instances, the per-block charge is unequal, and charge-weighted thresholds must be used instead, which is a separate subsequent task; currently Offline rejects instances with `linear_step != 0`.

---

## 7. Offline Facts Pipeline

The Offline runner (`lite_hit_main` + `OptimizerLiteHitConfig`) processes standard traces in batches:

```text
batch window = pipeline_worker_count * 256 entries
  ├─ parallel preprocessing (workers assigned by index stripe): parse + NormalizeRequest + prefix hash
  ├─ split into lanes by instance; within a lane, strictly serial ProcessRequest in input order
  └─ serially write out facts rows in input order
```

**Trace granularity and re-blocking**: the config field `block_size` (default 256) declares the native block granularity of the trace; each instance's `block_size` is the analysis granularity of that lane, which must be an integer multiple of it (only coarsening allowed; violation fails the whole thing at lane initialization), re-blocking per the sampling method of 1.1a.

**Write events**: `write` trace events are recognized and ignored. When a `get` is submitted, all blocks are treated as written back (the physical basis of §2: write-back itself is a touch), so `write` rows in split `get`/`write` traces have no effect on facts; delayed write modeling belongs only to the replay path.

**Fanout mode**: when `fanout_all_instances = true`, each request is broadcast to all lanes (each lane has independent LRU state and independent facts rows); combined with multiple instances of different `block_size`, a single replay can scan multiple analysis granities over the same trace; mutually exclusive with `override_instance_id`. The facts query summary is grouped by instance (one row per instance + one total row), and fanout results are directly readable.

**TTL replay**: when the instance group config has `ttl_seconds != 0`, the `LiteHit` core of that group's lanes layers a fixed TTL on top (consistent with the online `TtlCacheIndexerWrapper` semantics: a block survives while strictly less than TTL since its last access; expired blocks are misses at any capacity and truncate the prefix like cold blocks; every access (hit or miss) refreshes last_access; time is taken from the trace timestamp, and replay is deterministic). Age is monotonic along the LRU stack, and expired blocks do not raise the reuse distance of surviving blocks, so a single replay remains exact for the joint metric of "fixed TTL × arbitrary capacity", and facts are still ordinary hit curve rows. TTL is a replay-time parameter, taken directly from the group's `ttl_seconds`; to scan multiple TTLs, configure multiple groups each with a different `ttl_seconds` (can be combined with fanout to complete in a single replay).

**Fail-fast**: timestamp out of order, unknown instance, length validation failure, zero valid rows in the whole file — any of these fails the whole thing with a reason — facts are an all-or-nothing reconciliation ledger, and silent row loss is not allowed.

**Atomic publish**: first write `litehit_facts.csv.tmp`, then `rename` to `litehit_facts.csv` after all succeed; readers never see a half-finished product.

### 7.1 facts CSV Format

```text
trace_id,instance_id,timestamp_ns,input_token_len,block_size_tokens,block_bytes,hit_curve
```

`hit_curve` is a quoted JSON array `[[start_required_blocks, run_length], ...]`; string fields are quote-escaped per CSV rules. Each row is independently parseable, self-describing, and re-projectable.

### 7.2 facts query Tool

`lite_hit_facts_query_main` (`RunLiteHitFactsQuery`) performs after-the-fact capacity queries on published facts:

```text
input: facts CSV + capacity_gb list (order-preserving, duplicates and 0 allowed, negative = infinite capacity)
output: JSONL, one row per request (hit_blocks/hit_rates per slot)
     + one summary row per instance_id (instance_id in lexicographic order) + one total summary row
     (requests / total_input_tokens / total_hit_blocks / total_hit_tokens / hit_rates)
```

Memory is only O(number of instances × number of capacity slots) cumulative integers; any malformed row fails the entire query.

---

## 8. Online Integration

The Online Optimizer's full-attention `InstanceState` directly holds a `LiteHit` (when the group config has `ttl_seconds != 0`, a fixed TTL is layered on with wall-clock time, with metrics consistent with the linear path's `TtlCacheIndexerWrapper`). Each TraceQuery:

```text
NormalizeRequest → ProcessRequest → obtain RequestFact
  ├─ for each configured capacity slot: ProjectBlocks(fact, lite_hit_capacity_blocks[i])
  ├─ theoretical upper bound: ProjectInfinite(fact)
  └─ update cumulative integers: total_queries / total_input_tokens / total_hits per slot
```

Online does not persist facts (facts persistence is currently Offline-exclusive); the hit rate of `ListInstances` is derived from cumulative integers (`total_hits * block_size_tokens / total_input_tokens`). linear attention continues to go through the legacy indexer path (when prefix hash is enabled, only `ApplyPrefixHash` is performed).

---

## 9. Complexity and State Size

Let N = total number of block accesses, U = number of historically distinct blocks, Q = number of requests, S = number of segments per request (= 1 + number of queue-jumps).

```text
ProcessRequest: O(m log U) (m is the number of request blocks)
ProjectBlocks:  O(S), independent of capacity value
core persistent state: Fenwick + last_positions, O(U)
```

**No capacity-based pruning**: capacity is only known after the fact, and any key may be used by some future large-capacity query, so the core retains all historical unique keys (this is the inherent cost of capacity independence). The compaction in Section 4 only reclaims abandoned positions, it does not delete keys. `memory_usage_bytes()` / `current_unique_blocks()` provide observability.

---

## 10. End-to-End Example

```text
block_size_tokens = 4, block_bytes = 1024
five requests (satisfying the prefix hash contract, [A,B,C] and [A,B,D] share the first two blocks, [A,E] forks after one block)
```

| # | keys | len | snapshot threshold prefix_required | hit_curve (RLE) |
|---|---|---:|---|---|
| 1 | [A,B,C] | 13 | all cold | `[]` |
| 2 | [A,B,D] | 12 | [1, 2], D cold truncates | `[[1,2]]` |
| 3 | [A,B,C] | 13 | [1, 2, 4] (D's queue-jump makes C depth 4) | `[[1,2],[4,1]]` |
| 4 | [A,E] | 8 | [1], E cold truncates | `[[1,1]]` |
| 5 | [A,E] | 8 | [1, 2] | `[[1,2]]` |

The LRU after each request ends is respectively `[A,B,C]`, `[A,B,D,C]`, `[A,B,C,D]`, `[A,E,B,C,D]`, `[A,E,B,C,D]`.

After-the-fact projection of three capacities (2048 B → 2 blocks, 3072 B → 3 blocks, infinite):

| Capacity | hit_blocks per request | cumulative blocks | cumulative tokens | cumulative hit rate |
|---:|---|---:|---:|---:|
| 2 blocks | 0,2,2,1,2 | 7 | 28 | `28 / 54 = 51.85%` |
| 3 blocks | 0,2,2,1,2 | 7 | 28 | `28 / 54 = 51.85%` |
| ∞ | 0,2,3,1,2 | 8 | 32 | `32 / 54 = 59.26%` |

Request 3's curve `[[1,2],[4,1]]` reads directly: capacity 2, 3 hits 2 blocks (the second segment start=4 exceeds), capacity ≥ 4 and infinite hit 3 blocks. Compared with forward-order submission (chain head oldest): under the same trace, the cumulative hits at capacity 2 would drop from 7 blocks to 2 blocks — this is exactly the value inversion mentioned in 2.2.

---

## 11. Correctness Verification

Unit tests (`LiteHitTest` / `LiteHitOfflineRunnerTest`) cover:

1. **oracle comparison**: under contract inputs (random tree-shaped chains + `ApplyPrefixHash`), `ProjectBlocks` is **exactly identical** to naive multi-capacity LRU (snapshot evaluation + reverse-order per-block touch) at capacities {0,1,2,4,9,∞};
2. non-contract input projection ≤ oracle (monotonic defense is pessimistic, never optimistic), infinite capacity still exact;
3. RLE shape: whole-chain replay single segment, queue-jump breaks segments, adjacent segments cannot be merged;
4. projection boundaries: capacity 0, segment boundaries, byte floor conversion;
5. Offline facts and Online per-request projection cross-reconcile consistently; parallelism 4 and serial output are byte-for-byte identical;
6. fail-fast: out-of-order timestamps, unknown instance, length violation, zero valid rows, analysis granularity not an integer multiple of trace granularity;
7. prefix hash golden vectors are bit-for-bit identical to the Python producer;
8. after compaction all keys can still be hit (no history lost);
9. re-blocking: coarse-granularity sampled keys / tail discard / only coarsening allowed; fanout produces independent facts for multiple block_size in a single replay, query summary grouped by instance.

---

## 12. Core Conclusions

```text
Multiple (arbitrary) capacity LRU caches
    ↓ LRU stack inclusion property
one global most-recent-access order
    ↓ Fenwick / order statistics
minimum hit capacity per access
    ↓ in-request prefix max (strictly increasing under the contract)
arithmetic-segment RLE hit curve = capacity-independent fact
    ↓ HitCurveProjector (sole projection entry, byte conversion only at this boundary)
prefix hit_blocks at arbitrary capacity
    ↓ block_size_tokens / input_token_len
per-request and cumulative token hit rate
```

The three most easily confused points:

```text
1. The hit curve is a "fact", capacity is a "query" — the core is completely decoupled from capacity,
   at the cost of not being able to do capacity-based pruning.
2. block_size_tokens converts tokens, block_bytes converts capacity; they cannot replace each other.
3. The losslessness of arithmetic-segment RLE depends on the prefix hash contract + reverse-order submission + equal charge;
   mixed charge (linear/Mamba) must be designed separately.
```
