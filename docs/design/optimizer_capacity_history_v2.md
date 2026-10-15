# Revisable capacity history metrics v2

The capacity estimate applies MRC99 to the five-minute buckets overlapping
`[now - reuse_p99, now]`, retaining the maximum within one analyzer generation.
Four independent histories use MRC99 and reuse P99 from **non-overlapping 5min,
15min, 30min and 1h** sample windows aligned to their wall-clock boundaries. The first request or
report after a boundary finalizes the previous window exactly once. Its result
is anchored at the completed window end minus one millisecond, not the delayed
report timestamp. There are four additional 58 KiB reuse histograms and four
sparse MRC accumulators, cleared at their respective boundaries: no rolling slices, per-request
sample queue, or 30-second history estimates. Ordinary MRC/reuse gauges and history
exports still run every reporting interval (normally 30 seconds). Percentiles use
raw histogram counts and hit-curve deltas, never averages of per-report percentiles.
Only the 5min base receives per-request updates; before each drain, its raw counts
are merged into the three longer accumulators. Each history has its own 288-point
ring; no maximum is taken across different source windows. Historical plot buckets
remain 5min for comparison, but new estimates are produced only at each source
window's boundary. Startup windows can be partial; longer windows may have gaps
when the reuse interval is shorter than the source window. Do not fill those gaps
with fabricated values or reinterpret them as zero.
Local receipt time determines membership; producer timestamps determine reuse
duration. Startup uses available samples in the first partial window, empty windows
do not create observations, and delayed reports do not fabricate idle-gap points.
Reset clears the inputs along with history. This is a window-level estimate, not
an exact reconstruction of cache occupancy.

This source-window change requires an optimizer binary rollout; changing the
dashboard alone cannot apply it. Previously collected history is not recomputed.

## Stable identity and lifecycle

The 5min source preserves these two gauges (legacy ring-slot gauges are removed):

- `trace_query_capacity_history_p99_v2_bytes`
- `trace_query_capacity_history_p99_v2_observed_ms`

The other sources use `trace_query_capacity_history_p99_window_bytes` and
`trace_query_capacity_history_p99_window_observed_ms`, with an additional
`source_window_seconds` label (`900`, `1800`, `3600`). Queries must filter this label
to one value, preserving the same pairing/revision rules. The 5min series have no
new label, so already collected v2 history remains queryable without migration.

Both have identical `instance_group`, `instance_id`, `history_time_ms`,
`history_generation`, and `history_shard` labels. `history_time_ms` is an immutable
epoch-millisecond bucket timestamp. `history_generation` contains a process nonce
and a process-local counter; reset and instance recreation allocate a new value.
`history_shard = (history_time_ms / 300000) % 4` partitions queries without dropping
any points. Pair samples by the full immutable identity and evaluation time;
the process nonce prevents different sources from sharing a generation.

`observed_ms` identifies the report containing the value, not the last time that
the value increased. Across generations select the latest observed report, not
the largest capacity. A genuine zero capacity is valid; an absent bucket is not
reported. A generation with no observation does not erase an older observation.

The local reporter atomically replaces the four exclusively-owned gauge families.
Scrapes keep immutable snapshots, so capacity and report time cannot straddle a
report update. Expired/replaced tags are removed, never changed to zero. Each
5min family has at most 288 active series per instance; each multi-window family
has at most 864. The total is at most 2304 series per instance. Historical storage cardinality
grows by up to 1152 bucket identities per active instance/day plus reset overlap;
it is subject to the monitoring retention policy. Kmonitor reporting bypasses
the application tag cache for these changing timestamp tags.

## Query contract

Use Range queries throughout. The Grafana datasource rejects fractional-second
Instant timestamps and its lookback allowance differs from the CLI gateway's;
a successful CLI replay alone does not establish browser compatibility.

The display interval is **not** an as-of revision cutoff. A bucket at `h` may be
revised by reports until `h + 24h`, so reading reports only through the selected
display end is incorrect. Read the subsequent stored reports as well, including
reports from generations/processes that have already exited.

The deployed query evaluates hourly and reads 26 contiguous one-hour windows:
`max_over_time(metric[1h] offset -Nh)` for `N = 0..25`. These cover
`[evaluation - 1h, evaluation + 25h]`. Negative offsets read later **existing**
reports; they do not fabricate future samples. For each displayed five-minute
bucket `h`, the hourly evaluation `floor(h / 1h) * 1h` covers the entire 24h
revision lifetime plus at least five minutes of collection-boundary margin.
The hourly query step does not change the plotted five-minute bucket resolution.

Capacity is nondecreasing within a full immutable identity including
`history_generation`; `observed_ms` is also nondecreasing. Therefore each
identity's window maximum equals its final value. This invariant is essential:
**never maximize capacity across generations**. A reset can correctly produce a
lower capacity or zero, and the latest report must win.

To keep the query planner's expanded SQL bounded, combine at most four adjacent
windows per query with `or`, newest window first. Query all seven groups, four
shards, and both metrics (56 targets). Give capacity and observed-time targets in
the same group the same `alias("revision_window_N", ...)`. Preserve this alias
as `__name__` during the table merge: it is part of the pairing key and prevents
values from different revision windows being incorrectly paired. `metrics()`
with repeated selectors and different offsets was not reliable in testing and
must not replace this grouping without validating its actual semantics.

Keep evaluation time, window alias, and all immutable identity labels when
joining capacity and observed-time results. Filter historical timestamps to the displayed range,
sort by `observed_ms`, and keep the latest valid pair per logical instance and
historical timestamp. Query all four shards; a failed shard must remain an error,
not be treated as an empty result. Check backend series limits separately from
the UI's maximum data point setting. Do not apply a sparse subquery resolution
to obtain the last sample. Correctness is bounded by collected samples and the
backend's raw/rollup retention; the dashboard cannot recover uncollected data.

Use an actual query step of at most one hour; the current panel has a one-hour
minimum and `maxDataPoints = 10000`. If Grafana increases the step beyond one
hour, the fixed forward coverage needs to be redesigned rather than silently
claimed complete. The boundary margin assumes prompt collection and aligned
clocks; arbitrary late ingestion is not covered. Correctness is also bounded by
monitoring retention, backend series/response limits, and actual collected
samples. Grafana converts the immutable historical label into plot time; the
selected display end is used only for this final filtering, not as the latest
allowed report timestamp.

Legacy data has no immutable timestamp label and must not be silently merged as
if it were v2. Legacy reporting and its dashboard panel are removed; stored legacy
samples are not deleted. No historical samples for newly introduced windows are
fabricated during deployment. Preserve the existing 5min panel and its v2 history.
