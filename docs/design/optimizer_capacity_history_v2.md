# Revisable capacity history metrics v2

The capacity estimate itself is unchanged: every reporting window applies MRC99
to the five-minute buckets overlapping `[now - reuse_p99, now]`, retaining the
maximum within one analyzer generation. This is a window-level estimate, not
an exact reconstruction of cache occupancy.

## Stable identity and lifecycle

Two new gauges accompany the legacy ring-slot metrics:

- `trace_query_capacity_history_p99_v2_bytes`
- `trace_query_capacity_history_p99_v2_observed_ms`

Both have identical `instance_group`, `instance_id`, `history_time_ms`,
`history_generation`, and `history_shard` labels. `history_time_ms` is an immutable
epoch-millisecond bucket timestamp. `history_generation` contains a process nonce
and a process-local counter; reset and instance recreation allocate a new value.
`history_shard = (history_time_ms / 300000) % 4` partitions queries without dropping
any points. Platform source labels must also be retained when pairing samples.

`observed_ms` identifies the report containing the value, not the last time that
the value increased. Across generations select the latest observed report, not
the largest capacity. A genuine zero capacity is valid; an absent bucket is not
reported. A generation with no observation does not erase an older observation.

The local reporter atomically replaces the two exclusively-owned gauge families.
Scrapes keep immutable snapshots, so capacity and report time cannot straddle a
report update. Expired/replaced tags are removed, never changed to zero. Each
family has at most 288 active series per instance. Historical storage cardinality
grows by up to 288 bucket identities per active instance/day plus reset overlap;
it is subject to the monitoring retention policy. Kmonitor reporting bypasses
the application tag cache for these changing timestamp tags.

## Query contract

At the selected range end, evaluate `last_over_time(metric[range])` against stored
samples for each immutable identity, then join the two gauge results by all
identity/source labels. Filter historical timestamps to the displayed range,
sort by `observed_ms`, and keep the latest valid pair per logical instance and
historical timestamp. Query all four shards; a failed shard must remain an error,
not be treated as an empty result. Check backend series limits separately from
the UI's maximum data point setting. Do not apply a sparse subquery resolution
to obtain the last sample. Correctness is bounded by collected samples and the
backend's raw/rollup retention; the dashboard cannot recover uncollected data.

The query is instant only in the sense of evaluation at the selected end time:
the range vector scans historical storage, including samples from processes that
have exited and buckets older than the current in-memory ring. Grafana converts
the immutable historical label into the plot time rather than using query time.

Legacy data has no immutable timestamp label and must not be silently merged as
if it were v2. Retain the legacy panel for pre-migration inspection, and mark the
v2 rollout boundary. No historical v2 samples are fabricated during deployment.
