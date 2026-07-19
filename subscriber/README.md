# KVCacheEventSubscriber

A process that observes inference-engine KV cache state and forwards normalized
events to the KVCM service. vLLM uses its ZMQ event stream; RTP-LLM uses full
`GetCacheStatus` gRPC snapshots and an acknowledged local diff.

## Installation

```bash
uv sync
```

## Running

```bash
# With defaults
uv run python -m subscriber

# With CLI args and a directly addressed KVCM service
uv run python -m subscriber \
  --zmq-pub-endpoint tcp://localhost:5557 \
  --kvcm-base-url http://10.0.0.1:6382

# With config file
uv run python -m subscriber --config config.yaml

# RTP-LLM: point at the rank-0 GetCacheStatus gRPC endpoint
uv run python -m subscriber \
  --config config.yaml \
  --engine-type rtp_llm \
  --rtp-endpoints 127.0.0.1:8089 \
  --block-size 64 \
  --host-ip-port 10.0.0.8:8088
```

For RTP-LLM, multiple comma-separated endpoints may be supplied when one
subscriber must merge several DP cache snapshots. Every endpoint must return a
complete snapshot; a failed endpoint rejects the whole poll so a transient DP
failure is not interpreted as mass eviction. The subscriber retries a diff
until KVCM acknowledges it, confirms removals across consecutive snapshots,
and periodically refreshes the full add set for reconciliation.
The configured `block_size` must match every RTP endpoint because KVCM instance
registration happens before the first snapshot is forwarded.
For RTP-LLM, node registration and heartbeat are deferred until the cache API
returns a valid snapshot. With the default startup reset enabled, the
Subscriber clears any old host generation before registering the node and
reporting the first full add set.

By default the Subscriber discovers KVCM from `KVCM_VSERVICE_ID`. For local
tests or deployments with a fixed manager address, set `kvcm_base_url` in YAML
or pass `--kvcm-base-url`; the explicit address takes precedence over Spectrum
service discovery.

KVCM send failures are retried in event order with bounded-queue backpressure;
`kvcm_send_retry_interval_s` controls the retry interval. A queued batch is
discarded only after engine recovery advances the epoch, so events from an old
engine generation cannot leak into the new one.

The vLLM adapter currently accepts exactly one DP event endpoint. Multi-DP vLLM
configuration fails at startup instead of silently subscribing to DP rank 0.
RTP-LLM multi-DP snapshot aggregation is supported as described above.

RTP-LLM can launch this process automatically by setting
`KVCM_SUBSCRIBER_CONFIG` to the YAML path. RTP derives the default cache API
endpoint from its resolved `START_PORT`: rank-0 gRPC is `START_PORT + 1`.
Set `RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS` explicitly for multi-DP or multi-node
deployments where remote advertised addresses cannot be inferred locally.

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check subscriber/ tests/

# Type check
uv run mypy subscriber/
```
