# KVCacheEventSubscriber

A process that subscribes to inference engine's KV cache events and forwards them to the kvcm service.

## Installation

```bash
uv sync
```

## Running

```bash
# With defaults
uv run python -m subscriber

# With CLI args
uv run python -m subscriber --zmq-pub-endpoint tcp://localhost:5557 --kvcm-addr 10.0.0.1:50051

# With config file
uv run python -m subscriber --config config.yaml
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check subscriber/ tests/

# Type check
uv run mypy subscriber/
```
