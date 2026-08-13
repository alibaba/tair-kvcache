#!/usr/bin/env python3
"""Preprocess CacheBoard enriched traces for mrc_replay_main.

Reads one ``*.enriched.jsonl`` stream on stdin (decompress .zst upstream via
``zstd -dc``) and emits one line per request on stdout:

    <timestamp_ns>\t<service>/<pod>\t<hex_key>,<hex_key>,...

The formal LiteHit runner stable-sorts each replay bucket by timestamp before
replaying, so pipe the output through a stable numeric sort and strip the
first column before feeding mrc_replay_main:

    ... | sort -t$'\t' -k1,1n -s | cut -f2-

Semantics mirror the formal LiteHit run (formal-litehit-7model-h23.json):
  * rows with ambiguous service (len != 1) or ambiguous pod (len != 1) are
    dropped and counted separately;
  * only full 256-token blocks count: keys = input_block_hash_ids[:input_length // 256];
  * timestamp_ns uses the same int64 truncation as the runner.

Counters are printed to stderr as JSON for cross-checking against the formal
result's ``counters`` section.
"""
import json
import sys

BLOCK_TOKENS = 256


def main() -> int:
    input_rows = 0
    valid_rows = 0
    ambiguous_service = 0
    ambiguous_pod = 0
    invalid_rows = 0
    emitted_blocks = 0

    out = sys.stdout
    for line in sys.stdin:
        input_rows += 1
        try:
            row = json.loads(line)
            services = row["service_names"]
            pods = row["pods"]
            hashes = row["input_block_hash_ids"]
            input_length = int(row["input_length"])
            timestamp_ns = int(float(row["timestamp"]) * 1.0e9)
        except (KeyError, ValueError, json.JSONDecodeError):
            invalid_rows += 1
            continue
        if len(services) != 1:
            ambiguous_service += 1
            continue
        if len(pods) != 1:
            ambiguous_pod += 1
            continue
        valid_rows += 1
        n_full = input_length // BLOCK_TOKENS
        if n_full <= 0:
            continue
        keys = hashes[:n_full]
        if not keys:
            continue
        emitted_blocks += len(keys)
        out.write(str(timestamp_ns))
        out.write("\t")
        out.write(services[0])
        out.write("/")
        out.write(pods[0])
        out.write("\t")
        out.write(",".join(keys))
        out.write("\n")

    print(
        json.dumps(
            {
                "input_rows": input_rows,
                "valid_rows": valid_rows,
                "ambiguous_service": ambiguous_service,
                "ambiguous_pod": ambiguous_pod,
                "invalid_rows": invalid_rows,
                "emitted_blocks": emitted_blocks,
            }
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
