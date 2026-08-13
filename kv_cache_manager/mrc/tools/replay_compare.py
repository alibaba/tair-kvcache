#!/usr/bin/env python3
"""Compare mrc_replay_main output against the formal LiteHit result.

Usage:
    replay_compare.py --formal formal-litehit-7model-h23.json \
        --replay a.replay.json [b.replay.json ...] [--scope full_day] \
        [--output compare.json]

Per matched instance it checks:
  * exact engine: hit_blocks at every shared capacity grid point must match
    the formal LiteHit counts (bit-exact expectation);
  * bounded online engine: hit counts must match the formal exact result at
    every capacity inside its configured tracking boundary.
"""
import argparse
import json


def load_replay_lanes(paths):
    lanes = {}
    grids = set()
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        grids.add(tuple(doc["capacities_blocks"]))
        for lane in doc["lanes"]:
            lanes[lane["instance_id"]] = lane
    if len(grids) != 1:
        raise SystemExit("replay files use different capacity grids")
    return lanes, list(grids.pop())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", required=True)
    ap.add_argument("--replay", nargs="+", required=True)
    ap.add_argument("--scope", default="full_day", choices=["full_day", "steady_state"])
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    with open(args.formal) as f:
        formal = json.load(f)
    lanes, grid = load_replay_lanes(args.replay)

    formal_caps = formal["capacities_blocks"]
    shared_caps = [c for c in formal_caps if c in set(grid)]
    grid_index = {c: i for i, c in enumerate(grid)}

    matched = 0
    missing = []
    exact_mismatch_instances = 0
    exact_mismatch_points = 0
    total_points = 0
    block_count_mismatches = []
    online_mismatch_points = 0
    per_instance = []

    for inst in formal["instances"]:
        instance_id = inst["instance_id"]
        lane = lanes.get(instance_id)
        if lane is None:
            missing.append(instance_id)
            continue
        matched += 1
        fd = inst[args.scope]

        if fd["full_blocks"] != lane["total_blocks"]:
            block_count_mismatches.append(
                {"instance_id": instance_id, "formal": fd["full_blocks"], "replay": lane["total_blocks"]}
            )

        formal_points = {p["quota_blocks"]: p for p in fd["points"]}
        inst_mismatch = 0
        inst_online_mismatch = 0
        for cap in shared_caps:
            fp = formal_points.get(cap)
            if fp is None:
                continue
            total_points += 1
            idx = grid_index[cap]
            exact_hits = lane["exact_hit_blocks"][idx]
            if exact_hits != fp["hit_blocks"]:
                inst_mismatch += 1
                exact_mismatch_points += 1
            total = lane["total_blocks"]
            if total > 0 and cap > 0:
                online_rate = lane["online_exact"]["hit_rate"][idx]
                # The profiler's rate is an integer hit count divided by the
                # same total block count. Reconstruct that integer before the
                # comparison so JSON floating-point formatting cannot turn an
                # otherwise bit-exact result into a false mismatch.
                online_hits = round(online_rate * total)
                if online_hits != exact_hits:
                    inst_online_mismatch += 1
                    online_mismatch_points += 1
        if inst_mismatch:
            exact_mismatch_instances += 1
        per_instance.append(
            {
                "instance_id": instance_id,
                "total_blocks": lane["total_blocks"],
                "exact_mismatch_points": inst_mismatch,
                "online_mismatch_points": inst_online_mismatch,
                "tracked_blocks": lane["online_exact"]["tracked_blocks"],
            }
        )

    summary = {
        "formal_instances": len(formal["instances"]),
        "matched": matched,
        "missing_in_replay": len(missing),
        "block_count_mismatches": len(block_count_mismatches),
        "exact": {
            "compared_points": total_points,
            "mismatch_points": exact_mismatch_points,
            "mismatch_instances": exact_mismatch_instances,
            "bit_exact": exact_mismatch_points == 0 and len(block_count_mismatches) == 0,
        },
        "bounded_online_exact": {"mismatch_points": online_mismatch_points,
                                 "bit_exact": online_mismatch_points == 0},
    }
    print(json.dumps(summary, indent=2))
    if missing[:5]:
        print("missing sample:", missing[:5])
    if block_count_mismatches[:3]:
        print("block mismatch sample:", block_count_mismatches[:3])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {"summary": summary, "missing": missing, "block_count_mismatches": block_count_mismatches,
                 "per_instance": per_instance},
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
