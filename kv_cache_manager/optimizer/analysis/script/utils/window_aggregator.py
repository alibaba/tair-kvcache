#!/usr/bin/env python3
"""
Windowed aggregation for per-instance optimizer replay outputs.

The standard metric is token hit rate:

    HitRate = HitTokens / InputTokens

HitRate is computed from token counters. Block counters are aggregated as raw
diagnostic counters so readers can inspect read and hit volume directly.
"""

import os
from collections import defaultdict
from typing import Dict, Iterable, Optional

import pandas as pd


COUNTER_COLUMNS = [
    "ReadBlocks",
    "LocalHitBlocks",
    "RemoteHitBlocks",
    "HitBlocks",
    "LocalHitTokens",
    "RemoteHitTokens",
    "InputTokens",
    "HitTokens",
]


def collect_hit_rate_csvs(output_dir: str) -> Dict[str, str]:
    """Collect per-instance hit-rate CSVs from an optimizer output directory."""
    if not os.path.isdir(output_dir):
        return {}
    result = {}
    for fname in os.listdir(output_dir):
        if not fname.endswith("_hit_rates.csv") or fname.startswith("global_"):
            continue
        instance_id = fname[: -len("_hit_rates.csv")]
        result[instance_id] = os.path.join(output_dir, fname)
    return result


def aggregate_and_write(
    csv_files: Dict[str, str],
    output_dir: str,
    bucket_name: str = "",
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    window_ns: Optional[int] = None,
    chunksize: int = 1_000_000,
    include_instance_windows: bool = False,
) -> dict:
    """
    Aggregate instance hit-rate CSVs and write summary/window CSV files.

    Time interval semantics are [start_ns, end_ns).  Windows are aligned to
    absolute timestamp 0 so independent runs aggregate consistently.
    """
    if not csv_files:
        raise ValueError("No hit-rate CSV files to aggregate")
    if window_ns is not None and window_ns <= 0:
        raise ValueError("window_ns must be positive")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    per_instance = {}
    global_counter = _new_counter()
    global_windows = defaultdict(_new_counter)
    instance_windows = defaultdict(_new_counter) if include_instance_windows else None

    for instance_id, csv_path in sorted(csv_files.items()):
        instance_counter = _new_counter()
        for chunk in _iter_normalized_chunks(csv_path, chunksize):
            chunk = _filter_time_range(chunk, start_ns, end_ns)
            if chunk.empty:
                continue

            _update_counter_from_frame(instance_counter, chunk)
            _update_counter_from_frame(global_counter, chunk)

            if window_ns is not None:
                starts = (chunk["TimestampNs"].astype("int64") // window_ns) * window_ns
                work = pd.DataFrame({
                    "WindowStartNs": starts,
                    "ReadBlocks": chunk["ReadBlocks"].astype("int64"),
                    "LocalHitBlocks": chunk["LocalHitBlocks"].astype("int64"),
                    "RemoteHitBlocks": chunk["RemoteHitBlocks"].astype("int64"),
                    "HitBlocks": chunk["HitBlocks"].astype("int64"),
                    "LocalHitTokens": chunk["LocalHitTokens"].astype("int64"),
                    "RemoteHitTokens": chunk["RemoteHitTokens"].astype("int64"),
                    "InputTokens": chunk["InputTokens"].astype("int64"),
                    "HitTokens": chunk["HitTokens"].astype("int64"),
                    "RequestCount": 1,
                })
                grouped = work.groupby("WindowStartNs", sort=False).sum(numeric_only=True)
                for window_start, row in grouped.iterrows():
                    _update_counter_from_sums(global_windows[int(window_start)], row)
                    if instance_windows is not None:
                        _update_counter_from_sums(instance_windows[(instance_id, int(window_start))], row)

        if instance_counter["request_count"] > 0:
            per_instance[instance_id] = instance_counter

    instance_rows = [
        _counter_to_row(counter, bucket_name, instance_id, start_ns, end_ns)
        for instance_id, counter in sorted(per_instance.items())
    ]
    global_row = _counter_to_row(global_counter, bucket_name, "__GLOBAL__", start_ns, end_ns)
    global_row["InstanceCount"] = len(per_instance)

    instance_summary_path = os.path.join(aggregate_dir, "instance_aggregate.csv")
    global_summary_path = os.path.join(aggregate_dir, "global_aggregate.csv")
    pd.DataFrame(instance_rows, columns=_row_columns()).to_csv(instance_summary_path, index=False)
    pd.DataFrame([global_row], columns=_row_columns(include_instance_count=True)).to_csv(
        global_summary_path, index=False)

    result = {
        "instance_summary_path": instance_summary_path,
        "global_summary_path": global_summary_path,
        "instance_count": len(per_instance),
        "global": global_row,
    }

    if window_ns is not None:
        global_window_rows = [
            _counter_to_row(
                counter,
                bucket_name,
                "__GLOBAL__",
                window_start,
                window_start + window_ns,
                window_start,
                window_start + window_ns,
            )
            for window_start, counter in sorted(global_windows.items())
        ]
        global_window_path = os.path.join(aggregate_dir, "global_window_hit_rates.csv")
        pd.DataFrame(global_window_rows, columns=_row_columns(include_window=True)).to_csv(
            global_window_path, index=False)
        result["global_window_path"] = global_window_path

        if instance_windows is not None:
            instance_window_rows = [
                _counter_to_row(
                    counter,
                    bucket_name,
                    instance_id,
                    window_start,
                    window_start + window_ns,
                    window_start,
                    window_start + window_ns,
                )
                for (instance_id, window_start), counter in sorted(instance_windows.items())
            ]
            instance_window_path = os.path.join(aggregate_dir, "instance_window_hit_rates.csv")
            pd.DataFrame(instance_window_rows, columns=_row_columns(include_window=True)).to_csv(
                instance_window_path, index=False)
            result["instance_window_path"] = instance_window_path

    return result


def _row_columns(include_window: bool = False, include_instance_count: bool = False) -> list:
    columns = [
        "Bucket",
        "InstanceId",
        "StartNs",
        "EndNs",
        "FirstTimestampNs",
        "LastTimestampNs",
        "RequestCount",
        "ReadBlocks",
        "LocalHitBlocks",
        "RemoteHitBlocks",
        "HitBlocks",
        "LocalHitTokens",
        "RemoteHitTokens",
        "InputTokens",
        "HitTokens",
        "LocalHitRate",
        "RemoteHitRate",
        "HitRate",
    ]
    if include_window:
        columns.extend(["WindowStartNs", "WindowEndNs"])
    if include_instance_count:
        columns.append("InstanceCount")
    return columns


def _iter_normalized_chunks(csv_path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    header_set = set(header)
    for col in ["TimestampNs"] + COUNTER_COLUMNS:
        if col not in header_set:
            raise ValueError(f"{csv_path} is missing {col}")

    usecols = [
        col
        for col in ["TimestampNs"] + COUNTER_COLUMNS
        if col in header_set
    ]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        yield chunk[["TimestampNs"] + COUNTER_COLUMNS]


def _filter_time_range(df: pd.DataFrame, start_ns: Optional[int], end_ns: Optional[int]) -> pd.DataFrame:
    if start_ns is not None:
        df = df[df["TimestampNs"] >= start_ns]
    if end_ns is not None:
        df = df[df["TimestampNs"] < end_ns]
    return df


def _new_counter() -> dict:
    return {
        "request_count": 0,
        "read_blocks": 0,
        "local_hit_blocks": 0,
        "remote_hit_blocks": 0,
        "hit_blocks": 0,
        "local_hit_tokens": 0,
        "remote_hit_tokens": 0,
        "input_tokens": 0,
        "hit_tokens": 0,
        "first_timestamp_ns": None,
        "last_timestamp_ns": None,
    }


def _update_counter_from_frame(counter: dict, df: pd.DataFrame) -> None:
    counter["request_count"] += int(len(df))
    counter["read_blocks"] += int(df["ReadBlocks"].sum())
    counter["local_hit_blocks"] += int(df["LocalHitBlocks"].sum())
    counter["remote_hit_blocks"] += int(df["RemoteHitBlocks"].sum())
    counter["hit_blocks"] += int(df["HitBlocks"].sum())
    counter["local_hit_tokens"] += int(df["LocalHitTokens"].sum())
    counter["remote_hit_tokens"] += int(df["RemoteHitTokens"].sum())
    counter["input_tokens"] += int(df["InputTokens"].sum())
    counter["hit_tokens"] += int(df["HitTokens"].sum())
    _update_timestamp_bounds(counter, int(df["TimestampNs"].min()), int(df["TimestampNs"].max()))


def _update_counter_from_sums(counter: dict, row: pd.Series) -> None:
    counter["request_count"] += int(row["RequestCount"])
    counter["read_blocks"] += int(row["ReadBlocks"])
    counter["local_hit_blocks"] += int(row["LocalHitBlocks"])
    counter["remote_hit_blocks"] += int(row["RemoteHitBlocks"])
    counter["hit_blocks"] += int(row["HitBlocks"])
    counter["local_hit_tokens"] += int(row["LocalHitTokens"])
    counter["remote_hit_tokens"] += int(row["RemoteHitTokens"])
    counter["input_tokens"] += int(row["InputTokens"])
    counter["hit_tokens"] += int(row["HitTokens"])


def _update_timestamp_bounds(counter: dict, first_ts: int, last_ts: int) -> None:
    if counter["first_timestamp_ns"] is None or first_ts < counter["first_timestamp_ns"]:
        counter["first_timestamp_ns"] = first_ts
    if counter["last_timestamp_ns"] is None or last_ts > counter["last_timestamp_ns"]:
        counter["last_timestamp_ns"] = last_ts


def _counter_to_row(
    counter: dict,
    bucket_name: str,
    instance_id: str,
    start_ns: Optional[int],
    end_ns: Optional[int],
    window_start_ns: Optional[int] = None,
    window_end_ns: Optional[int] = None,
) -> dict:
    local_hit_tokens = int(counter["local_hit_tokens"])
    remote_hit_tokens = int(counter["remote_hit_tokens"])
    input_tokens = int(counter["input_tokens"])
    hit_tokens = int(counter["hit_tokens"])
    read_blocks = int(counter["read_blocks"])
    hit_blocks = int(counter["hit_blocks"])
    row = {
        "Bucket": bucket_name,
        "InstanceId": instance_id,
        "StartNs": start_ns if start_ns is not None else "",
        "EndNs": end_ns if end_ns is not None else "",
        "FirstTimestampNs": counter["first_timestamp_ns"] if counter["first_timestamp_ns"] is not None else "",
        "LastTimestampNs": counter["last_timestamp_ns"] if counter["last_timestamp_ns"] is not None else "",
        "RequestCount": int(counter["request_count"]),
        "ReadBlocks": read_blocks,
        "LocalHitBlocks": int(counter["local_hit_blocks"]),
        "RemoteHitBlocks": int(counter["remote_hit_blocks"]),
        "HitBlocks": hit_blocks,
        "LocalHitTokens": local_hit_tokens,
        "RemoteHitTokens": remote_hit_tokens,
        "InputTokens": input_tokens,
        "HitTokens": hit_tokens,
        "LocalHitRate": local_hit_tokens / input_tokens if input_tokens else 0.0,
        "RemoteHitRate": remote_hit_tokens / input_tokens if input_tokens else 0.0,
        "HitRate": hit_tokens / input_tokens if input_tokens else 0.0,
    }
    if window_start_ns is not None:
        row["WindowStartNs"] = window_start_ns
    if window_end_ns is not None:
        row["WindowEndNs"] = window_end_ns
    return row
