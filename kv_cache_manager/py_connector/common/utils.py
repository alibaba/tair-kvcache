import hashlib
import time


def calc_md5_hash(input: str):
    return hashlib.md5(input.encode()).hexdigest()


def deadline_ms_from_now(timeout_ms: int) -> int:
    """返回当前时刻 + timeout_ms 对应的绝对 deadline（steady_clock 毫秒）。

    与 C++ SteadyClockMs() 同源（time.monotonic_ns() 与 steady_clock 均为 CLOCK_MONOTONIC）。
    """
    return time.monotonic_ns() // 1_000_000 + timeout_ms