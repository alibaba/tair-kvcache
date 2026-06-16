"""
Request-level time predictor.
Predicts total prefill time for a request (not per-iteration).
Supports: lookup table from pkl (1D or 2D), constant value, or custom callable.
"""
import math
import joblib
from typing import Optional, Callable

from schedule_simulator.infer_time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)


class RequestLevelTimePredictor:
    """
    Predicts total prefill compute time for a request.
    Does NOT inherit InferTimePredictor (no ModelInfo/AcceleratorInfo needed).
    """

    def __init__(
        self,
        lookup_table_path: Optional[str] = None,
        constant_ms_per_token: Optional[float] = None,
        predict_fn: Optional[Callable] = None,
    ):
        self.table = None
        self.train_table = None
        self.bins = None
        self.mode = "1d"  # "1d" or "2d"
        self._predict_fn = predict_fn
        self._constant_rate = constant_ms_per_token

        if lookup_table_path:
            bundle = joblib.load(lookup_table_path)
            self.train_table = bundle.get("train_table", bundle.get("table", {}))
            self.bins = bundle.get("bins")
            self.mode = bundle.get("mode", "1d")
            # Auto-detect 2D if keys are 4-tuples
            if self.train_table:
                first_key = next(iter(self.train_table.keys()))
                if len(first_key) == 4:
                    self.mode = "2d"
            # Pre-sort table once for fast lookup (avoid sorting on every call)
            self._sorted_items = sorted(self.train_table.items()) if self.train_table else []

    def predict_request_time(self, uncached_tokens: int, cached_tokens: int = 0) -> float:
        """Returns total prefill time in SECONDS."""
        if self._predict_fn is not None:
            return self._predict_fn(uncached_tokens, cached_tokens)

        if self._constant_rate is not None:
            return max(uncached_tokens * self._constant_rate / 1000.0, 0.001)

        if self.train_table:
            ms = self._lookup(uncached_tokens, cached_tokens)
            return ms / 1000.0

        return max(uncached_tokens * 0.1 / 1000.0, 0.001)

    def _lookup(self, uncached: int, cached: int = 0) -> float:
        if self._sorted_items:
            if self.mode == "2d":
                # 2D lookup: (uncached_lo, uncached_hi, cached_lo, cached_hi)
                for key, med in self._sorted_items:
                    u_lo, u_hi, c_lo, c_hi = key
                    if u_lo <= uncached < u_hi and c_lo <= cached < c_hi:
                        return med
                # Fallback: find closest uncached bin
                for key, med in self._sorted_items:
                    u_lo, u_hi, c_lo, c_hi = key
                    if u_lo <= uncached < u_hi:
                        return med
            else:
                # 1D lookup: (uncached_lo, uncached_hi)
                for (lo, hi), med in self._sorted_items:
                    if lo <= uncached < hi:
                        return med
            return self._sorted_items[-1][1]
        return 500.0

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        """Fallback for iteration-level interface compatibility."""
        total = 0
        for req in batch.reqs:
            total += self.predict_request_time(req.input_length, req.past_kv_length)
        return total
