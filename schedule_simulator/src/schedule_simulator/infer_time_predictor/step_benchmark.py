from schedule_simulator.schedule_emulator.types import (
    SchedulerConfig,
)
from schedule_simulator.infer_time_predictor.base import ScheduleBatch
from schedule_simulator._compat import ModelInfo, AcceleratorInfo, get_logger

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


from schedule_simulator.infer_time_predictor.deepestim import (
    LLMPerfTimePredictor,
)

logger = get_logger("schedule_simulator")


class StepBenchmarkTimePredictor(LLMPerfTimePredictor):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
    ):
        super().__init__(model, hw, config)
        df = pd.read_csv(database_path)
        df["mean_iter_latency_ms"] /= 1e3  # ms -> s
        df["min_iter_latency_ms"] /= 1e3  # ms -> s

        decode_df = df[df["num_context_requests"] == 0].copy()
        decode_df = decode_df.groupby(
            ["num_generation_requests", "mean_input_length"], as_index=False
        ).agg("mean")
        decode_df["total_past_kv_length"] = (
            decode_df["num_generation_requests"] * decode_df["mean_input_length"]
        )

        # During the Decode phase, for each Batch Size, construct an interpolation function for the KV Cache.
        self.decode_interp1d_models = {}
        for batch_size in pd.unique(decode_df["num_generation_requests"]):
            data = decode_df[decode_df["num_generation_requests"] == batch_size]
            data = data.sort_values("total_past_kv_length")
            self.decode_interp1d_models[int(batch_size)] = interp1d(
                x=data["total_past_kv_length"].values,
                y=data["min_iter_latency_ms"].values,
                kind="linear",
                fill_value="extrapolate",
            )

        # During the Prefill phase, for each request, establish a quadratic function for the New Token and Prefix Cache.
        prefill_df = df[
            (df["num_context_requests"] == 1) & (df["num_generation_requests"] == 0)
        ].copy()
        self.prefill_curve_model_params = {}

        for plan, lower_bound, upper_bound in [
            ("plan1", 1, 256),
            ("plan2", 257, 624),
            ("plan3", 513, 1024),
            ("plan4", 513, 2048),  # extend the lower bound
            ("plan5", 1537, 4096),
            ("plan6", 4097, float("inf")),
        ]:
            data = prefill_df[
                (prefill_df["mean_context_tokens"] >= lower_bound)
                & (prefill_df["mean_context_tokens"] <= upper_bound)
            ]
            if data.empty:
                logger.warning(
                    f"The data in plan(lower={lower_bound}, upper={upper_bound}) is empty."
                )
                self.prefill_curve_model_params[plan] = None
                continue

            X = data[["mean_context_tokens", "mean_reused_tokens"]].values.T
            y = data["mean_iter_latency_ms"].values

            def prefill_curve_model(x, a, b, c) -> float:
                attn, mlp = self._prefill_curve_model(x, a, b, c)
                return attn + mlp

            try:
                params, covariance = curve_fit(prefill_curve_model, X, y)
                y_pred = prefill_curve_model(X, *params)
                residuals = y - y_pred
                mse = np.sum(residuals**2) / (
                    len(y) - len(params)
                )  # Bessel's correction
                logger.info(
                    f"Fit curve for plan(lower={lower_bound}, upper={upper_bound}) with {len(data)} items. MSE={mse}"
                )
            except Exception as e:
                logger.warning(
                    f"Fail to fit the plan(lower={lower_bound}, upper={upper_bound}). Error: {e}"
                )
                self.prefill_curve_model_params[plan] = None
                continue
            self.prefill_curve_model_params[plan] = params

    def _choose_prefill_plan(self, context_len: int):
        if context_len <= 256:
            params = self.prefill_curve_model_params["plan1"]
        elif context_len <= 624:
            params = self.prefill_curve_model_params["plan2"]
        elif context_len <= 1248:
            params = self.prefill_curve_model_params["plan3"]
        elif context_len <= 2496:
            params = self.prefill_curve_model_params["plan4"]
        elif context_len <= 4992:
            params = self.prefill_curve_model_params["plan5"]
        else:
            params = self.prefill_curve_model_params["plan6"]
        return params

    @staticmethod
    def _prefill_curve_model(x, a, b, c) -> tuple[float, float]:
        r"""
        The attention latency is relative to past KV length and the context tokens.
        So, assume the attention latency can be fit by $attn\_latency = a \cdot \sum_{i=1}^{context\_length}(past\_kv\_length + i) \cdot i$
        and other operator latency (mostly MLP) can be fit by $mlp\_latency = context\_length \cdot b + c$
        """
        context_len = x[0]
        past_kv_len = x[1]
        i = context_len
        attn_latency = a * (
            i / 2 * (i - 1) * past_kv_len + i * (i + 1) * (2 * i + 1) / 6
        )
        mlp_latency = context_len * b + c

        return abs(attn_latency), abs(mlp_latency)

    def _predict_decode_time(self, batch: ScheduleBatch):
        if batch.batch_size in self.decode_interp1d_models:
            total_past_kv_length = 0
            for req in batch.reqs:
                total_past_kv_length += req.past_kv_length
            return self.decode_interp1d_models[batch.batch_size](total_past_kv_length)
        else:
            logger.warning(
                f"Interp1d model is not found for current batch(batch_size={batch.batch_size})."
            )
            return super().predict_infer_time(batch)

    def _predict_prefill_time(self, batch: ScheduleBatch):
        attn_latency = 0
        total_context_len = 0
        # The attention latency will accumulate with each request.
        for req in batch.reqs:
            params = self._choose_prefill_plan(req.input_length)
            if params is None:
                return super().predict_infer_time(batch)
            attn, _ = self._prefill_curve_model(
                (req.input_length, req.past_kv_length), *params
            )
            attn_latency += attn
            total_context_len += req.input_length

        # The mlp latency will be predicted with all total context len.
        params = self._choose_prefill_plan(total_context_len)
        if params is None:
            return super().predict_infer_time(batch)
        _, mlp_latency = self._prefill_curve_model((total_context_len, 0), *params)

        return attn_latency + mlp_latency

    def predict_infer_time(self, batch):
        if batch.is_prefill():
            return self._predict_prefill_time(batch)
        else:
            return self._predict_decode_time(batch)
