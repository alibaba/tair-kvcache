from schedule_simulator.schedule_emulator.types import (
    SchedulerConfig,
)
from kunlun_commons.model_info import ModelInfo
from kunlun_commons.hardwares.accelerator import AcceleratorInfo
from kunlun_commons.data_type import DataType
from kunlun_commons.utils import get_logger

from deepestim.llmperf import TheoreticalExecutionAnalyzer, RequestInfo, InferenceConfig

from schedule_simulator.infer_time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
)


logger = get_logger("schedule_simulator")


class LLMPerfTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        *args,
        **kwargs,
    ):
        super().__init__(model, hw, config)

        if self.config.data_type is None:
            self.config.data_type = DataType.from_torch_dtype(self.model.torch_dtype)
            logger.warning(
                f"Data type is not set. Use the model's torch data type, {self.config.data_type.value}, default."
            )

        self.llm_analyzer = TheoreticalExecutionAnalyzer(
            model=self.model,
            hw=self.hw,
            infer_config=InferenceConfig(
                framework="sglang",
                max_num_seqs=self.model.max_seq_len,
                dt=self.config.data_type,
                tp=self.config.tp_size,
                dp=self.config.dp_size,
                ep=self.config.ep_size,
                pp=self.config.pp_size,
            ),
        )

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        req_infos = []
        if batch.is_prefill():
            for req in batch.reqs:
                # When the input tokens of a prefill request are fully cached in the key-value (KV) cache,
                # the context_prefill_length is set to zero.
                # TODO: remove to decoding batch?
                req_infos.append(
                    RequestInfo(max(req.input_length, 1), req.past_kv_length)
                )
        else:
            for req in batch.reqs:
                req_infos.append(RequestInfo(1, req.past_kv_length))
        latency = self.llm_analyzer.analyze_step_latency(req_infos)
        if latency < 0:
            latency = -latency
            logger.warning("Encountered OOM (Out of Memory) for the current batch.")
        return latency
