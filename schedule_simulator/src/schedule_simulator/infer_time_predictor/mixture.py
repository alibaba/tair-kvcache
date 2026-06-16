from schedule_simulator.infer_time_predictor.base import (
    InferTimePredictor,
)
from schedule_simulator.infer_time_predictor.aiconfigurator import (
    AIConfiguratorTimePredictor,
)
from schedule_simulator.infer_time_predictor.step_benchmark import (
    StepBenchmarkTimePredictor,
)


class MixtureTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model,
        hw,
        config,
        step_benchmark_database_path: str,
        aic_database_path: str,
        aic_database_mode: str = "SILICON",
        *args,
        **kwargs,
    ):
        super().__init__(model, hw, config, *args, **kwargs)

        self.aiconfigurator_predictor = AIConfiguratorTimePredictor(
            model=model,
            hw=hw,
            config=config,
            database_path=aic_database_path,
            database_mode=aic_database_mode,
        )

        self.step_predictor = StepBenchmarkTimePredictor(
            model=model,
            hw=hw,
            config=config,
            database_path=step_benchmark_database_path,
        )

    def predict_infer_time(self, batch):
        if batch.is_prefill():
            return self.aiconfigurator_predictor.predict_infer_time(batch)
        else:
            return self.step_predictor.predict_infer_time(batch)
