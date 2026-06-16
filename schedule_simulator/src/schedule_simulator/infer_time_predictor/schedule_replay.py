import json
import os

from schedule_simulator.infer_time_predictor.base import (
    InferTimePredictor,
    ScheduleRequest,
    ScheduleBatch,
)
from kunlun_commons.utils.logger import get_logger


logger = get_logger("schedule_simulator")


class ScheduleReplayTimePredictor(InferTimePredictor):
    def __init__(self, model, hw, config, database_path: str, *args, **kwargs):
        super().__init__(model, hw, config, *args, **kwargs)

        self.data: list[tuple[ScheduleBatch, float]] = []
        for db_path in database_path.split(","):
            if os.path.exists(db_path):
                self.data.extend(self._load_data(db_path))
            else:
                raise RuntimeError(f"{db_path} is not exist.")
        self.iteration = 0

    def _load_data(self, database_path: str) -> list[tuple[ScheduleBatch, float]]:
        # Load the scheduled batch information which collected by `examples/benchmark`
        data = []
        with open(database_path) as f:
            line = f.readline()
            while line:
                item = json.loads(line)
                reqs: list[ScheduleRequest] = []
                for req in item["request_infos"]:
                    if req["output_ids_len"] == 0:
                        input_len = req["extend_input_len"]
                        past_kv_len = req["prefix_indices_len"]
                    else:
                        input_len = 1
                        past_kv_len = req["prefix_indices_len"] + req["output_ids_len"]
                    reqs.append(
                        ScheduleRequest(
                            input_length=input_len,
                            past_kv_length=past_kv_len,
                        )
                    )
                data.append((ScheduleBatch(reqs), item["iter_latency"]))
                line = f.readline()
        return data

    def predict_infer_time(self, batch):
        if self.iteration >= len(self.data):
            logger.warning("Iteration index is out of range for the dataset.")
            return -1
        if batch != self.data[self.iteration][0]:
            logger.debug(
                f"Scheduler replay mismatch: Current={batch} -> Real={self.data[self.iteration][0]}"
            )
        latency = self.data[max(self.iteration, 0)][1]
        self.iteration += 1
        return latency
