from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from hisim.simulation.types import SchedulerConfig
from hisim.spec.model import ModelInfo
from hisim.spec.accelerator import AcceleratorInfo
from hisim.utils import get_logger


logger = get_logger("hisim")


@dataclass
class FakeRequest:
    input_length: int = 0
    past_kv_length: int = 0


@dataclass
class ScheduleBatch:
    reqs: list[FakeRequest] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"batch_size={len(self.reqs)},reqs={[(req.input_length, req.past_kv_length) for req in self.reqs]}"

    def __eq__(self, batch: "ScheduleBatch"):
        if self.batch_size != batch.batch_size:
            return False

        req1, req2 = [], []
        for idx in range(self.batch_size):
            req1.append((self.reqs[idx].input_length, self.reqs[idx].past_kv_length))

            req2.append((batch.reqs[idx].input_length, batch.reqs[idx].past_kv_length))

        return sorted(req1) == sorted(req2)

    def request_info(self) -> list[list[int, int]]:
        # The request information organized in the format `(input_len, past_kv_len)`
        return [[req.input_length, req.past_kv_length] for req in self.reqs]

    @property
    def num_context_tokens(self) -> int:
        return sum(req.input_length for req in self.reqs)

    @property
    def total_past_kv_length(self) -> int:
        return sum(req.past_kv_length for req in self.reqs)

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def is_prefill(self) -> bool:
        return not self.is_decode()

    def is_decode(self) -> bool:
        for req in self.reqs:
            if req.input_length > 1:
                return False
        return True

    @property
    def num_ctx_requests(self) -> int:
        return self.batch_size if self.is_prefill() else 0

    @property
    def num_gen_requests(self) -> int:
        return self.batch_size if self.is_decode() else 0


class InferTimePredictor(ABC):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        *args,
        **kwargs,
    ):
        self.model: ModelInfo = model
        self.hw: AcceleratorInfo = hw
        self.config: SchedulerConfig = config

    @abstractmethod
    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        # Return the inference time in seconds. Return a negative value if an exception occurs (e.g., out of memory).
        pass
