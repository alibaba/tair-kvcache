from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetArgs:
    name: str = ""
    filepath: str = ""
    num_prompts: int = -1
    min_input_len: int = -1
    max_input_len: int = -1
    min_output_len: int = -1
    max_output_len: int = -1
    prefix_hit_rate: Optional[float] = None

    @property
    def mean_input_length(self):
        return (self.min_input_len + self.max_input_len) // 2

    @property
    def mean_output_length(self):
        return (self.min_output_len + self.max_output_len) // 2
