from hisim.dataset.dataset_args import DatasetArgs
from hisim.dataset import get_dataset

from transformers import AutoTokenizer
from env import MODEL_PATH
import numpy as np


def longest_common_prefix(str1, str2):
    prefix = []
    for a, b in zip(str1, str2):
        if a == b:
            prefix.append(a)
        else:
            break
    return prefix


def test_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # normal dataset
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=20,
        min_input_len=100,
        max_input_len=200,
        min_output_len=8,
        max_output_len=16,
    )

    dataset = get_dataset(dataset_args, tokenizer)
    for req in dataset:
        assert (
            dataset_args.min_input_len <= req.input_length <= dataset_args.max_input_len
        )
        assert (
            dataset_args.min_output_len
            <= req.output_length
            <= dataset_args.max_output_len
        )

    # Prefix Cache
    dataset_args = DatasetArgs(
        "random",
        num_prompts=20,
        min_input_len=100,
        max_input_len=200,
        min_output_len=8,
        max_output_len=16,
        prefix_hit_rate=0.8,
    )
    dataset = get_dataset(dataset_args, tokenizer)
    last_req = None
    hit_rates = []
    for req in dataset:
        if last_req is not None:
            curr_prompt_ids = tokenizer.encode(req.prompt)
            last_prompt_ids = tokenizer.encode(last_req.prompt)
            prefix = longest_common_prefix(curr_prompt_ids, last_prompt_ids)
            hit_rates.append(len(prefix) / len(curr_prompt_ids))
        last_req = req
    assert 0.75 < np.average(hit_rates) < 0.85


if __name__ == "__main__":
    test_dataset()
