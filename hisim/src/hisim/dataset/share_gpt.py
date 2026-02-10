from hisim.dataset.base_dataset import BaseDataset, GenericRequest
from random import randint
import os
import json
from hisim.utils import get_logger


logger = get_logger("hisim")


class ShareGPTDataset(BaseDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)

        self._name = "sample_sharegpt"
        if self.args.name != self.name:
            logger.warning(
                f"The required dataset is {self.args.name}, while the current dataset is {self.name}"
            )

        if not os.path.exists(self.args.filepath):
            raise FileNotFoundError(f"The file[{self.args.filepath}] does not exist.")

        self.dataset = self.load_file(self.args.filepath)
        self.index = 0

    def __len__(self):
        return self.args.num_prompts

    def __getitem__(self, index) -> GenericRequest:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if index >= len(self):
            raise IndexError

        input_len = randint(self.args.min_input_len, self.args.max_input_len)

        prompt: str = self.dataset[self.index % len(self.dataset)]
        self.index += 1  # new sequence

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = self.fix_size(input_ids, input_len)

        return GenericRequest(
            prompt=self.tokenizer.decode(input_ids, skip_special_tokens=True),
            input_length=input_len,
            output_length=randint(self.args.min_output_len, self.args.max_output_len),
        )

    def load_file(
        self,
        dataset_path: str,
    ) -> list[str]:
        # Load the dataset.
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)
        if not dataset:
            raise ValueError("The dataset file is empty or contains no valid data.")
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # The first sequences might be too short
        dataset = [data["conversations"][1]["value"] for data in dataset]
        # Ignore sequences with fewer than 20 words.
        dataset = list(filter(lambda item: len(item.split(" ")) > 20, dataset))
        if not dataset:
            raise ValueError("No valid sequences found in the dataset file.")
        return dataset

    def fix_size(self, data: list, size: int) -> list:
        while len(data) < size:
            data = data + data
        return data[:size]
