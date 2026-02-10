import json
from hisim.dataset.base_dataset import BaseDataset, GenericRequest
from hisim.utils import get_logger


logger = get_logger("hisim")


class HisimCollectionDataset(BaseDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self._name = "hisim_collection"
        self.dataset = self._load_dataset(args.filepath)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if index >= len(self):
            raise IndexError

        return self.dataset[index]

    def _load_dataset(self, filepath: str) -> list[GenericRequest]:
        raw_data = []
        with open(filepath) as f:
            line = f.readline()
            while line:
                raw_data.append(json.loads(line))
                line = f.readline()

        if len(raw_data) == 0:
            logger.warning(f"Dataset is empty: {filepath}")
            return []

        min_created_ts = float("inf")
        # Adapted with the old version.
        time_field_name = (
            "created_time" if "created_time" in raw_data[0] else "timestamp"
        )
        for req in raw_data:
            min_created_ts = min(min_created_ts, req[time_field_name])

        dataset = []
        # align timestamp
        for req in raw_data:
            dataset.append(
                GenericRequest(
                    token_ids=req["input_ids"],
                    input_length=len(req["input_ids"]),
                    output_length=req["output_length"],
                    custom_params={
                        "created_time": req[time_field_name] - min_created_ts,
                    },
                )
            )

        return dataset
