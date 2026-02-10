from transformers import PreTrainedTokenizer

from hisim.dataset.base_dataset import BaseDataset, GenericRequest
from hisim.dataset.random import (
    RandomDataset,
    RandomIDsDataset,
    IdenticalIDsDataset,
)
from hisim.dataset.share_gpt import ShareGPTDataset
from hisim.dataset.hisim_collection import HisimCollectionDataset
from hisim.dataset.dataset_args import DatasetArgs
from hisim.dataset.prefix_cache import PrefixCacheDecorator
from hisim.utils import get_logger


logger = get_logger("hisim")


dataset_registry: dict[str, BaseDataset] = {
    "random": RandomDataset,
    "random_ids": RandomIDsDataset,
    "sample_sharegpt": ShareGPTDataset,
    "identical_ids": IdenticalIDsDataset,
    "hisim_collection": HisimCollectionDataset,
}


def get_dataset(
    dataset_args: DatasetArgs, tokenizer: PreTrainedTokenizer | None = None
) -> BaseDataset:
    if dataset_args.name not in dataset_registry:
        raise ValueError(f"unknown dataset name: {dataset_args.name}")

    dataset: BaseDataset = dataset_registry[dataset_args.name](
        args=dataset_args, tokenizer=tokenizer
    )

    logger.info(f"Initialized dataset[{dataset.name}] with {len(dataset)} requests.")
    if dataset_args.prefix_hit_rate is not None and dataset_args.prefix_hit_rate > 0:
        return PrefixCacheDecorator(dataset)
    else:
        return dataset


__all__ = (
    "BaseDataset",
    "GenericRequest",
    "get_dataset",
)
