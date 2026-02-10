from hisim.dataset.base_dataset import BaseDataset, GenericRequest
from typing import Union


class PrefixCacheDecorator(BaseDataset):
    """
    Decorator for dataset classes to handle prefix_hit_rate functionality.
    Uses the Decorator design pattern to add prefix caching capabilities
    to any dataset implementation without modifying the original classes.
    """

    def __init__(self, dataset: BaseDataset):
        super().__init__(dataset.tokenizer, dataset.args)
        self._dataset = dataset
        self._name = dataset.name
        self.cached: list[GenericRequest] = []

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[GenericRequest, list[GenericRequest]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if index >= len(self):
            raise IndexError

        # If prefix_hit_rate is not set, delegate directly to the wrapped dataset
        if self.args.prefix_hit_rate is None:
            return self._dataset[index]
        else:
            # Handle prefix cache logic
            return self._get_item_with_prefix_cache(index)

    def _get_item_with_prefix_cache(self, index: int) -> GenericRequest:
        if index == 0:
            # The first item
            item = self._dataset[index]
            self.cached.append(item)
            return item
        else:
            new_item = self._dataset[index]
            old_item = self.cached[-1]

            if old_item.prompt is not None:
                old_prompt_ids = self.tokenizer.encode(
                    old_item.prompt, add_special_tokens=False
                )
            else:
                old_prompt_ids = old_item.token_ids

            if new_item.prompt is not None:
                new_prompt_ids = self.tokenizer.encode(
                    new_item.prompt, add_special_tokens=False
                )
            else:
                new_prompt_ids = new_item.token_ids

            final_prompt_ids = old_prompt_ids[
                : int(len(old_prompt_ids) * self.args.prefix_hit_rate)
            ]
            if new_prompt_ids:
                while len(final_prompt_ids) < old_item.input_length:
                    final_prompt_ids.extend(new_prompt_ids)
            final_prompt_ids = final_prompt_ids[
                : min(old_item.input_length, len(final_prompt_ids))
            ]

            new_item.input_length = old_item.input_length

            if new_item.prompt is not None:
                new_item.prompt = self.tokenizer.decode(
                    final_prompt_ids, skip_special_tokens=True
                )
            else:
                new_item.token_ids = final_prompt_ids

            self.cached.append(new_item)
            return new_item
