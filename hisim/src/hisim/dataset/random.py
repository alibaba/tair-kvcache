from hisim.dataset.base_dataset import BaseDataset, GenericRequest
from random import randint


class RandomIDsDataset(BaseDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self._name = "random_ids"

    def __len__(self):
        return self.args.num_prompts

    def _get_single_item(self, index: int) -> GenericRequest:
        min_id, max_id = (
            int(self.tokenizer.vocab_size * 0.25),
            int(self.tokenizer.vocab_size * 0.75),
        )

        input_len = randint(self.args.min_input_len, self.args.max_input_len)
        input_ids = [randint(min_id, max_id) for _ in range(input_len)]

        return GenericRequest(
            token_ids=input_ids,
            input_length=input_len,
            output_length=randint(self.args.min_output_len, self.args.max_output_len),
        )


class RandomDataset(RandomIDsDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.cached: list[GenericRequest] = []
        self._name = "random"

    def __len__(self):
        return self.args.num_prompts

    def _get_single_item(self, index: int) -> GenericRequest:
        req = super()._get_single_item(index)
        if req.token_ids is not None:
            req.prompt = self.tokenizer.decode(req.token_ids, skip_special_tokens=True)
            req.token_ids = None
        return req


class IdenticalIDsDataset(BaseDataset):
    """
    A dataset where all requests have the exact same content.

    This is used to simulate scenarios where identical input tokens are used,
    such as benchmarking decoding latency without prefill overhead.
    """

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self._name = "identical_ids"

    def __len__(self):
        return self.args.num_prompts

    def _get_single_item(self, index: int) -> GenericRequest:
        req_id = int(self.tokenizer.vocab_size * 0.5)  # avoid the special tokens.
        input_len = randint(self.args.min_input_len, self.args.max_input_len)
        input_ids = [req_id] * input_len

        return GenericRequest(
            token_ids=input_ids,
            input_length=input_len,
            output_length=randint(self.args.min_output_len, self.args.max_output_len),
        )
