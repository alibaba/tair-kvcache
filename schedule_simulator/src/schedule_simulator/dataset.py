from typing import Optional, Any, overload
from dataclasses import dataclass, field


@dataclass
class GenericRequest:
    session_id: Optional[Any] = None
    timestamp: float = 0
    prompt: Optional[str] = None
    token_ids: Optional[list[int]] = None
    output_ids: Optional[list[int]] = None
    input_length: int = -1
    output_length: int = -1
    custom_params: dict = field(default_factory=dict)


class BaseDataset:
    @overload
    def __getitem__(self, index: int) -> GenericRequest: ...

    @overload
    def __getitem__(self, index: slice) -> list[GenericRequest]: ...

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class MultiTurnConversationDataset(BaseDataset):
    pass
