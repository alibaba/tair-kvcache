from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class KvCacheGroupMetadataErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    KV_CACHE_GROUP_METADATA_OK: _ClassVar[KvCacheGroupMetadataErrorCode]
    KV_CACHE_GROUP_METADATA_UNAVAILABLE: _ClassVar[KvCacheGroupMetadataErrorCode]
KV_CACHE_GROUP_METADATA_OK: KvCacheGroupMetadataErrorCode
KV_CACHE_GROUP_METADATA_UNAVAILABLE: KvCacheGroupMetadataErrorCode

class KvCacheGroupMetadataPB(_message.Message):
    __slots__ = ("group_idx", "kind", "block_size", "sliding_window")
    GROUP_IDX_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    SLIDING_WINDOW_FIELD_NUMBER: _ClassVar[int]
    group_idx: int
    kind: str
    block_size: int
    sliding_window: int
    def __init__(self, group_idx: _Optional[int] = ..., kind: _Optional[str] = ..., block_size: _Optional[int] = ..., sliding_window: _Optional[int] = ...) -> None: ...

class KvCacheGroupListPB(_message.Message):
    __slots__ = ("items", "err_code", "err_msg")
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    ERR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERR_MSG_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[KvCacheGroupMetadataPB]
    err_code: KvCacheGroupMetadataErrorCode
    err_msg: str
    def __init__(self, items: _Optional[_Iterable[_Union[KvCacheGroupMetadataPB, _Mapping]]] = ..., err_code: _Optional[_Union[KvCacheGroupMetadataErrorCode, str]] = ..., err_msg: _Optional[str] = ...) -> None: ...

class KvCacheGroupsRequestPB(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
