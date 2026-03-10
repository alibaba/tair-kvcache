from typing import List, Union, Optional, Any
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import psutil
import os
import threading
from functools import wraps

from hisim.utils.logger import get_logger
from hisim.simulation.manager import StateManager, ConfigManager, Envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
)

try:
    from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
except ImportError:
    kvcm_py_optimizer = None

logger = get_logger("hisim")
_CURRENT_DIR = Path(__file__).parent.resolve()


def alloc_extend_cpu(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,  # Pre-allocated output tensor (consistent with Triton kernel)
    bs_upper: int,  # CPU doesn't need this, but kept for interface consistency (can be ignored)
    page_size: int,
    max_num_extend_tokens: int,  # Also kept but not used
):
    # Convert to Python list or scalar for processing
    pre_lens = pre_lens_ptr.cpu().tolist()
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()

    batch_size = len(pre_lens)

    def ceil_div(a, b):
        return (a + b - 1) // b

    extend_lens = [seq_lens[i] - pre_lens[i] for i in range(batch_size)]
    num_new_pages_per_seq = []
    for i in range(batch_size):
        pages_before = ceil_div(pre_lens[i], page_size)
        pages_after = ceil_div(seq_lens[i], page_size)
        num_new_pages_per_seq.append(pages_after - pages_before)

    # Initialize offsets
    output_offset = 0
    free_page_offset = 0

    # Process each sequence
    for pid in range(batch_size):
        pre_len = pre_lens[pid]
        seq_len = seq_lens[pid]
        extend_len = extend_lens[pid]
        last_token_pos = last_loc[pid]

        if extend_len <= 0:
            # No extension, skip (but still need to advance free_page_offset)
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 1: Fill the remaining space of the current incomplete page ===
        current_page_end = ceil_div(pre_len, page_size) * page_size
        part1_end = min(seq_len, current_page_end)
        num_part1 = part1_end - pre_len

        if num_part1 > 0:
            # out_indices[output_offset : output_offset + num_part1] = last_token_pos + 1 + i
            for i in range(num_part1):
                out_indices[output_offset + i] = last_token_pos + 1 + i
            output_offset += num_part1

        if pre_len + num_part1 == seq_len:
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 2: Fill complete new pages ===
        full_pages_start = current_page_end
        full_pages_end = (seq_len // page_size) * page_size
        num_part2 = full_pages_end - full_pages_start

        if num_part2 > 0:
            for i in range(num_part2):
                page_idx_in_free = free_page_offset + (i // page_size)
                page_id = free_pages[page_idx_in_free]
                token_in_page = i % page_size
                out_indices[output_offset + i] = page_id * page_size + token_in_page
            output_offset += num_part2

        if pre_len + num_part1 + num_part2 == seq_len:
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 3: Fill the last incomplete new page ===
        num_part3 = seq_len - full_pages_end
        if num_part3 > 0:
            last_page_idx = free_page_offset + num_new_pages_per_seq[pid] - 1
            last_page_id = free_pages[last_page_idx]
            for i in range(num_part3):
                out_indices[output_offset + i] = last_page_id * page_size + i
            output_offset += num_part3

        # Push forward free_page_offset
        free_page_offset += num_new_pages_per_seq[pid]


def alloc_decode_cpu(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,  # Reserved parameter (not used in CPU)
    page_size: int,
):
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()

    batch_size = len(seq_lens)

    def ceil_div(a, b):
        return (a + b - 1) // b

    # Calculate the number of new pages needed for each sequence (used to determine free_page offset)
    num_new_pages_per_seq = []
    for i in range(batch_size):
        pre_len_i = seq_lens[i] - 1
        pages_before = ceil_div(pre_len_i, page_size)
        pages_after = ceil_div(seq_lens[i], page_size)
        num_new_pages_per_seq.append(pages_after - pages_before)

    # Calculate prefix sum to determine the starting position in free_page_ptr for each sequence
    prefix_sum = 0
    for pid in range(batch_size):
        num_new_pages_self = num_new_pages_per_seq[pid]
        new_page_start_loc = (
            prefix_sum  # Starting index in free_pages for current sequence
        )
        prefix_sum += num_new_pages_self

        seq_len = seq_lens[pid]
        pre_len = seq_len - 1

        num_page_start_loc_self = ceil_div(seq_len, page_size) - ceil_div(
            pre_len, page_size
        )

        if num_page_start_loc_self == 0:
            # Reuse current page, directly write last_loc + 1
            out_indices[pid] = last_loc[pid] + 1
        else:
            # Allocate new page, take the first new page ID
            page_id = free_pages[new_page_start_loc]
            out_indices[pid] = page_id * page_size


def get_num_new_pages(
    seq_lens: torch.Tensor,
    page_size: int,
    prefix_lens: Optional[torch.Tensor] = None,
    decode: bool = False,
) -> torch.Tensor:
    """
    Get the number of new pages for the given prefix and sequence lengths.
    We use cpu tensors to avoid blocking kernel launch.
    """
    cpu_device = torch.device("cpu")
    assert seq_lens.device == cpu_device

    if prefix_lens is None or decode:
        # NOTE: Special case for handling decode, which prefix lens is `seq_lens - 1`.
        assert decode
        return (seq_lens % page_size == 1).int().sum().item()

    assert prefix_lens.device == cpu_device
    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (prefix_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before
    sum_num_new_pages = torch.sum(num_new_pages).to(torch.int64)
    return sum_num_new_pages.item()


class MockReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    # from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        self.size = size
        self.max_context_len = 1  # overwrite
        self.device = device
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )

        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class MockTokenToKVPool:
    # from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # self.head_num = head_num
        # self.head_dim = head_dim
        # Overwrite
        self.head_num = 1
        self.head_dim = 1

        self.custom_mem_pool = None

        self._create_buffers()
        _is_cuda = True

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None
        # self._finalize_allocation_log(size)

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def register_layer_transfer_counter(self, layer_transfer_counter):
        # required by the hicache backend.
        self.layer_transfer_counter = layer_transfer_counter

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer,  # "RadixAttention"
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        return None


class MockBaseTokenToKVPoolAllocator:
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: MockTokenToKVPool,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def clear(self):
        raise NotImplementedError()

    def alloc(self, need_size: int):
        raise NotImplementedError()

    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class MockTokenToKVPoolAllocator(MockBaseTokenToKVPoolAllocator):
    # from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: MockTokenToKVPool,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.is_not_in_free_group = True
        self.free_group = []

    def clear(self):
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def alloc(self, need_size: int):
        # It is not necessary to sort the slots during simulation.
        # if self.need_sort and need_size > len(self.free_pages):
        #     self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        self.free_pages = torch.cat((self.free_pages, free_index))


class MockPagedTokenToKVPoolAllocator(MockBaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: MockTokenToKVPool,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

        self.num_pages = size // page_size
        # self.seen_max_num_extend_tokens_next_power_of_2 = 1
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        # self.seen_max_num_extend_tokens_next_power_of_2 = max(
        #     self.seen_max_num_extend_tokens_next_power_of_2,
        #     next_power_of_2(extend_num_tokens),
        # )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_cpu(
            pre_lens_ptr=prefix_lens,
            seq_lens_ptr=seq_lens,
            last_loc_ptr=last_loc,
            free_page_ptr=self.free_pages,
            out_indices=out_indices,
            bs_upper=None,
            page_size=self.page_size,
            max_num_extend_tokens=None,
        )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_cpu(
            seq_lens_ptr=seq_lens,
            last_loc_ptr=last_loc,
            free_page_ptr=self.free_pages,
            out_indices=out_indices,
            bs_upper=None,  # Reserved parameter (not used in CPU)
            page_size=self.page_size,
        )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class MockTokenToKVPoolHost:
    KV_CACHE_BYTES: int = None
    KV_CACHE_BYTES_PER_LAYER: int = None

    MEMORY_READ_BANDWIDTH_BYTES: float = None
    MEMORY_WRITE_BANDWIDTH_BYTES: float = None

    # from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = False
        self.device = device

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert self.size > device_pool.size, (
            "The host memory should be larger than the device memory with the current protocol"
        )

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    def get_size_per_token(self):
        # MHA implementation
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        buffer = torch.empty(
            dims,
            dtype=self.dtype,
            device=self.device,
        )
        if self.pin_memory:
            torch.cuda.cudart().cudaHostRegister(
                buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
            )
        return buffer

    def est_bandwidth_batch(self, size_bytes_arr: np.ndarray, cat: str):
        if MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES is None:
            MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES = (
                ConfigManager.get_platform_config().memory_read_bandwidth
            )
        if MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES is None:
            MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES = (
                ConfigManager.get_platform_config().memory_write_bandwidth
            )
        x = size_bytes_arr.astype(np.float64)
        if cat == "H2D":
            eff = 0.85
            t0 = 6.67e-6
            bw = MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES * eff
        else:
            eff = 0.85
            t0 = 4e-6
            bw = MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES * eff
        return x * bw / (t0 * bw + x)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        # update global clock
        # Merge cache indices
        # https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/kvcacheio/transfer.cu#L713
        assert len(host_indices) == len(device_indices)
        num_indices = len(host_indices)

        host = np.asarray(host_indices, dtype=np.int64)
        dev = np.asarray(device_indices, dtype=np.int64)
        cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
        cut = np.flatnonzero(~cont) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, num_indices]
        seg_len = (ends - starts).astype(np.float64)

        if MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER is None:
            MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER = (
                ConfigManager.get_kv_cache_bytes_per_layer()
            )

        size_bytes_arr = seg_len * float(MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER)
        bandwidth_arr = self.est_bandwidth_batch(size_bytes_arr, cat="H2D")
        total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
        # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
        StateManager.inc_hicache_l2_load_dur(total_time_cost)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        # update global clock
        num_indices = len(host_indices)

        host = np.asarray(host_indices, dtype=np.int64)
        dev = np.asarray(device_indices, dtype=np.int64)
        cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
        cut = np.flatnonzero(~cont) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, num_indices]
        seg_len = (ends - starts).astype(np.float64)

        if MockTokenToKVPoolHost.KV_CACHE_BYTES is None:
            MockTokenToKVPoolHost.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()

        size_bytes_arr = seg_len * float(MockTokenToKVPoolHost.KV_CACHE_BYTES)
        bandwidth_arr = self.est_bandwidth_batch(size_bytes_arr, cat="D2H")
        total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
        # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead

        StateManager.inc_hicache_l2_backup_dur(total_time_cost)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        return torch.ones(size=(1, 1)) * index

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        pass

    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert need_size % self.page_size == 0, (
            "The requested size should be a multiple of the page size."
        )
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


def _pass_str_to_block_ids(hash_key):
    int_hash = int(hash_key, 16)
    MAX_18_DIGITS = 10**18
    return int_hash % MAX_18_DIGITS


class MockHiCacheStorage:
    def __init__(self, *args, **kwargs):
        if kvcm_py_optimizer is not None:
            logger.info("Using KVCM HiCache storage")
            self.init_kvcm()
        else:
            logger.info("Using common set HiCache storage")
            self.storage: set = set()
            self.storage_file_path: str = "/tmp/hisim/hicache/storage_keys.txt"
            os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)

            if os.path.exists(self.storage_file_path):
                with open(self.storage_file_path) as f:
                    line = f.readline()
                    while line:
                        self.storage.add(line.strip())
                        line = f.readline()

        if Envs.reset_hicache_storage():
            logger.info(
                "Cleared KV cache saved in the storage backend because the system environment variable (`HISIM_RESET_HICACHE_STORAGE`) is set."
            )
            with open(self.storage_file_path, "w") as f:
                pass

    def init_kvcm(self):
        # Initialize kvcm based on reference examples
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {self.temp_dir}")

        project_root = _CURRENT_DIR.parents[4]
        config_path = (
            project_root
            / "kv_cache_manager"
            / "optimizer"
            / "test"
            / "testdata"
            / "optimizer_startup_config_load.json"
        )
        config_path = config_path.resolve()
        self.config_loader = kvcm_py_optimizer.OptimizerConfigLoader()

        if not self.config_loader.load(str(config_path)):
            raise RuntimeError(f"Failed to load optimizer config from {config_path}")
        self.config = self.config_loader.config()
        self.storage_manager = kvcm_py_optimizer.OptimizerManager(self.config)
        self.storage_manager.Init()

        # Multi-instance not supported yet; using a single shared instance_id.
        self.instance_id = "3780643326877293460"

    def tearDown(self):
        if hasattr(self, "temp_dir"):
            import shutil

            logger.info("Cleaning up temporary directory")
            shutil.rmtree(self.temp_dir)

    def register_mem_pool_host(self, mem_pool_host):
        pass

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            return True
        self.storage.add(key)
        with open(self.storage_file_path, "a+") as f:
            f.write(key + "\n")
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        extra_info: HiCacheStorageExtraInfo = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if hasattr(self, "storage_manager"):
            if extra_info and extra_info.prefix_keys is not None:
                complete_prefix_hashs = extra_info.prefix_keys + keys
            else:
                complete_prefix_hashs = keys
            int_hash_keys = [
                _pass_str_to_block_ids(key) for key in complete_prefix_hashs
            ]
            # insert to kvcm
            trace_id = "1"
            write_timestamp = int(time.time() * 1000)
            write_token_ids = [1]
            self.storage_manager.WriteCache(
                self.instance_id,
                trace_id,
                write_timestamp,
                int_hash_keys,
                write_token_ids,
            )
            return True
        else:
            for key, value in zip(keys, values):
                if not self.set(key, value):
                    return False
            return True

    def exists(self, key: str) -> bool:
        return key in self.storage

    def batch_exists(self, keys: List[str], extra_info) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        if hasattr(self, "storage_manager"):
            int_hash_keys = [_pass_str_to_block_ids(key) for key in keys]
            # Call the kvcm interface to match L3 prefix
            trace_id = "2"
            read_timestamp = int(time.time() * 1000)
            read_token_ids = [1]
            mask_offset = 0
            res = self.storage_manager.GetCacheLocation(
                self.instance_id,
                trace_id,
                read_timestamp,
                int_hash_keys,
                read_token_ids,
                mask_offset,
            )
            logger.debug(f"{res.kvcm_hit_length=}")
            return res.kvcm_hit_length
        else:
            for i in range(len(keys)):
                if not self.exists(keys[i]):
                    return i
            return len(keys)

    def clear(self) -> bool:
        if hasattr(self, "storage_manager"):
            logger.info("Clear all storage cache in kvcm.")
            self.storage_manager.ClearAllCaches()
            return True
        else:
            self.storage.clear()
            with open(self.storage_file_path, "w"):
                pass
            return True
