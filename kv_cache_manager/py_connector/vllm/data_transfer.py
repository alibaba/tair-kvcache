import time
import threading
from concurrent.futures.thread import ThreadPoolExecutor

from typing import Any

import torch
from kv_cache_manager.client.pybind import kvcm_py_client

from kv_cache_manager.py_connector.common.tp_coordinator import CoordinateMsgSerializer, TpCoordinatorClient, \
    CoordinateMessage, SendBlockFinishedEvent, LoadBlockFinishedEvent
from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.types import KVCacheInfo
from kv_cache_manager.py_connector.kernel import batch_gather_scatter_helper
from kv_cache_manager.py_connector.kernel.gather_scatter_helper import CopyBufferAllocator


class MultiResult:
    """多任务结果管理类
    
    用于管理多个异步任务的结果, 当所有任务完成时触发回调
    """
    def __init__(self, size: int, callback):
        self._size: int = size
        self._results = [None] * size
        self._lock = threading.Lock()
        self._finished_num: int = 0
        self._finished_callback = callback

    def submit_result(self, idx: int, result):
        with self._lock:
            assert self._results[idx] is None
            self._results[idx] = result
            self._finished_num += 1
            if self._finished_num == self._size:
                self._finished_callback(self._results)


class DataTransferManager:
    """KVCache数据传输核心类
    
    负责实际的KV缓存保存和加载操作, 包括:
    1. 保存任务（save_task）
    2. 加载任务（load_task）
    3. 回调创建（_create_save_done_callback, _create_load_done_callback）
    """
    
    def __init__(self, 
                 kvcache_info: KVCacheInfo,
                 manager_block_size: int,
                 copy_buffer_allocator: CopyBufferAllocator,
                 transfer_client: Any,
                 coordinator_client: TpCoordinatorClient,
                 extra_config: Any):
        """
        初始化KV数据传输器
        
        Args:
            kvcache_info: KV缓存信息
            manager_block_size: instance的block_size
            copy_buffer_allocator: 复制缓冲区分配器
            transfer_client: 传输客户端
            coordinator_client: 协调器客户端
            extra_config: 额外配置
        """
        self._kvcache_info = kvcache_info
        self._manager_block_size = manager_block_size
        self._copy_buffer_allocator = copy_buffer_allocator
        self._transfer_client = transfer_client
        self._coordinator_client = coordinator_client
        self._extra_config = extra_config

        # 创建内部线程池执行器
        self._io_executor = self._create_io_executor()
        
        # 保存和加载流
        self._save_stream = torch.cuda.Stream()
        self._load_stream = torch.cuda.Stream()
    
    def _create_io_executor(self) -> ThreadPoolExecutor:
        """创建IO线程池执行器"""
        from concurrent.futures import ThreadPoolExecutor
        
        # 初始化线程池，设置线程名和初始化函数
        def init_worker():
            import torch
            torch.cuda.set_device(self._kvcache_info.device)
        
        return ThreadPoolExecutor(
            max_workers=32, 
            thread_name_prefix="kvcm_io_",
            initializer=init_worker
        )
    
    def submit_task(self, func, *args, **kwargs):
        """提交任务到内部线程池
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            Future对象
        """
        return self._io_executor.submit(func, *args, **kwargs)

    def load_task(self, multi_result: MultiResult, task_idx, remote_uris, block_token_indices):
        """加载任务
        
        Args:
            multi_result: 多任务结果管理器
            task_idx: 任务索引
            remote_uris: 远程URI列表
            block_token_indices: 块令牌索引列表
        """
        logger.debug("load remote_uris:%s, block_token_indices:%s", remote_uris, block_token_indices)

        copy_buffer_indices = self._copy_buffer_allocator.alloc_buffer_idx_blocking(len(remote_uris))
        copy_buffers = self._copy_buffer_allocator.get_buffer_by_idx(copy_buffer_indices)

        buffers = []
        for copy_buffer in copy_buffers:
            buffer = kvcm_py_client.BlockBuffer()
            iovs = []
            iov = kvcm_py_client.Iov()
            iov.type = kvcm_py_client.MemoryType.CPU
            iov.base = copy_buffer.data_ptr()
            iov.size = copy_buffer.nbytes
            iov.ignore = False
            iovs.append(iov)
            buffer.iovs = iovs
            buffers.append(buffer)
        logger.debug("start transfer")
        transfer_result = self._transfer_client.LoadKvCaches(remote_uris, buffers)
        logger.debug("done transfer,result:%s", transfer_result)
        if transfer_result == kvcm_py_client.ClientErrorCode.ER_OK:
            with torch.cuda.stream(self._load_stream):
                batch_gather_scatter_helper.batch_scatter_kv_caches(
                    self._kvcache_info.all_kvcache_ptr_tensor_gpu,
                    self._copy_buffer_allocator._raw_buffer,
                    block_token_indices,
                    copy_buffer_indices,
                    self._manager_block_size,
                    self._kvcache_info.per_token_per_layer_dim_size,
                )

                copy_done_event = torch.cuda.Event()
                copy_done_event.record(self._load_stream)
            copy_done_event.synchronize()

            logger.debug("done scatter")
        else:
            logger.warning("load task failed, remote_uris:%s, block_token_indices:%s, transfer_result:%s",
                           remote_uris,
                           block_token_indices, transfer_result)
        self._copy_buffer_allocator.free_buffer(copy_buffer_indices)
        multi_result.submit_result(task_idx, [transfer_result] * len(remote_uris))

    def create_load_done_callback(self, req_id, tp_rank, epoch, local_block_ids):
        """创建加载完成回调函数

        Args:
            req_id: 请求ID
            tp_rank: TP rank
            epoch
            local_block_ids: 本地块ID列表

        Returns:
            回调函数
        """
        def generate_message(task_results):
            failed_block_idxs = []
            idx = 0
            for task_result in task_results:
                for block_result in task_result:
                    if block_result != kvcm_py_client.ClientErrorCode.ER_OK:
                        failed_block_idxs.append(local_block_ids[idx])
                    idx += 1

            msg = CoordinateMessage(
                time.time(),
                LoadBlockFinishedEvent(request_id=req_id, tp_rank=tp_rank,
                                       epoch=epoch, failed_block_idxs=failed_block_idxs)
            )
            self._coordinator_client.send(CoordinateMsgSerializer.dumps(msg))

        return generate_message

    def save_task(self, multi_result: MultiResult, task_idx, remote_uris, block_token_indices,
                  kvcache_ready_event: torch.cuda.Event):
        """保存任务
        
        Args:
            multi_result: 多任务结果管理器
            task_idx: 任务索引
            remote_uris: 远程URI列表
            block_token_indices: 块令牌索引列表
            kvcache_ready_event: KV缓存就绪事件
        """
        logger.debug("save remote_uris:%s, block_token_indices:%s", remote_uris, block_token_indices)

        with torch.cuda.stream(self._save_stream):
            kvcache_ready_event.wait()
            copy_buffer_indices = self._copy_buffer_allocator.alloc_buffer_idx_blocking(len(remote_uris))
            batch_gather_scatter_helper.batch_gather_kv_caches(
                self._kvcache_info.all_kvcache_ptr_tensor_gpu,
                self._copy_buffer_allocator._raw_buffer,
                block_token_indices,
                copy_buffer_indices,
                self._manager_block_size,
                self._kvcache_info.per_token_per_layer_dim_size,
            )
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(self._save_stream)

        copy_done_event.synchronize()

        logger.debug("done gather")

        copy_buffers = self._copy_buffer_allocator.get_buffer_by_idx(copy_buffer_indices)
        buffers = []
        for copy_buffer in copy_buffers:
            buffer = kvcm_py_client.BlockBuffer()
            iovs = []
            iov = kvcm_py_client.Iov()
            iov.type = kvcm_py_client.MemoryType.CPU
            iov.base = copy_buffer.data_ptr()
            iov.size = copy_buffer.nbytes
            iov.ignore = False
            iovs.append(iov)
            buffer.iovs = iovs
            buffers.append(buffer)
        logger.debug("start transfer")

        transfer_result = self._transfer_client.SaveKvCaches(remote_uris, buffers)
        logger.debug("done transfer,result:%s", transfer_result)
        if transfer_result[0] != kvcm_py_client.ClientErrorCode.ER_OK:
            logger.warning("save task failed, remote_uris:%s, block_token_indices:%s, transfer_result:%s", remote_uris,
                           block_token_indices, transfer_result)

        self._copy_buffer_allocator.free_buffer(copy_buffer_indices)
        # TODO: submit uri when enable local alloc
        multi_result.submit_result(task_idx, [transfer_result[0]] * len(remote_uris))

    def create_save_done_callback(self, req_id, tp_rank, write_session_id):
        """创建保存完成回调函数
        
        Args:
            req_id: 请求ID
            tp_rank: TP rank
            write_session_id: 写入会话ID
            
        Returns:
            回调函数
        """
        def generate_message(task_results):
            is_successes = []
            # TODO: report uri when enable local alloc
            # remote_uris = []
            for task_result in task_results:
                for block_result in task_result:
                    if block_result != kvcm_py_client.ClientErrorCode.ER_OK:
                        is_successes.append(False)
                        # remote_uris.append(None)
                    else:
                        is_successes.append(True)
                        # remote_uris.extend(future_result[1])

            msg = CoordinateMessage(
                time.time(),
                SendBlockFinishedEvent(request_id=req_id, tp_rank=tp_rank,
                                       write_session_id=write_session_id,
                                       is_success_list=is_successes)
            )
            self._coordinator_client.send(CoordinateMsgSerializer.dumps(msg))

        return generate_message
