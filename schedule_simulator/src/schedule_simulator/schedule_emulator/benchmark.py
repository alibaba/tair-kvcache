import os
import json
try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads
import numpy as np
import asyncio
from typing import AsyncGenerator, Optional
from tqdm import tqdm

import logging

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    FakeRequest,
    TimelineMode,
)
from schedule_simulator.schedule_emulator.utils import calc_metrics
from schedule_simulator.dataset import MultiTurnConversationDataset
from schedule_simulator.schedule_emulator.block_id_converter import convert_block_ids

logger = logging.getLogger("kunlun.schedule_simulator")


class BenchmarkEmulator:
    def __init__(
        self,
        config: BenchmarkConfig,
        request_queue: asyncio.Queue,
        response_queue: asyncio.Queue,
        use_real_token_ids: bool = False,
        kvcm_block_size: Optional[int] = None,
        page_size: Optional[int] = None,
        timeline_mode: TimelineMode = TimelineMode.DISABLED,
    ):
        self.config = config
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.use_real_token_ids = use_real_token_ids
        self.kvcm_block_size = kvcm_block_size
        self.page_size = page_size
        self.timeline_mode = timeline_mode
        self.response_lock = asyncio.Lock()
        self.global_clock = 0
        self.pending_requests: dict[
            int, tuple[asyncio.Future, FakeRequest]
        ] = {}  # id -> (future, request)
        self.response_results: list[FakeRequest] = []

    async def request_func(self, req: FakeRequest) -> FakeRequest:
        # Set the created time of the request before adding it to the queue.
        # If the request_rate != float("inf"), the time of the last event might be greater than the global clock.
        req.last_event_time = max(self.global_clock, req.last_event_time)
        self.request_queue.put_nowait(req)

        future = asyncio.get_event_loop().create_future()
        async with self.response_lock:
            self.pending_requests[req.id] = (future, req)

        try:
            result = await future
            return result
        finally:
            async with self.response_lock:
                if req.id in self.pending_requests:
                    del self.pending_requests[req.id]

    def _maybe_convert_block_ids(self, block_ids):
        """Convert block_ids if data_block_size differs from page_size."""
        if block_ids is None:
            return None
        data_bs = self.config.data_block_size
        target_bs = self.page_size
        if data_bs is None or target_bs is None or data_bs == target_bs:
            return block_ids
        return convert_block_ids(block_ids, data_bs, target_bs)

    @staticmethod
    def _to_int64(val) -> int:
        """Convert a value to int64. Handles both hex strings and integers."""
        if isinstance(val, int):
            return val % (2**63)
        return int(val, 16) % (2**63)

    def _get_block_ids(self, req: dict):
        """Get block_ids, auto-converting from input_block_hash_ids if needed."""
        block_ids = req.get("block_ids")
        if block_ids is not None:
            return self._maybe_convert_block_ids(block_ids)
        # Fallback: convert from input_block_hash_ids (may be hex strings or ints)
        raw_ids = req.get("input_block_hash_ids")
        if raw_ids:
            # Check first element to determine format
            if raw_ids and isinstance(raw_ids[0], str):
                int_ids = self._hex_list_to_int64(raw_ids)
            else:
                int_ids = raw_ids  # already int list
            return self._maybe_convert_block_ids(int_ids)
        return None

    @staticmethod
    def _hex_list_to_int64(hex_list):
        """Batch convert hex strings to int64 using numpy (3x faster than loop)."""
        hex_concat = ''.join(hex_list)
        byte_data = bytes.fromhex(hex_concat)
        arr = np.frombuffer(byte_data, dtype='>u8') % np.uint64(2**63)
        return arr.astype(np.int64).tolist()

    @staticmethod
    def _normalize_timestamp(ts) -> float:
        """Normalize timestamp to seconds. Auto-detect format:
        - ts > 1e15: epoch microseconds (e.g. 1780923600400578) -> /1e6
        - ts > 1e12: epoch milliseconds (e.g. 1780923600400) -> /1e3
        - 1e9 < ts <= 1e12: epoch seconds (e.g. 1780923600.4) -> use directly
        - ts <= 1e9: sim.jsonl milliseconds (e.g. 1000.0) -> /1e3
        """
        if ts is None:
            return 0.0
        ts = float(ts)
        if ts > 1e15:
            # Epoch microseconds (e.g. 1780923600400578)
            return ts / 1e6
        elif ts > 1e12:
            # Epoch milliseconds (e.g. 1780923600400)
            return ts / 1e3
        elif ts > 1e9:
            # Epoch seconds (Unix time for years 2001+)
            return ts
        else:
            # Legacy sim.jsonl format: milliseconds
            return ts / 1e3

    def _validate_timeline_fields(self, first_req: dict):
        """Validate that required timeline fields exist in the first record."""
        errors = []
        pods = first_req.get("pods")
        if not pods or not isinstance(pods, list) or len(pods) == 0:
            errors.append("'pods' field missing or empty")
        prefill = first_req.get("prefill")
        if not prefill or not isinstance(prefill, dict):
            errors.append("'prefill' field missing or not a dict")
        elif "first_latency_ms" not in prefill:
            errors.append("'prefill.first_latency_ms' field missing")
        if errors:
            msg = (
                f"Timeline mode '{self.timeline_mode.value}' requires timeline fields "
                f"in dataset, but validation failed on first record: "
                + "; ".join(errors)
                + f". File: {self.config.dataset_path}"
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(
            "Timeline field validation passed (pods=%s, first_latency_ms=%.1f)",
            pods[0], float(prefill["first_latency_ms"])
        )

    async def get_request(self) -> AsyncGenerator[FakeRequest, None]:
        if self.config.dataset_path is not None and os.path.exists(
            self.config.dataset_path
        ):
            async for item in self.sample_request_from_file():
                yield item
        elif (
            self.config.min_input_length is not None
            and self.config.max_input_length is not None
            and self.config.min_output_length is not None
            and self.config.max_output_length is not None
        ):
            async for item in self.gen_random_request():
                yield item
        else:
            raise ValueError("No dataset path or min/max input/output length specified")

    async def gen_random_request(self) -> AsyncGenerator[FakeRequest, None]:
        assert (
            self.config.max_input_length is not None
            and self.config.max_output_length is not None
        )

        input_requests: list[FakeRequest] = []
        for i in range(self.config.num_prompts):
            input_len = np.random.randint(
                self.config.min_input_length, self.config.max_input_length
            )
            output_len = np.random.randint(
                self.config.min_output_length, self.config.max_output_length
            )
            if (
                self.config.min_prefix_disk_hit_rate is not None
                and self.config.max_prefix_disk_hit_rate is not None
            ):
                hit_rate = np.random.uniform(
                    self.config.min_prefix_disk_hit_rate,
                    self.config.max_prefix_disk_hit_rate,
                )
                disk_hit_len = int(hit_rate * input_len)
            else:
                disk_hit_len = 0

            if (
                self.config.min_prefix_host_hit_rate is not None
                and self.config.max_prefix_host_hit_rate is not None
            ):
                hit_rate = np.random.uniform(
                    self.config.min_prefix_host_hit_rate,
                    self.config.max_prefix_host_hit_rate,
                )
                host_hit_len = int(hit_rate * input_len)
            else:
                host_hit_len = 0

            input_requests.append(
                FakeRequest(
                    id=i,
                    input_token_length=input_len,
                    output_token_length=output_len,
                    disk_cache_hit_length=disk_hit_len,
                    host_cache_hit_length=host_hit_len,
                )
            )

        yield_delay_time = 0
        for index, request in enumerate(input_requests):
            if index % self.config.num_instances != 0:
                continue
            request.last_event_time = yield_delay_time
            yield request

            if self.config.request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / self.config.request_rate)
            # The next request will be sent after the interval.
            # await asyncio.sleep(interval)
            yield_delay_time = yield_delay_time + interval

    async def sample_request_from_file(self) -> AsyncGenerator[FakeRequest, None]:
        input_requests = []
        with open(self.config.dataset_path, "rb") as f:
            line = f.readline()
            id = 0
            index = 0
            while line:
                # Instance router: round robin
                if index % self.config.num_instances != 0:
                    line = f.readline()
                    index += 1
                    continue
                index += 1
                req: dict = _json_loads(line)
                # Filter by pod prefix if specified
                if self.config.pod_prefix:
                    pods = req.get("pods", [])
                    if not pods or not pods[0].startswith(self.config.pod_prefix):
                        line = f.readline()
                        continue
                if self.use_real_token_ids:
                    new_req = FakeRequest(
                        id=id,
                        input_token_length=len(req.get("input_ids", [])),
                        output_token_length=len(req.get("output_ids", [])),
                        last_event_time=req.get("timestamp")
                        / 1e3,  # convert to seconds
                        origin_input_ids=req.get("input_ids", []),
                        output_ids=req.get("output_ids", []),
                    )
                    if self.kvcm_block_size:
                        new_req.input_token_length = (
                            new_req.input_token_length * self.kvcm_block_size
                        )
                        new_req.output_token_length = (
                            new_req.output_token_length * self.kvcm_block_size
                        )
                        if new_req.output_token_length == 0:
                            new_req.output_token_length = 1
                    input_requests.append(new_req)
                else:
                    # Validate timeline fields on first record
                    if id == 0 and self.timeline_mode != TimelineMode.DISABLED:
                        self._validate_timeline_fields(req)

                    new_req = FakeRequest(
                        id=id,
                        input_token_length=int(req.get("input_length")),
                        output_token_length=int(req.get("output_length", 1)),
                        last_event_time=self._normalize_timestamp(
                            req.get("timestamp")
                        ),
                        device_cache_hit_length=int(
                            req.get("device_cache_hit_length", 0)
                        ),
                        host_cache_hit_length=int(
                            req.get("host_cache_hit_length", 0)
                        ),
                        disk_cache_hit_length=int(
                            req.get("disk_cache_hit_length", 0)
                        ),
                        origin_input_ids=self._get_block_ids(req),
                    )
                    if req.get("instance_id"):
                        new_req.extra_key = req["instance_id"]
                    elif req.get("pods"):
                        new_req.extra_key = req["pods"][0]
                    # Inject timeline replay fields
                    if self.timeline_mode != TimelineMode.DISABLED:
                        pods = req.get("pods", [])
                        if pods:
                            new_req.timeline_pod_name = pods[0]
                        prefill_info = req.get("prefill")
                        if prefill_info:
                            fl_ms = prefill_info.get("first_latency_ms")
                            if fl_ms is not None:
                                new_req.timeline_prefill_ms = float(fl_ms)
                            cached_tokens = prefill_info.get("cached_input_tokens")
                            if cached_tokens is not None:
                                new_req.timeline_cached_tokens = int(cached_tokens)
                    input_requests.append(new_req)
                id += 1
                line = f.readline()

        if self.config.num_prompts is not None:
            input_requests = input_requests[
                : min(self.config.num_prompts, len(input_requests))
            ]

        input_requests.sort(key=lambda x: x.last_event_time)

        # Normalize timestamps to zero-based (subtract minimum arrival time)
        if input_requests:
            min_time = input_requests[0].last_event_time
            if min_time > 0:
                for req in input_requests:
                    req.last_event_time -= min_time

        for request in input_requests:
            yield request

    async def handle_responses(
        self,
        pbar: Optional[tqdm] = None,
    ):
        while True:
            try:
                resp: FakeRequest = await self.response_queue.get()
                async with self.response_lock:
                    if resp.id in self.pending_requests:
                        future, req = self.pending_requests.pop(resp.id)
                        # update the global clock
                        self.global_clock = max(self.global_clock, req.last_event_time)
                        if not future.done():
                            future.set_result(req)
                            if pbar:
                                pbar.update(1)
            except asyncio.CancelledError:
                break
            await asyncio.sleep(0)

    async def benchmark(self) -> dict:
        if self.config.dataset is not None and isinstance(
            self.config.dataset, MultiTurnConversationDataset
        ):
            return await self.benchmark_multi_turn()
        else:
            return await self.benchmark_normal()

    async def benchmark_normal(self) -> dict:
        semaphore = None
        if self.config.max_concurrency is not None:
            instance_concurrency = (
                self.config.max_concurrency + self.config.num_instances - 1
            ) // self.config.num_instances
            semaphore = asyncio.Semaphore(instance_concurrency)

        async def limited_request_func(request_func_input):
            if semaphore is None:
                return await self.request_func(request_func_input)
            async with semaphore:
                return await self.request_func(request_func_input)

        tasks: list[asyncio.Task] = []
        async for req in self.get_request():
            tasks.append(
                asyncio.create_task(limited_request_func(request_func_input=req))
            )

        pbar = None if self.config.disable_tqdm else tqdm(total=len(tasks))
        resp_task = asyncio.create_task(self.handle_responses(pbar))
        self.response_results = await asyncio.gather(*tasks)
        # Cancel the response task when all requests have been received from the response queue.
        resp_task.cancel()

        metrics = calc_metrics(self.response_results)
        return metrics

    async def benchmark_multi_turn(self) -> dict:
        if self.config.dataset is None or not isinstance(
            self.config.dataset, MultiTurnConversationDataset
        ):
            raise ValueError(
                "dataset should be provided and should be an instance of MultiTurnConversationDataset for multi-turn benchmark."
            )
        # Group the requests by session_id.
        sessions: dict[Optional[int], list[FakeRequest]] = {}
        req_id = 0
        for req in self.config.dataset:
            sessions.setdefault(req.session_id, []).append(
                FakeRequest(
                    id=req_id,
                    session_id=req.session_id,
                    input_token_length=req.input_length,
                    output_token_length=req.output_length,
                    origin_input_ids=req.token_ids,
                    output_ids=req.output_ids,
                )
            )
            req_id += 1

        # Process each session sequentially, but process different sessions concurrently.
        semaphore = None
        if self.config.max_concurrency is not None:
            instance_concurrency = (
                self.config.max_concurrency + self.config.num_instances - 1
            ) // self.config.num_instances
            semaphore = asyncio.Semaphore(instance_concurrency)

        async def process_session(session_reqs: list[FakeRequest]):
            session_responses = []
            for req in session_reqs:
                resp = await self.request_func(req)
                session_responses.append(resp)
            return session_responses

        async def limited_process_session(session_reqs):
            if semaphore is None:
                return await process_session(session_reqs)
            async with semaphore:
                return await process_session(session_reqs)

        tasks: list[asyncio.Task] = []
        for session_reqs in sessions.values():
            tasks.append(asyncio.create_task(limited_process_session(session_reqs)))

        pbar = (
            None if self.config.disable_tqdm else tqdm(total=len(self.config.dataset))
        )
        resp_task = asyncio.create_task(self.handle_responses(pbar))
        session_response_results = await asyncio.gather(*tasks)
        for session_resps in session_response_results:
            self.response_results.extend(session_resps)

        # Cancel the response task when all requests have been received from the response queue.
        resp_task.cancel()

        metrics = calc_metrics(self.response_results)
        return metrics

    def get_response_results(self) -> list[FakeRequest]:
        return self.response_results
