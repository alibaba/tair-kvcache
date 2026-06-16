import os
import json
import numpy as np
import asyncio
from typing import AsyncGenerator, Optional
from tqdm import tqdm

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    FakeRequest,
)
from schedule_simulator.schedule_emulator.utils import calc_metrics
from schedule_simulator.dataset import MultiTurnConversationDataset
from schedule_simulator.schedule_emulator.block_id_converter import convert_block_ids


class BenchmarkEmulator:
    def __init__(
        self,
        config: BenchmarkConfig,
        request_queue: asyncio.Queue,
        response_queue: asyncio.Queue,
        use_real_token_ids: bool = False,
        kvcm_block_size: Optional[int] = None,
        page_size: Optional[int] = None,
    ):
        self.config = config
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.use_real_token_ids = use_real_token_ids
        self.kvcm_block_size = kvcm_block_size
        self.page_size = page_size
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
        with open(self.config.dataset_path) as f:
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
                req: dict = json.loads(line)
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
                    new_req = FakeRequest(
                        id=id,
                        input_token_length=int(req.get("input_length")),
                        output_token_length=int(req.get("output_length")),
                        last_event_time=req.get("timestamp")
                        / 1e3,  # convert to seconds
                        device_cache_hit_length=int(
                            req.get("device_cache_hit_length", 0)
                        ),
                        host_cache_hit_length=int(
                            req.get("host_cache_hit_length", 0)
                        ),
                        disk_cache_hit_length=int(
                            req.get("disk_cache_hit_length", 0)
                        ),
                        origin_input_ids=self._maybe_convert_block_ids(
                            req.get("block_ids")
                        ),
                    )
                    if req.get("instance_id"):
                        new_req.extra_key = req["instance_id"]
                    input_requests.append(new_req)
                id += 1
                line = f.readline()

        if self.config.num_prompts is not None:
            input_requests = input_requests[
                : min(self.config.num_prompts, len(input_requests))
            ]

        input_requests.sort(key=lambda x: x.last_event_time)
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
