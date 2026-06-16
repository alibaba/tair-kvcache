from abc import ABC, abstractmethod
import asyncio

from schedule_simulator.schedule_emulator.types import (
    FakeRequest,
    SchedulerConfig,
    PlatformConfig,
    IterationStats,
)


class GlobalValues:
    clock: float = 0
    last_batch_run_time: float = 0
    iteration: int = 0

    @property
    def last_batch_start(self) -> float:
        return self.clock - self.last_batch_run_time

    def reset(self):
        self.clock = 0
        self.last_batch_run_time = 0
        self.iteration = 0


class ScheduleEmulator(ABC):
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        platform_config: PlatformConfig,
        request_queue: asyncio.Queue,
        response_queue: asyncio.Queue,
        name: str,
    ):
        self.scheduler_config: SchedulerConfig = scheduler_config
        self.platform_config: PlatformConfig = platform_config
        self.request_queue: asyncio.Queue[FakeRequest] = request_queue
        self.response_queue: asyncio.Queue[FakeRequest] = response_queue
        self.name = name

        self.global_values = GlobalValues()
        self.iter_stats: list[IterationStats] = []

    @abstractmethod
    def event_loop(self):
        pass

    def get_iteration_stats(self) -> list[IterationStats]:
        return self.iter_stats
