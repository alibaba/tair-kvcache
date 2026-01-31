from integration_test.testlib.worker import Worker
from typing import List, Optional
import sys
import os
from integration_test.testlib.module_base import ModuleBase


class WorkerManager(ModuleBase):
    def __init__(self):
        self.workers: List[Worker] = []
        self.pids: List[Optional[int]] = []

    def get_worker(self, worker_id=0) -> Worker:
        return self.workers[worker_id]

    def add_worker(self, worker: Worker):
        self.workers.append(worker)
        self.pids.append(None)

    def start_all(self, **kwargs):
        for i in range(len(self.workers)):
            if not self.start_worker(i, **kwargs):
                return False
        return True

    def start_worker(self, idx, **kwargs):
        pid = self.workers[idx].start_worker_get_pid(**kwargs)
        if pid:
            self.pids[idx] = pid
            return True
        return False

    def stop_worker(self, idx):
        if self.pids[idx] is not None:
            self.workers[idx].stop_worker(self.pids[idx])
            self.pids[idx] = None

    def stop_all(self):
        for i in range(len(self.workers)):
            self.stop_worker(i)

    def suspend_worker(self, idx):
        self.suspend(self.pids[idx])

    def resume_worker(self, idx):
        self.resume(self.pids[idx])
