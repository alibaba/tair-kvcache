import unittest
import os
import shutil
import logging
import subprocess
from typing import List, Dict

from integration_test.testlib.worker_manager import WorkerManager
from integration_test.testlib.worker import WorkerEnv, Worker
from integration_test.testlib.module_base import ModuleBase


class TestBase(object):
    def init_default(self):
        # TODO(qisa.cb) 先强制INFO，方便排查
        logging.basicConfig(level=logging.INFO)
        self.clean_workdir()
        self.prepare_test_resource(1)
        self.start_worker()

    def cleanup(self):
        logging.info("test is finished, begin cleanup.")
        self.clean_test_resource()
        self.stop_worker()

    def prepare_test_resource(self, worker_num, work_dir=None, worker_mode='normal'):
        self._init_dirs(work_dir)
        self.worker_manager = WorkerManager()
        self.envs: List[WorkerEnv] = []
        self.mode = worker_mode

        port_range_from, port_range_to = self.get_hash_range(os.getcwd())
        for i in range(worker_num):
            env = WorkerEnv(workdir=os.path.join(self.workdir, 'worker_' + str(i)), path_root=self.workdir)
            env.set_port_range(port_range_from, port_range_to)
            env.set_mode(worker_mode)
            self.envs.append(env)
            logging.info(f"add worker {i} workdir: {env.workdir}")
            self.worker_manager.add_worker(Worker(i, env))

    def start_worker(self, **kwargs):
        self.assertTrue(self.worker_manager.start_all(**kwargs))

    def clean_test_resource(self):
        pass

    def stop_worker(self):
        self.worker_manager.stop_all()

    def get_hash_range(self, hash_str):
        # 8u ephemeral-port-range
        # $sysctl net.ipv4.ip_local_port_range
        # net.ipv4.ip_local_port_range = 20000    60999
        available_range_from = 10000
        available_range_to = 19999
        part_cnt = 16
        part_offset = (available_range_to - available_range_from + 1) // part_cnt
        hash_value = hash(hash_str)
        part_id = hash_value % part_cnt
        range_from = available_range_from + part_offset * part_id
        range_to = available_range_from + part_offset * (part_id + 1) - 1
        logging.info("case will use port range [%d, %d]" % (range_from, range_to))
        return range_from, range_to

    def get_workdir(self):
        return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')), self._testMethodName)

    def clean_workdir(self):
        workdir = self.get_workdir()
        if os.path.exists(workdir):
            shutil.rmtree(workdir)

    def _init_dirs(self, work_dir):
        self.workdir = work_dir if work_dir is not None else self.get_workdir()
        self.path_root = os.path.abspath(os.path.join(self.workdir, '../'))
        self.global_install_root = os.path.join(self.path_root, 'install_root')
        self.worker_install_root = os.path.join(self.workdir, 'install_root')

        ModuleBase.create_symlink(self.global_install_root, self.worker_install_root)
        logging.info(f"[WORKER ROOT DIR]: {self.workdir}")
