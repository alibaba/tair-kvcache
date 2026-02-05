import os
import logging
import shutil
import subprocess
import signal
import sys
import time
import socket

from typing import List, Optional
from integration_test.testlib.ranged_port_util import RangedPortUtil
from integration_test.testlib.module_base import ModuleBase


class WorkerEnv(ModuleBase):
    def __init__(self, workdir, path_root, binary_name='kv_cache_manager_bin'):
        ModuleBase.__init__(self, path_root)
        self.binary_name = binary_name
        self.parameters = {}
        self.init(workdir)

    def init(self, workdir):
        self.set_debug_mode(True)
        self.set_workdir(workdir)
        self.set_binary_path(self.install_root)
        self.ip = socket.gethostbyname(socket.gethostname())
        self.port_range_from = None
        self.port_range_to = None
        self.ports_holder = None

    def gen_start_cmd(self, **kwargs):
        start_cmd = '''
        kv_cache_manager_bin
        --env kvcm.service.rpc_port=%(rpc_port)d
        --env kvcm.service.http_port=%(http_port)d
        --env kvcm.service.admin_rpc_port=%(admin_rpc_port)d
        --env kvcm.service.admin_http_port=%(admin_http_port)d
        --env kvcm.service.enable_debug_service=true
        --env kvcm.logger.log_level=5
        -d
        '''
        # 0: auto, 1: fatal, 2: error, 3: warn, 4: info, 5: debug
        replace = {
            'binpath': self.binary_path,
            'rpc_port': self.rpc_port,
            'http_port': self.http_port,
            'admin_rpc_port': self.admin_rpc_port,
            'admin_http_port': self.admin_http_port,
        }

        self.update_parameters(**kwargs)
        cmd_complete = ' '.join([self.ld_library_cmd(),
                                self.__replace(start_cmd, replace), self.__gen_parameters_str(), ])
        cmd_complete += ' > stdout 2> stderr'
        return cmd_complete

    def __replace(self, cmd, replace_args):
        return cmd.replace('\n', ' ') % replace_args

    def __gen_parameters_str(self):
        cmd_head = ""
        for k, v in self.parameters.items():
            cmd_head = ' '.join([cmd_head, "--env", str(k) + "=" + str(v)])
        return cmd_head

    def allocate_free_ports(self):
        return RangedPortUtil(self.port_range_from, self.port_range_to).get_unused_ports(4)

    def update_ports(self):
        self.ports_holder = self.allocate_free_ports()
        self.ports_holder.release_socket()
        free_ports = self.ports_holder.get_ports()
        self.set_port(free_ports[0], free_ports[1], free_ports[2], free_ports[3])

    def set_port_range(self, port_range_from, port_range_to):
        self.port_range_from = port_range_from
        self.port_range_to = port_range_to

    def set_port(self, rpc_port, http_port, admin_rpc_port, admin_http_port):
        self.rpc_port = rpc_port
        self.http_port = http_port
        # 在弹内，Meta服务和Admin接口端口保持一直
        # self.admin_rpc_port = admin_rpc_port
        self.admin_rpc_port = rpc_port
        self.admin_http_port = admin_http_port

        logging.info("rpc_port = %s, http_port = %s" % (self.rpc_port, self.http_port))

    def update_parameters(self, **kwargs):
        for k, v in kwargs.items():
            self.add_env_parameter(k, v)

    def add_env_parameter(self, key, value):
        if type(value) is dict:
            value = r"""'%s'""" % str(value).replace('\'', '\"')
        self.parameters.update({key: value})

    def set_debug_mode(self, debug_mode):
        self.debug_mode = debug_mode

    def set_mode(self, mode):
        self.mode = mode

    def set_workdir(self, workdir):
        self.workdir = workdir

    def set_binary_path(self, binary_path):
        self.binary_path = binary_path

    def set_rpc_port(self, rpc_port):
        self.rpc_port = rpc_port

    def set_http_port(self, http_port):
        self.http_port = http_port


class Worker(ModuleBase):
    def __init__(self, worker_id, worker_env: WorkerEnv):
        self.worker_id = worker_id
        self.env = worker_env
        self.worker_name = 'kv_cache_manager_' + str(worker_id)

    def start_worker(self, **kwargs):
        self.env.update_ports()
        start_cmd = self.env.gen_start_cmd(**kwargs)
        logging.info('start [%s] worker at:[%s] with cmd:[%s]', self.worker_name, self.env.workdir, start_cmd)
        p = subprocess.Popen(start_cmd, shell=True, cwd=self.env.workdir)
        p.communicate()
        ret = p.returncode
        if ret != 0:
            logging.error('start worker failed, cwd:[%s] cmd:[%s] ret:[%d]' % (self.env.workdir, start_cmd, ret))
            logging.error("error msg: %s", open(os.path.join(self.env.workdir, "stderr")).read())
            logging.error("stdout msg: %s", open(os.path.join(self.env.workdir, "stdout")).read())
            logging.error("path: " + str(sys.path))
            return False
        logging.info('finish start [%s] worker at:[%s] with cmd:[%s]', self.worker_name, self.env.workdir, start_cmd)

        time.sleep(2)
        return True

    def start_worker_get_pid(self, **kwargs) -> Optional[int]:
        if not os.path.exists(self.env.workdir):
            os.makedirs(self.env.workdir)

        retry = 5
        while retry > 0:
            if not self.start_worker(**kwargs):
                return None
            pid = self.get_pid()
            if not pid:
                logging.error("get pid failed")
                retry -= 1
                time.sleep(0.1)
                continue
            return pid
        return None

    def get_work_id(self):
        return self.worker_id

    def get_work_dir(self):
        return self.env.workdir

    def stop_worker(self, pid=None):
        logging.info(f"stop worker {self.worker_id}, pid={pid}")
        self._stop(pid)
        self._wait_exit('kv_cache_manager_bin', pid)

    def get_pid(self):
        pids = self._get_pid('kv_cache_manager_bin', extra="rpc_port=%d" % int(self.env.rpc_port))
        return pids[0] if len(pids) > 0 else None
