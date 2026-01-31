import os
import logging
import shutil
import subprocess
import signal


class ModuleBase(object):
    def __init__(self, path_root=None):
        self.path_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '../')) if path_root is None else path_root
        self.install_root = os.path.join(self.path_root, 'install_root')
        self.local_path = os.path.join(self.install_root, 'usr/local')
        self.bin_path = os.path.join(self.local_path, 'bin')
        self.etc_path = os.path.join(self.local_path, 'etc')
        self.suspend_flag = False

    def remove_symlinks(self):
        # TODO
        file_list = []
        for f in file_list:
            ModuleBase.remove_symlink(f)

    def ld_library_cmd(self):
        ld_cmd = 'LD_LIBRARY_PATH=%s/lib64:%s/lib:$LD_LIBRARY_PATH PATH=%s:$PATH ' % (
            self.local_path, self.local_path, self.bin_path)
        return ld_cmd

    @staticmethod
    def create_symlink(source_dir, target_dir):
        if not os.path.exists(source_dir):
            logging.info("source file[{}] not exist, ignored.".format(source_dir))
            return
        if os.path.exists(target_dir):
            logging.info("target file[{}] exist, remove first.".format(target_dir))
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir, symlinks=True)

    @staticmethod
    def remove_symlink(file_path):
        if os.path.islink(file_path):
            try:
                src = os.readlink(file_path)
                os.unlink(file_path)
                shutil.copy(src, file_path)
            except FileNotFoundError:
                logging.info("file[{}] not exist, maybe unlink already.".format(file_path))

    def _get_pid(self, module_name, extra=None):
        cmd = 'ps uxww | grep %s | grep -v grep' % (module_name)
        if extra:
            if type(extra) is str:
                cmd = cmd + ' | grep %s' % extra
            if type(extra) is list:
                for s in extra:
                    cmd = cmd + ' | grep %s' % s
        p = subprocess.Popen(cmd, shell=True, close_fds=False, stdout=subprocess.PIPE)
        out, err = p.communicate()

        pids = []
        for line in out.splitlines():
            parts = line.split()
            pids.append(int(parts[1]))

        return pids

    def _wait_exit(self, module_name, pid):
        while True:
            if not self._get_pid(module_name, extra=['defunct', '%s' % pid]):
                break

    @staticmethod
    def _stop(pid=None):
        if pid is None:
            return
        os.kill(pid, signal.SIGTERM)

    @staticmethod
    def _suspend(pid=None):
        if pid is None:
            return
        if isinstance(pid, int):
            cmd = 'kill -STOP %d' % pid
            os.system(cmd)
        if isinstance(pid, str):
            cmd = 'kill -STOP %s' % pid
            os.system(cmd)

    @staticmethod
    def _resume(pid=None):
        if pid is None:
            return
        if isinstance(pid, int):
            cmd = 'kill -CONT %d' % pid
            os.system(cmd)
        if isinstance(pid, str):
            cmd = 'kill -CONT %s' % pid
            os.system(cmd)

    def kill_stop(self):
        if hasattr(self, 'pid'):
            ModuleBase._stop(self.pid)
            self.pid = None
        if hasattr(self, 'pids'):
            for i in range(len(self.pids)):
                ModuleBase._stop(self.pids[i])
                self.pids[i] = None

    def suspend(self):
        if not self.suspend_flag:
            if hasattr(self, 'pid'):
                ModuleBase._suspend(self.pid)
                self.suspend_flag = True
            if hasattr(self, 'pids'):
                for pid in self.pids:
                    ModuleBase._suspend(pid)
                self.suspend_flag = True

    def resume(self):
        if self.suspend_flag:
            if hasattr(self, 'pid'):
                ModuleBase._resume(self.pid)
                self.suspend_flag = False
            if hasattr(self, 'pids'):
                for pid in self.pids:
                    ModuleBase._resume(pid)
                self.suspend_flag = False
