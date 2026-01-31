import random
import time
import socket
import logging
import math


class AllocatedPortsHolder(object):
    def __init__(self, exclusive_socket, port_start, sockets):
        self.exclusive_socket = exclusive_socket
        self.port_start = port_start
        self.port_num = len(sockets)
        self.sockets = sockets

    def __del__(self):
        self.release_socket()
        self.do_release(self.exclusive_socket)
        logging.info(f'__del__ for ports {self.get_ports()}')

    def nums(self):
        return self.port_num

    def get_ports(self):
        return list(range(self.port_start, self.port_start + self.nums()))

    def release_socket(self):
        self.sockets, sockets = [], self.sockets
        sockets and logging.info(f'release socket for ports {self.get_ports()}')
        for s in sockets:
            self.do_release(s)

    @staticmethod
    def do_release(s):
        if s:
            try:
                s.shutdown()
            except BaseException:
                pass
            try:
                s.close()
            except BaseException:
                pass


class RangedPortUtil(object):
    def __init__(self, range_from, range_to):
        self.range_from = range_from or 10000
        self.range_to = range_to or 32767

    def get_unused_ports(self, num, retry=20):
        holder = None
        num += 1
        assert num > 0
        assert self.range_from < self.range_to - num
        for i in range(retry):
            port = random.randint(0, (self.range_to - self.range_from) // num) * num + self.range_from
            port = int(min(port, self.range_to - num))
            sockets = []
            for i in range(num):
                s = self.listen_port(port + i)
                if not s:
                    break
                sockets.append(s)
            holder = AllocatedPortsHolder(sockets[0] if sockets else None, port + 1, sockets[1:])
            if len(sockets) == num:
                logging.info(f'hold socket for ports {holder.get_ports()}')
                break
            time.sleep(1)
        return holder

    def listen_port(self, port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", port))
            return s
        except OSError as e:
            logging.info("port [%d] already in use, retry" % port)
            s.close()
        return None
