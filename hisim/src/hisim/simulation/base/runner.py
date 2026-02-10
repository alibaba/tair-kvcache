from abc import ABC, abstractmethod


class BaseBenchmarkRunner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def benchmark():
        pass

    @abstractmethod
    def flush_cache(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass
