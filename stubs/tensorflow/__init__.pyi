from typing import Any

from . import train

class gpu_options:
    def __init__(self):
        self.allow_growth: bool

class ConfigProto:
    def __init__(self, log_device_placement: bool) -> None:
        self.gpu_options: gpu_options
        #self.gpu_options.allow_growth: bool

class Graph:
    pass

class Session:
    def __init__(self, graph: Graph = None, config: ConfigProto = None) -> None:
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass
