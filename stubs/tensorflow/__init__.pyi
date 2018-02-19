from typing import Any

class gpu_options:
    def __init__(self):
        self.allow_growth: bool

class ConfigProto:
    def __init__(self):
        self.gpu_options: gpu_options
        #self.gpu_options.allow_growth: bool
