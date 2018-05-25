from typing import Any, Dict, Optional

from . import train
from . import errors

class gpu_options:
    def __init__(self):
        self.allow_growth: bool

class ConfigProto:
    def __init__(self, log_device_placement: bool) -> None:
        self.gpu_options: gpu_options
        #self.gpu_options.allow_growth: bool

class Graph:
    pass

class BaseSession:
    #TODO: options is of type RunOption, run_metadata is of type RunMetadata
    # Return type is option of:
    # single graph element if fetches is a single graph element  OR
    # list of graph elements if fetches is a list of single graph elements OR
    # a dictionary
    # Leaving it as Any for now
    def run(self, fetches: Any, feed_dict: Optional[Dict[Any, Any]] = None, run_options: Any = None, run_metadata: Any = None) -> Any: ...

    def close(self) -> None: ...

class Session(BaseSession):
    def __init__(self, graph: Graph = None, config: ConfigProto = None) -> None:
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass
    def close(self) -> None: ...

# Original function definition for global_variables_initializer here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/variables.py#L1565
def global_variables_initializer() -> Any: ...