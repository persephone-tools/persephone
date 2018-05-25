from typing import Any, Dict, Optional

from . import errors
from . import train
from . import nn

class dtype: ...


# Integer types
class number(): ...
class integer(number): ...
class signedinteger(integer): ...
class int8(signedinteger): ...
class int16(signedinteger): ...
class int32(signedinteger): ...
class int64(signedinteger): ...

class unsignedinteger(integer): ...
class uint8(unsignedinteger): ...
class uint16(unsignedinteger): ...
class uint32(unsignedinteger): ...
class uint64(unsignedinteger): ...

class floating(number): ...
class float16(floating): ...
class float32(floating): ...
class float64(floating): ...


class dtypes:
    float32 = float32


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


# Original function definition for edit_distance here:
# https://github.com/tensorflow/tensorflow/blob/faff6f2a60a01dba57cf3a3ab832279dbe174798/tensorflow/python/ops/array_ops.py#L2049
# return type is Tensor
def edit_distance(hypothesis: Any, truth: Any, normalize: Optional[bool]=True, name: Optional[str]="edit_distance") -> Any: ...

# Original function definition for global_variables_initializer here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/variables.py#L1565
def global_variables_initializer() -> Any: ...

# Original function definition for reset_default_graph here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/framework/ops.py#L5531
def reset_default_graph() -> Graph: ...


# Original function definition for placeholder here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/array_ops.py#L1693
# TODO: improve types
def placeholder(dtype: Any, shape: Any = None, name: Optional[str] = None) -> Any: ...

# Original function definition for sparse_placeholder here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/array_ops.py#L1749
# TODO: improve types
def sparse_placeholder(dtype: Any, shape: Any = None, name: Optional[str] = None) -> Any: ...


# Original function definition for shape here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/array_ops.py#L197
# TODO: improve types. return type of None here is a hack
# input is `Tensor` or `SparsTensor`
# out_type is an optional integral data-type (`int32` or `int64`).
# returns a `Tensor` of type specified by `out_type`
def shape(input: Any, name: Optional[str] = None, out_type: Any = None) -> Any: ...

# # Original function definition for truncated_normal here:
# https://github.com/tensorflow/tensorflow/blob/70cd9ed2d2ea37a6da6f813a99b32c03e90736a4/tensorflow/python/ops/random_ops.py#L139
def truncated_normal(shape, mean: Any=0.0, stddev: Any=1.0, dtype: Any=dtypes.float32, seed: Any=None, name: Optional[str]=None) -> Any: ...