from typing import Any, Optional, Tuple

# ctc_beam_search_decoder implemented here:
# https://github.com/tensorflow/tensorflow/blob/bb4e724f429ae5c9afad3a343dc1f483ecde1f74/tensorflow/python/ops/ctc_ops.py#L234
def ctc_beam_search_decoder(inputs : Any , sequence_length: Any, beam_width: int =100,
                            top_paths: int = 1, merge_repeated: bool = True) -> Tuple[Any, Any]: ...

# bidirectional_dynamic_rnn implemented here:
# https://github.com/tensorflow/tensorflow/blob/d8f9538ab48e3c677aaf532769d29bc29a05b76e/tensorflow/python/ops/rnn.py#L314
# TODO: types
# scope VariableScope
def bidirectional_dynamic_rnn(cell_fw: Any, cell_bw: Any, inputs: Any, sequence_length: Any = None,
                          initial_state_fw: Any = None, initial_state_bw: Any = None,
                          dtype: Any = None, parallel_iterations: Optional[int] = None,
                          swap_memory: Optional[bool]=False, time_major:Optional[bool]=False, scope: Any=None) -> Tuple[Any, Any]: ...

# ctc_loss implemented here:
# https://github.com/tensorflow/tensorflow/blob/bb4e724f429ae5c9afad3a343dc1f483ecde1f74/tensorflow/python/ops/ctc_ops.py#L32
# TODO: types
def ctc_loss(labels: Any, inputs: Any, sequence_length: Any,
             preprocess_collapse_repeated: bool=False,
             ctc_merge_repeated: bool=True, ignore_longer_outputs_than_inputs: bool=False,
             time_major: bool=True) -> Any: ...

# log_softmax implemented here:
# https://github.com/tensorflow/tensorflow/blob/95c8f92947c6a420b70759d9d0d7825f2f5de368/tensorflow/python/ops/nn_ops.py#L1741
# TODO: types
# Returns Tensor
def log_softmax(logits: Any, axis: Optional[int] = None, name: Optional[str]=None, dim: Optional[int]=None) -> Any: ...