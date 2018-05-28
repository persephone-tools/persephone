# Base tensorflow exception class
# implemented here: https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/framework/errors_impl.py#L32
class OpError(Exception): ...


class ResourceExhaustedError(OpError): ...

