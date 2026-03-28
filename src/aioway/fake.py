# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
from collections.abc import Callable
from typing import TypeIs

from torch import Tensor, _guards
from torch._subclasses import FakeTensor, FakeTensorMode

__all__ = ["enable", "enable_func", "is_fake_tensor", "is_real_tensor"]

_FAKE_MODE: FakeTensorMode = FakeTensorMode()


def to_fake_tensor(tensor: Tensor) -> FakeTensor:
    """
    Move a possibly real tensor to a fake Tensor
    """

    if is_fake_tensor(tensor):
        return tensor

    with enable() as mode:
        converter = mode.fake_tensor_converter
        return converter.from_real_tensor(mode, tensor)


def is_real_tensor(tensor: object) -> TypeIs[Tensor]:
    """
    Detect if a tensor is a normal tensor.
    """

    return isinstance(tensor, Tensor) and not is_fake_tensor(tensor)


def is_fake_tensor(tensor: object) -> TypeIs[FakeTensor]:
    """
    Detect if a tensor is a fake tensor.
    """

    return isinstance(tensor, FakeTensor)


def detect_fake_mode():
    """
    Get the current fake mode, is available.

    This can be used in an `if` or a `with`.
    """

    return _guards.detect_fake_mode()


@ctxl.contextmanager
def enable():
    """
    Enable `torch`'s fake mode s.t. we can do symbolic processing easily.

    Since fake mode doesn't nest (it seems), if fake mode is already on, yield that.
    """

    with _FAKE_MODE:
        yield _FAKE_MODE


def enable_func[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator on a function, s.t. when the function is being called, fake mode is enabled.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with enable():
            return func(*args, **kwargs)

    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__

    return wrapper
