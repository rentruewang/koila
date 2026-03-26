# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
from collections.abc import Callable
from typing import TypeIs

from torch import Tensor, _guards
from torch._subclasses import FakeTensor, FakeTensorMode

__all__ = ["fake_mode_scope", "fake_mode_func", "is_fake_tensor", "is_real_tensor"]


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


@ctxl.contextmanager
def fake_mode_scope():
    """
    Enable `torch`'s fake mode s.t. we can do symbolic processing easily.

    Since fake mode doesn't nest (it seems), if fake mode is already on, yield that.
    """

    if fake := _guards.detect_fake_mode():
        yield fake

    else:
        with FakeTensorMode() as fake:
            yield fake


def fake_mode_func[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator on a function, s.t. when the function is being called, fake mode is enabled.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with fake_mode_scope():
            return func(*args, **kwargs)

    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__

    return wrapper
