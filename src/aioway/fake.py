# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
from typing import TypeIs

from torch import Tensor
from torch._subclasses import FakeTensor, FakeTensorMode

__all__ = ["enable_fake_mode", "is_fake_tensor", "is_real_tensor"]


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
def enable_fake_mode():
    """
    Enable `torch`'s fake mode s.t. we can do symbolic processing easily.
    """

    with FakeTensorMode() as fake:
        yield fake
