# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import _subclasses as _S

__all__ = ["enable", "enable_func", "is_fake_tensor", "is_real_tensor"]


@dcls.dataclass
class FakeModeRc:
    """
    Do "reference counting" for fake mode.
    """

    mode: _S.FakeTensorMode = dcls.field(default_factory=_S.FakeTensorMode)
    "The fake mode instance that shall be entered."

    count: int = 0
    "The enter count."

    @ctxl.contextmanager
    def __call__(self):
        with self.mode, self._count_mode():
            yield self.mode

    @ctxl.contextmanager
    def _count_mode(self):
        try:
            self.count += 1
            yield
        finally:
            self.count -= 1

    def active(self) -> bool:
        return self.count != 0


_FAKE_MODE = FakeModeRc()


def to_fake_tensor(tensor: torch.Tensor) -> _S.FakeTensor:
    """
    Move a possibly real tensor to a fake torch.Tensor
    """

    if is_fake_tensor(tensor):
        return tensor

    with enable() as mode:
        converter = mode.fake_tensor_converter
        return converter.from_real_tensor(mode, tensor)


def is_real_tensor(tensor: object) -> typing.TypeIs[torch.Tensor]:
    """
    Detect if a tensor is a normal tensor.
    """

    return isinstance(tensor, torch.Tensor) and not is_fake_tensor(tensor)


def is_fake_tensor(tensor: object) -> typing.TypeIs[_S.FakeTensor]:
    """
    Detect if a tensor is a fake tensor.
    """

    return isinstance(tensor, _S.FakeTensor)


def is_enabled() -> _S.FakeTensorMode | None:
    """
    Get the current fake mode, is available.

    This can be used in an `if` or a `with`.
    """

    if _FAKE_MODE.active():
        return _FAKE_MODE.mode
    else:
        return None


@ctxl.contextmanager
def enable():
    """
    Enable `torch`'s fake mode s.t. we can do symbolic processing easily.

    Since fake mode doesn't nest (it seems), if fake mode is already on, yield that.
    """

    with _FAKE_MODE():
        yield _FAKE_MODE.mode


def enable_func[**P, T](func: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
    """
    Decorator on a function, s.t. when the function is being called, fake mode is enabled.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with enable():
            return func(*args, **kwargs)

    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__

    return wrapper
