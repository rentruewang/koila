# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Callable, Iterator
from typing import Any, override

from torch import Tensor
from torch._tensor import Tensor

from aioway import _common
from aioway.fn import Fn

from .fn import TensorFn

__all__ = ["thunk1"]


def thunk1(func: Callable[[Tensor], Tensor], fn: TensorFn):
    return UFunc1Thunk(func=func, arg=fn)


class AnyThunk(TensorFn):
    "Represents some computation that is deferred."

    __match_args__ = "func", "args", "kwargs"

    def __init__(
        self, func: Callable[..., Tensor], *args: Fn[Any], **kwargs: Fn[Any]
    ) -> None:
        super().__init__()

        self._func = func

        for arg in self.args:
            if not isinstance(arg, Fn):
                raise TypeError(f"{arg} is not a `Thunk`.")

        for key, value in self.kwargs.items():
            if not isinstance(value, Fn):
                raise TypeError(f"{key}={value} is not a `Thunk`")

        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return _common.format_function(self.func, *self.args, **self.kwargs)

    @typing.override
    def forward(self) -> Tensor:
        args = [arg.do() for arg in self.args]
        kwargs = {key: val.do() for key, val in self.kwargs.items()}
        return self.func(*args, **kwargs)

    @typing.override
    def _deps(self) -> Iterator[Fn[object]]:
        yield from self.args
        yield from self.kwargs.values()

    @property
    def func(self) -> Callable[..., Tensor]:
        return self._func

    @property
    def args(self) -> tuple[Fn[Any], ...]:
        return self._args

    @property
    def kwargs(self) -> dict[str, Fn[Any]]:
        return self._kwargs


class UFunc1Thunk(TensorFn):
    """
    Thunk for unary function.
    """

    __match_args__ = "func", "arg"

    def __init__(self, func: Callable[[Tensor], Tensor], arg: TensorFn) -> None:
        super().__init__()
        self._func = func
        self._arg = arg

    @typing.override
    def forward(self) -> Tensor:
        return self.func(self.arg.do())

    @property
    def func(self):
        return self._func

    @property
    def arg(self):
        return self._arg

    @override
    def _deps(self) -> Iterator[TensorFn]:
        # If it's a primitive or `Tensor`, do not recurse.
        if isinstance(self.arg, TensorFn):
            yield self.arg


type BinaryTensorFnRhs = TensorFn | Tensor | int | float | bool


class UFunc2Thunk(TensorFn):
    pass
