# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Callable, Iterator
from typing import Any

from torch import Tensor

from aioway import _common
from aioway.fn import Fn

from .bases import TensorFn

__all__ = ["AnyThunkTensorFn"]


class AnyThunkTensorFn(TensorFn):
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
