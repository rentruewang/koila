# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable, Iterator
from types import NotImplementedType
from typing import Any, override

from torch import Tensor
from torch._tensor import Tensor

from aioway import _common
from aioway.fn import Fn

from .fn import TensorFn

__all__ = ["thunk"]


@typing.overload
def thunk(func: Callable[[Tensor], Tensor], arg: TensorFn, /) -> UFunc1Thunk: ...


@typing.overload
def thunk(
    func: Callable[[Tensor, BinaryTensorFnRhs], Tensor], left: TensorFn, right: Any, /
) -> UFunc2Thunk: ...


@typing.overload
def thunk(*args) -> NotImplementedType: ...


def thunk(func, *args):
    try:
        return UFunc1Thunk(func, *args)
    except TypeError:
        pass

    try:
        return UFunc2Thunk(func, *args)
    except TypeError:
        pass

    return NotImplemented


@dcls.dataclass
class AnyThunk(TensorFn):
    "Represents some computation that is deferred."

    __match_args__ = "func", "args", "kwargs"

    func: Callable[..., Tensor]
    args: tuple[Fn[Any], ...]
    kwargs: dict[str, Fn[Any]]

    def __post_init__(self) -> None:
        super().__init__()

        for arg in self.args:
            if not isinstance(arg, Fn):
                raise TypeError(f"{arg} is not a `Thunk`.")

        for key, value in self.kwargs.items():
            if not isinstance(value, Fn):
                raise TypeError(f"{key}={value} is not a `Thunk`")

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


@dcls.dataclass
class UFunc1Thunk(TensorFn):
    """
    Thunk for unary function.
    """

    func: Callable[[Tensor], Tensor]
    arg: TensorFn

    def __post_init__(self):
        super().__init__()

        if not callable(self.func):
            raise TypeError

        if not isinstance(self.arg, TensorFn):
            raise TypeError

    @typing.override
    def forward(self) -> Tensor:
        return self.func(self.arg.do())

    @override
    def _deps(self) -> Iterator[TensorFn]:
        # If it's a primitive or `Tensor`, do not recurse.
        yield self.arg


type BinaryTensorFnRhs = TensorFn | Tensor | int | float | bool


@dcls.dataclass
class UFunc2Thunk(TensorFn):
    """
    Thunk for binary function.
    """

    __match_args__ = "func", "left", "right"

    func: Callable[[Tensor, Any], Tensor]
    left: TensorFn
    right: BinaryTensorFnRhs

    def __post_init__(self) -> None:
        super().__init__()

        if not callable(self.func):
            raise TypeError

        if not isinstance(self.left, TensorFn):
            raise TypeError

        if not isinstance(self.right, TensorFn | Tensor | int | float | bool):
            raise TypeError

    @typing.override
    def forward(self) -> Tensor:
        left_do = self.left.do()

        match right := self.right:
            case TensorFn():
                return self.func(left_do, right.do())
            case _:
                return self.func(left_do, right)

    @override
    def _deps(self) -> Iterator[TensorFn]:
        # If it's a primitive or `Tensor`, do not recurse.
        yield self.left

        if isinstance(right := self.right, TensorFn):
            yield right
