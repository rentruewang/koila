# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Callable, Iterator
from typing import Any

from torch import Tensor
from torch._tensor import Tensor

from aioway import _common as _common
from aioway.fn import Fn

from .fn import TensorFn

__all__ = ["UFunc1Thunk", "UFunc2Thunk", "GatherThunk"]


@_common.dcls_no_eq
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


@_common.dcls_no_eq
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

    @typing.override
    def _deps(self) -> Iterator[TensorFn]:
        # If it's a primitive or `Tensor`, do not recurse.
        yield self.arg


type BinaryTensorFnRhs = TensorFn | Tensor | int | float | bool


@_common.dcls_no_eq
class UFunc2Thunk(TensorFn):
    """
    Thunk for binary function.
    """

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

    @typing.override
    def _deps(self) -> Iterator[TensorFn]:
        # If it's a primitive or `Tensor`, do not recurse.
        yield self.left

        if isinstance(right := self.right, TensorFn):
            yield right


class GatherThunk(TensorFn):
    tensor: TensorFn | Tensor
    index: TensorFn | Tensor

    def __post_init__(self):
        super().__init__()

        # One of them should be `TensorFn`.
        if not isinstance(self.tensor, TensorFn) and isinstance(self.index, TensorFn):
            raise TypeError

    @typing.override
    def forward(self) -> Tensor:
        raise NotImplementedError

    @typing.override
    def _deps(self) -> Iterator[TensorFn]:
        if isinstance(self.tensor, TensorFn):
            yield self.tensor

        if isinstance(self.index, TensorFn):
            yield self.index
