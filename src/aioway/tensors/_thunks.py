# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import torch

from aioway import _common, fn

from . import tensors

__all__ = ["UFunc1Thunk", "UFunc2Thunk", "GatherThunk"]


@_common.dcls_no_eq
class AnyThunk(tensors.BasicPreviewFn):
    "Represents some computation that is deferred."

    __match_args__ = "func", "args", "kwargs"

    func: cabc.Callable[..., torch.Tensor]
    args: tuple[fn.Fn[typing.Any], ...]
    kwargs: dict[str, fn.Fn[typing.Any]]

    def __post_init__(self) -> None:
        super().__init__()

        for arg in self.args:
            if not isinstance(arg, fn.Fn):
                raise TypeError(f"{arg} is not a `Thunk`.")

        for key, value in self.kwargs.items():
            if not isinstance(value, fn.Fn):
                raise TypeError(f"{key}={value} is not a `Thunk`")

    def __repr__(self) -> str:
        return _common.format_function(self.func, *self.args, **self.kwargs)

    @typing.override
    def do(self) -> torch.Tensor:
        args = [arg.do() for arg in self.args]
        kwargs = {key: val.do() for key, val in self.kwargs.items()}
        return self.func(*args, **kwargs)

    @typing.override
    def deps(self):
        return tuple(self._deps())

    def _deps(self) -> cabc.Iterator[fn.Fn[object]]:
        yield from self.args
        yield from self.kwargs.values()


@_common.dcls_no_eq
class UFunc1Thunk(tensors.BasicPreviewFn):
    """
    Thunk for unary function.
    """

    func: cabc.Callable[[torch.Tensor], torch.Tensor]
    arg: tensors.TensorFn

    def __post_init__(self):
        super().__init__()

        if not callable(self.func):
            raise TypeError

        if not isinstance(self.arg, tensors.TensorFn):
            raise TypeError

    @typing.override
    def do(self) -> torch.Tensor:
        return self.func(self.arg.do())

    @typing.override
    def deps(self):
        # If it's a primitive or `torch.Tensor`, do not recurse.
        return (self.arg,)


type BinaryTensorFnRhs = tensors.TensorFn | torch.Tensor | int | float | bool


@_common.dcls_no_eq
class UFunc2Thunk(tensors.BasicPreviewFn):
    """
    Thunk for binary function.
    """

    func: cabc.Callable[[torch.Tensor, typing.Any], torch.Tensor]
    left: tensors.TensorFn
    right: BinaryTensorFnRhs

    def __post_init__(self) -> None:

        if not callable(self.func):
            raise TypeError

        if not isinstance(self.left, tensors.TensorFn):
            raise TypeError

        if not isinstance(
            self.right, tensors.TensorFn | torch.Tensor | int | float | bool
        ):
            raise TypeError

        super().__init__()

    @typing.override
    def do(self) -> torch.Tensor:
        left_do = self.left.do()

        match right := self.right:
            case tensors.TensorFn():
                return self.func(left_do, right.do())
            case _:
                return self.func(left_do, right)

    @typing.override
    def deps(self):
        return tuple(self._deps())

    def _deps(self):
        # If it's a primitive or `torch.Tensor`, do not recurse.
        yield self.left

        if isinstance(right := self.right, tensors.TensorFn):
            yield right


@_common.dcls_no_eq
class GatherThunk(tensors.BasicPreviewFn):
    tensor: tensors.TensorFn | torch.Tensor
    index: tensors.TensorFn | torch.Tensor

    def __post_init__(self):
        super().__init__()

        # One of them should be `tensors.TensorFn`.
        if not isinstance(self.tensor, tensors.TensorFn) and isinstance(
            self.index, tensors.TensorFn
        ):
            raise TypeError

    @typing.override
    def do(self) -> torch.Tensor:
        tensor = (
            self.tensor.do()
            if isinstance(self.tensor, tensors.TensorFn)
            else self.tensor
        )
        index = (
            self.index.do() if isinstance(self.index, tensors.TensorFn) else self.index
        )
        return tensor[index]

    @typing.override
    def deps(self):
        return tuple(self._deps())

    def _deps(self) -> cabc.Iterator[tensors.TensorFn]:
        if isinstance(self.tensor, tensors.TensorFn):
            yield self.tensor

        if isinstance(self.index, tensors.TensorFn):
            yield self.index
