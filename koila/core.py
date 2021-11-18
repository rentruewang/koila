from __future__ import annotations

import dataclasses as dcls
import functools
import inspect
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Dict, Tuple, overload

import torch
from torch import Tensor

from .protocols import Runnable


@dataclass
class LazyFunction(Runnable):
    function: Callable[..., Tensor]
    args: Tuple[LazyTensor | Tensor | Any, ...] = dcls.field(default_factory=tuple)
    kwargs: Dict[str, LazyTensor | Tensor | Any] = dcls.field(default_factory=dict)

    def run(self) -> Tensor:
        args = [eager(arg) for arg in self.args]
        kwargs = {k: eager(w) for (k, w) in self.kwargs.items()}
        return self.function(*args, **kwargs)

    def tensor(self) -> LazyTensor:
        return LazyTensor(self)


@dataclass(init=False)
class LazyTensor(Runnable):
    data: Tensor | LazyFunction
    retain_forward: bool

    def __init__(
        self, data: Tensor | LazyFunction, retain_forward: bool = False
    ) -> None:
        self.data = data
        self.retain_forward = retain_forward

    def __getattr__(self, name: str) -> Callable[..., Any]:
        return self._getattr_global_func(name)

    def __add__(self, other: LazyTensor | Tensor | Number) -> LazyTensor:
        return add(self, other)

    def __radd__(self, other: LazyTensor | Tensor | Number) -> LazyTensor:
        return add(other, self)

    def __sub__(self, other: LazyTensor | Tensor | Number) -> LazyTensor:
        return add(self, other)

    def __rsub__(self, other: LazyTensor | Tensor | Number) -> LazyTensor:
        return sub(other, self)

    def __pow__(self, other: LazyTensor | Tensor | Number) -> LazyTensor:
        return pow(self, other)

    def __rpow__(self, other: LazyTensor | Tensor | Number) -> LazyTensor:
        return pow(other, self)

    def _getattr_global_func(self, name: str) -> Callable[..., Any]:
        if (
            (func := globals().get(name, None)) is not None
            and callable(func)
            and len(inspect.signature(func).parameters) >= 1
        ):
            wrapper = functools.wraps(func)
            partial = functools.partial(func, self)
            return wrapper(partial)

        raise AttributeError

    def run(self) -> Tensor:
        data = self.data
        if isinstance(data, Tensor):
            return data

        tensor = data.run()
        if not self.retain_forward:
            self.data = tensor
        return tensor


@overload
def lazy(tensor: LazyTensor | Tensor) -> LazyTensor:
    ...


@overload
def lazy(tensor: Number) -> Number:
    ...


@overload
def lazy(tensor: None) -> None:
    ...


def lazy(tensor: LazyTensor | Tensor | Number | None) -> LazyTensor | Number | None:
    if isinstance(tensor, Tensor):
        return LazyTensor(tensor)
    return tensor


@overload
def eager(tensor: LazyTensor | Tensor) -> Tensor:
    ...


@overload
def eager(tensor: Number) -> Number:
    ...


@overload
def eager(tensor: None) -> None:
    ...


def eager(tensor: LazyTensor | Tensor | Number | None) -> Tensor | Number | None:
    if isinstance(tensor, LazyTensor):
        return tensor.run()
    return tensor


def add(
    input: LazyTensor | Tensor | Number, other: LazyTensor | Tensor | Number
) -> LazyTensor:
    input = lazy(input)
    other = lazy(other)
    return LazyFunction(torch.add, (input, other)).tensor()


def sub(
    input: LazyTensor | Tensor | Number, other: LazyTensor | Tensor | Number
) -> LazyTensor:
    input = lazy(input)
    other = lazy(other)
    return LazyFunction(torch.sub, (input, other)).tensor()


def mul(
    input: LazyTensor | Tensor | Number, other: LazyTensor | Tensor | Number
) -> LazyTensor:
    input = lazy(input)
    other = lazy(other)
    return LazyFunction(torch.mul, (input, other)).tensor()


def div(
    input: LazyTensor | Tensor | Number, other: LazyTensor | Tensor | Number
) -> LazyTensor:
    input = lazy(input)
    other = lazy(other)
    return LazyFunction(torch.div, (input, other)).tensor()


def pow(
    input: LazyTensor | Tensor | Number, other: LazyTensor | Tensor | Number
) -> LazyTensor:
    input = lazy(input)
    other = lazy(other)
    return LazyFunction(torch.pow, (input, other)).tensor()


def all(input: LazyTensor | Tensor) -> LazyTensor:
    input = lazy(input)
    return LazyFunction(torch.all, (input,)).tensor()


def any(input: LazyTensor | Tensor) -> LazyTensor:
    input = lazy(input)
    return LazyFunction(torch.any, (input,)).tensor()


def amin(
    input: LazyTensor | Tensor, dim: int | Tuple[int, ...], keepdim: bool = False
) -> LazyTensor:
    input = lazy(input)
    return LazyFunction(torch.amin, (input, dim), {"keepdim": keepdim}).tensor()


def amax(
    input: LazyTensor | Tensor, dim: int | Tuple[int, ...], keepdim: bool = False
) -> LazyTensor:
    input = lazy(input)
    return LazyFunction(torch.amax, (input, dim), {"keepdim": keepdim}).tensor()


def minimum(input: LazyTensor | Tensor, other: LazyTensor | Tensor) -> LazyTensor:
    input = lazy(input)
    other = lazy(other)
    return LazyFunction(torch.minimum, (input, other)).tensor()


def maximum(input: LazyTensor | Tensor, other: LazyTensor | Tensor) -> LazyTensor:
    input = lazy(input)
    other = lazy(other)
    return LazyFunction(torch.maximum, (input, other)).tensor()


@overload
def clamp(input: LazyTensor | Tensor, min: Tensor, max: Tensor) -> LazyTensor:
    ...


@overload
def clamp(input: LazyTensor | Tensor, min: Number, max: Number) -> LazyTensor:
    ...


def clamp(
    input: LazyTensor | Tensor, min: Tensor | Number, max: Tensor | Number
) -> LazyTensor:
    input = lazy(input)
    return LazyFunction(torch.clamp, (input, min, max)).tensor()
