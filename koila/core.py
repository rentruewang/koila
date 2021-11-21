from __future__ import annotations

import functools
from typing import Any, Callable, Dict, NamedTuple, Tuple, Type, overload
from unittest.mock import Mock

import torch
from torch import Tensor

from .protocols import Lazy, LazyFunction, Runnable


class LazyTensor(Lazy[Tensor], Mock):
    def __init__(self, data: Tensor | Runnable[Tensor] | Lazy[Tensor]) -> None:
        super().__init__(data)

    # Magic methods

    def __pos__(self) -> LazyTensor:
        return _pos(self)

    def __neg__(self) -> LazyTensor:
        return _neg(self)

    def __add__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _add(self, other)

    def __radd__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _add(other, self)

    def __sub__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _sub(self, other)

    def __rsub__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _sub(other, self)

    def __mul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mul(self, other)

    def __rmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mul(other, self)

    def __truediv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _div(self, other)

    def __rtruediv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _div(other, self)

    def __floordiv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _div(self, other, rounding_mode="trunc")

    def __rfloordiv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _div(other, self, rounding_mode="trunc")

    def __pow__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _pow(self, other)

    def __rpow__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _pow(other, self)

    def __mod__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mod(self, other)

    def __rmod__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mod(other, self)

    def __matmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _matmul(self, other)

    def __rmatmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _matmul(other, self)

    def __getattr__(self, name: str) -> LazyFunction:
        func = getattr(torch, name)
        method = getattr(Tensor, name)
        wrapper = functools.wraps(method)
        partial = functools.partial(func, self)
        return LazyFunction(wrapper(partial))

    def __bool__(self) -> bool:
        return bool(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    # Arithmetic operations

    def positive(self) -> LazyTensor:
        return _pos(self)

    def neg(self) -> LazyTensor:
        return _neg(self)

    def add(self, other: Tensor | LazyTensor, *, alpha: float = 1) -> LazyTensor:
        return _add(self, other, alpha=alpha)

    def sub(self, other: Tensor | LazyTensor, *, alpha: float = 1) -> LazyTensor:
        return _sub(self, other, alpha=alpha)

    def mul(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mul(self, other)

    def div(self, other: Tensor | LazyTensor, *, rounding_mode=None) -> LazyTensor:
        return _div(self, other, rounding_mode=rounding_mode)

    def pow(self, exponent: Tensor | LazyTensor) -> LazyTensor:
        return _pow(self, exponent)

    def remainder(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mod(self, other)

    def matmul(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _matmul(self, other)

    # Slicing operations

    @overload
    def min(self) -> Lazy[Tensor]:
        ...

    @overload
    def min(self, dim: int, keepdim: bool = False) -> Lazy[MinMaxResult]:
        ...

    @overload
    def min(self, other: Tensor | LazyTensor) -> Lazy[Tensor]:
        ...

    def min(self, *args: Any, **kwargs: Any) -> Lazy[Any]:
        return _min(self, *args, **kwargs)

    @overload
    def max(self) -> Lazy[Tensor]:
        ...

    @overload
    def max(self, dim: int, keepdim: bool = False) -> Lazy[MinMaxResult]:
        ...

    @overload
    def max(self, other: Tensor | LazyTensor) -> Lazy[Tensor]:
        ...

    def max(self, *args: Any, **kwargs: Any) -> Lazy[Any]:
        return _max(self, *args, **kwargs)

    def minimum(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _minimum(self, other)

    def maximum(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _maximum(self, other)

    def item(self) -> bool | int | float:
        return self.run().item()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Any], ...],
        args: Tuple[Tensor | LazyTensor, ...] = (),
        kwargs: Dict[str, Tensor | LazyTensor] | None = None,
    ) -> LazyTensor:
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(typ, (LazyTensor, Tensor, int, float)) for typ in types):
            return NotImplemented

        return LazyTensor(LazyFunction(func)(*args, **kwargs))


def _lazy_eval(func: Callable[..., Any], *args: Any, **kwargs: Any) -> LazyTensor:
    return LazyTensor(LazyFunction(func)(*args, **kwargs))


def _pos(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.positive, input)


def _neg(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.neg, input)


def _add(
    input: Tensor | LazyTensor, other: Tensor | LazyTensor, *, alpha: float = 1
) -> LazyTensor:
    return _lazy_eval(torch.add, input, other, alpha=alpha)


def _sub(
    input: Tensor | LazyTensor, other: Tensor | LazyTensor, *, alpha: float = 1
) -> LazyTensor:
    return _lazy_eval(torch.sub, input, other, alpha=alpha)


def _mul(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.mul, input, other)


def _div(
    input: Tensor | LazyTensor, other: Tensor | LazyTensor, *, rounding_mode=None
) -> LazyTensor:
    return _lazy_eval(torch.div, input, other, rounding_mode=rounding_mode)


def _pow(input: Tensor | LazyTensor, exponent: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.pow, input, exponent)


def _mod(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.fmod, input, other)


def _matmul(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.matmul, input, other)


class MinMaxResult(NamedTuple):
    values: Tensor
    indices: Tensor


@overload
def _min(input: Tensor | LazyTensor) -> Lazy[Tensor]:
    ...


@overload
def _min(
    input: Tensor | LazyTensor, dim: int, keepdim: bool = False
) -> Lazy[MinMaxResult]:
    ...


@overload
def _min(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> Lazy[Tensor]:
    ...


def _min(input: Tensor | LazyTensor, *args: Any, **kwargs: Any) -> Lazy[Any]:
    return LazyFunction(torch.min)(input, *args, **kwargs)


@overload
def _max(input: Tensor | LazyTensor) -> Lazy[Tensor]:
    ...


@overload
def _max(
    input: Tensor | LazyTensor, dim: int, keepdim: bool = False
) -> Lazy[MinMaxResult]:
    ...


@overload
def _max(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> Lazy[Tensor]:
    ...


def _max(input: Tensor | LazyTensor, *args: Any, **kwargs: Any) -> Lazy[Any]:
    return LazyFunction(torch.max)(input, *args, **kwargs)


def _minimum(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.minimum, input, other)


def _maximum(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.maximum, input, other)
