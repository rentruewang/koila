from __future__ import annotations
from types import FunctionType

import typing, functools
from typing import Any, Callable, ClassVar, Dict, Protocol, Set, Tuple, Type, TypeVar, Union

import torch
from torch import Tensor

from .protocols import Lazy, LazyFunction,Runnable

T = TypeVar("T")


class LazyTensor(Lazy[Tensor]):
    def __init__(self, data: Tensor | Runnable[Tensor] | Lazy) -> None:
        super().__init__(data)
        LazyTensor.init()


    def __add__(self, other: TensorLike) -> LazyTensor:
        return torch.add(self, other)   # type: ignore

    def __radd__(self, other: TensorLike) -> LazyTensor:
        return torch.add(other, self)   # type: ignore

    def __sub__(self, other: TensorLike) -> LazyTensor:
        return torch.sub(self, other)   # type: ignore

    def __rsub__(self, other: TensorLike) -> LazyTensor:
        return torch.sub(other, self)   # type: ignore

    def __mul__(self, other: TensorLike) -> LazyTensor:
        return torch.mul(self, other)   # type: ignore

    def __rmul__(self, other: TensorLike) -> LazyTensor:
        return torch.mul(other, self)   # type: ignore

    def __truediv__(self, other: TensorLike) -> LazyTensor:
        return torch.div(self, other)   # type: ignore

    def __rtruediv__(self, other: TensorLike) -> LazyTensor:
        return torch.div(other, self)   # type: ignore

    def __floordiv__(self, other: TensorLike) -> LazyTensor:
        return torch.trunc(self, other) # type: ignore

    def __rfloordiv__(self, other: TensorLike) -> LazyTensor:
        return torch.trunc(other, self) # type: ignore

    def __pow__(self, other: TensorLike) -> LazyTensor:
        return torch.pow(self, other)   # type: ignore

    def __rpow__(self, other: TensorLike) -> LazyTensor:
        return torch.pow(other, self)   # type: ignore

    def __matmul__(self, other: TensorLike) -> LazyTensor:
        return torch.matmul(self, other)    # type: ignore

    def __rmatmul__(self, other: TensorLike) -> LazyTensor:
        return torch.matmul(other, self)    # type: ignore

    def __getattr__(self, name: str) -> LazyFunction:
        func = getattr(torch, name)

        if func not in self._methods:
            raise AttributeError

        return LazyFunction(func)

    @classmethod
    def register(cls, func: Callable[..., Tensor]) -> None:
        cls._methods.add(func)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Any], ...],
        args: Tuple[TensorLike, ...] = (),
        kwargs: Dict[str, TensorLike] | None = None,
    ) -> LazyTensor:
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(typ, (LazyTensor, Tensor, int, float)) for typ in types):
            return NotImplemented

        if func not in cls._methods:
            return NotImplemented

        return LazyTensor(LazyFunction(func)(*args, **kwargs))

    _methods: ClassVar[Set[Callable[...,Tensor]]] = set()

    @classmethod
    def init(cls) -> None:
        cls.register(torch.add)
        cls.register(torch.sub)
        cls.register(torch.mul)
        cls.register(torch.div)
        cls.register(torch.trunc)
        cls.register(torch.multiply)
        cls.register(torch.divide)

        cls.register(torch.pow)
        cls.register(torch.abs)
        cls.register(torch.min)
        cls.register(torch.max)
        cls.register(torch.absolute)
        cls.register(torch.minimum)
        cls.register(torch.maximum)

        cls.register(torch.sin)
        cls.register(torch.cos)
        cls.register(torch.tan)
        cls.register(torch.asin)
        cls.register(torch.acos)
        cls.register(torch.atan)
        cls.register(torch.arcsin)
        cls.register(torch.arccos)
        cls.register(torch.arctan)

        cls.register(torch.matmul)
        cls.register(torch.mm)
        cls.register(torch.bmm)


TensorLike = Union[LazyTensor, Tensor, int, float]
