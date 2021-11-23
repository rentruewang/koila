from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from dataclasses import dataclass
from torch import Tensor
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    overload,
    runtime_checkable,
)

from numpy.core.fromnumeric import size

T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...

        ...

    @overload
    @abstractmethod
    def size(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    @overload
    def size(self, dim: int) -> int:
        ...

    @abstractmethod
    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.shape, 1)


@runtime_checkable
class CalculateShape(Protocol):
    def __call__(
        self, *args: Tuple[int, ...], **kwargs: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        ...


@dataclass(frozen=True)
class Lazy(Runnable[T]):
    data: T | Runnable[T]

    def __init__(self, data: T | Runnable[T] | Lazy[T]) -> None:
        if isinstance(data, Lazy):
            object.__setattr__(self, "data", data.data)
        else:
            object.__setattr__(self, "data", data)

    def run(self) -> T:
        data = self.data
        if isinstance(data, Runnable):
            return data.run()
        return data

    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        if dim is None:
            return ()
        else:
            raise ()[dim]


@dataclass(frozen=True)
class LazyFunction(Generic[V]):
    func: Callable[..., Tensor]
    shape: CalculateShape

    def __call__(self, *args: Any, **kwargs: Any) -> Lazy[Tensor]:
        lazy_args = tuple(Lazy(arg) for arg in args)
        lazy_kwargs = dict((k, Lazy(v)) for (k, v) in kwargs.items())

        shape_args = [arg.shape for arg in lazy_args]
        shape_kwargs = {k: v.shape for (k, v) in lazy_kwargs.items()}
        shape = self.shape(*shape_args, **shape_kwargs)

        return Lazy(Evaluation(self.func, shape, lazy_args, lazy_kwargs))

    def __get__(self, obj: V, objtype: Type[V]) -> Callable[..., Lazy[Tensor]]:
        assert isinstance(obj, objtype), [type(obj), objtype]
        if obj is None:
            return self
        else:
            return functools.partial(self, obj)

    def calculate_shape(self, *args: Lazy[Any], **kwargs: Lazy[Any]) -> Tuple[int, ...]:
        arg_shapes = [arg.shape for arg in args]
        kwarg_shapes = {k: v.shape for (k, v) in kwargs.items()}
        return self.shape(*arg_shapes, **kwarg_shapes)


@dataclass
class Evaluation(Runnable[Tensor]):
    func: Callable[..., Tensor]
    shape: Tuple[int, ...]
    args: Tuple[Lazy[Any], ...]
    kwargs: Dict[str, Lazy[Any]]

    def run(self) -> Tensor:
        real_args = [arg.run() for arg in self.args]
        real_kwargs = {k: v.run() for (k, v) in self.kwargs.items()}
        result = self.func(*real_args, **real_kwargs)
        assert result.shape == self.shape

        return result

    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        if dim is not None:
            return self.shape[dim]
        else:
            return self.shape
